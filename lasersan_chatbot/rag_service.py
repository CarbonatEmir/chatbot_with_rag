from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import ollama
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .embeddings import embed_text
from .guardrails import NO_PRODUCT_FOUND_MESSAGE, REFUSAL_MESSAGE, is_product_question, sanitize_user_question
from .products_repo import (
    ProductRow,
    get_product_by_exact_name,
    list_all_products,
    list_product_names,
    vector_search_products,
)
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .query_filters import (
    NumericConstraint,
    interval_satisfies_constraint,
    parse_generic_constraint_from_question,
    parse_weight_constraint_from_question,
    parse_weight_grams_from_text,
    parse_weight_range_from_question,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Context formatters
# ──────────────────────────────────────────────

def _format_specs(specs: Optional[Dict[str, Any]], brief: bool = False) -> str:
    if not specs:
        return "  Özellik bilgisi yok."
    lines = []
    for k, v in specs.items():
        if v is None or str(v).strip() in ("", "None", "Belirtilmemiş", "Yok"):
            continue
        label = k.replace("_", " ").title()
        lines.append(f"  • {label}: {v}")
        if brief and len(lines) >= 6:
            lines.append("  ...")
            break
    return "\n".join(lines) if lines else "  Özellik bilgisi yok."


def _format_products_context(products: List[ProductRow], brief: bool = False) -> str:
    parts: List[str] = []
    for p in products:
        parts.append(f"ÜRÜN: {p.product_name}")
        if p.category:
            parts.append(f"  Kategori: {p.category}")
        if p.description:
            parts.append(f"  Açıklama: {p.description}")
        if p.specifications:
            parts.append(_format_specs(p.specifications, brief=brief))
        parts.append("---")
    return "\n".join(parts).strip()


# ──────────────────────────────────────────────
# Intent helpers
# ──────────────────────────────────────────────

def _is_list_all_query(q: str) -> bool:
    t = q.lower()
    return any(x in t for x in [
        "tüm cihaz", "tum cihaz", "tüm ürün", "tum urun",
        "hepsini listele", "adını listele", "adini listele",
        "bütün cihaz", "butun cihaz",
    ])


def _is_count_query(q: str) -> bool:
    t = q.lower()
    return any(x in t for x in [
        "kaç cihaz", "kac cihaz", "kaç ürün", "kac urun",
        "toplam cihaz", "toplam ürün", "toplam urun",
        "kaç tane", "kac tane",
    ])


def _is_all_specs_query(q: str) -> bool:
    """User wants specs for all products."""
    t = q.lower()
    has_all = any(x in t for x in ["tüm", "tum", "hep", "bütün", "butun", "hepsinin"])
    has_spec = any(x in t for x in ["özellik", "ozellik", "spec", "teknik", "bilgi"])
    return has_all and has_spec


def _is_category_filter(q: str) -> Optional[str]:
    """Returns category name if user filters by category, else None."""
    t = q.lower()
    categories = {
        "gece görüş": "Gece Görüş Sistemleri",
        "gece gorus": "Gece Görüş Sistemleri",
        "termal": "Termal El Dürbünü",
        "gimbal": "Gimbal",
        "radar": "Radar Sistemleri",
        "silah": "Silah Üstü Nişangahlar",
        "nişangah": "Silah Üstü Nişangahlar",
        "nisangah": "Silah Üstü Nişangahlar",
        "sürüş": "Sürüş Görüş Sistemleri",
        "surus": "Sürüş Görüş Sistemleri",
        "keşif": "Keşif ve Gözetleme Sistemleri",
        "kesif": "Keşif ve Gözetleme Sistemleri",
        "elektro-optik": "Elektro-Optik Nişangahlar",
        "elektro optik": "Elektro-Optik Nişangahlar",
        "optik haberleş": "Optik Haberleşme Sistemi",
        "renkli gece": "Renkli Gece Görüş Sistemleri",
    }
    for key, cat in categories.items():
        if key in t:
            return cat
    return None


# Keywords that indicate semantic/description-based search
# (not strict category filter — use vector search + description scan)
_SEMANTIC_TOPIC_HINTS: Dict[str, List[str]] = {
    "deniz": ["deniz", "naval", "maritime", "sualtı", "su altı", "gemi", "tekne", "sahil",
              "kıyı", "kiyi", "coast", "offshore"],
    "hava": ["hava", "aerial", "insansız hava", "iha", "uav", "drone"],
    "kara": ["kara", "ground", "araç", "bron", "tank", "piyade"],
    "kentsel": ["kentsel", "urban", "şehir", "sehir", "bina"],
    "sınır": ["sınır", "sinir", "border", "hudut"],
}


def _get_semantic_topics(q: str) -> List[str]:
    """Return list of semantic topic keywords found in question."""
    t = q.lower()
    found = []
    for topic, hints in _SEMANTIC_TOPIC_HINTS.items():
        if any(h in t for h in hints):
            found.append(topic)
    return found


# ──────────────────────────────────────────────
# Corrections persistence
# ──────────────────────────────────────────────

def save_pending_correction(
    engine: Engine,
    *,
    user_question: str,
    original_answer: str,
    correction_text: Optional[str],
    product_name: Optional[str],
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO pending_corrections
                    (user_question, original_answer, correction_text, product_name)
                VALUES (:q, :a, :c, :p);
            """),
            {"q": user_question, "a": original_answer, "c": correction_text, "p": product_name},
        )


def get_pending_corrections(engine: Engine) -> List[Dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, user_question, original_answer, correction_text, product_name, created_at
                FROM pending_corrections
                WHERE status = 'pending'
                ORDER BY created_at DESC;
            """)
        ).mappings().fetchall()
    return [dict(r) for r in rows]


def approve_correction(engine: Engine, correction_id: int, admin_note: str = "") -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE pending_corrections
                SET status='approved', admin_note=:note, reviewed_at=CURRENT_TIMESTAMP
                WHERE id=:id;
            """),
            {"id": correction_id, "note": admin_note},
        )


def apply_correction_to_product(
    engine: Engine,
    *,
    correction_id: int,
    product_name: str,
    correction_text: str,
    spec_updates: Optional[Dict[str, str]] = None,
    embedding_model: str,
) -> bool:
    """
    Apply an approved correction to a product:
    - Optionally updates individual spec fields in specifications JSON
    - Appends correction_text to description so LLM context is semantically enriched
    - Rebuilds the embedding with all description + spec content so vector search
      immediately reflects the corrected/learned knowledge
    - Marks the correction as applied in pending_corrections
    """
    from .embeddings import embed_text as _embed
    from .products_repo import get_product_by_exact_name, upsert_product

    pr = get_product_by_exact_name(engine, product_name.strip())
    if not pr:
        return False

    # Merge spec field updates (e.g. {"agirlik": "2.5 kg"}) into existing specs
    new_specs = dict(pr.specifications or {})
    if spec_updates:
        for k, v in spec_updates.items():
            if k.strip() and v.strip():
                new_specs[k.strip()] = v.strip()

    new_desc = (pr.description or "") + f"\n[Onaylı Düzeltme]: {correction_text}"

    # Build a rich embedding that includes ALL textual knowledge about the product:
    # product name, category, full description (with corrections), and every spec value.
    # This ensures semantic search can find this product even with varied query phrasing.
    spec_text = "\n".join(f"  {k}: {v}" for k, v in new_specs.items() if v)
    embed_input = "\n".join([
        f"ProductName: {pr.product_name}",
        f"Category: {pr.category or ''}",
        f"Description: {new_desc}",
        f"Specifications:\n{spec_text}",
        f"LearnedCorrection: {correction_text}",
    ])
    vec = _embed(embed_input, model=embedding_model)

    upsert_product(
        engine,
        product_name=pr.product_name,
        category=pr.category,
        description=new_desc,
        specifications=new_specs,
        embedding_vector=vec,
    )
    approve_correction(engine, correction_id, admin_note="Ürün güncellendi ve embedding yeniden üretildi.")
    return True


def update_product_specs(
    engine: Engine,
    *,
    product_name: str,
    spec_updates: Dict[str, str],
    new_category: Optional[str] = None,
    description_append: Optional[str] = None,
    embedding_model: str,
) -> bool:
    """
    Directly update a product's spec fields and optionally its category / description.
    Rebuilds the embedding afterwards so vector search reflects the changes.
    An empty string value removes the key from specs.
    """
    from .embeddings import embed_text as _embed
    from .products_repo import get_product_by_exact_name, upsert_product

    pr = get_product_by_exact_name(engine, product_name.strip())
    if not pr:
        return False

    new_specs = dict(pr.specifications or {})
    for k, v in spec_updates.items():
        k = k.strip()
        if not k:
            continue
        if v.strip():
            new_specs[k] = v.strip()
        else:
            new_specs.pop(k, None)  # empty value = delete the field

    category = new_category.strip() if new_category and new_category.strip() else pr.category
    description = pr.description
    if description_append and description_append.strip():
        description = (description or "") + f"\n{description_append.strip()}"

    # Build rich embedding: all textual content + every spec value
    spec_text = "\n".join(f"  {k}: {v}" for k, v in new_specs.items() if v)
    embed_input = "\n".join([
        f"ProductName: {pr.product_name}",
        f"Category: {category or ''}",
        f"Description: {description or ''}",
        f"Specifications:\n{spec_text}",
    ])
    vec = _embed(embed_input, model=embedding_model)

    upsert_product(
        engine,
        product_name=pr.product_name,
        category=category,
        description=description,
        specifications=new_specs,
        embedding_vector=vec,
    )
    return True


def get_approved_corrections(engine: Engine, limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent approved corrections to be included in RAG context."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT user_question, correction_text, product_name, reviewed_at
                FROM pending_corrections
                WHERE status = 'approved' AND correction_text IS NOT NULL
                ORDER BY reviewed_at DESC
                LIMIT :lim;
            """),
            {"lim": limit},
        ).mappings().fetchall()
    return [dict(r) for r in rows]


def reject_correction(engine: Engine, correction_id: int, admin_note: str = "") -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE pending_corrections
                SET status='rejected', admin_note=:note, reviewed_at=CURRENT_TIMESTAMP
                WHERE id=:id;
            """),
            {"id": correction_id, "note": admin_note},
        )


# ──────────────────────────────────────────────
# Weight filter
# ──────────────────────────────────────────────

def _weight_raw(specs: Dict[str, Any]) -> str:
    return (
        specs.get("agirlik")
        or specs.get("ağırlık")
        or specs.get("weight")
        or specs.get("Agirlik")
        or ""
    )


def _filter_by_weight_range(
    products: List[ProductRow],
    min_c: NumericConstraint,
    max_c: NumericConstraint,
) -> List[ProductRow]:
    """Filter products that satisfy BOTH a min and max weight constraint."""
    result = []
    for p in products:
        specs = p.specifications or {}
        raw = _weight_raw(specs)
        if not raw:
            continue
        interval = parse_weight_grams_from_text(str(raw))
        if interval is None:
            continue
        if (
            interval_satisfies_constraint(interval=interval, constraint=min_c)
            and interval_satisfies_constraint(interval=interval, constraint=max_c)
        ):
            result.append(p)
    return result


def _filter_by_weight(products: List[ProductRow], constraint: NumericConstraint) -> List[ProductRow]:
    result = []
    for p in products:
        specs = p.specifications or {}
        raw = _weight_raw(specs)
        if not raw:
            continue
        interval = parse_weight_grams_from_text(str(raw))
        if interval and interval_satisfies_constraint(interval=interval, constraint=constraint):
            result.append(p)
    return result


def _filter_by_generic(products: List[ProductRow], constraint: NumericConstraint) -> List[ProductRow]:
    result = []
    for p in products:
        specs = p.specifications or {}
        raw = specs.get(constraint.field) or specs.get(constraint.field.lower()) or ""
        m = re.search(r"(\d+(?:[.,]\d+)?)", str(raw))
        if not m:
            continue
        val = float(m.group(1).replace(",", "."))
        ok = False
        if constraint.op in ("=", "=="):
            ok = val == constraint.value
        elif constraint.op == "<":
            ok = val < constraint.value
        elif constraint.op == "<=":
            ok = val <= constraint.value
        elif constraint.op == ">":
            ok = val > constraint.value
        elif constraint.op == ">=":
            ok = val >= constraint.value
        if ok:
            result.append(p)
    return result


# ──────────────────────────────────────────────
# Main answer function
# ──────────────────────────────────────────────

def answer_question(
    *,
    engine: Engine,
    user_question: str,
    embedding_model: str,
    llm_model: str,
    top_k: int,
    min_score: float,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (answer_text, retrieved_products_list).
    Guardrails: product-only gate → DB retrieval → LLM generation from context only.
    """
    user_question = sanitize_user_question(user_question)
    q_lower = user_question.lower()

    product_names = list_product_names(engine)
    known_categories = [
        "Gece Görüş Sistemleri", "Renkli Gece Görüş Sistemleri",
        "Termal", "Radar", "Gimbal",
        "Keşif ve Gözetleme Sistemleri", "Elektro-Optik Nişangahlar",
        "Optik Haberleşme Sistemi", "Sürüş Görüş Sistemleri", "Silah Üstü Nişangahlar",
    ]

    # ── Guard: must be product question ──────────────
    if not is_product_question(user_question, product_names, known_categories):
        return REFUSAL_MESSAGE, []

    # ── Special: count ────────────────────────────────
    if _is_count_query(q_lower):
        all_p = list_all_products(engine)
        return f"Veritabanında toplam {len(all_p)} ürün bulunuyor.", [
            {"id": p.id, "product_name": p.product_name} for p in all_p
        ]

    # ── Special: list ALL products with names ─────────
    if _is_list_all_query(q_lower) and "özellik" not in q_lower and "ozellik" not in q_lower:
        all_p = list_all_products(engine)
        # Apply category filter if present (e.g. "gece görüş sistemleri adını listele")
        cat_filter = _is_category_filter(q_lower)
        if cat_filter:
            all_p = [p for p in all_p if p.category and cat_filter.lower() in p.category.lower()]
        if not all_p:
            return NO_PRODUCT_FOUND_MESSAGE, []
        header = f"**{cat_filter}** kategorisindeki ürünler:" if cat_filter else "Veritabanındaki tüm ürünler:"
        lines = [header]
        for p in all_p:
            cat_label = f" ({p.category})" if p.category and not cat_filter else ""
            lines.append(f"- {p.product_name}{cat_label}")
        return "\n".join(lines), [{"id": p.id, "product_name": p.product_name} for p in all_p]

    # ── Special: all products with specs ─────────────
    brief = "kısaca" in q_lower or "kısa" in q_lower or "ozet" in q_lower or "özet" in q_lower
    if _is_all_specs_query(q_lower):
        cat_filter = _is_category_filter(q_lower)
        all_p = list_all_products(engine)
        if cat_filter:
            all_p = [p for p in all_p if p.category and cat_filter.lower() in p.category.lower()]
        if not all_p:
            return NO_PRODUCT_FOUND_MESSAGE, []
        lines = []
        for p in all_p:
            lines.append(f"\n### {p.product_name}")
            if p.category:
                lines.append(f"  Kategori: {p.category}")
            lines.append(_format_specs(p.specifications, brief=brief))
        ret = [{"id": p.id, "product_name": p.product_name} for p in all_p]
        return "\n".join(lines), ret

    # ── Numeric filter: weight range / single / generic ──────────────────
    weight_range = parse_weight_range_from_question(user_question)
    weight_c = None if weight_range else parse_weight_constraint_from_question(user_question)
    generic_c = parse_generic_constraint_from_question(user_question)

    is_listing = any(w in q_lower for w in [
        "hangileri", "listele", "say", "tümü", "tüm", "kaç tane", "sırala",
        "var mı", "varmı", "göster", "goster", "hangi", "bul",
    ])

    if weight_range or weight_c or generic_c:
        all_p = list_all_products(engine)
        cat_filter = _is_category_filter(q_lower)
        if cat_filter:
            all_p = [p for p in all_p if p.category and cat_filter.lower() in p.category.lower()]

        if weight_range:
            min_c, max_c = weight_range
            filtered = _filter_by_weight_range(all_p, min_c, max_c)
            label = f"Ağırlık {min_c.value / 1000:.3g} kg – {max_c.value / 1000:.3g} kg arasındaki ürünler"
        elif weight_c:
            filtered = _filter_by_weight(all_p, weight_c)
            label = "Şartı sağlayan ürünler"
        else:
            filtered = _filter_by_generic(all_p, generic_c)
            label = "Şartı sağlayan ürünler"

        if not filtered:
            return NO_PRODUCT_FOUND_MESSAGE, []

        lines = [label + ":"]
        for p in filtered:
            specs = p.specifications or {}
            if weight_range or weight_c:
                raw = _weight_raw(specs)
                lines.append(f"- **{p.product_name}**  |  Ağırlık: {raw or '?'}")
            else:
                raw = specs.get(generic_c.field) or "?"
                lines.append(f"- **{p.product_name}**  |  {generic_c.field}: {raw}")
        ret = [{"id": p.id, "product_name": p.product_name, "score": p.score} for p in filtered]
        return "\n".join(lines), ret

    # ── Category filter + listing ─────────────────────
    cat_filter = _is_category_filter(q_lower)
    if cat_filter and is_listing:
        all_p = list_all_products(engine)
        filtered = [p for p in all_p if p.category and cat_filter.lower() in p.category.lower()]
        if not filtered:
            return NO_PRODUCT_FOUND_MESSAGE, []
        brief2 = "kısaca" in q_lower or "özet" in q_lower
        lines = [f"**{cat_filter}** kategorisindeki ürünler:"]
        for p in filtered:
            lines.append(f"\n### {p.product_name}")
            lines.append(_format_specs(p.specifications, brief=brief2))
        return "\n".join(lines), [{"id": p.id, "product_name": p.product_name} for p in filtered]

    # ── RAG: exact-name match first ───────────────────
    hits: List[ProductRow] = []
    for pname in product_names:
        if pname and pname.lower() in q_lower:
            pr = get_product_by_exact_name(engine, pname)
            if pr:
                hits = [pr]
            break

    # Vector search (always run, merge results)
    qvec = embed_text(user_question, model=embedding_model)
    vec_hits = vector_search_products(engine, query_embedding=qvec, top_k=top_k)
    vec_hits = [h for h in vec_hits if (h.score is None or h.score >= min_score)]
    seen = {p.product_name for p in hits}
    for h in vec_hits:
        if h.product_name not in seen:
            hits.append(h)
            seen.add(h.product_name)
    hits = hits[:max(top_k, 3)]

    # ── Semantic topic fallback: search description text for topic keywords ──
    # e.g. "deniz cihazları" → find products whose description mentions "deniz"
    if not hits or len(hits) < 2:
        semantic_topics = _get_semantic_topics(q_lower)
        if semantic_topics:
            try:
                all_p = list_all_products(engine)
                topic_hits: List[ProductRow] = []
                for p in all_p:
                    search_text = " ".join([
                        (p.description or ""),
                        " ".join(str(v) for v in (p.specifications or {}).values()),
                        (p.category or ""),
                    ]).lower()
                    if any(
                        any(hint in search_text for hint in _SEMANTIC_TOPIC_HINTS[topic])
                        for topic in semantic_topics
                    ):
                        if p.product_name not in seen:
                            topic_hits.append(p)
                            seen.add(p.product_name)
                hits = hits + topic_hits
            except Exception:
                pass

    if not hits:
        return NO_PRODUCT_FOUND_MESSAGE, []

    product_context = _format_products_context(hits, brief=False)

    # Append any admin-approved corrections for the retrieved products
    hit_names = {p.product_name for p in hits}
    try:
        corrections = get_approved_corrections(engine, limit=20)
        relevant = [
            c for c in corrections
            if c.get("product_name") in hit_names or c.get("product_name") is None
        ]
        if relevant:
            corr_lines = ["\n--- ONAYLANAN DÜZELTMELER ---"]
            for c in relevant[:5]:
                corr_lines.append(f"Soru: {c['user_question']}")
                corr_lines.append(f"Düzeltme: {c['correction_text']}")
            product_context += "\n" + "\n".join(corr_lines)
    except Exception:
        pass

    prompt = build_user_prompt(user_question=user_question, product_context=product_context)
    res = ollama.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0},
    )
    answer = ((res.get("message") or {}).get("content") or "").strip()
    answer = answer or NO_PRODUCT_FOUND_MESSAGE

    ret = [{"id": p.id, "product_name": p.product_name, "score": p.score} for p in hits]
    return answer, ret


def log_conversation(
    *, engine: Engine, user_message: str, chatbot_response: str, retrieved_products: List[Dict[str, Any]]
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO conversation_logs (user_message, chatbot_response, retrieved_products)
                VALUES (:u, :a, CAST(:rp AS jsonb));
            """),
            {"u": user_message, "a": chatbot_response, "rp": json.dumps(retrieved_products)},
        )
