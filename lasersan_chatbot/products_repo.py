from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class ProductRow:
    id: int
    product_name: str
    category: Optional[str]
    description: Optional[str]
    specifications: Optional[Dict[str, Any]]
    score: Optional[float] = None


def list_product_names(engine: Engine) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT product_name FROM products ORDER BY product_name")).fetchall()
    return [r[0] for r in rows]


def list_all_products(engine: Engine) -> List[ProductRow]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, product_name, category, description, specifications
                FROM products
                ORDER BY product_name
                """
            )
        ).mappings().fetchall()
    return [
        ProductRow(
            id=int(r["id"]),
            product_name=r["product_name"],
            category=r["category"],
            description=r["description"],
            specifications=r["specifications"],
            score=None,
        )
        for r in rows
    ]


def get_product_by_exact_name(engine: Engine, product_name: str) -> Optional[ProductRow]:
    pname = (product_name or "").strip()
    if not pname:
        return None
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT id, product_name, category, description, specifications
                FROM products
                WHERE product_name = :n
                LIMIT 1;
                """
            ),
            {"n": pname},
        ).mappings().fetchone()
    if not row:
        return None
    return ProductRow(
        id=int(row["id"]),
        product_name=row["product_name"],
        category=row["category"],
        description=row["description"],
        specifications=row["specifications"],
        score=1.0,
    )


def upsert_product(
    engine: Engine,
    *,
    product_name: str,
    category: Optional[str],
    description: Optional[str],
    specifications: Optional[Dict[str, Any]],
    embedding_vector: Optional[List[float]],
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO products (product_name, category, description, specifications, embedding_vector)
                VALUES (:product_name, :category, :description, CAST(:specifications AS jsonb), :embedding_vector)
                ON CONFLICT (product_name) DO UPDATE SET
                    category = EXCLUDED.category,
                    description = EXCLUDED.description,
                    specifications = EXCLUDED.specifications,
                    embedding_vector = EXCLUDED.embedding_vector,
                    updated_at = CURRENT_TIMESTAMP;
                """
            ),
            {
                "product_name": product_name,
                "category": category,
                "description": description,
                "specifications": json.dumps(specifications) if specifications is not None else None,
                "embedding_vector": embedding_vector,
            },
        )


def _has_pgvector(engine: Engine) -> bool:
    try:
        with engine.connect() as conn:
            v = conn.execute(text("SELECT extname FROM pg_extension WHERE extname='vector';")).fetchone()
        return v is not None
    except Exception:
        return False


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def vector_search_products(
    engine: Engine, *, query_embedding: List[float], top_k: int
) -> List[ProductRow]:
    """
    If pgvector is installed, uses DB-side cosine distance.
    Otherwise, loads candidate embeddings and ranks in Python via cosine similarity.
    """
    if _has_pgvector(engine):
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT
                        id,
                        product_name,
                        category,
                        description,
                        specifications,
                        1 - (embedding_vector <=> :qvec) AS score
                    FROM products
                    WHERE embedding_vector IS NOT NULL
                    ORDER BY embedding_vector <=> :qvec
                    LIMIT :top_k;
                    """
                ),
                {"qvec": query_embedding, "top_k": top_k},
            ).mappings().fetchall()
    else:
        q = np.asarray(query_embedding, dtype=np.float32)
        with engine.connect() as conn:
            candidates = conn.execute(
                text(
                    """
                    SELECT id, product_name, category, description, specifications, embedding_vector
                    FROM products
                    WHERE embedding_vector IS NOT NULL;
                    """
                )
            ).mappings().fetchall()

        scored = []
        for r in candidates:
            ev = r["embedding_vector"]
            if ev is None:
                continue
            v = np.asarray(ev, dtype=np.float32)
            scored.append((r, _cosine_sim(q, v)))

        scored.sort(key=lambda x: x[1], reverse=True)
        rows = []
        for r, s in scored[:top_k]:
            rr = dict(r)
            rr["score"] = s
            rows.append(rr)

    out: List[ProductRow] = []
    for r in rows:
        out.append(
            ProductRow(
                id=int(r["id"]),
                product_name=r["product_name"],
                category=r["category"],
                description=r["description"],
                specifications=r["specifications"],
                score=float(r["score"]) if r["score"] is not None else None,
            )
        )
    return out

