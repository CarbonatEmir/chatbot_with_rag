"""
Lasersan Chatbot – ana uygulama
Çalıştır: python -m streamlit run app.py
"""
from __future__ import annotations

import base64
import glob
import json
import os
import re
from typing import Any, Dict, Optional

import ollama
import pdfplumber
import streamlit as st
import streamlit.components.v1 as components
from PyPDF2 import PdfReader
from sqlalchemy import text

from lasersan_chatbot.config import load_settings
from lasersan_chatbot.db import create_db_engine
from lasersan_chatbot.embeddings import embed_text
from lasersan_chatbot.feedback_repo import save_feedback
from lasersan_chatbot.logging_utils import configure_logging
from lasersan_chatbot.products_repo import list_all_products, upsert_product
from lasersan_chatbot.rag_service import (
    answer_question,
    apply_correction_to_product,
    approve_correction,
    get_pending_corrections,
    log_conversation,
    reject_correction,
    save_pending_correction,
    update_product_specs,
)
from lasersan_chatbot.schema import ensure_schema

# ──────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────

settings = load_settings()
configure_logging(settings.log_level)
engine = create_db_engine(settings.database_url)
ensure_schema(engine)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Lasersan AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────

for key, default in {
    "show_pdf": False,
    "current_product": None,
    "aktif_urun": "",
    "mesajlar": [],
    "pending_feedback": None,
    "answer_nonce": 0,
    "show_feedback_form": False,
    "feedback_saved_msg": None,
    "extracted_data": None,
    "approve_expand": {},
    "admin_authenticated": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

page = st.query_params.get("page", "chat")
urun_secimi = st.query_params.get("product")
if urun_secimi != st.session_state.current_product:
    st.session_state.current_product = urun_secimi
    st.session_state.show_pdf = False

# ──────────────────────────────────────────────
# CSS  – ChatGPT-inspired compact design
# ──────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-size: 14px !important;
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif !important;
}
.stApp { background-color: #0f172a !important; color: #e2e8f0 !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
}
[data-testid="stChatMessage"] h1 { font-size: 1.1rem !important; }
[data-testid="stChatMessage"] h2 { font-size: 1rem !important; }
[data-testid="stChatMessage"] h3 { font-size: 0.95rem !important; font-weight: 700 !important; }
[data-testid="chatAvatarIcon-assistant"] { background-color: #003B70 !important; color: white !important; }

/* ── Chat input ── */
.stChatInputContainer > div {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    background-color: #1e293b !important;
    font-size: 0.875rem !important;
}
.stChatInputContainer textarea {
    font-size: 0.875rem !important;
    color: #e2e8f0 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background-color: #1e2d3d !important; }
[data-testid="stSidebar"] * { font-size: 0.8rem !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background-color: #162030 !important;
    border: 1px solid #2d3f52 !important;
    border-radius: 6px !important;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-size: 0.78rem !important;
    color: #94a3b8 !important;
    padding: 6px 10px !important;
}

/* ── Sidebar buttons / links ── */
.sidebar-link-btn {
    display: block;
    padding: 4px 8px;
    background-color: #1e4a6e;
    color: #e2e8f0 !important;
    text-align: center;
    border-radius: 5px;
    text-decoration: none !important;
    font-size: 0.75rem !important;
    font-weight: 600;
    margin-bottom: 4px;
    transition: background .2s;
}
.sidebar-link-btn:hover { background-color: #003B70; color: white !important; text-decoration: none !important; }

.nav-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    color: #f0f6ff !important;
    text-align: left;
    border-radius: 10px;
    text-decoration: none !important;
    font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
    font-weight: 600;
    font-size: 0.82rem !important;
    letter-spacing: 0.2px;
    margin-bottom: 5px;
    transition: filter .18s ease, transform .18s ease, box-shadow .18s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.25);
    border: none;
    outline: none;
}
.nav-btn:hover {
    text-decoration: none !important;
    filter: brightness(1.18);
    transform: translateX(3px);
    box-shadow: 0 3px 10px rgba(0,0,0,0.35);
}
.nav-btn-chat  { background: linear-gradient(135deg,#0e8fa3,#107F93); }
.nav-btn-add   { background: linear-gradient(135deg,#d41432,#C8102E); }
.nav-btn-admin { background: linear-gradient(135deg,#1a5585,#1e4a6e); }

/* ── Product page titles ── */
.prod-title-large {
    text-align: center; color: #C8102E;
    font-size: 2rem !important; font-weight: 800;
    margin: 0 0 12px 0;
}
.prod-title-small {
    text-align: center; color: #C8102E;
    font-size: 1.3rem !important; font-weight: 800;
    margin: 0 0 10px 0;
}
.fancy-pdf-header {
    text-align: center; font-size: 1rem !important;
    color: #e2e8f0; margin: 0 0 10px 0;
    padding-bottom: 8px; border-bottom: 1px dashed #334155;
    letter-spacing: 1px;
}
.fancy-pdf-header span { color: #C8102E; font-weight: 800; }

/* ── PDF transitions ── */
@keyframes slideFromRight {
    from { opacity: 0; transform: translateX(40px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeOut {
    from { opacity: 1; }
    to   { opacity: 0; }
}
.pdf-slide-in  { animation: slideFromRight 0.42s cubic-bezier(.2,.8,.2,1) both; }
.pdf-fade-in   { animation: fadeInUp 0.38s cubic-bezier(.2,.8,.2,1) both; }
.page-fade-in  { animation: fadeInUp 0.3s ease both; }

/* ── Admin / analytics ── */
.admin-title {
    text-align: center; color: #C8102E;
    font-size: 1.55rem !important; font-weight: 800; margin-bottom: 4px;
    letter-spacing: -0.3px;
}
.admin-subtitle {
    text-align: center; color: #94a3b8;
    font-size: 0.82rem !important; margin-bottom: 18px;
}
.correction-card {
    background: #1a2740;
    border: 1px solid #2d4060;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    font-size: 0.84rem !important;
    transition: border-color .2s;
}
.correction-card:hover { border-color: #4a7fa5; }

/* ── Admin stat card ── */
[data-testid="metric-container"] {
    background: #1a2740 !important;
    border: 1px solid #2d4060 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}

/* ── Password gate ── */
.pw-gate {
    max-width: 360px;
    margin: 80px auto;
    background: #1a2740;
    border: 1px solid #2d4060;
    border-radius: 14px;
    padding: 32px 28px;
    text-align: center;
}
.pw-gate h2 {
    font-size: 1.2rem !important;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 6px;
}
.pw-gate p {
    font-size: 0.8rem !important;
    color: #64748b;
    margin-bottom: 20px;
}

/* ── Feedback box ── */
.feedback-box {
    background: linear-gradient(135deg, #1e3a4a, #1e4a6e);
    padding: 12px 16px;
    border-radius: 8px;
    margin-top: 12px;
    font-size: 0.85rem !important;
}
.feedback-box h4 { font-size: 0.9rem !important; margin: 0 0 8px 0; }

/* ── Chat page header ── */
.chat-title {
    text-align: center; margin-bottom: 4px;
    font-size: 1.4rem !important; font-weight: 800;
}
.chat-subtitle {
    text-align: center; color: #64748b;
    font-size: 0.8rem !important; margin-bottom: 16px;
}

/* ── Streamlit default sizing fixes ── */
h1 { font-size: 1.4rem !important; }
h2 { font-size: 1.15rem !important; }
h3 { font-size: 1rem !important; }
p, li, div.stMarkdown { font-size: 0.875rem !important; }
[data-testid="metric-container"] label { font-size: 0.75rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
button { font-size: 0.8rem !important; }

/* ── Images ── */
[data-testid="stImage"] img {
    max-height: 300px !important;
    object-fit: contain !important;
    margin: 0 auto !important;
}
[data-testid="stSidebar"] [data-testid="stImage"] img { max-height: none !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _find_file(name: str, directory: str, ext: str) -> Optional[str]:
    """Locate a file by product name, tolerating Turkish char encoding variations.
    For images (.png/.jpg) also checks the img/ subdirectory automatically."""
    search_dirs = [directory]
    if ext.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        img_sub = os.path.join(BASE_DIR, "img")
        if img_sub != directory and os.path.isdir(img_sub):
            search_dirs.insert(0, img_sub)  # prefer img/ first

    target = f"{name.lower()}{ext.lower()}"
    for d in search_dirs:
        direct = os.path.join(d, f"{name}{ext}")
        if os.path.exists(direct):
            return direct
        for path in glob.glob(os.path.join(d, f"*{ext}")):
            if os.path.basename(path).lower() == target:
                return path
    return None


def _feedback_kaydet(*, soru: str, cevap: str, feedback_type: str, yorum: Optional[str]) -> None:
    try:
        save_feedback(engine, user_question=soru, chatbot_answer=cevap,
                      feedback_type=feedback_type, user_comment=yorum)
    except Exception as e:
        st.error(f"Feedback kaydedilemedi: {e}")


def _extract_pdf_text(uploaded_file) -> str:
    """Extract full text from an uploaded PDF using pdfplumber (primary) with PyPDF2 fallback.
    pdfplumber handles complex column layouts, embedded fonts and Turkish characters better.
    Reads raw bytes once so the stream can be rewound for both extractors."""
    import io
    raw_bytes = uploaded_file.read() if hasattr(uploaded_file, "read") else bytes(uploaded_file)

    # Primary: pdfplumber
    text_parts: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                text_parts.append(t)
        full = "\n".join(text_parts).strip()
        if len(full) > 100:
            return full
    except Exception:
        pass

    # Fallback: PyPDF2
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        text_parts = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(text_parts).strip()
    except Exception:
        return ""


def pdften_bilgi_cek(uploaded_file_or_reader) -> Optional[Dict]:
    try:
        # Accept either raw uploaded file or a PdfReader (legacy path)
        if isinstance(uploaded_file_or_reader, PdfReader):
            metin = " ".join(p.extract_text() or "" for p in uploaded_file_or_reader.pages)
        else:
            metin = _extract_pdf_text(uploaded_file_or_reader)

        metin = re.sub(r'[ \t]+', ' ', metin.replace('\n', ' ')).strip()
        tam_metin = metin[:10000]  # store up to 10k chars for embedding

        prompt = f"""Aşağıdaki ürün broşürünü analiz et. Cihazın TÜM teknik özelliklerini VE genel tanıtım metnini eksiksiz çıkar.
SADECE JSON formatında çıktı ver, başka hiçbir açıklama yapma.
KURALLAR:
1. Çerçeve: kategori, aciklama, agirlik, boyut, calisma_sicakligi, fov, kare_hizi, cozunurluk, ip_seviyesi, sensor, lazer_mesafe_olcer, optik_zoom, insan_tespit_menzili, arac_tespit_menzili, pil_omru, donus_acisi, kullanim_alanlari, ek_ozellikler
2. "aciklama": cihazın ne olduğunu, hangi ortamlarda (KARA / DENİZ / HAVA / KENTSEl / SINIR vb.) kullanıldığını ve temel faydalarını açıklayan tam bir paragraf. Broşürdeki tanıtım metnini AYNEN dahil et.
3. "kullanim_alanlari": "Kara harekâtı, deniz gözetleme, sınır güvenliği" gibi virgülle ayrılmış liste.
4. Fazladan teknik özellik varsa yeni anahtar oluştur.
5. Çift kamera varsa özellikleri birleştir.
Broşür metni:
{tam_metin[:7000]}"""

        res = ollama.generate(model=settings.ollama_model, prompt=prompt,
                               options={"temperature": 0}).get("response", "")
        m = re.search(r'\{.*\}', res, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        cihaz_adi = st.session_state.get("aktif_urun", "BILINMIYOR")
        sonuc: Dict[str, Any] = {"cihaz_adi": cihaz_adi}
        std_keys = ["kategori", "aciklama", "agirlik", "boyut", "calisma_sicakligi", "fov",
                    "kare_hizi", "cozunurluk", "ip_seviyesi", "sensor", "lazer_mesafe_olcer",
                    "optik_zoom", "insan_tespit_menzili", "arac_tespit_menzili", "pil_omru",
                    "donus_acisi", "kullanim_alanlari", "ek_ozellikler"]
        for k in std_keys:
            sonuc[k] = ""
        for k, v in data.items():
            safe = re.sub(r'[^a-z0-9_]', '', k.lower().replace(' ', '_'))
            if safe and safe != "cihaz_adi":
                sonuc[safe] = str(v).strip()
        sonuc["_tam_metin"] = tam_metin
        return sonuc
    except Exception as e:
        st.error(f"Yapay Zeka Analiz Hatası: {e}")
        return None


def veritabanina_kaydet(bilgiler: Dict[str, Any]) -> bool:
    try:
        product_name = (bilgiler.get("cihaz_adi") or "").strip().upper()
        if not product_name:
            st.warning("Ürün adı boş olamaz.")
            return False
        category = (bilgiler.get("kategori") or "").strip() or None

        # Build rich description: AI-extracted summary + usage areas + raw PDF text
        desc_parts = []
        for field in ("aciklama", "kullanim_alanlari", "ek_ozellikler"):
            val = (bilgiler.get(field) or "").strip()
            if val:
                desc_parts.append(val)
        tam_metin = (bilgiler.get("_tam_metin") or "").strip()
        if tam_metin:
            desc_parts.append(f"[Tam Metin]: {tam_metin[:3000]}")
        description = "\n".join(desc_parts) or None

        # Specs: exclude meta/description fields and private keys
        exclude_keys = {"cihaz_adi", "kategori", "aciklama", "kullanim_alanlari",
                        "ek_ozellikler", "_tam_metin"}
        specs = {k: v for k, v in bilgiler.items() if k not in exclude_keys and v}

        # Rich embedding: product name + category + description + specs + full PDF text
        embed_parts = [
            f"ProductName: {product_name}",
            f"Category: {category or ''}",
            f"Description: {description or ''}",
            f"Specifications: {json.dumps(specs, ensure_ascii=False)}",
        ]
        if tam_metin:
            embed_parts.append(f"FullText: {tam_metin[:4000]}")
        embed_text_val = "\n".join(embed_parts)

        vec = embed_text(embed_text_val, model=settings.embedding_model)
        upsert_product(engine, product_name=product_name, category=category,
                       description=description, specifications=specs, embedding_vector=vec)
        return True
    except Exception as e:
        st.error(f"Veritabanı kaydetme hatası: {e}")
        return False

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    logo_file = next(
        (f for f in ["logo3.png", "logo.png"] if os.path.exists(os.path.join(BASE_DIR, f))),
        None
    )
    if logo_file:
        with open(os.path.join(BASE_DIR, logo_file), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f'<div style="text-align:center;margin-bottom:6px">'
            f'<a href="/" target="_self">'
            f'<img src="data:image/png;base64,{b64}" style="width:72%;cursor:pointer">'
            f'</a></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="text-align:center;color:#64748b;font-size:.7rem;font-weight:600;'
        'letter-spacing:.5px;margin-bottom:10px;">Lasersan Advanced Technology</div>',
        unsafe_allow_html=True,
    )

    # Navigation buttons
    st.markdown(
        '<a href="/" target="_self" class="nav-btn nav-btn-chat">'
        '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>'
        'Chatbot</a>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<a href="?page=admin" target="_self" class="nav-btn nav-btn-add">'
        '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg>'
        'Yeni Ürün Ekle</a>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<a href="?page=analytics" target="_self" class="nav-btn nav-btn-admin">'
        '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>'
        'Admin Panel</a>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;color:#475569;font-size:.72rem;font-weight:700;"
        "text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Ürünler</div>",
        unsafe_allow_html=True,
    )

    try:
        all_prods = list_all_products(engine)
        cat_map: Dict[str, list] = {}
        for p in all_prods:
            cat = p.category or "Diğer"
            cat_map.setdefault(cat, []).append(p.product_name)
        for cat, names in sorted(cat_map.items()):
            with st.expander(cat):
                for n in sorted(names):
                    st.markdown(
                        f'<a href="?product={n}" target="_self" class="sidebar-link-btn">{n}</a>',
                        unsafe_allow_html=True,
                    )
    except Exception:
        st.caption("Ürün listesi yüklenemedi.")

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#475569;font-size:.65rem;'>© 2026 Lasersan Savunma Sanayii</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────

# ── ANALYTICS & ADMIN PAGE ───────────────────
if page == "analytics":

    # ── Password gate ──────────────────────────
    if not st.session_state.admin_authenticated:
        st.markdown("""
        <div class='pw-gate'>
            <div style='font-size:2rem;margin-bottom:8px'>🔐</div>
            <h2>Admin Paneli</h2>
            <p>Devam etmek için yönetici şifresi gereklidir.</p>
        </div>
        """, unsafe_allow_html=True)
        _, pw_col, _ = st.columns([1, 1.2, 1])
        with pw_col:
            with st.form("pw_form", clear_on_submit=True):
                pw_input = st.text_input("Şifre", type="password", placeholder="••••••••")
                if st.form_submit_button("Giriş Yap", use_container_width=True, type="primary"):
                    if pw_input == settings.admin_password:
                        st.session_state.admin_authenticated = True
                        st.rerun()
                    else:
                        st.error("Hatalı şifre. Lütfen tekrar deneyin.")
        st.stop()

    # ── Header ────────────────────────────────
    col_hdr, col_logout = st.columns([6, 1])
    with col_hdr:
        st.markdown(
            "<h1 class='admin-title' style='text-align:left;margin-bottom:2px'>"
            "⚙️ Analiz & Admin Paneli</h1>"
            "<p class='admin-subtitle' style='text-align:left'>"
            "Feedback istatistikleri · Ürün yönetimi · Düzeltme onayları</p>",
            unsafe_allow_html=True,
        )
    with col_logout:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("🚪 Çıkış", use_container_width=True):
            st.session_state.admin_authenticated = False
            st.rerun()

    st.markdown("<hr style='border-color:#2d4060;margin:0 0 16px 0'>", unsafe_allow_html=True)

    tabs = st.tabs(["📈 İstatistikler", "📦 Ürün Düzenle", "✅ Düzeltme Onayları", "📝 Tüm Feedbackler", "🔄 Embedding Yenile"])

    # ── Tab 1: Stats ──────────────────────────
    with tabs[0]:
        try:
            with engine.connect() as conn:
                total = conn.execute(text("SELECT COUNT(*) FROM user_feedback")).fetchone()[0]
                wrong = conn.execute(text("SELECT COUNT(*) FROM user_feedback WHERE feedback_type='incorrect'")).fetchone()[0]
                acc = ((total - wrong) / total * 100) if total else 0
                by_type = dict(conn.execute(text(
                    "SELECT feedback_type, COUNT(*) FROM user_feedback GROUP BY feedback_type"
                )).fetchall())
                pending_cnt = conn.execute(
                    text("SELECT COUNT(*) FROM pending_corrections WHERE status='pending'")
                ).fetchone()[0]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Toplam Feedback", total)
            c2.metric("Yanlış Cevaplar", wrong)
            c3.metric("Doğruluk", f"{acc:.1f}%")
            c4.metric("Doğru (helpful)", by_type.get("helpful", 0))
            c5.metric("Onay Bekleyen", pending_cnt, delta_color="inverse")

            st.markdown("#### Dağılım")
            st.json({"helpful": by_type.get("helpful", 0),
                     "incorrect": by_type.get("incorrect", 0),
                     "needs_improvement": by_type.get("needs_improvement", 0)})
        except Exception as e:
            st.error(f"İstatistik yüklenemedi: {e}")

    # ── Tab 2: Product Editor ─────────────────
    with tabs[1]:
        st.markdown("#### Ürün Düzenle")

        try:
            all_prods_admin = list_all_products(engine)
        except Exception as e:
            st.error(f"Ürünler yüklenemedi: {e}")
            all_prods_admin = []

        if all_prods_admin:
            cat_map_admin: Dict[str, list] = {}
            for p in all_prods_admin:
                cat_map_admin.setdefault(p.category or "Diğer", []).append(p)

            mode = st.radio(
                "Düzenleme modu:",
                ["Tek ürün düzenle", "Kategori toplu güncelle"],
                horizontal=True,
            )

            if mode == "Tek ürün düzenle":
                col_cat, col_prod = st.columns(2)
                with col_cat:
                    selected_cat = st.selectbox(
                        "Kategori:", sorted(cat_map_admin.keys()), key="edit_cat"
                    )
                with col_prod:
                    cat_products = cat_map_admin.get(selected_cat, [])
                    prod_names = [p.product_name for p in cat_products]
                    selected_prod_name = st.selectbox("Ürün:", prod_names, key="edit_prod")

                selected_prod = next(
                    (p for p in cat_products if p.product_name == selected_prod_name), None
                )
                if selected_prod:
                    st.markdown(f"**{selected_prod.product_name}** — mevcut özellikler:")
                    current_specs = dict(selected_prod.specifications or {})

                    with st.form(f"prod_edit_{selected_prod.product_name}"):
                        new_cat = st.text_input(
                            "Kategori", value=selected_prod.category or "", key="new_cat_field"
                        )
                        # Editable spec fields
                        updated_specs: Dict[str, str] = {}
                        existing_keys = list(current_specs.keys())
                        cols = st.columns(2)
                        for i, k in enumerate(existing_keys):
                            with cols[i % 2]:
                                new_val = st.text_input(
                                    k.replace("_", " ").title(),
                                    value=str(current_specs[k]),
                                    key=f"spec_{selected_prod.product_name}_{k}",
                                )
                                updated_specs[k] = new_val

                        st.markdown("**Yeni alan ekle** *(opsiyonel)*")
                        nc1, nc2 = st.columns(2)
                        with nc1:
                            new_key = st.text_input("Yeni alan adı", key="new_spec_key")
                        with nc2:
                            new_key_val = st.text_input("Değer", key="new_spec_val")
                        if new_key.strip() and new_key_val.strip():
                            updated_specs[new_key.strip()] = new_key_val.strip()

                        if st.form_submit_button("💾 Kaydet ve Embedding Yenile", use_container_width=True):
                            with st.spinner("Güncelleniyor..."):
                                ok = update_product_specs(
                                    engine,
                                    product_name=selected_prod.product_name,
                                    spec_updates=updated_specs,
                                    new_category=new_cat or None,
                                    embedding_model=settings.embedding_model,
                                )
                            if ok:
                                st.success(f"✅ **{selected_prod.product_name}** güncellendi!")
                                st.rerun()
                            else:
                                st.error("Güncelleme başarısız.")

            else:  # Kategori toplu güncelle
                st.info(
                    "Seçilen kategorideki **tüm ürünlere** aynı alan güncellemesini uygular. "
                    "Örn: tüm Gece Görüş cihazları için `ip_seviyesi = IP67` gibi."
                )
                bulk_cat = st.selectbox(
                    "Kategori seç:", sorted(cat_map_admin.keys()), key="bulk_cat"
                )
                bulk_products = cat_map_admin.get(bulk_cat, [])
                st.caption(f"Bu kategoride {len(bulk_products)} ürün var: "
                           f"{', '.join(p.product_name for p in bulk_products)}")

                with st.form("bulk_update_form"):
                    st.markdown("**Güncellenecek alanlar** (birden fazla satır doldurabilirsin):")
                    bulk_updates: Dict[str, str] = {}
                    for idx in range(5):
                        c1, c2 = st.columns(2)
                        with c1:
                            bk = st.text_input(
                                f"Alan {idx+1}", key=f"bk_{idx}",
                                placeholder="ör. ip_seviyesi",
                                label_visibility="collapsed",
                            )
                        with c2:
                            bv = st.text_input(
                                f"Değer {idx+1}", key=f"bv_{idx}",
                                placeholder="ör. IP67",
                                label_visibility="collapsed",
                            )
                        if bk.strip() and bv.strip():
                            bulk_updates[bk.strip()] = bv.strip()

                    if st.form_submit_button(
                        f"🔄 {bulk_cat} — {len(bulk_products)} ürünü güncelle",
                        use_container_width=True,
                    ):
                        if not bulk_updates:
                            st.warning("En az bir alan doldurun.")
                        else:
                            ok_count = 0
                            fail_list = []
                            with st.spinner(f"{len(bulk_products)} ürün güncelleniyor..."):
                                for prod in bulk_products:
                                    ok = update_product_specs(
                                        engine,
                                        product_name=prod.product_name,
                                        spec_updates=bulk_updates,
                                        embedding_model=settings.embedding_model,
                                    )
                                    if ok:
                                        ok_count += 1
                                    else:
                                        fail_list.append(prod.product_name)
                            st.success(
                                f"✅ {ok_count}/{len(bulk_products)} ürün güncellendi. "
                                f"Alanlar: {list(bulk_updates.keys())}"
                            )
                            if fail_list:
                                st.warning(f"Başarısız: {fail_list}")
                            st.rerun()

    # ── Tab 3: Correction Approvals ───────────
    with tabs[2]:
        st.markdown("#### Kullanıcı Düzeltmeleri – Admin Onayı")
        st.info(
            "Kullanıcının düzelttiği bilgileri burada onaylayabilirsin. "
            "**Onayla + Ürün Güncelle** butonuna tıklayarak ilgili ürünün "
            "açıklamasına düzeltme eklenir ve embedding yeniden üretilir — "
            "chatbot bir sonraki soruda bu bilgiyi kullanır."
        )

        corrections = get_pending_corrections(engine)
        if not corrections:
            st.success("Onay bekleyen düzeltme yok.")
        else:
            all_product_names = [p.product_name for p in list_all_products(engine)]

            for c in corrections:
                cid = c['id']
                ts = c['created_at'].strftime('%Y-%m-%d %H:%M') if c.get('created_at') else '?'
                with st.container():
                    st.markdown(f"<div class='correction-card'>", unsafe_allow_html=True)
                    st.markdown(f"**#{cid} — {ts}**")
                    st.write(f"**Soru:** {c['user_question']}")
                    st.write(f"**Bot Cevabı:** {(c['original_answer'] or '')[:300]}")
                    st.write(f"**Kullanıcı Düzeltmesi:** {c['correction_text'] or '—'}")
                    if c.get('product_name'):
                        st.write(f"**İlgili Ürün:** {c['product_name']}")
                    st.markdown("</div>", unsafe_allow_html=True)

                col_a, col_au, col_r = st.columns([1, 2, 1])
                with col_a:
                    if st.button("✅ Onayla", key=f"approve_{cid}"):
                        approve_correction(engine, cid)
                        st.success(f"#{cid} onaylandı.")
                        st.rerun()
                with col_au:
                    # Expand an inline form to pick product + apply correction
                    if st.button("🧠 Onayla + Ürün Güncelle", key=f"update_{cid}", type="primary"):
                        st.session_state.approve_expand[cid] = True
                with col_r:
                    if st.button("❌ Reddet", key=f"reject_{cid}"):
                        reject_correction(engine, cid)
                        st.warning(f"#{cid} reddedildi.")
                        st.rerun()

                # Inline product update form
                if st.session_state.approve_expand.get(cid):
                    with st.form(f"update_form_{cid}"):
                        st.markdown("**Hangi ürünü güncellemek istiyorsunuz?**")
                        default_idx = 0
                        if c.get('product_name') and c['product_name'] in all_product_names:
                            default_idx = all_product_names.index(c['product_name'])
                        selected_product = st.selectbox(
                            "Ürün seçin:", all_product_names,
                            index=default_idx, key=f"sel_{cid}"
                        )
                        correction_to_apply = st.text_area(
                            "Düzeltme açıklaması (description'a eklenir):",
                            value=c.get('correction_text') or "",
                            key=f"txt_{cid}",
                            height=70,
                        )

                        st.markdown(
                            "**Spec alanı güncellemeleri** *(isteğe bağlı — doğrudan teknik özelliği günceller)*"
                        )
                        st.caption(
                            "Örn: Alan adı = `agirlik`, Değer = `2.5 kg`  |  "
                            "veya `cozunurluk` = `640x512`"
                        )
                        spec_cols = st.columns([1, 1])
                        spec_updates: Dict[str, str] = {}
                        for idx in range(4):
                            with spec_cols[idx % 2]:
                                k = st.text_input(f"Alan {idx+1}", key=f"sk_{cid}_{idx}", label_visibility="collapsed", placeholder=f"alan_adı_{idx+1}")
                                v = st.text_input(f"Değer {idx+1}", key=f"sv_{cid}_{idx}", label_visibility="collapsed", placeholder="değer")
                                if k.strip() and v.strip():
                                    spec_updates[k.strip()] = v.strip()

                        if st.form_submit_button("💾 Güncelle ve Kaydet"):
                            if correction_to_apply.strip() or spec_updates:
                                with st.spinner("Ürün güncelleniyor ve embedding üretiliyor..."):
                                    ok = apply_correction_to_product(
                                        engine,
                                        correction_id=cid,
                                        product_name=selected_product,
                                        correction_text=correction_to_apply.strip() or "(admin düzeltmesi)",
                                        spec_updates=spec_updates or None,
                                        embedding_model=settings.embedding_model,
                                    )
                                if ok:
                                    st.success(
                                        f"✅ **{selected_product}** güncellendi! "
                                        f"Embedding yeniden üretildi. "
                                        f"Güncellenen alanlar: {list(spec_updates.keys()) if spec_updates else 'yalnızca açıklama'}"
                                    )
                                    st.session_state.approve_expand.pop(cid, None)
                                    st.rerun()
                                else:
                                    st.error("Ürün bulunamadı veya güncelleme başarısız.")
                            else:
                                st.warning("En az bir düzeltme metni veya spec alanı doldurun.")

                st.markdown("---")

    # ── Tab 4: All feedback ───────────────────
    with tabs[3]:
        try:
            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT user_question, chatbot_answer, feedback_type, user_comment, created_at
                    FROM user_feedback ORDER BY created_at DESC LIMIT 50
                """)).fetchall()
            if rows:
                for i, (q, a, t, cmt, ts) in enumerate(rows, 1):
                    with st.expander(f"{i}. [{t}] {q[:60]}... ({ts.strftime('%Y-%m-%d %H:%M')})"):
                        st.write(f"**Soru:** {q}")
                        st.write(f"**Bot cevabı:** {a[:400]}")
                        st.write(f"**Tür:** {t}")
                        st.write(f"**Yorum:** {cmt or '—'}")
            else:
                st.info("Henüz feedback kaydı yok.")
        except Exception as e:
            st.error(f"Feedback listesi yüklenemedi: {e}")

    # ── Tab 5: Rebuild embeddings ─────────────
    with tabs[4]:
        st.markdown("#### Tüm Ürün Embeddinglerini Yeniden Oluştur")
        st.info(
            "Mevcut ürünlerin embedding vektörlerini **açıklama + tüm özellikler** "
            "birleştirilerek yeniden üretir. Yeni PDF eklendikten veya toplu "
            "düzeltme yapıldıktan sonra çalıştırılması önerilir. "
            "Böylece semantik arama (ör. *'deniz cihazları'*) daha doğru sonuç verir."
        )
        if st.button("🔄 Tüm Embeddingleri Yenile", type="primary", use_container_width=True):
            try:
                all_for_rebuild = list_all_products(engine)
                progress = st.progress(0)
                ok_cnt, fail_cnt = 0, 0
                for idx, prod in enumerate(all_for_rebuild):
                    try:
                        specs = prod.specifications or {}
                        spec_text = "\n".join(f"  {k}: {v}" for k, v in specs.items() if v)
                        embed_parts = [
                            f"ProductName: {prod.product_name}",
                            f"Category: {prod.category or ''}",
                            f"Description: {prod.description or ''}",
                            f"Specifications:\n{spec_text}",
                        ]
                        vec = embed_text("\n".join(embed_parts), model=settings.embedding_model)
                        upsert_product(
                            engine,
                            product_name=prod.product_name,
                            category=prod.category,
                            description=prod.description,
                            specifications=specs,
                            embedding_vector=vec,
                        )
                        ok_cnt += 1
                    except Exception:
                        fail_cnt += 1
                    progress.progress((idx + 1) / len(all_for_rebuild))
                st.success(f"✅ {ok_cnt} ürün güncellendi. {'⚠️ ' + str(fail_cnt) + ' hata.' if fail_cnt else ''}")
            except Exception as e:
                st.error(f"Embedding yenileme hatası: {e}")

# ── ADMIN / NEW PRODUCT PAGE ─────────────────
elif page == "admin":

    # ── Password gate ──────────────────────────
    if not st.session_state.admin_authenticated:
        st.markdown("""
        <div class='pw-gate'>
            <div style='font-size:2rem;margin-bottom:8px'>🔐</div>
            <h2>Yeni Ürün Ekle</h2>
            <p>Devam etmek için yönetici şifresi gereklidir.</p>
        </div>
        """, unsafe_allow_html=True)
        _, pw_col, _ = st.columns([1, 1.2, 1])
        with pw_col:
            with st.form("pw_form_admin", clear_on_submit=True):
                pw_input = st.text_input("Şifre", type="password", placeholder="••••••••")
                if st.form_submit_button("Giriş Yap", use_container_width=True, type="primary"):
                    if pw_input == settings.admin_password:
                        st.session_state.admin_authenticated = True
                        st.rerun()
                    else:
                        st.error("Hatalı şifre.")
        st.stop()

    col_hdr2, col_logout2 = st.columns([6, 1])
    with col_hdr2:
        st.markdown(
            "<h1 class='admin-title' style='text-align:left;margin-bottom:2px'>"
            "➕ Yeni Ürün Ekle</h1>"
            "<p class='admin-subtitle' style='text-align:left'>"
            "PDF broşürünü yükleyin — yapay zeka bilgileri otomatik çıkarır</p>",
            unsafe_allow_html=True,
        )
    with col_logout2:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("🚪 Çıkış", key="logout_admin", use_container_width=True):
            st.session_state.admin_authenticated = False
            st.rerun()
    st.markdown("<hr style='border-color:#2d4060;margin:0 0 16px 0'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        urun_ismi = st.text_input("1. Cihaz Adı:", placeholder="Örn: YALMAN-150PT")
        uploaded_file = st.file_uploader("2. PDF Broşürü:", type="pdf")

        if st.button("Bilgileri Yapay Zeka ile Çek", type="primary", use_container_width=True):
            if not urun_ismi or not uploaded_file:
                st.warning("Lütfen cihaz adı girin ve PDF yükleyin.")
            else:
                with st.spinner("Analiz ediliyor..."):
                    st.session_state.aktif_urun = urun_ismi.strip().upper()
                    bilgiler = pdften_bilgi_cek(uploaded_file)
                    if bilgiler:
                        st.session_state.extracted_data = bilgiler

    if st.session_state.extracted_data:
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#e2e8f0;text-align:center;font-size:1rem'>📋 Çıkarılan Veriler (Düzenleyebilirsiniz)</h3>", unsafe_allow_html=True)
        with st.form("kayit_formu"):
            cols = st.columns(3)
            guncel = {}
            # Exclude multiline/private fields from the column grid
            exclude_from_grid = {"ek_ozellikler", "aciklama", "kullanim_alanlari", "_tam_metin"}
            keys = [k for k in st.session_state.extracted_data if k not in exclude_from_grid]
            for i, k in enumerate(keys):
                with cols[i % 3]:
                    label = k.replace("_", " ").title()
                    val = st.session_state.extracted_data.get(k) or ""
                    guncel[k] = st.text_input(label, value=val)

            st.markdown("<br>", unsafe_allow_html=True)
            guncel["aciklama"] = st.text_area(
                "Genel Açıklama (kullanım amacı, ortam, faydalar)",
                value=st.session_state.extracted_data.get("aciklama") or "", height=80)
            guncel["kullanim_alanlari"] = st.text_area(
                "Kullanım Alanları (kara, deniz, hava, kentsel vb.)",
                value=st.session_state.extracted_data.get("kullanim_alanlari") or "", height=60)
            guncel["ek_ozellikler"] = st.text_area("Ek Özellikler",
                value=st.session_state.extracted_data.get("ek_ozellikler") or "", height=70)
            # Carry over the raw PDF full text silently
            guncel["_tam_metin"] = st.session_state.extracted_data.get("_tam_metin") or ""

            st.markdown("#### ➕ Yeni Özellik Ekle")
            c1n, c2n = st.columns(2)
            with c1n:
                yeni_ad = st.text_input("Sütun Adı (ör. Lazer İşaretleyici)")
            with c2n:
                yeni_val = st.text_input("Değer (ör. Var)")

            if st.form_submit_button("✅ Onayla ve Veritabanına Kaydet", use_container_width=True):
                if yeni_ad and yeni_val:
                    tr = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
                    safe = re.sub(r'[^a-z0-9_]', '', yeni_ad.translate(tr).lower().replace(" ", "_"))
                    if safe:
                        guncel[safe] = yeni_val
                if veritabanina_kaydet(guncel):
                    st.balloons()
                    st.success(f"✅ {guncel.get('cihaz_adi', '?')} veritabanına kaydedildi ve embedding üretildi!")
                    st.session_state.extracted_data = None

# ── PRODUCT DETAIL PAGE ──────────────────────
elif urun_secimi:
    img_path = _find_file(urun_secimi, BASE_DIR, ".png")
    pdf_path = _find_file(urun_secimi, os.path.join(BASE_DIR, "urunpdf"), ".pdf")

    if not st.session_state.show_pdf:
        st.markdown(
            "<div class='page-fade-in' style='height:20px'></div>",
            unsafe_allow_html=True,
        )
        _, col_main, _ = st.columns([1, 1.2, 1])
        with col_main:
            st.markdown(
                f"<div class='prod-title-large page-fade-in'>{urun_secimi}</div>",
                unsafe_allow_html=True,
            )
            if img_path and os.path.exists(img_path):
                st.markdown("<div class='page-fade-in'>", unsafe_allow_html=True)
                st.image(img_path, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            if pdf_path:
                if st.button("📄 Broşürü Görüntüle", use_container_width=True):
                    st.session_state.show_pdf = True
                    st.rerun()
            else:
                st.info("Bu ürün için PDF broşürü bulunamadı.")
    else:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        col_sol, col_sag = st.columns([1, 2.5])

        with col_sol:
            st.markdown(
                f"<div class='pdf-fade-in'>"
                f"<div class='prod-title-small'>{urun_secimi}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if img_path and os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button("← Broşürü Kapat", type="primary", use_container_width=True):
                st.session_state.show_pdf = False
                st.rerun()

            if pdf_path:
                with open(pdf_path, "rb") as f:
                    pdf_b64 = base64.b64encode(f.read()).decode()
                components.html(
                    f"""<!DOCTYPE html><html><head><style>
                    body{{margin:0;padding:0;background:transparent}}
                    .tab-btn{{
                        background:linear-gradient(135deg,#1a5585,#1e4a6e);
                        color:#fff;padding:9px;border-radius:8px;
                        font-weight:600;font-size:.8rem;border:none;
                        cursor:pointer;width:100%;font-family:'Segoe UI',sans-serif;
                        transition:filter .2s;margin-top:4px;letter-spacing:.3px;
                    }}
                    .tab-btn:hover{{filter:brightness(1.2)}}
                    </style></head>
                    <body><button onclick="openPdf()" class="tab-btn">🔗 Yeni Sekmede Aç</button>
                    <script>function openPdf(){{
                        var b='{pdf_b64}';var c=atob(b);
                        var d=new Uint8Array(c.length);
                        for(var i=0;i<c.length;i++)d[i]=c.charCodeAt(i);
                        var e=new Blob([d],{{type:'application/pdf'}});
                        window.open(URL.createObjectURL(e),'_blank')
                    }}</script></body></html>""",
                    height=52,
                )

        with col_sag:
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    pdf_b64_view = base64.b64encode(f.read()).decode()
                st.markdown(
                    f"<div class='fancy-pdf-header'><span>{urun_secimi}</span> — ÜRÜN BROŞÜRÜ</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="pdf-slide-in">'
                    f'<iframe src="data:application/pdf;base64,{pdf_b64_view}#toolbar=0&view=FitH"'
                    f' width="100%" height="820" type="application/pdf"'
                    f' style="border-radius:8px;border:1px solid #334155;'
                    f'display:block;min-height:600px"></iframe>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Bu ürün için PDF bulunamadı.")

# ── MAIN CHAT PAGE ───────────────────────────
else:
    _, col2, _ = st.columns([2, 1.8, 2])
    with col2:
        logo = next(
            (f for f in ["logo3.png", "logo.png"] if os.path.exists(os.path.join(BASE_DIR, f))),
            None,
        )
        if logo:
            st.image(os.path.join(BASE_DIR, logo), use_container_width=True)

    st.markdown(
        "<h1 class='chat-title'>"
        "<span style='color:#003B70;'>Laser</span>"
        "<span style='color:#C8102E;'>san Akıllı Asistan</span>"
        "</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='chat-subtitle'>Tüm şirket cihazları hakkında anında ve güvenilir bilgi alın.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.mesajlar:
        st.markdown(
            "<p style='text-align:center;color:#475569;font-size:.85rem;margin-top:8px;'>"
            "Bugün size nasıl yardımcı olabilirim?</p>",
            unsafe_allow_html=True,
        )

    for msg in st.session_state.mesajlar:
        with st.chat_message(msg["rol"]):
            st.markdown(msg["icerik"])

    soru = st.chat_input("Lasersan cihazlarıyla ilgili bir soru sorun...")

    if soru:
        with st.chat_message("user"):
            st.markdown(soru)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("⏳ Veritabanı taranıyor...")

            cevap, retrieved = answer_question(
                engine=engine,
                user_question=soru,
                embedding_model=settings.embedding_model,
                llm_model=settings.ollama_model,
                top_k=settings.rag_top_k,
                min_score=settings.rag_min_score,
            )
            try:
                log_conversation(engine=engine, user_message=soru,
                                 chatbot_response=cevap, retrieved_products=retrieved)
            except Exception:
                pass

            placeholder.markdown(cevap)

        st.session_state.answer_nonce += 1
        st.session_state.pending_feedback = {
            "soru": soru, "cevap": cevap,
            "nonce": st.session_state.answer_nonce,
        }
        st.session_state.show_feedback_form = False
        st.session_state.mesajlar.append({"rol": "user", "icerik": soru})
        st.session_state.mesajlar.append({"rol": "assistant", "icerik": cevap})

    # ── Persistent feedback widget ──
    if st.session_state.pending_feedback:
        nonce = st.session_state.pending_feedback["nonce"]
        pf = st.session_state.pending_feedback

        if st.session_state.feedback_saved_msg:
            st.success(st.session_state.feedback_saved_msg)
            st.session_state.feedback_saved_msg = None

        st.markdown("<div class='feedback-box'><h4>Bu cevap yardımcı oldu mu?</h4></div>", unsafe_allow_html=True)
        fb1, fb2, fb3, _ = st.columns(4)

        with fb1:
            if st.button("👍 Evet", key=f"yes_{nonce}"):
                _feedback_kaydet(soru=pf["soru"], cevap=pf["cevap"],
                                 feedback_type="helpful", yorum=None)
                st.session_state.feedback_saved_msg = "✅ Teşekkürler! Feedback kaydedildi."
                st.session_state.pending_feedback = None
                st.session_state.show_feedback_form = False
                st.rerun()

        with fb2:
            if st.button("👎 Yanlış", key=f"no_{nonce}"):
                st.session_state.show_feedback_form = True

        with fb3:
            if st.button("⭐ Mükemmel", key=f"star_{nonce}"):
                _feedback_kaydet(soru=pf["soru"], cevap=pf["cevap"],
                                 feedback_type="helpful", yorum="Mükemmel cevap!")
                st.session_state.feedback_saved_msg = "⭐ Teşekkürler!"
                st.session_state.pending_feedback = None
                st.session_state.show_feedback_form = False
                st.rerun()

        if st.session_state.get("show_feedback_form"):
            with st.form(f"fb_form_{nonce}"):
                st.markdown("**Neyi yanlış/eksik buldunuz?**")
                feedback_text = st.text_area("Açıklama:", height=70)
                dogrulmasi = st.text_area("Doğru bilgi ne olmalıydı?", height=70)
                secim = st.selectbox(
                    "Tür:",
                    options=["incorrect", "needs_improvement"],
                    format_func=lambda x: "❌ Yanlış cevap" if x == "incorrect" else "🔧 Geliştirilmeli",
                )
                if st.form_submit_button("📨 Gönder"):
                    yorum = "\n".join(p for p in [feedback_text.strip(), dogrulmasi.strip()] if p)
                    _feedback_kaydet(soru=pf["soru"], cevap=pf["cevap"],
                                     feedback_type=secim, yorum=yorum or None)
                    save_pending_correction(
                        engine,
                        user_question=pf["soru"],
                        original_answer=pf["cevap"],
                        correction_text=dogrulmasi.strip() or feedback_text.strip() or None,
                        product_name=None,
                    )
                    st.session_state.feedback_saved_msg = "📨 Feedback gönderildi! Admin inceleyecek."
                    st.session_state.pending_feedback = None
                    st.session_state.show_feedback_form = False
                    st.rerun()
