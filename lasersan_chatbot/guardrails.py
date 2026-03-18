from __future__ import annotations

import re
from typing import Iterable


REFUSAL_MESSAGE = "Ben Lasersan Chatbotuyum. Sadece Lasersan ürünleri hakkında bilgi verebilirim."
NO_PRODUCT_FOUND_MESSAGE = "Bu konuda ürün bilgisi bulunamadı. Lütfen ürün adıyla birlikte tekrar sorunuz."


_PRODUCT_INTENT_HINTS = [
    "ürün",
    "cihaz",
    "model",
    "kategori",
    "teknik",
    "özellik",
    "spesifik",
    "spesifikasyon",
    "fov",
    "çözünürlük",
    "kare hızı",
    "menzil",
    # weight / size
    "ağırl",
    "agirl",
    "kilo",
    "kg",
    "gram",
    "boyut",
    "ip",
    # usage / environment queries
    "deniz",
    "naval",
    "gemi",
    "tekne",
    "sahil",
    "sualtı",
    "su altı",
    "kara harekât",
    "hava",
    "iha",
    "uav",
    "drone",
    "kentsel",
    "şehir",
    "sınır",
    "hudut",
    "gözetleme",
    "gece",
    "termal",
    "gimbal",
    "radar",
    "nişangah",
    "lasersan",
    "kullanım",
    "kullanim",
    "hangi",
    "var mı",
    "varmı",
]


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def is_product_question(user_question: str, known_product_names: Iterable[str], known_categories: Iterable[str]) -> bool:
    q = _normalize(user_question)
    if not q:
        return False

    # If question explicitly mentions a known product name or category, it's in-scope.
    for name in known_product_names:
        n = _normalize(name)
        if n and n in q:
            return True

    for cat in known_categories:
        c = _normalize(cat)
        if c and c in q:
            return True

    # Otherwise, allow only if user shows clear "product intent" (still DB-only via retrieval).
    return any(h in q for h in _PRODUCT_INTENT_HINTS)


def sanitize_user_question(q: str) -> str:
    # Prevent prompt injection via extreme length / odd control chars.
    q = (q or "").replace("\x00", "").strip()
    q = re.sub(r"\s+", " ", q)
    return q[:1000]

