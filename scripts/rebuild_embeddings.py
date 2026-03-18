import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

from sqlalchemy import text

from lasersan_chatbot.config import load_settings
from lasersan_chatbot.db import create_db_engine
from lasersan_chatbot.embeddings import embed_text
from lasersan_chatbot.logging_utils import configure_logging
from lasersan_chatbot.schema import ensure_schema


def _product_to_embedding_text(product_name: str, category: str | None, description: str | None, specs: dict | None) -> str:
    parts: list[str] = [f"ProductName: {product_name}"]
    if category:
        parts.append(f"Category: {category}")
    if description:
        parts.append(f"Description: {description}")
    if specs:
        parts.append(f"Specifications: {json.dumps(specs, ensure_ascii=False)}")
    return "\n".join(parts)


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    engine = create_db_engine(settings.database_url)
    ensure_schema(engine)

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, product_name, category, description, specifications FROM products ORDER BY id")
        ).mappings().fetchall()

    with engine.begin() as conn:
        for r in rows:
            text_for_embed = _product_to_embedding_text(
                r["product_name"], r["category"], r["description"], r["specifications"]
            )
            vec = embed_text(text_for_embed, model=settings.embedding_model)
            conn.execute(
                text("UPDATE products SET embedding_vector = :v, updated_at=CURRENT_TIMESTAMP WHERE id = :id"),
                {"v": vec, "id": r["id"]},
            )

    print(f"OK: embeddings rebuilt for {len(rows)} products")


if __name__ == "__main__":
    main()

