import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text

from lasersan_chatbot.config import load_settings
from lasersan_chatbot.db import create_db_engine
from lasersan_chatbot.embeddings import embed_text
from lasersan_chatbot.products_repo import upsert_product
from lasersan_chatbot.schema import ensure_schema


def main() -> None:
    settings = load_settings()
    engine = create_db_engine(settings.database_url)
    ensure_schema(engine)

    with engine.connect() as conn:
        # Check legacy table exists
        exists = conn.execute(
            text(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema='public' AND table_name='cihaz_ozellikleri'
                """
            )
        ).fetchone()
        if not exists:
            print("Legacy table cihaz_ozellikleri not found. Nothing to migrate.")
            return

        cols = conn.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema='public' AND table_name='cihaz_ozellikleri'
                ORDER BY ordinal_position
                """
            )
        ).fetchall()
        colnames = [c[0] for c in cols]
        if "cihaz_adi" not in colnames:
            print("Legacy table missing cihaz_adi. Aborting.")
            return

        rows = conn.execute(text("SELECT * FROM cihaz_ozellikleri")).mappings().fetchall()

    migrated = 0
    for r in rows:
        product_name = (r.get("cihaz_adi") or "").strip().upper()
        if not product_name:
            continue

        category = (r.get("kategori") or "").strip() or None
        description = (r.get("ek_ozellikler") or "").strip() or None

        specs = dict(r)
        specs.pop("id", None)
        specs.pop("cihaz_adi", None)
        specs.pop("kategori", None)
        specs.pop("ek_ozellikler", None)

        embedding_text = "\n".join(
            [
                f"ProductName: {product_name}",
                f"Category: {category or ''}",
                f"Description: {description or ''}",
                f"Specifications: {json.dumps(specs, ensure_ascii=False)}",
            ]
        )
        vec = embed_text(embedding_text, model=settings.embedding_model)

        upsert_product(
            engine,
            product_name=product_name,
            category=category,
            description=description,
            specifications=specs,
            embedding_vector=vec,
        )
        migrated += 1

    print(f"OK: migrated {migrated} products from cihaz_ozellikleri -> products")


if __name__ == "__main__":
    main()

