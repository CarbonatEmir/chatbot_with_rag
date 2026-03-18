import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text

from lasersan_chatbot.config import load_settings
from lasersan_chatbot.db import create_db_engine


def main() -> None:
    s = load_settings()
    e = create_db_engine(s.database_url)
    with e.connect() as c:
        tables = c.execute(
            text("select tablename from pg_tables where schemaname='public' order by tablename")
        ).fetchall()
        print("tables:", [t[0] for t in tables])

        for tname in ["products", "cihaz_ozellikleri", "user_feedback", "conversation_logs"]:
            try:
                cnt = c.execute(text(f"select count(*) from {tname}")).fetchone()[0]
                print(f"{tname}: {cnt}")
            except Exception as ex:
                print(f"{tname}: not available ({ex.__class__.__name__})")


if __name__ == "__main__":
    main()

