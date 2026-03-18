import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lasersan_chatbot.config import load_settings
from lasersan_chatbot.db import create_db_engine
from lasersan_chatbot.logging_utils import configure_logging
from lasersan_chatbot.schema import ensure_schema


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    engine = create_db_engine(settings.database_url)
    ensure_schema(engine)
    print("OK: schema ensured")


if __name__ == "__main__":
    main()

