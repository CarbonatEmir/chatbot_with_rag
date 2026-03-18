from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def create_db_engine(database_url: str) -> Engine:
    # pool_pre_ping helps avoid stale connections in prod.
    return create_engine(database_url, pool_pre_ping=True, future=True)

