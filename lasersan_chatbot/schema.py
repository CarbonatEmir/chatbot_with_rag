from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


def ensure_schema(engine: Engine) -> None:
    """
    Creates/migrates all required tables.
    Uses pgvector if available, falls back to DOUBLE PRECISION[].
    """
    has_pgvector = True
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    except Exception:
        has_pgvector = False

    with engine.begin() as conn:

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS products (
                id BIGSERIAL PRIMARY KEY,
                product_name TEXT UNIQUE NOT NULL,
                category TEXT,
                description TEXT,
                specifications JSONB,
                embedding_vector DOUBLE PRECISION[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        if has_pgvector:
            try:
                conn.execute(text("""
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='products' AND column_name='embedding_vector'
                        ) THEN
                            BEGIN
                                ALTER TABLE products
                                ALTER COLUMN embedding_vector TYPE vector(1024)
                                USING embedding_vector::vector(1024);
                            EXCEPTION WHEN others THEN NULL;
                            END;
                        END IF;
                    END $$;
                """))
            except Exception:
                pass

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id BIGSERIAL PRIMARY KEY,
                user_question TEXT NOT NULL,
                chatbot_answer TEXT NOT NULL,
                feedback_type TEXT NOT NULL CHECK (feedback_type IN ('helpful','incorrect','needs_improvement')),
                user_comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversation_logs (
                id BIGSERIAL PRIMARY KEY,
                user_message TEXT NOT NULL,
                chatbot_response TEXT NOT NULL,
                retrieved_products JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS rag_improvement_notes (
                id BIGSERIAL PRIMARY KEY,
                note_type TEXT NOT NULL,
                payload JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        # Admin-approval correction workflow:
        # User submits correction → status='pending'
        # Admin approves → product DB updated + embedding rebuilt → status='approved'
        # Admin rejects → status='rejected'
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pending_corrections (
                id BIGSERIAL PRIMARY KEY,
                user_question TEXT NOT NULL,
                original_answer TEXT NOT NULL,
                correction_text TEXT,
                product_name TEXT,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','approved','rejected')),
                admin_note TEXT,
                reviewed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
