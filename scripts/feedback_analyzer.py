import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from collections import Counter

from sqlalchemy import text

from lasersan_chatbot.config import load_settings
from lasersan_chatbot.db import create_db_engine
from lasersan_chatbot.logging_utils import configure_logging
from lasersan_chatbot.schema import ensure_schema


def main() -> None:
    """
    Minimal “self-improving” loop:
    - Looks at recent negative feedback
    - Stores aggregate notes to DB so you can review (admin dashboard later)

    This does NOT auto-edit product data; it flags weak areas safely.
    """
    settings = load_settings()
    configure_logging(settings.log_level)
    engine = create_db_engine(settings.database_url)
    ensure_schema(engine)

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT user_question, chatbot_answer, feedback_type, user_comment, created_at
                FROM user_feedback
                WHERE feedback_type IN ('incorrect','needs_improvement')
                ORDER BY created_at DESC
                LIMIT 200;
                """
            )
        ).mappings().fetchall()

    if not rows:
        print("No negative feedback to analyze.")
        return

    type_counts = Counter(r["feedback_type"] for r in rows)
    common_questions = [r["user_question"] for r in rows[:50]]

    payload = {
        "summary": dict(type_counts),
        "examples": [
            {
                "user_question": r["user_question"],
                "feedback_type": r["feedback_type"],
                "user_comment": r["user_comment"],
            }
            for r in rows[:20]
        ],
        "heuristics": {
            "recommendations": [
                "Consider increasing RAG_TOP_K if users report missing details.",
                "Lower RAG_MIN_SCORE if retrieval returns empty too often (but watch false positives).",
                "Add/normalize product descriptions/specifications for weak products.",
            ],
            "top_recent_questions": common_questions,
        },
    }

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO rag_improvement_notes (note_type, payload)
                VALUES ('feedback_analysis', CAST(:p AS jsonb));
                """
            ),
            {"p": json.dumps(payload, ensure_ascii=False)},
        )

    print("OK: feedback analysis note stored")


if __name__ == "__main__":
    main()

