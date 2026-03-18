from __future__ import annotations

from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine


def save_feedback(
    engine: Engine,
    *,
    user_question: str,
    chatbot_answer: str,
    feedback_type: str,
    user_comment: Optional[str] = None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO user_feedback (user_question, chatbot_answer, feedback_type, user_comment)
                VALUES (:q, :a, :t, :c);
                """
            ),
            {"q": user_question, "a": chatbot_answer, "t": feedback_type, "c": user_comment},
        )

