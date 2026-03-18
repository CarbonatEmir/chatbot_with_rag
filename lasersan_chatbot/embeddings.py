from __future__ import annotations

from typing import List

import numpy as np
import ollama


def embed_texts(texts: List[str], model: str) -> np.ndarray:
    """
    Returns a 2D numpy array shape: (n_texts, dim)
    Uses Ollama embeddings endpoint.
    """
    vectors: List[List[float]] = []
    for t in texts:
        t = (t or "").strip()
        res = ollama.embeddings(model=model, prompt=t)
        vectors.append(res["embedding"])
    return np.asarray(vectors, dtype=np.float32)


def embed_text(text: str, model: str) -> List[float]:
    return embed_texts([text], model=model)[0].tolist()

