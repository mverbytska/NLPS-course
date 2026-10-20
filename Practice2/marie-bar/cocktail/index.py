import hashlib
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from cocktail.data_io import DATA_DIR
from cocktail.llm import OllamaLLM
from cocktail.schemas import Recipe

INDEX_PATH = Path(DATA_DIR) / "embeddings.npz"
BATCH = 64


def _text(r: Recipe) -> str:
    core = ", ".join(i.name for i in r.items if not i.optional)
    return f"{r.name}. {r.taste or ''} {r.category or ''}. {', '.join(r.flavors)}. Ingredients: {core}. {(r.instructions or '')[:200]}"[:700]


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9, None)


class RecipeIndex:
    def __init__(self, recipes: List[Recipe], llm: OllamaLLM, path: Path = INDEX_PATH):
        self.llm = llm
        self.ids = [r.id for r in recipes]
        texts = [_text(r) for r in recipes]
        sig = hashlib.sha1((llm.embed_model + "\n".join(texts)).encode("utf-8")).hexdigest()
        self.vecs = self._load(path, sig)
        if self.vecs is None:
            print(f"[marie] indexing {len(texts)} recipes with {llm.embed_model} (first start only)...", flush=True)
            self.vecs = self._build(texts, path, sig)

    @staticmethod
    def _load(path: Path, sig: str):
        try:
            with np.load(path, allow_pickle=False) as z:
                return z["vecs"] if str(z["sig"]) == sig else None
        except Exception:
            return None

    def _build(self, texts: List[str], path: Path, sig: str) -> np.ndarray:
        vecs: List[List[float]] = []
        for i in range(0, len(texts), BATCH):
            vecs += self.llm.embed(["search_document: " + t for t in texts[i:i + BATCH]])
        arr = _normalize(np.asarray(vecs, dtype=np.float32))
        try:
            np.savez(path, vecs=arr, sig=np.array(sig))
        except OSError:
            pass
        return arr

    def similarity(self, text: str) -> Dict[str, float]:
        try:
            q = _normalize(np.asarray(self.llm.embed(["search_query: " + text])[0], dtype=np.float32))
        except Exception:
            return {}
        return dict(zip(self.ids, (self.vecs @ q).tolist()))
