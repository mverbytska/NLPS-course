from typing import Dict, List, Optional, Tuple

from cocktail.index import RecipeIndex
from cocktail.inventory import Vocabulary
from cocktail.schemas import Recipe


class Toolbox:
    def __init__(self, recipes: List[Recipe], vocab: Vocabulary, index: RecipeIndex):
        self.recipes, self.vocab, self.index = recipes, vocab, index
        self.by_id = {r.id: r for r in recipes}
        self.trace: List[str] = []

    def _log(self, name: str, arg: str = "") -> None:
        self.trace.append(f"{name}({arg})")

    def semantic(self, text: str) -> Dict[str, float]:
        self._log("semantic_search", text[:40])
        return self.index.similarity(text)

    def search(self, text: str, pool: Optional[List[str]] = None, k: int = 20) -> List[Recipe]:
        sims = self.semantic(text)
        ids = [i for i in (pool if pool is not None else self.by_id) if i in sims]
        return [self.by_id[i] for i in sorted(ids, key=lambda i: -sims[i])[:k]]

    def nearest(self, recipe: Recipe) -> Tuple[Optional[Recipe], float]:
        self._log("nearest", recipe.name)
        mine = {i.name for i in recipe.items if not i.optional}
        best, score = None, 0.0
        for r in self.recipes:
            theirs = {i.name for i in r.items if not i.optional}
            sim = len(mine & theirs) / len(mine | theirs) if mine | theirs else 0.0
            if sim > score:
                best, score = r, sim
        return best, score
