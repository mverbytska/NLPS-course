import json
from pathlib import Path
from typing import List, Optional

from cocktail import taxonomy as T
from cocktail.schemas import Recipe

DATA_DIR = Path(__file__).resolve().parent / "data"
RECIPES_PATH = DATA_DIR / "recipes.json"


def load_recipes(path: Optional[Path] = None) -> List[Recipe]:
    path = Path(path) if path else RECIPES_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/refine_data.py --fetch`.")
    recipes = [Recipe(**d) for d in json.loads(path.read_text(encoding="utf-8"))]
    for recipe in recipes:
        for item in recipe.items:
            alias = T.ALIASES.get(item.name)
            if alias and alias != item.name:
                item.name, item.family = alias, T.family_of(alias)
    return recipes
