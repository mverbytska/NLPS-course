from typing import Iterable, List, Optional, Sequence

from cocktail import taxonomy as T
from cocktail.schemas import Item, Recipe


def canonicalize(raw: str, vocab: Optional[Iterable[str]] = None) -> str:
    name = T.ALIASES.get(T.clean(raw), T.clean(raw))
    if vocab is not None and name not in vocab:
        for suffix in ("es", "s"):
            if name.endswith(suffix) and name[: -len(suffix)] in vocab:
                return name[: -len(suffix)]
    return name


def build_items(raws: Sequence[str], measures: Optional[Sequence[Optional[str]]] = None) -> List[Item]:
    items: List[Item] = []
    seen = set()
    for idx, raw in enumerate(raws):
        if not raw or not raw.strip():
            continue
        measure = (measures[idx].strip() if measures and idx < len(measures) and measures[idx] else None)
        name = canonicalize(raw)
        key = (name, measure)
        if key in seen:
            continue
        seen.add(key)
        items.append(Item(raw=raw.strip(), name=name, family=T.family_of(name),
                          measure=measure, optional=T.is_optional(name, measure or "")))
    return items


def infer_flavors(recipe: Recipe) -> List[str]:
    core = [i.name for i in recipe.items if not i.optional]
    text = " | ".join(core)
    tags = {tag for tag, words in T.FLAVOR_KEYWORDS.items() if any(T.has_word(text, w) for w in words)}
    if "refreshing" in tags and ({"creamy", "coffee"} & tags):
        tags.discard("refreshing")
    instr = (recipe.instructions or "").lower()
    if any(h in instr for h in T.WARM_HINTS) or (recipe.category or "") == "Coffee / Tea":
        tags.add("warm")
        tags.discard("refreshing")
    if (recipe.category or "") in ("Shake", "Cocoa"):
        tags.add("dessert")
    return [t for t in T.FLAVOR_VOCAB if t in tags]


def infer_strength(recipe: Recipe) -> str:
    if recipe.alcoholic is False:
        return "none"
    names = [i.name for i in recipe.items if not i.optional]
    spirits = sum(any(T.has_word(n, w) for w in T.SPIRIT_WORDS) for n in names)
    mixers = sum(any(T.has_word(n, w) for w in T.MIXER_WORDS) for n in names)
    glass = (recipe.glass or "").lower()
    if spirits == 0:
        return "light" if any(T.has_word(" ".join(names), w) for w in ("wine", "beer", "champagne", "prosecco", "cider")) else "medium"
    if spirits >= 3 or (spirits >= 2 and mixers == 0) or recipe.category == "Shot":
        return "strong"
    if spirits == 1 and ("highball" in glass or "collins" in glass or mixers >= 2):
        return "light"
    return "medium"


def enrich(recipe: Recipe) -> Recipe:
    recipe.flavors = infer_flavors(recipe)
    recipe.strength = infer_strength(recipe)
    return recipe
