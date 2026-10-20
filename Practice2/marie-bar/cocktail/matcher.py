import re

from cocktail.schemas import Intent, Recipe


def has_ingredient(recipe: Recipe, term: str) -> bool:
    pat = re.compile(rf"(?<![a-z]){re.escape(term)}(?![a-z])")
    return any(term in (i.name, i.family) or pat.search(i.name) for i in recipe.items)


def passes_constraints(recipe: Recipe, intent: Intent) -> bool:
    if any(has_ingredient(recipe, x) for x in intent.exclude):
        return False
    if intent.alcohol == "none" and not recipe.can_be_alcohol_free:
        return False
    if intent.alcohol == "only" and recipe.alcoholic is False:
        return False
    return True