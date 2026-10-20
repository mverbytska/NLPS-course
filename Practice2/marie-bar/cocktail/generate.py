import difflib
from typing import List, Optional, Tuple

from cocktail import taxonomy as T
from cocktail.enrich import enrich
from cocktail.inventory import Inventory, Vocabulary
from cocktail.llm import OllamaLLM
from cocktail.matcher import has_ingredient
from cocktail.prompts import generate_user
from cocktail.schemas import Draft, Intent, Item, Recipe

UNIT_ML = {"ml": 1.0, "cl": 10.0, "oz": 29.57, "tsp": 4.93, "tbsp": 14.79, "dash": 0.9, "piece": 0.0}


def draft_to_recipe(d: Draft, vocab: Vocabulary) -> Recipe:
    items = []
    for it in d.items:
        name = vocab.canon(it.name)
        measure = f"{it.amount:g}" if it.unit == "piece" else f"{it.amount:g} {it.unit}"
        items.append(Item(raw=it.name.strip(), name=name, family=T.family_of(name), measure=measure,
                          optional=T.is_optional(name, measure)))
    slug = "".join(c if c.isalnum() else "-" for c in d.name.lower()).strip("-") or "drink"
    alcoholic = any(T.is_alcoholic(i.name) for i in items)
    return enrich(Recipe(id=f"gen-{slug}", name=d.name.strip(), category="Invented", alcoholic=alcoholic,
                         glass=d.glass.strip() or None, instructions=d.method.strip(), items=items,
                         source="generated"))


def validate(d: Draft, recipe: Recipe, vocab: Vocabulary, inv: Inventory, intent: Intent, allow: int,
             existing_names: set) -> List[str]:
    p: List[str] = []
    if not d.name.strip():
        p.append("give the drink a name")
    elif d.name.strip().casefold() in existing_names:
        p.append(f"'{d.name}' is the name of an existing cocktail; invent a new name")
    if not 2 <= len(d.items) <= 7:
        p.append(f"use 2 to 7 ingredients (you used {len(d.items)})")
    if len(d.method.strip()) < 15:
        p.append("the method is too short")
    names = [i.name for i in recipe.items]
    if len(set(names)) < len(names):
        p.append("an ingredient is listed twice")

    for it, item in zip(d.items, recipe.items):
        if item.name not in vocab.names:
            close = difflib.get_close_matches(item.name, vocab.names, n=3, cutoff=0.6)
            p.append(f"'{it.name}' is not an ingredient I know" + (f" (did you mean: {', '.join(close)}?)" if close else ""))
    for x in intent.exclude:
        if has_ingredient(recipe, x):
            p.append(f"it uses {x}, which the user excluded")
    for m in intent.must_include:
        if not has_ingredient(recipe, m):
            p.append(f"it must use {m}")

    alcoholic = [i.name for i in recipe.items if T.is_alcoholic(i.name)]
    if intent.alcohol == "none" and alcoholic:
        p.append(f"it must be alcohol-free but contains {', '.join(alcoholic)}")
    if intent.alcohol == "only" and not alcoholic:
        p.append("it must contain alcohol")

    ml = [it.amount * UNIT_ML[it.unit] for it in d.items]
    if any(it.amount <= 0 for it in d.items):
        p.append("every amount must be greater than zero")
    total = sum(ml)
    if not 45 <= total <= 300:
        p.append(f"the liquid adds up to {total:.0f} ml; one serving is roughly 60-250 ml")
    for it, v in zip(recipe.items, ml):
        if T.is_alcoholic(it.name) and v > 90:
            p.append(f"{v:.0f} ml of {it.name} is too much for one serving")

    cov = inv.check(recipe)
    if len(cov.missing) > allow:
        p.append(f"it needs {', '.join(cov.missing)}, which the bar doesn't have (at most {allow} may be missing)")
    return p


def constraint_lines(intent: Intent, allow: int) -> List[str]:
    lines = []
    if intent.must_include:
        lines.append("must use " + ", ".join(intent.must_include))
    if intent.exclude:
        lines.append("never use " + ", ".join(intent.exclude))
    lines.append({"none": "no alcohol at all", "only": "must contain alcohol"}.get(intent.alcohol, "alcohol is optional"))
    if intent.flavors:
        lines.append("flavours: " + ", ".join(intent.flavors))
    if intent.avoid_flavors:
        lines.append("less: " + ", ".join(intent.avoid_flavors))
    if intent.strength:
        lines.append("strength: " + intent.strength)
    lines.append(f"at most {allow} ingredient(s) outside the bar")
    return lines


def invent(llm: OllamaLLM, vocab: Vocabulary, inv: Inventory, intent: Intent, message: str, refs: List[Recipe],
           allow: int, bar: List[str], existing_names: set, trace: List[str], prev: Optional[Draft] = None,
           tries: int = 3) -> Tuple[Optional[Draft], Optional[Recipe], List[str]]:
    allowed = list(dict.fromkeys(list(bar) + intent.must_include))
    ref_lines = [(r.name, ", ".join(f"{i.measure or ''} {i.raw}".strip() for i in r.items)) for r in refs]
    problems: List[str] = []
    last: Optional[Draft] = prev
    for attempt in range(1, tries + 1):
        prompt = generate_user(message, constraint_lines(intent, allow), allowed, ref_lines, last, problems)
        draft = llm.draft(prompt, temperature=0.6 if attempt == 1 else 0.3)
        if draft is None:
            trace.append(f"draft#{attempt}:no-output")
            return None, None, problems
        recipe = draft_to_recipe(draft, vocab)
        problems = validate(draft, recipe, vocab, inv, intent, allow, existing_names)
        trace.append(f"draft#{attempt}:" + ("ok" if not problems else f"rejected({len(problems)})"))
        if not problems:
            return draft, recipe, []
        last = draft
    return None, None, problems
