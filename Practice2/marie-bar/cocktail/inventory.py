import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from cocktail import taxonomy as T
from cocktail.enrich import canonicalize
from cocktail.schemas import Recipe


class Vocabulary:

    def __init__(self, recipes: Iterable[Recipe]):
        names = set(T.STAPLES) | set(T.DERIVES) | set(T.ALIASES.values())
        for r in recipes:
            for i in r.items:
                names.add(i.name)
                names.add(i.family)
        self.names = names
        self.tokens = {tok for n in names for tok in re.findall(r"[a-z]+", n)}

    def canon(self, raw: str) -> str:
        return canonicalize(raw, self.names)

    def is_known(self, term: str) -> bool:
        c = self.canon(term)
        return c in self.names or c in self.tokens


def parse_inventory_text(text: str) -> List[str]:
    parts = re.split(r"[,;\n]|\band\b|&", text or "", flags=re.I)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class Coverage:
    missing: List[str] = field(default_factory=list)
    optional_missing: List[str] = field(default_factory=list)
    substitutions: Dict[str, str] = field(default_factory=dict)


class Inventory:
    def __init__(self, vocab: Vocabulary, items: Iterable[str]):
        given = [i for i in items if i and i.strip()]
        self.unknown = [i for i in given if not vocab.is_known(i)]
        have = {vocab.canon(i) for i in given} | set(T.STAPLES)
        for c in list(have):
            have.update(T.DERIVES.get(c, ()))
        self.have = have
        self.families = {T.family_of(c) for c in have}

    def has(self, name: str) -> bool:
        return name in self.have or T.family_of(name) in self.families

    def check(self, recipe: Recipe) -> Coverage:
        cov = Coverage()
        seen = set()
        for item in recipe.items:
            if item.name in seen:
                continue
            seen.add(item.name)
            if self.has(item.name):
                continue
            alt = next((a for a in T.SUBSTITUTES.get(item.name, []) + T.SUBSTITUTES.get(item.family, [])
                        if self.has(a)), None)
            if alt:
                cov.substitutions[item.name] = alt
            elif item.optional:
                cov.optional_missing.append(item.name)
            else:
                cov.missing.append(item.name)
        return cov
