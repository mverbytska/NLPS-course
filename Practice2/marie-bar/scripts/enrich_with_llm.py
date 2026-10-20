#!/usr/bin/env python3
"""Let the local model add knowledge to cocktail/data/recipes.json (build time, not at run time).

  python scripts/enrich_with_llm.py --limit 20     # try it on 20 recipes first
  python scripts/enrich_with_llm.py                # one taste sentence + flavour tags per recipe (resumable)
  python scripts/enrich_with_llm.py --names        # also merge rare ingredient names into common ones

What the model writes:
  taste    one sentence per drink, used to build the search embeddings and shown to the judge
  flavors  extra flavour tags, added to the rule-based ones
  names    "baileys irish cream" -> "irish cream" and similar (only among near-identical names, only with --names)

Answers are cached in cocktail/data/llm_cache.json, so re-running costs nothing for finished recipes
"""
import argparse
import difflib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cocktail.data_io import DATA_DIR, RECIPES_PATH, load_recipes
from cocktail.llm import OllamaLLM
from cocktail.taxonomy import FLAVOR_VOCAB, family_of

CACHE_PATH = DATA_DIR / "llm_cache.json"


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", action="store_true", help="also merge rare ingredient names into common ones")
    ap.add_argument("--limit", type=int, default=0, help="process at most N new recipes this run (0 = all)")
    args = ap.parse_args()

    llm = OllamaLLM()
    ready, msg = llm.status()
    if not ready:
        sys.exit(msg)
    recipes = load_recipes()
    try:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        cache = {"recipes": {}, "names": {}}

    todo = [r for r in recipes if r.id not in cache["recipes"]]
    todo = todo[:args.limit] if args.limit else todo
    print(f"describing {len(todo)} recipes with {llm.model} ({len(recipes) - len(todo)} already done)...")
    failed = 0
    for n, r in enumerate(todo, 1):
        tags = llm.tag_recipe(r.name, [i.name for i in r.items if not i.optional], r.instructions or "")
        if tags is None:
            failed += 1
        else:
            cache["recipes"][r.id] = {"flavors": list(tags.flavors), "taste": tags.taste.strip()}
        if n % 25 == 0 or n == len(todo):
            save_json(CACHE_PATH, cache)
            print(f"  {n}/{len(todo)}")
    if failed:
        print(f"  {failed} recipes failed (last error: {llm.last_error}); run again to retry them")

    if args.names:
        counts = Counter(i.name for r in recipes for i in r.items)
        common = [n for n, c in counts.items() if c >= 3]
        rare = [n for n, c in counts.items() if c <= 2 and n not in cache["names"]]
        print(f"checking {len(rare)} rare ingredient names...")
        for name in rare:
            close = difflib.get_close_matches(name, common, n=8, cutoff=0.5)
            cache["names"][name] = llm.same_ingredient(name, close) if close else ""
            if cache["names"][name]:
                print(f"  {name!r} -> {cache['names'][name]!r}")
        save_json(CACHE_PATH, cache)

    renamed = 0
    for r in recipes:
        found = cache["recipes"].get(r.id)
        if found:
            r.taste = found["taste"]
            r.flavors = [f for f in FLAVOR_VOCAB if f in set(r.flavors) | set(found["flavors"])]
        for item in r.items:
            target = cache["names"].get(item.name)
            if target:
                item.name, item.family = target, family_of(target)
                renamed += 1
    save_json(RECIPES_PATH, [r.model_dump(mode="json") for r in recipes])
    described = sum(1 for r in recipes if r.taste)
    print(f"Wrote {RECIPES_PATH.name}: {described}/{len(recipes)} with a taste line, {renamed} ingredient names merged.")
    print("The search index rebuilds itself on the next start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())