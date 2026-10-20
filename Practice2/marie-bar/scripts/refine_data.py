#!/usr/bin/env python3
"""Build cocktail/data/recipes.json from TheCocktailDB, fast and incrementally.

  python scripts/refine_data.py                # rebuild recipes.json (downloads everything first if raw_cocktaildb.json is missing)
  python scripts/refine_data.py --fetch        # top up the raw cache from the API (only what is missing)
  python scripts/refine_data.py --fetch --refresh   # also re-download the letter pages (picks up edited recipes)

How the download works now (about 40 calls for an update instead of hundreds):
  1. letter / digit pages  - 36 calls, up to 25 full recipes each (only on a first run or with --refresh)
  2. category + alcoholic filters - ~15 calls that list EVERY recipe id, uncapped
  3. lookup by id          - only for ids that are not in the raw cache yet
Calls run in parallel behind one global rate limiter, and the raw cache is never thrown away.
"""
import argparse
import json
import string
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cocktail.data_io import DATA_DIR
from cocktail.enrich import build_items, enrich
from cocktail.schemas import Recipe

BASE_URL = "https://www.thecocktaildb.com/api/json/v1/1"
FIRST_CHARS = string.ascii_lowercase + string.digits
RAW_PATH = DATA_DIR / "raw_cocktaildb.json"
OUT_PATH = DATA_DIR / "recipes.json"


class Client:
    """requests session with retries and ONE rate limiter shared by all worker threads."""

    def __init__(self, rps: float = 8.0, workers: int = 4):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self.interval, self._next, self._lock, self.calls = 1.0 / rps, 0.0, threading.Lock(), 0
        self.session = requests.Session()
        retry = Retry(total=4, backoff_factor=0.7, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
        self.session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=workers * 2))

    def drinks(self, endpoint: str, **params) -> list:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
            self.calls += 1
        time.sleep(wait)
        r = self.session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=20)
        r.raise_for_status()
        try:
            found = (r.json() or {}).get("drinks")
        except ValueError:
            return []
        return found if isinstance(found, list) else []


def update_raw(client: Client, raw: Dict[str, dict], workers: int, refresh: bool, log=print) -> dict:
    stats = {"from_letters": 0, "from_filters": 0}
    pool = ThreadPoolExecutor(workers)

    if refresh or not raw:
        for found in pool.map(lambda c: client.drinks("search.php", f=c), FIRST_CHARS):
            for d in found:
                if d.get("idDrink"):
                    stats["from_letters"] += d["idDrink"] not in raw
                    raw[d["idDrink"]] = d
        log(f"[1] letter pages: {len(raw)} recipes")

    cats = [d["strCategory"] for d in client.drinks("list.php", c="list")]
    alcs = [d["strAlcoholic"] for d in client.drinks("list.php", a="list")]
    jobs = [{"c": v} for v in cats] + [{"a": v} for v in alcs]
    ids = {d["idDrink"] for found in pool.map(lambda p: client.drinks("filter.php", **p), jobs) for d in found}
    missing = sorted(ids - set(raw), key=int)
    log(f"[2] filters know {len(ids)} recipes, {len(missing)} not in the cache yet")

    for found in pool.map(lambda i: client.drinks("lookup.php", i=i), missing):
        for d in found:
            raw[d["idDrink"]] = d
            stats["from_filters"] += 1
    pool.shutdown()
    stats["queries"] = client.calls
    stats["known_ids"] = len(ids)
    return stats


def _norm_glass(g: Optional[str]) -> Optional[str]:
    g = " ".join((g or "").split())
    return g[:1].upper() + g[1:].lower() if g else None


def parse_drink(d: dict) -> Recipe:
    raws = [d.get(f"strIngredient{i}") or "" for i in range(1, 16)]
    measures = [d.get(f"strMeasure{i}") for i in range(1, 16)]
    label = (d.get("strAlcoholic") or "").strip()
    low = label.lower()
    alcoholic = True if low == "alcoholic" else False if low.startswith("non") else None
    recipe = Recipe(
        id=str(d["idDrink"]),
        name=(d.get("strDrink") or "").strip(),
        category=(d.get("strCategory") or "").strip() or None,
        alcoholic=alcoholic,
        alcohol_label=label or None,
        glass=_norm_glass(d.get("strGlass")),
        instructions=" ".join((d.get("strInstructions") or "").split()) or None,
        items=build_items(raws, measures),
        tags=[t.strip().lower() for t in (d.get("strTags") or "").split(",") if t.strip()],
        iba=(d.get("strIBA") or None),
        thumb=(d.get("strDrinkThumb") or None),
    )
    return enrich(recipe)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="top up the raw cache from the API (needs network)")
    ap.add_argument("--refresh", action="store_true", help="with --fetch: also re-download the letter pages")
    ap.add_argument("--workers", type=int, default=4, help="parallel API calls")
    ap.add_argument("--rps", type=float, default=8.0, help="max API calls per second (be kind to the free API)")
    args = ap.parse_args()

    raw = {str(d["idDrink"]): d for d in load_json(RAW_PATH, [])}
    if args.fetch or not raw:
        stats = update_raw(Client(args.rps, args.workers), raw, args.workers, args.refresh)
        save_json(RAW_PATH, [raw[k] for k in sorted(raw, key=int)])
        print(f"Raw cache: {len(raw)} drinks, {stats['queries']} API calls, "
              f"{stats['from_letters'] + stats['from_filters']} new -> {RAW_PATH.name}")

    recipes = sorted((parse_drink(d) for d in raw.values()), key=lambda r: r.name.casefold())

    save_json(OUT_PATH, [r.model_dump(mode="json") for r in recipes])
    print(f"Wrote {len(recipes)} recipes -> {OUT_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
