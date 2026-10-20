import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from cocktail.schemas import Draft, Intent

STORE = Path(os.environ.get("MARIE_HOME", Path(__file__).resolve().parents[1])) / "marie_bar.json"
UNSURE_LOG = Path(os.environ.get("MARIE_HOME", Path(__file__).resolve().parents[1])) / "marie_unsure.jsonl"
DEFAULT_BAR = ["rum", "limes", "mint", "sugar", "soda water", "gin", "lemons"]
ALCOHOL_TEXT = {"any": "", "none": "alcohol-free", "only": "with alcohol"}


@dataclass
class Profile:
    bar: List[str] = field(default_factory=lambda: list(DEFAULT_BAR))

    @classmethod
    def load(cls, path: Path = STORE) -> "Profile":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return cls()
        bar = data if isinstance(data, list) else data.get("bar", [])
        return cls(bar=[str(x) for x in bar])

    def save(self, path: Path = STORE) -> None:
        try:
            Path(path).write_text(json.dumps({"bar": self.bar}, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass


@dataclass
class State:
    alcohol: str = "any"
    exclude: List[str] = field(default_factory=list)
    avoid_flavors: List[str] = field(default_factory=list)

    def clear(self) -> None:
        self.alcohol, self.exclude, self.avoid_flavors = "any", [], []

    def describe(self) -> str:
        bits = [ALCOHOL_TEXT.get(self.alcohol, "")] + [f"no {x}" for x in self.exclude] \
            + [f"less {x}" for x in self.avoid_flavors]
        return ", ".join(b for b in bits if b)


@dataclass
class Session:
    state: State = field(default_factory=State)
    last_intent: Optional[Intent] = None
    last_route: str = ""
    last_draft: Optional[Draft] = None
    last_ids: List[str] = field(default_factory=list)
    shown_ids: List[str] = field(default_factory=list)
    turns: List[dict] = field(default_factory=list)

    def remember(self, user: str, assistant: str) -> None:
        self.turns += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        self.turns = self.turns[-8:]


def log_unsure(message: str, route: str, reason: str, trace: List[str]) -> None:
    try:
        with open(UNSURE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"message": message, "route": route, "reason": reason, "trace": trace}, ensure_ascii=False) + "\n")
    except OSError:
        pass
