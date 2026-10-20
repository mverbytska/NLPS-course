import re
from typing import List, Optional

from cocktail.inventory import Vocabulary
from cocktail.llm import OllamaLLM
from cocktail.memory import State
from cocktail.render import intent_summary
from cocktail.schemas import Intent

RESET_WORDS = ("start over", "start fresh", "forget", "reset")
ALCOHOL_WORDS = ("alcohol", "alcoholic", "booze", "boozy", "mocktail", "virgin", "sober", "proof")


def _dedupe(xs) -> List[str]:
    return list(dict.fromkeys(xs))


def _in_text(term: str, message: str) -> bool:
    return any(re.search(rf"\b{re.escape(w)}(?:e?s)?\b", message.lower()) for w in term.replace("-", " ").split())


def _said(terms: List[str], message: str, vocab: Vocabulary) -> List[str]:
    return _dedupe(vocab.canon(t) for t in terms if t.strip() and _in_text(t, message))


def _context(prev: Optional[Intent], shown: List[str], state: State) -> str:
    bits = []
    if prev is not None:
        bits.append(intent_summary(prev))
        if shown:
            bits.append("drinks already shown: " + ", ".join(shown))
    if state.describe():
        bits.append("standing: " + state.describe())
    return "; ".join(bits) or "none"


def understand(message: str, vocab: Vocabulary, llm: OllamaLLM, prev: Optional[Intent] = None,
               shown: Optional[List[str]] = None, state: Optional[State] = None) -> Intent:
    state = state if state is not None else State()
    parsed = llm.understand(message, _context(prev, shown or [], state))
    if parsed is None:
        return Intent(query=message, action="search", model_failed=True)
    exclude = _said(parsed.exclude, message, vocab)
    intent = Intent(
        query=message,
        action=parsed.action,
        flavors=_dedupe(parsed.flavors),
        avoid_flavors=_dedupe(parsed.avoid_flavors),
        exclude=exclude,
        must_include=[m for m in _said(parsed.must_include, message, vocab) if vocab.is_known(m) and m not in exclude],
        strength=None if parsed.strength == "any" else parsed.strength,
        alcohol=parsed.alcohol if any(_in_text(w, message) for w in ALCOHOL_WORDS) else "keep",
        have=[h for h in _said(parsed.have, message, vocab) if vocab.is_known(h)],
        vibe=parsed.vibe.strip(),
        lookup=parsed.lookup.strip(),
        followup=bool(prev and parsed.followup),
    )
    reset = parsed.reset and any(w in message.lower() for w in RESET_WORDS)
    _apply_state(intent, state, reset, set(_said(parsed.lift, message, vocab)))
    return _carry_over(intent, prev) if intent.followup and prev else intent


def _apply_state(intent: Intent, state: State, reset: bool, lifted: set) -> None:
    if reset:
        state.clear()
    if intent.alcohol != "keep":
        state.alcohol = intent.alcohol
    intent.alcohol = state.alcohol
    state.exclude = [x for x in state.exclude if x not in lifted and x not in intent.must_include]
    state.avoid_flavors = [x for x in state.avoid_flavors if x not in lifted and x not in intent.flavors]
    intent.exclude = [x for x in intent.exclude if x not in lifted]
    intent.avoid_flavors = [x for x in intent.avoid_flavors if x not in lifted]

    intent.exclude = _dedupe(state.exclude + intent.exclude)
    intent.avoid_flavors = _dedupe(state.avoid_flavors + intent.avoid_flavors)
    intent.flavors = [f for f in intent.flavors if f not in intent.avoid_flavors]

    state.exclude, state.avoid_flavors = intent.exclude, intent.avoid_flavors


def _carry_over(new: Intent, prev: Intent) -> Intent:
    return new.model_copy(update={
        "flavors": [f for f in (new.flavors or prev.flavors) if f not in new.avoid_flavors],
        "must_include": new.must_include or prev.must_include,
        "strength": new.strength or prev.strength,
        "vibe": new.vibe or prev.vibe,
        "lookup": new.lookup or prev.lookup,
    })
