from typing import List, Optional, Tuple

from cocktail.inventory import parse_inventory_text
from cocktail.memory import Profile, Session
from cocktail.pipeline import CocktailPipeline
from cocktail.schemas import Reply
from cocktail.stt import hint_words, transcribe

ALCOHOL_CHOICES = [("Any drink", "any"), ("Alcohol-free", "none"), ("With alcohol", "only")]
ALCOHOL_ALIASES = {"any": "any", "all": "any", "none": "none", "free": "none", "off": "none",
                   "only": "only", "alcoholic": "only", "on": "only"}


def set_exclude(pipe: CocktailPipeline, session: Session, text: str) -> None:
    session.state.exclude = list(dict.fromkeys(pipe.vocab.canon(x) for x in parse_inventory_text(text)))


def turn(pipe: CocktailPipeline, message: str, session: Session, bar: List[str], allow_missing: int = 1,
         top_k: int = 3, alcohol: Optional[str] = None, exclude_text: Optional[str] = None) -> Tuple[Reply, List[str]]:
    if alcohol in ("any", "none", "only"):
        session.state.alcohol = alcohol
    if exclude_text is not None:
        set_exclude(pipe, session, exclude_text)
    profile = Profile(bar=list(bar))
    reply = pipe.respond(message, session, profile, allow_missing=allow_missing, top_k=top_k)
    profile.save()
    return reply, profile.bar


def hear(pipe: CocktailPipeline, audio_path: str) -> str:
    return transcribe(audio_path, hint_words(pipe.recipes))
