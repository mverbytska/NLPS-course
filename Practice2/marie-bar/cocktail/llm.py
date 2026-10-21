import json
import os
import sys
from typing import List, Literal, Optional, Tuple

import requests
from pydantic import BaseModel

from cocktail.prompts import (CHAT_SYSTEM, EXPLAIN_SYSTEM, GENERATE_SYSTEM, INTENT_SYSTEM, INTRO_SYSTEM, JUDGE_SYSTEM,
                              SAME_SYSTEM, TAG_SYSTEM, explain_user, intent_user, intro_user, judge_user, same_user, tag_user)
from cocktail.schemas import Draft
from cocktail.taxonomy import FLAVOR_VOCAB

CHAT_MODEL = "qwen2.5:3b-instruct"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

Flavor = Literal[tuple(FLAVOR_VOCAB)]


class LLMIntent(BaseModel):
    action: Literal["suggest", "search", "invent", "chat"]
    flavors: List[Flavor]
    avoid_flavors: List[Flavor]
    exclude: List[str]
    must_include: List[str]
    strength: Literal["any", "light", "medium", "strong"]
    alcohol: Literal["keep", "none", "any", "only"]
    have: List[str]
    followup: bool
    vibe: str
    lookup: str
    lift: List[str]
    reset: bool


class Judged(BaseModel):
    verdict: Literal["exact", "related", "none"]
    picks: List[int]
    note: str


class LLMTags(BaseModel):
    flavors: List[Flavor]
    taste: str


class SameAs(BaseModel):
    same_as: str


def extract_json(raw: str) -> str:
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


class OllamaLLM:
    def __init__(self, base_url: str = OLLAMA_URL, timeout: int = 180, keep_alive: str = "30m"):
        self.model = CHAT_MODEL
        self.embed_model = EMBED_MODEL
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.last_error = ""

    def status(self) -> Tuple[bool, str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            return False, f"Ollama is not reachable at {self.base_url}. Start it with `ollama serve`."
        for model in (self.model, self.embed_model):
            if model not in names and f"{model}:latest" not in names:
                return False, f"Model '{model}' not found. Run: ollama pull {model}"
        return True, "ok"

    def _chat(self, messages: List[dict], schema: Optional[dict] = None,
              max_tokens: int = 160, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 4096},
        }
        if schema:
            payload["format"] = schema
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama {r.status_code}: {r.text[:200]}")
        return r.json()["message"]["content"].strip()

    def _ask(self, system: str, user: str, **kw) -> str:
        return self._chat([{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)

    def _structured(self, system: str, user: str, cls, max_tokens: int, temperature: float = 0.1):
        self.last_error = ""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(2):
            try:
                raw = self._chat(messages, schema=cls.model_json_schema(), max_tokens=max_tokens, temperature=temperature)
            except Exception as e:
                self._fail(f"{type(e).__name__}: {e}")
                return None
            try:
                return cls(**json.loads(extract_json(raw)))
            except Exception as e:
                self._fail(f"unusable output ({type(e).__name__}): {raw[:120]!r}")
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": "That was not valid. Reply with ONLY one JSON object that "
                                                         f"matches the schema. Problem: {str(e)[:200]}"}]
        return None

    def _fail(self, reason: str) -> None:
        self.last_error = reason
        print(f"[marie] model call problem: {reason}", file=sys.stderr)

    def understand(self, message: str) -> Optional[LLMIntent]:
        return self._structured(INTENT_SYSTEM, intent_user(message), LLMIntent, max_tokens=300)

    def explain(self, name: str, ingredients: List[str], flavors: List[str], request: str) -> Optional[str]:
        try:
            return self._ask(EXPLAIN_SYSTEM, explain_user(name, ingredients, flavors, request),
                             max_tokens=90, temperature=0.3)
        except Exception:
            return None

    def intro(self, request: str, names: List[str]) -> Optional[str]:
        try:
            return self._ask(INTRO_SYSTEM, intro_user(request, names), max_tokens=50, temperature=0.5)
        except Exception:
            return None

    def chat_reply(self, history: List[dict], message: str, context: str) -> Optional[str]:
        messages = [{"role": "system", "content": f"{CHAT_SYSTEM}\n\n{context}"}]
        messages += history[-6:] + [{"role": "user", "content": message}]
        try:
            return self._chat(messages, max_tokens=180, temperature=0.5)
        except Exception:
            return None

    def judge(self, message: str, lines: List[str], context: str = "") -> Optional[Judged]:
        return self._structured(JUDGE_SYSTEM, judge_user(message, lines, context), Judged, max_tokens=220)

    def draft(self, user: str, temperature: float = 0.5) -> Optional[Draft]:
        return self._structured(GENERATE_SYSTEM, user, Draft, max_tokens=500, temperature=temperature)

    def tag_recipe(self, name: str, ingredients: List[str], instructions: str) -> Optional[LLMTags]:
        return self._structured(TAG_SYSTEM, tag_user(name, ingredients, instructions), LLMTags, max_tokens=120)

    def same_ingredient(self, name: str, candidates: List[str]) -> str:
        answer = self._structured(SAME_SYSTEM, same_user(name, candidates), SameAs, max_tokens=40)
        return answer.same_as.strip() if answer and answer.same_as.strip() in candidates else ""

    def embed(self, texts: List[str]) -> List[List[float]]:
        r = requests.post(f"{self.base_url}/api/embed", timeout=self.timeout,
                          json={"model": self.embed_model, "input": texts, "keep_alive": self.keep_alive})
        if r.status_code != 200:
            raise RuntimeError(f"Ollama {r.status_code}: {r.text[:200]}")
        return r.json()["embeddings"]
