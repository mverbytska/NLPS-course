from collections import Counter
from functools import lru_cache
from typing import List

from cocktail.schemas import Recipe

STT_MODEL = "small"


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel
    return WhisperModel(STT_MODEL, device="auto", compute_type="auto")


def hint_words(recipes: List[Recipe], n: int = 40) -> List[str]:
    return [name for name, _ in Counter(i.name for r in recipes for i in r.items).most_common(n)]


def transcribe(audio_path: str, hints: List[str] = ()) -> str:
    kw = dict(vad_filter=True, beam_size=1, initial_prompt="Cocktail request. Ingredients: " + ", ".join(hints))
    segments, info = _model().transcribe(audio_path, task="transcribe", **kw)
    if info.language != "en":
        segments, info = _model().transcribe(audio_path, task="translate", language=info.language, **kw)
    return " ".join(s.text.strip() for s in segments).strip()


def record(seconds: int = 8) -> str:
    import tempfile
    import wave
    try:
        import sounddevice as sd
    except ImportError as e:
        raise RuntimeError("Recording needs `pip install sounddevice` (or use /voice with an audio file).") from e
    rate = 16000
    data = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="int16")
    sd.wait()
    path = tempfile.mkstemp(suffix=".wav")[1]
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())
    return path
