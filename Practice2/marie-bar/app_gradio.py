#!/usr/bin/env python3
import traceback
from functools import lru_cache

import gradio as gr

from cocktail.chat import ALCOHOL_CHOICES, hear, turn
from cocktail.data_io import load_recipes
from cocktail.inventory import parse_inventory_text
from cocktail.llm import OllamaLLM
from cocktail.memory import Profile, Session
from cocktail.pipeline import CocktailPipeline
from cocktail.render import to_markdown, unlocks_markdown

EXAMPLES = [
    ["Something sour and refreshing, no gin"],
    ["No alcohol tonight, something fruity"],
    ["A strong, boozy drink with whiskey"],
    ["Something cozy for a rainy evening"],
    ["Creamy dessert drink, no rum or vodka"],
    ["Surprise me, only what I have"],
]
GREETING = ("Hey, I'm Marie 🍸 Tell me what you feel like (type it or say it), what you have in your bar, "
            "or ask me about a drink. Say *no alcohol tonight* or *no gin* once and I'll keep it in mind "
            "until you take it back (or use the controls on the left). Follow up with *another one* or *less sweet*.")

RECIPES = load_recipes()
NO_CHANGE = gr.update()


@lru_cache(maxsize=1)
def get_pipeline() -> CocktailPipeline:
    return CocktailPipeline(recipes=RECIPES, llm=OllamaLLM())


def chat(message, history, bar_text, allow, top_k, alcohol, exclude_text, session, display=None):
    history, message = list(history or []), (message or "").strip()
    unchanged = (session, bar_text, alcohol, exclude_text)
    if not message:
        return history, "", *unchanged, NO_CHANGE, NO_CHANGE
    history.append({"role": "user", "content": display or message})
    try:
        pipe = get_pipeline()
    except Exception as e:
        history.append({"role": "assistant", "content": f"⚠️ {e}"})
        return history, "", *unchanged, NO_CHANGE, NO_CHANGE

    reply, bar = turn(pipe, message, session, parse_inventory_text(bar_text), int(allow), int(top_k),
                      alcohol=alcohol, exclude_text=exclude_text)
    history.append({"role": "assistant", "content": to_markdown(reply)})
    shop = unlocks_markdown(reply) if reply.unlocks else ""
    debug = {"route": reply.route, "tools": reply.trace, "intent": reply.intent.model_dump()}
    return (history, "", session, ", ".join(bar), session.state.alcohol, ", ".join(session.state.exclude),
            shop, debug)


def voice(audio, auto_send, history, bar_text, allow, top_k, alcohol, exclude_text, session):
    history = list(history or [])
    skip = (history, NO_CHANGE, session, bar_text, alcohol, exclude_text, NO_CHANGE, NO_CHANGE)
    if audio is None:
        return skip
    try:
        text = hear(get_pipeline(), audio)
    except Exception as e:
        traceback.print_exc()
        history.append({"role": "assistant", "content": f"⚠️ I couldn't process the audio: {e}"})
        return (history, *skip[1:])
    if not text:
        history.append({"role": "assistant", "content": "I couldn't hear anything. Try again?"})
        return (history, *skip[1:])
    if not auto_send:
        return (history, text, *skip[2:])
    return chat(text, history, bar_text, allow, top_k, alcohol, exclude_text, session, display=f"🎙️ {text}")


def save_bar(text: str) -> None:
    Profile(bar=parse_inventory_text(text)).save()


with gr.Blocks(title="marie bar", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍸 marie bar\nYour bar, your mood, one conversation.")
    session = gr.State(Session())
    with gr.Row():
        with gr.Column(scale=1):
            bar_box = gr.Textbox(label="My bar (comma or line separated, saved automatically)", lines=5)
            alcohol = gr.Radio(ALCOHOL_CHOICES, value="any", label="Alcohol")
            exclude_box = gr.Textbox(label="Never include (comma separated)", placeholder="e.g. gin, cream")
            allow = gr.Radio([0, 1, 2], value=1, label="Missing ingredients I'm OK with")
            top_k = gr.Slider(1, 5, value=3, step=1, label="How many suggestions")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(value=[{"role": "assistant", "content": GREETING}], type="messages", height=480)
            msg = gr.Textbox(label="What are you in the mood for?", placeholder="something sour and refreshing, no gin", lines=1)
            mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="…or say it")
            auto_send = gr.Checkbox(value=True, label="Send voice requests right away (untick to edit the transcript first)")
            gr.Examples(EXAMPLES, inputs=[msg], label="Try")
            shop = gr.Markdown()
            with gr.Accordion("What I understood", open=False):
                intent_json = gr.JSON()

    inputs = [msg, chatbot, bar_box, allow, top_k, alcohol, exclude_box, session]
    outputs = [chatbot, msg, session, bar_box, alcohol, exclude_box, shop, intent_json]
    msg.submit(chat, inputs, outputs)
    voice_in = [mic, auto_send, chatbot, bar_box, allow, top_k, alcohol, exclude_box, session]
    mic.stop_recording(voice, voice_in, outputs)
    mic.upload(voice, voice_in, outputs)
    chatbot.clear(lambda: (Session(), "any", ""), None, [session, alcohol, exclude_box])

    demo.load(lambda: ", ".join(Profile.load().bar), None, bar_box)
    bar_box.blur(save_bar, bar_box, None)


if __name__ == "__main__":
    try:
        import faster_whisper
    except ImportError:
        print("[warn] voice input is off: pip install faster-whisper")
    try:
        get_pipeline()
    except Exception as e:
        print(f"[warn] {e}\n[warn] the UI will start anyway and tell you again when you ask for a drink.")
    demo.launch()
