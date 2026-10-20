# marie bar 🍸

hey, i'm marie, a cocktail assistant that lives on your computer. tell me what you feel like (type it or say it), and what's in your bar, and i'll suggest what you can make tonight, look drinks up in my recipe book, or invent a new one for you. i remember your bar and the rules you set ("no alcohol tonight", "no gin"), and everything runs locally, so you can still chat with Claude when having a nice drink.

## what i can do

* **suggest from your bar.** "something sour and refreshing", "something cozy for a rainy evening". python works out which drinks your bar can make (you choose how many missing ingredients are OK), and the model picks the ones that fit your mood best.
* **look things up.** "how do i make a negroni?", "Aperol", "what can i make with campari and gin?". i search my recipe book by meaning, not by exact words, and i tell you honestly when a drink isn't in it.
* **invent new drinks.** "invent a drink with campari and grapefruit juice". i draft a recipe, check it, and only show it if it passes.
* **keep a conversation going.** "another one, but less sweet" works. drinks i already showed don't come back.
* **remember your rules.** say "no alcohol tonight" or "no gin" once and it stays on until you take it back ("alcohol is fine again", "gin is ok now", "start over").
* **learn your bar while we talk.** "i just bought some campari" and it's in your bar.
* **listen.** use the microphone (or an audio file) instead of typing. other languages are translated to english first.
* **say when i'm not sure.** every unsure or failed turn is written to `marie_unsure.jsonl` so you can see exactly what went wrong.

## how it works

```
message ─► understand ─► ROUTE ─┬─ suggest  drinks your bar can make ─►  semantic top 20 ─► model judge picks the best fits
           the model reads it   │           (nothing makeable? loosen the bar rule step by step, then invent)
           into structured JSON ├─ search   semantic top 20 from the whole book ─► model judge: exact / related / none
           + your standing rules├─ invent   model drafts ─► python validates ─► problems go back to the model (up to 3 tries)
                                └─ chat     conversation with memory
```

## who decides what

the model decides everything that needs understanding or taste. python only keeps the hard facts.

| LLM | code |
|---|---|
| reads every message into structured JSON: route, flavours, exclusions, alcohol setting, ingredients, follow-ups, a search phrase | checks that anything the model reports about exclusions, alcohol, ingredients and resets is really in your message, so it can't invent "no gin" |
| judges which drinks fit (search and suggest) | filters by your exclusions and alcohol setting, works out what your bar covers |
| writes the intros, explanations, chat answers, descriptive human-like phrases | checks the text mentions no ingredient the drink doesn't have |
| invents new drinks | validates them: known ingredients, your rules, alcohol, amounts, your bar |

the embedding model only fetches the 20 closest candidates.

## system parts

| part | file | what it does |
|---|---|---|
| understanding and routing | `cocktail/intent.py` | the model reads your message into a structured request and picks the route. no keyword or regex parsing: python only drops things the model reports that aren't in your message, and keeps your standing rules |
| the router and writer | `cocktail/llm.py`, `prompts.py` | talks to Ollama. every model task lives here: understand, judge search results, write intros and explanations, chat, invent drinks |
| semantic search | `cocktail/index.py`, `tools.py` | turns every recipe into a vector once, then finds the recipes closest in meaning to what you asked |
| recipe book + bar logic | `cocktail/inventory.py`, `matcher.py`, `taxonomy.py`, `enrich.py` | ingredient names and substitutes (limes cover lime juice), what your bar covers, the hard rules (exclusions, alcohol setting) |
| inventing | `cocktail/generate.py` | model drafts a drink as JSON, python checks it: known ingredients, your rules, alcohol setting, sane amounts, your bar. problems go back to the model |
| memory | `cocktail/memory.py` | your bar (saved in `marie_bar.json`) and the conversation state: alcohol setting, never-include list, flavours to avoid, last drink |
| voice | `cocktail/stt.py` | speech to text |
| the pipeline | `cocktail/pipeline.py`, `chat.py` | ties it all together; the web UI and the terminal both use it, so they behave the same |
| the two skins | `app_gradio.py`, `app_cli.py` | web UI and terminal chat |
| data builder | `scripts/refine_data.py` | optional: rebuilds the recipe book from TheCocktailDB |
| input enrichment | `scripts/enrich_with_llm.py` | adding more natural spoken context to the data for a better semantic search |

## the models

all three run locally.

| job | model | via |
|---|---|---|
| understanding, routing, judging search results, writing, chatting, inventing | `qwen2.5:3b-instruct` | Ollama |
| searching by meaning | `nomic-embed-text` | Ollama |
| listening | Whisper `small` | faster-whisper |

the chat and embedding models are set at the top of `cocktail/llm.py`, the voice model in `cocktail/stt.py`. the chat model is a small, non-thinking instruct model on purpose: it follows JSON formats reliably and answers fast, which matters because one message can trigger several model calls.

## run it

```bash
ollama pull qwen2.5:3b-instruct
ollama pull nomic-embed-text
pip install -r requirements.txt

python app_gradio.py     # web UI at http://127.0.0.1:7860
python app_cli.py        # terminal chat, /help lists the commands
```

on the first start i index the recipes once (cached in `cocktail/data/embeddings.npz`). please note that audio input is disabled for the cli setup.

## talking to me

things you can say:

```
something sour and refreshing, no gin
no alcohol tonight, something fruity
another one, but less sweet
Aperol
how do i make a negroni?
i just bought some campari, what can i do with it?
invent a new drink with campari and grapefruit juice
only with ingredients i have, what can i do?
```

the web UI and the terminal do the same things (except for audio):

| | web UI | terminal |
|---|---|---|
| your bar | "My bar" box (saved automatically) | `/bar`, `/add`, `/remove` |
| alcohol | "Alcohol" buttons | `/alcohol any\|none\|only` |
| never include | "Never include" box | `/exclude gin, cream` |
| missing ingredients allowed | 0 / 1 / 2 buttons | `/missing 0\|1\|2` |
| how many suggestions | slider | `/top N` |
| voice | microphone or upload | n/a |
| what i understood | "What I understood" panel (route, tools used) | `/state`, plus a line under each answer |
| new conversation | clear icon on the chat | `/new` |

## the data

the app only needs `cocktail/data/recipes.json`: 637 drinks from [TheCocktailDB](https://www.thecocktaildb.com/). to rebuild or refresh it (for example after editing `cocktail/taxonomy.py`), run `python scripts/refine_data.py`. it downloads first if the raw cache is missing, so it needs a network.

## where i can be wrong

* **a small model misreads things.** routing and understanding are the model's job, so the 3B model can misroute or misread a message. the checks only catch things it invents, not every misunderstanding. 
* **search can miss.** i fetch the 20 closest recipes before judging them, so the right drink can occasionally fall outside that list and i'll say "not in my book" when it actually is.
* **new combinations are legal, but not necessarily tasty.** the checker proves a drink uses known ingredients, respects your rules and has sane amounts. gin 60 ml + milk 120 ml + Worcestershire + Tabasco passes every check easily, but most probably it's not one's drink of choice.

if something looks off, check `marie_unsure.jsonl`: it records the message, the route, the tools used and the reason.

cheers! 🥂

