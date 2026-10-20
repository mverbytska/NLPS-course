import json

from cocktail.taxonomy import FLAVOR_VOCAB

_BASE = dict(action="suggest", flavors=[], avoid_flavors=[], exclude=[], must_include=[], strength="any",
             alcohol="keep", have=[], followup=False, vibe="", lookup="", lift=[], reset=False)


def _ex(request: str, context: str = "none", **kw) -> str:
    out = json.dumps({**_BASE, **kw}, separators=(",", ":"))
    return f"Previous request: {context}\nRequest: {request}\n{out}"


_EXAMPLES = "\n\n".join([
    _ex("something sour and refreshing, no gin", flavors=["sour", "refreshing"], exclude=["gin"], vibe="sour and refreshing"),
    _ex("a strong drink with whiskey, not too sweet", avoid_flavors=["sweet"], must_include=["whiskey"],
        strength="strong", vibe="strong whiskey drink"),
    _ex("something cozy for a rainy evening", flavors=["warm"], vibe="cozy drink for a rainy evening"),
    _ex("no alcohol tonight, something fruity", flavors=["fruity"], alcohol="none", vibe="fruity"),
    _ex("actually alcohol is fine again", "flavours: fruity; standing: alcohol-free", alcohol="any", followup=True),
    _ex("gin is ok now, something sour", "none; standing: no gin", flavors=["sour"], lift=["gin"], vibe="sour"),
    _ex("another one, but less sweet", "flavours: sour; without: gin", avoid_flavors=["sweet"], followup=True),
    _ex("Aperol", action="search", must_include=["aperol"], lookup="aperol"),
    _ex("what can I make with campari and gin?", action="search", must_include=["campari", "gin"],
        lookup="drinks with campari and gin"),
    _ex("I just bought some campari, what can I do with it?", action="search", must_include=["campari"],
        have=["campari"], lookup="drinks with campari"),
    _ex("how do I make a negroni?", action="search", lookup="negroni"),
    _ex("is there a drink called blue lagoon?", action="search", lookup="blue lagoon"),
    _ex("invent a new drink with grapefruit and campari", action="invent", must_include=["grapefruit", "campari"],
        vibe="new bitter grapefruit drink"),
    _ex("what can I make with what I have?", action="suggest"),
    _ex("come up with something new using tequila", action="invent", must_include=["tequila"], vibe="new tequila drink"),
    _ex("make it less bitter", "flavours: bitter; must use: campari", avoid_flavors=["bitter"], followup=True),
    _ex("thanks, that was great!", action="chat"),
])

INTENT_SYSTEM = f"""You are the ears of a bartender called Marie. Read the user's message and answer with JSON only.

Fields:
- action: the route for this message.
    "suggest"    mood or taste based recommendations from the user's bar (also follow-ups like "another one")
    "search"     look something up in the recipe book: a drink by name ("how do I make a negroni?"), drinks featuring an ingredient ("Aperol"), or a style
    "invent"     the user wants a NEW or custom drink invented ("invent something with grapefruit")
    "chat"       greetings, thanks, general questions
- flavors: zero or more of {FLAVOR_VOCAB} the user WANTS. Use only these words.
- avoid_flavors: flavors from the same list the user does NOT want ("not too sweet" -> ["sweet"]).
- exclude: ingredients the user does NOT want (lower-case, singular).
- must_include: ingredients the user explicitly wants to use or asks about (not drink names).
- strength: "any", "light", "medium" or "strong".
- alcohol: "none" if the user wants no alcohol; "only" if they insist on drinks with alcohol; "any" if they take back an earlier alcohol preference ("alcohol is fine again"); "keep" if the message says nothing about alcohol.
- have: ingredients the user says they own or just bought ("I just got gin"). Empty otherwise.
- followup: true if the message refers to the previous request or to drinks already shown.
- vibe: 3-8 words with the mood, occasion or taste in the user's own words. Empty string if none.
- lookup: for action "search", the short phrase to look up in the recipe book (a drink name, or what the drinks should contain), without filler words. Empty otherwise.
- lift: ingredients or flavours the user stops avoiding ("gin is ok now" -> ["gin"]). Empty otherwise.
- reset: true only if the user wants to forget all earlier preferences ("start over").

"Previous request" also lists standing preferences that stay on for the whole conversation.

Only report what the user wrote in THIS message. The examples show the output format only: never copy their ingredients, flavours or settings.

Examples:
{_EXAMPLES}
"""


def intent_user(message: str, context: str) -> str:
    return f"Previous request: {context}\nRequest: {message}"


EXPLAIN_SYSTEM = (
    "You are a concise bartender. Write exactly 2 short sentences explaining why the cocktail fits the request. "
    "Mention ONLY ingredients from the given list. Do not invent flavors, measures or facts."
)


def explain_user(name: str, ingredients: list, flavors: list, request: str) -> str:
    return (
        f"Request: {request}\n"
        f"Cocktail: {name}\n"
        f"Ingredients: {', '.join(ingredients)}\n"
        f"Flavor tags: {', '.join(flavors) or 'none'}"
    )


INTRO_SYSTEM = (
    "You are Marie, a warm, playful bartender. Write ONE short sentence (max 20 words) introducing the cocktails "
    "below for the user's request. Use only the cocktail names given; do not mention ingredients or invent facts."
)


def intro_user(request: str, names: list) -> str:
    return f"Request: {request}\nCocktails: {', '.join(names)}"


def judge_user(message: str, lines: list, context: str = "") -> str:
    cands = "\n".join(f"{n}. {l}" for n, l in enumerate(lines, 1))
    return f"Message: {message}\nPreferences: {context or 'none'}\nCandidates:\n{cands}"


def _judge_ex(message: str, lines: list, verdict: str, picks: list, note: str, context: str = "") -> str:
    out = json.dumps({"verdict": verdict, "picks": picks, "note": note}, separators=(",", ":"))
    return f"{judge_user(message, lines, context)}\n{out}"


JUDGE_SYSTEM = (
    "You pick drinks from a cocktail recipe book for the user. You get their message, their preferences and numbered "
    "candidate recipes found by semantic search, so some candidates may be irrelevant. Each candidate lists its "
    "ingredients and [tags]. Answer with JSON only.\n"
    "- verdict: 'exact' if the user asks for ONE specific drink and it is among the candidates (picks[0] is that drink); "
    "'related' if some candidates fit (drinks built around an asked-for ingredient, a similar style, or the mood or "
    "taste described, like 'something cozy' or 'sour and refreshing'); 'none' if nothing really fits, including when "
    "the named drink is not in the list.\n"
    "- picks: candidate numbers that fit, best first, at most 8. Empty for 'none'. Never pick a candidate that breaks "
    "the preferences (for example 'less: sweet'). Prefer drinks where an asked-for ingredient is a main ingredient, "
    "not a passing mention. Use only numbers from the list.\n"
    "- note: one short sentence about what you found, or why you are not sure.\n\nExamples:\n\n"
    + _judge_ex("how do I make a negroni?", ["Negroni: gin, campari, sweet vermouth [bitter, herbal; strong] - bitter and boozy, a classic aperitivo",
                                            "Boulevardier: bourbon, campari, sweet vermouth [bitter, herbal; strong] - a richer, whiskey-based cousin of the Negroni",
                                            "Americano: campari, sweet vermouth, soda water [bitter, refreshing; light] - light, bitter and fizzy"],
                "exact", [1], "Negroni is in my book.")
    + "\n\n"
    + _judge_ex("Aperol", ["Aperol Spritz: aperol, prosecco, soda water [bitter, refreshing; light] - light, bittersweet and bubbly for a summer afternoon",
                           "Mojito: rum, mint, lime, sugar, soda water [refreshing, herbal; light] - cool, minty and citrusy",
                           "Imperial Cocktail: lime juice, gin, aperol [sour, bitter; medium] - tart and bitter"],
                "related", [1, 3], "Two drinks here are built around Aperol.")
    + "\n\n"
    + _judge_ex("something cozy for a rainy evening", ["Hot Toddy: whiskey, honey, lemon juice [warm, sweet; medium] - a warm, sweet drink for cold evenings",
                                                      "Mojito: rum, mint, lime, sugar, soda water [refreshing, herbal; light] - cool, minty and citrusy",
                                                      "Irish Coffee: irish whiskey, coffee, cream [warm, coffee, creamy; medium] - warm, creamy and coffee-rich"],
                "related", [3], "A warm, creamy drink for a rainy evening.", "less: sweet")
    + "\n\n"
    + _judge_ex("blue meadow cocktail", ["Gin Fizz: gin, lemon juice, sugar, soda water [sour, refreshing; light] - sharp, lemony and fizzy",
                                        "Gimlet: gin, lime juice, sugar [sour, sweet; medium] - tart lime with a sweet edge"],
                "none", [], "I have no drink called Blue Meadow, only loosely similar ones.")
)


CHAT_SYSTEM = (
    "You are Marie, a warm, concise bartender living inside a home-bar app. Answer in at most 3 short sentences. "
    "You do not have the recipe book in front of you: if the user asks about a specific drink, tell them to ask "
    "'how do I make <drink>' so you can look it up. If they want a recommendation, invite them to say what they feel "
    "like (mood, flavours, what to avoid)."
)

GENERATE_SYSTEM = (
    "You are a careful bartender inventing ONE new cocktail. Answer with JSON only.\n"
    "Rules:\n"
    "- Use ONLY ingredients from the allowed list (plus ice and water). Use simple common names exactly as listed.\n"
    "- 3 to 6 ingredients. Realistic amounts: one serving is roughly 60-200 ml of liquid in total, "
    "a base spirit at most 60 ml. Units: ml, cl, oz, tsp, tbsp, dash, piece (piece for leaves, wedges, slices).\n"
    "- Respect every constraint in the request. Give the drink an original name.\n"
    "- 'method' is 1-3 short sentences. 'why' is one sentence about the taste, mentioning only your ingredients.\n"
    "The reference recipes are real cocktails: borrow their proportions, do not copy them."
)


def generate_user(message: str, constraints: list, allowed: list, refs: list, prev=None, problems=None) -> str:
    parts = [f"Request: {message}", "Constraints: " + ("; ".join(constraints) or "none"),
             "Allowed ingredients: " + ", ".join(allowed)]
    if refs:
        parts.append("Reference recipes:\n" + "\n".join(f"- {n}: {i}" for n, i in refs))
    if prev is not None:
        parts.append("The drink to adjust (change it as the request says, keep what works):\n" + prev.model_dump_json())
    if problems:
        parts.append("Your last attempt was rejected. Fix ALL of these problems:\n" + "\n".join(f"- {p}" for p in problems))
    return "\n".join(parts)


TAG_SYSTEM = (
    "You describe cocktails for a search index. Answer with JSON only.\n"
    f"- flavors: 1 to 4 tags that clearly describe the drink, only from {FLAVOR_VOCAB}.\n"
    "- taste: ONE sentence of at most 18 words about how it tastes and when you would drink it. "
    "Mention only ingredients from the list; never invent facts."
)


def tag_user(name: str, ingredients: list, instructions: str) -> str:
    return f"Cocktail: {name}\nIngredients: {', '.join(ingredients)}\nMethod: {instructions[:300]}"


def same_user(name: str, candidates: list) -> str:
    return f"Rare name: {name}\nCommon names: {', '.join(candidates)}"


def _same_ex(name: str, candidates: list, same_as: str) -> str:
    return f"{same_user(name, candidates)}\n" + json.dumps({"same_as": same_as})


SAME_SYSTEM = (
    "You clean up ingredient names in a cocktail database. You get a rare name and a list of common names. "
    "Say whether the rare name is the SAME ingredient as one of them, only written differently (brand, wording, "
    "plural, fresh or bottled). Different ingredients must stay unmatched even when the words are similar. "
    "Answer with JSON only: the common name, or an empty string.\n\nExamples:\n\n"
    + _same_ex("baileys irish cream", ["irish cream", "cream", "bitters"], "irish cream") + "\n\n"
    + _same_ex("maple syrup", ["simple syrup", "sugar syrup", "grenadine"], "") + "\n\n"
    + _same_ex("fresh lime juice", ["lime juice", "lime", "lemon juice"], "lime juice")
)
