import re

STAPLES = {"ice", "water"}

ALIASES = {
    "limes": "lime", "lemons": "lemon", "oranges": "orange", "cherries": "cherry",
    "lime juice": "lime juice", "fresh lime juice": "lime juice", "juice of lime": "lime juice",
    "fresh lemon juice": "lemon juice",
    "sugar syrup": "simple syrup", "sugar": "sugar", "superfine sugar": "sugar",
    "caster sugar": "sugar", "granulated sugar": "sugar",
    "club soda": "soda water", "carbonated water": "soda water", "sparkling water": "soda water",
    "seltzer": "soda water", "seltzer water": "soda water",
    "coca-cola": "cola", "coke": "cola", "sprite": "lemon-lime soda", "7-up": "lemon-lime soda",
    "whisky": "whiskey", "blended whiskey": "whiskey", "blended whisky": "whiskey",
    "baileys irish cream": "irish cream", "baileys": "irish cream",
    "yoghurt": "yogurt", "kahlua": "coffee liqueur", "tia maria": "coffee liqueur",
    "sour mix": "sweet and sour", "sweet n sour": "sweet and sour",
    "ice cubes": "ice", "crushed ice": "ice", "cracked ice": "ice",
    "mint leaves": "mint", "fresh mint": "mint",
    "jagermeister": "jägermeister", "peachtree schnapps": "peach schnapps",
    "cointreau": "triple sec", "grand marnier": "orange liqueur",
}

FAMILY_SUFFIXES = ("rum", "vodka", "whiskey", "tequila", "gin")
FAMILY_EXCEPTIONS = {"sloe gin", "ginger"}

DERIVES = {
    "lime": ["lime juice", "lime peel"],
    "lemon": ["lemon juice", "lemon peel"],
    "orange": ["orange juice", "orange peel"],
    "grapefruit": ["grapefruit juice", "grapefruit peel"],
    "sugar": ["simple syrup", "powdered sugar"],
    "egg": ["egg white", "egg yolk"],
    "bourbon": ["whiskey"], "scotch": ["whiskey"], "rye": ["whiskey"],
    "cognac": ["brandy"],
    "angostura bitters": ["bitters"], "orange bitters": ["bitters"],
    "sweet vermouth": ["vermouth"], "dry vermouth": ["vermouth"],
    "heavy cream": ["light cream", "whipping cream"],
}

SUBSTITUTES = {
    "lime juice": ["lemon juice"], "lemon juice": ["lime juice"],
    "scotch": ["whiskey"], "bourbon": ["whiskey"],
    "light cream": ["heavy cream", "half-and-half", "milk"],
    "whipping cream": ["heavy cream", "light cream"],
    "ginger beer": ["ginger ale"], "ginger ale": ["ginger beer"],
    "simple syrup": ["honey"],
}

OPTIONAL_WORDS = ("peel", "twist", "wedge", "slice", "cherry", "olive", "nutmeg", "cinnamon", "salt")

FLAVOR_VOCAB = [
    "sour", "sweet", "bitter", "fruity", "citrus", "refreshing", "creamy",
    "herbal", "spicy", "coffee", "tropical", "warm", "dessert",
]
FLAVOR_KEYWORDS = {
    "sour": ["lemon juice", "lime juice", "lime", "lemon", "sweet and sour", "grapefruit juice", "bitter lemon"],
    "citrus": ["lemon", "lime", "orange", "grapefruit", "triple sec", "curacao", "bitter lemon"],
    "sweet": ["syrup", "sugar", "grenadine", "liqueur", "schnapps", "honey", "cola", "ginger ale", "amaretto", "cream", "chocolate", "vanilla"],
    "bitter": ["campari", "aperol", "bitters", "tonic water", "bitter lemon", "fernet", "amaro"],
    "fruity": ["nectar", "peach", "strawberry", "raspberry", "cranberry", "apple", "pineapple", "banana", "mango", "kiwi", "melon", "apricot", "grape", "orange juice", "passion", "pomegranate", "blackberry", "blackcurrant"],
    "refreshing": ["soda water", "tonic water", "ginger beer", "mint", "cucumber", "lemon-lime soda"],
    "creamy": ["cream", "milk", "irish cream", "egg", "ice cream", "yogurt", "coconut cream"],
    "herbal": ["vermouth", "chartreuse", "absinthe", "benedictine", "mint", "basil", "thyme", "rosemary", "bitters", "amaro", "fernet", "galliano", "wormwood"],
    "spicy": ["fresh ginger", "ginger syrup", "ginger liqueur", "chili", "pepper", "cinnamon", "clove", "tabasco", "cayenne", "jalapeno"],
    "coffee": ["coffee", "espresso", "coffee liqueur"],
    "tropical": ["pineapple", "coconut", "banana", "mango", "passion", "papaya", "blue curacao", "malibu"],
    "dessert": ["chocolate", "cocoa", "ice cream", "creme de cacao", "irish cream", "vanilla", "coffee liqueur"],
}
WARM_HINTS = ("hot ", "heat", "boil", "warm", "mulled", "steam")
MIXER_WORDS = ("juice", "soda water", "tonic", "milk", "cream", "coffee", "cola", "ginger beer", "ginger ale", "lemonade", "tea", "water")
SPIRIT_WORDS = ("rum", "vodka", "whiskey", "gin", "tequila", "brandy", "cognac", "scotch", "bourbon", "mezcal", "sake", "absinthe")


def has_word(text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", text) is not None


def clean(name: str) -> str:
    return " ".join((name or "").lower().strip().split())


def family_of(canonical: str) -> str:
    if canonical in FAMILY_EXCEPTIONS:
        return canonical
    for suf in FAMILY_SUFFIXES:
        if canonical == suf or canonical.endswith(" " + suf):
            return suf
    return canonical


def is_optional(canonical: str, measure: str = "") -> bool:
    if any(has_word(canonical, w) for w in OPTIONAL_WORDS):
        return True
    return "garnish" in (measure or "").lower()


ALCOHOLIC_WORDS = SPIRIT_WORDS + (
    "liqueur", "vermouth", "wine", "beer", "champagne", "prosecco", "cava", "aperol", "campari", "amaretto",
    "schnapps", "curacao", "triple sec", "amaro", "sherry", "port", "kahlua", "baileys", "jägermeister",
    "midori", "chambord", "cointreau", "pisco", "irish cream", "creme de", "fernet", "chartreuse",
)
NON_ALCOHOLIC_LOOKALIKES = ("ginger beer", "root beer", "non-alcoholic", "nonalcoholic", "alcohol-free", "wine vinegar")


def is_alcoholic(name: str) -> bool:
    n = clean(name)
    if any(x in n for x in NON_ALCOHOLIC_LOOKALIKES):
        return False
    return any(has_word(n, w) for w in ALCOHOLIC_WORDS)
