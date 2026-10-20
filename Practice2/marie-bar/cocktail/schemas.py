from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Item(BaseModel):
    raw: str
    name: str
    family: str
    measure: Optional[str] = None
    optional: bool = False


class Recipe(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    category: Optional[str] = None
    alcoholic: Optional[bool] = None
    alcohol_label: Optional[str] = None
    glass: Optional[str] = None
    instructions: Optional[str] = None
    items: List[Item] = Field(default_factory=list)
    flavors: List[str] = Field(default_factory=list)
    strength: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    iba: Optional[str] = None
    thumb: Optional[str] = None
    taste: Optional[str] = None
    source: str = "cocktaildb"

    @property
    def ingredients(self) -> List[str]:
        return list(dict.fromkeys(i.name for i in self.items))

    @property
    def can_be_alcohol_free(self) -> bool:
        return self.alcoholic is False or (self.alcohol_label or "").lower() == "optional alcohol"


class Intent(BaseModel):
    query: str = ""
    action: str = "suggest"
    flavors: List[str] = Field(default_factory=list)
    avoid_flavors: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)
    must_include: List[str] = Field(default_factory=list)
    strength: Optional[str] = None
    alcohol: str = "keep"
    allow_missing: Optional[int] = None
    have: List[str] = Field(default_factory=list)
    vibe: str = ""
    lookup: str = ""
    followup: bool = False
    model_failed: bool = False


class Match(BaseModel):
    recipe: Recipe
    score: float = 0.0
    missing: List[str] = Field(default_factory=list)
    optional_missing: List[str] = Field(default_factory=list)
    substitutions: Dict[str, str] = Field(default_factory=dict)
    explanation: str = ""


class Reply(BaseModel):
    kind: str = "suggest"
    route: str = ""
    trace: List[str] = Field(default_factory=list)
    intent: Intent
    text: str = ""
    intro: str = ""
    matches: List[Match] = Field(default_factory=list)
    unlocks: List[dict] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    added: List[str] = Field(default_factory=list)
    unsure: str = ""


class DraftItem(BaseModel):
    name: str
    amount: float
    unit: Literal["ml", "cl", "oz", "tsp", "tbsp", "dash", "piece"]


class Draft(BaseModel):
    name: str
    items: List[DraftItem]
    method: str
    glass: str
    why: str
