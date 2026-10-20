from typing import Dict, List, Optional

from cocktail import generate
from cocktail import taxonomy as T
from cocktail.data_io import load_recipes
from cocktail.index import RecipeIndex
from cocktail.intent import understand
from cocktail.inventory import Inventory, Vocabulary
from cocktail.llm import OllamaLLM
from cocktail.matcher import passes_constraints
from cocktail.memory import Profile, Session, log_unsure
from cocktail.render import intent_summary
from cocktail.schemas import Intent, Match, Recipe, Reply
from cocktail.tools import Toolbox


class CocktailPipeline:
    def __init__(self, recipes: Optional[List[Recipe]] = None, llm: Optional[OllamaLLM] = None, index_path=None):
        self.llm = llm or OllamaLLM()
        ready, msg = self.llm.status()
        if not ready:
            raise RuntimeError(msg)
        self.recipes = recipes if recipes is not None else load_recipes()
        self.by_id = {r.id: r for r in self.recipes}
        self.names = {r.name.casefold() for r in self.recipes}
        self.vocab = Vocabulary(self.recipes)
        index = RecipeIndex(self.recipes, self.llm, index_path) if index_path else RecipeIndex(self.recipes, self.llm)
        self.tools = Toolbox(self.recipes, self.vocab, index)

    def respond(self, message: str, session: Session, profile: Profile,
                allow_missing: int = 1, top_k: int = 3) -> Reply:
        self.tools.trace = []
        shown = [self.by_id[i].name for i in session.last_ids if i in self.by_id]
        intent = understand(message, self.vocab, self.llm, session.last_intent, shown, session.state)

        added = [h for h in intent.have if h not in profile.bar]
        profile.bar = list(dict.fromkeys(profile.bar + added))
        inv = Inventory(self.vocab, profile.bar)

        route = "invent" if intent.followup and session.last_route == "invent" else intent.action
        intent.action, intent.allow_missing = route, allow_missing
        self.tools.trace.append(f"route:{route}")

        if route == "chat":
            text = self._chat(message, session, profile)
            session.remember(message, text)
            reply = Reply(kind="chat", intent=intent, text=text)
        elif route in ("suggest", "invent") and not profile.bar:
            text = "Your bar is empty. Tell me what you have (\"I have rum and limes\") or list it on the left."
            reply = Reply(kind="chat", intent=intent, text=text)
        elif route == "search":
            reply = self._search(message, intent, session, inv, top_k)
        elif route == "invent":
            reply = self._invent(message, intent, session, inv, profile, allow_missing)
        else:
            reply = self._suggest(message, intent, session, inv, profile, allow_missing, top_k)

        if intent.model_failed and not reply.unsure:
            reply.unsure = f"the model could not read the message; routed by default: {self.llm.last_error}"
        reply.added, reply.route, reply.trace = added, route, list(self.tools.trace)
        if reply.unsure:
            log_unsure(message, route, reply.unsure, reply.trace)
        if reply.kind == "suggest":
            session.last_intent, session.last_route = intent, route
            session.last_ids = [m.recipe.id for m in reply.matches]
            session.shown_ids = (session.shown_ids if intent.followup else []) + session.last_ids
            session.remember(message, (reply.intro + " " if reply.intro else "") + "Suggested: "
                             + (", ".join(m.recipe.name for m in reply.matches) or "nothing") + ".")
        elif route != "chat":
            session.remember(message, reply.text)
        return reply

    def _search(self, message: str, intent: Intent, session: Session, inv: Inventory, top_k: int) -> Reply:
        conflict = self._conflict(intent)
        if conflict:
            return Reply(kind="chat", intent=intent, text=conflict)
        skip = set(session.shown_ids) if intent.followup else set()
        pool = [r.id for r in self.recipes if r.id not in skip and passes_constraints(r, intent)]
        cands = self.tools.search(intent.lookup or message, pool, k=20)
        if not cands:
            return Reply(kind="chat", intent=intent, text="No drink in my book satisfies your rules.")
        return self._judged(message, intent, inv, cands, top_k, exact_ok=True)

    def _suggest(self, message, intent, session, inv, profile, allow, top_k) -> Reply:
        notes: List[str] = []
        if inv.unknown:
            notes.append("Not in my ingredient list (ignored for matching): " + ", ".join(inv.unknown))
        candidates = [r for r in self.recipes if passes_constraints(r, intent)]
        if not candidates:
            return Reply(kind="chat", intent=intent,
                         text=self._conflict(intent) or "No drink satisfies those requirements together.")
        covs = {r.id: inv.check(r) for r in candidates}
        skip = set(session.shown_ids) if intent.followup else set()
        fresh = [r for r in candidates if r.id not in skip]
        if not fresh:
            fresh = candidates
            notes.append("That's everything that fits right now, so some picks repeat.")
        for extra in (0, 1, 2):
            pool = [r.id for r in fresh if len(covs[r.id].missing) <= allow + extra]
            if pool:
                break
        else:
            self.tools.trace.append("fallback:invent")
            return self._invent(message, intent, session, inv, profile, allow)
        if extra:
            self.tools.trace.append(f"relax(+{extra})")
            notes.append("Nothing is makeable with your bar and your rules; these need up to "
                         f"{max(len(covs[i].missing) for i in pool)} more ingredient(s).")
        cands = self.tools.search(intent.vibe or message, pool, k=20)
        reply = self._judged(message, intent, inv, cands, top_k, exact_ok=False, notes=notes)
        reply.unlocks = self._unlocks(candidates, covs)
        return reply

    def _judged(self, message: str, intent: Intent, inv: Inventory, cands: List[Recipe], top_k: int,
                exact_ok: bool, notes: Optional[List[str]] = None) -> Reply:
        notes, unsure = list(notes or []), ""
        judged = self.llm.judge(message, [self._line(r) for r in cands], intent_summary(intent))
        self.tools.trace.append("judge:" + (judged.verdict if judged else "failed"))
        if judged is None:
            verdict, picks = "related", cands[:top_k]
            unsure = f"the model could not check the search results: {self.llm.last_error}"
            notes.append(f"I couldn't double-check these, so they are only the closest by meaning "
                         f"({self.llm.last_error[:100]}).")
        else:
            picks = [cands[n - 1] for n in dict.fromkeys(judged.picks) if 1 <= n <= len(cands)][:8]
            verdict = judged.verdict if picks else "none"
            if judged.note and self._grounded(judged.note, picks or cands[:3]):
                notes.append(judged.note)
        if verdict == "none":
            picks, unsure = cands[:3], "no confident match in the recipe book"
            notes.append("I'm not sure anything fits. These are the closest by meaning; "
                         "say \"invent ...\" and I'll create one instead.")

        matches = []
        for r in picks:
            cov = inv.check(r)
            matches.append(Match(recipe=r, missing=cov.missing, optional_missing=cov.optional_missing,
                                 substitutions=cov.substitutions))
        if verdict == "exact" and exact_ok:
            return Reply(intent=intent, matches=matches[:1], notes=notes,
                         intro=f"Here's the recipe for {matches[0].recipe.name}.")
        matches = sorted(matches, key=lambda m: len(m.missing))[:top_k]
        intro = ""
        if verdict != "none":
            for m in matches:
                m.explanation = self._explain(m.recipe, intent, message)
            intro = self._intro(message, matches)
        return Reply(intent=intent, matches=matches, notes=notes, intro=intro, unsure=unsure)

    def _invent(self, message, intent, session, inv, profile, allow) -> Reply:
        conflict = self._conflict(intent)
        if conflict:
            return Reply(kind="chat", intent=intent, text=conflict)
        if not profile.bar:
            return Reply(kind="chat", intent=intent, text="To invent something I need to know your bar. List what you have first.")
        refs = self.tools.search(" ".join(intent.must_include + intent.flavors) or message, None, k=3)
        prev = session.last_draft if (intent.followup and session.last_route == "invent") else None
        draft, recipe, problems = generate.invent(self.llm, self.vocab, inv, intent, message, refs, allow, profile.bar,
                                                  self.names, self.tools.trace, prev)
        if recipe is None:
            why = " What the checks said last time: " + "; ".join(problems[:3]) + "." if problems else ""
            return Reply(kind="chat", intent=intent, unsure="invention failed validation: " + "; ".join(problems[:3]),
                         text="I tried to invent something, but nothing I came up with passed my checks." + why
                              + " Try loosening a rule, allowing a missing ingredient, or naming different ingredients.")
        cov = inv.check(recipe)
        near, sim = self.tools.nearest(recipe)
        notes = ["✨ Invented by me, not from the cocktail database. Checked for ingredients, your rules, alcohol "
                 "and amounts, but never tasted."]
        if near and sim >= 0.75:
            notes.append(f"Very close to {near.name} from my book.")
        why = draft.why if draft.why and self._grounded(draft.why, [recipe]) else self._template(recipe, intent)
        session.last_draft = draft
        return Reply(intent=intent, intro="Here's something new:", notes=notes,
                     matches=[Match(recipe=recipe, missing=cov.missing, optional_missing=cov.optional_missing,
                                    substitutions=cov.substitutions, explanation=why)])

    @staticmethod
    def _conflict(intent: Intent) -> str:
        if intent.alcohol == "none":
            bad = [m for m in intent.must_include if T.is_alcoholic(m)]
            if bad:
                return (f"{', '.join(bad)} contains alcohol, but you asked for alcohol-free drinks. "
                        "Say \"alcohol is fine again\" (or switch the alcohol setting), or pick a different ingredient.")
        return ""

    @staticmethod
    def _line(r: Recipe) -> str:
        info = "; ".join(x for x in (", ".join(r.flavors), r.strength) if x)
        taste = f" - {r.taste}" if r.taste else ""
        return f"{r.name}: {', '.join(i.name for i in r.items if not i.optional)} [{info}]{taste}"

    @staticmethod
    def _unlocks(candidates: List[Recipe], covs, limit: int = 3) -> List[dict]:
        buckets: Dict[str, List[str]] = {}
        for r in candidates:
            miss = covs[r.id].missing
            if len(miss) == 1:
                buckets.setdefault(miss[0], []).append(r.name)
        ranked = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:limit]
        return [{"ingredient": ing, "unlocks_count": len(names), "examples": sorted(names)[:3]}
                for ing, names in ranked]

    def _explain(self, recipe: Recipe, intent: Intent, message: str) -> str:
        text = self.llm.explain(recipe.name, [i.name for i in recipe.items if not i.optional],
                                recipe.flavors, intent.vibe or message)
        return text if text and self._grounded(text, [recipe]) else self._template(recipe, intent)

    def _intro(self, message: str, top: List[Match]) -> str:
        if not top:
            return ""
        text = self.llm.intro(message, [m.recipe.name for m in top])
        return text if text and self._grounded(text, [m.recipe for m in top]) else "Here's what I'd make:"

    def _chat(self, message: str, session: Session, profile: Profile) -> str:
        standing = session.state.describe()
        context = (f"The user's bar: {', '.join(profile.bar) or 'empty'}."
                   + (f"\nStanding preferences for this conversation: {standing}." if standing else ""))
        return (self.llm.chat_reply(session.turns, message, context)
                or "Hmm, my mind went blank. Tell me what you feel like drinking and I'll find something.")

    def _grounded(self, text: str, recipes: List[Recipe]) -> bool:
        allowed = set()
        for recipe in recipes:
            for i in recipe.items:
                allowed.update(T.clean(i.name).replace("-", " ").split())
                allowed.update(T.clean(i.family).split())
            allowed.update(T.clean(recipe.name).split())
        low = text.lower()
        for phrase in self.vocab.names:
            if phrase in T.STAPLES or not T.has_word(low, phrase):
                continue
            if not all(tok in allowed for tok in phrase.replace("-", " ").split()):
                return False
        return True

    @staticmethod
    def _template(recipe: Recipe, intent: Intent) -> str:
        core = [i.name for i in recipe.items if not i.optional][:4]
        hits = [f for f in recipe.flavors if f in intent.flavors] or recipe.flavors[:2]
        vibe = " and ".join(hits) if hits else "balanced"
        return f"A {vibe} pick made with {', '.join(core)}."