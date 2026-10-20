from cocktail.schemas import Intent, Match, Reply


def _ingredient_lines(m: Match):
    for it in m.recipe.items:
        label = f"{it.measure} {it.raw}".strip() if it.measure else it.raw
        tag = ""
        if it.name in m.missing:
            tag = "  <- missing"
        elif it.name in m.substitutions:
            tag = f"  <- use {m.substitutions[it.name]} instead"
        elif it.name in m.optional_missing:
            tag = "  (optional)"
        yield label, tag


def intent_summary(i: Intent) -> str:
    bits = []
    if i.flavors:
        bits.append("flavours: " + ", ".join(i.flavors))
    if i.avoid_flavors:
        bits.append("less: " + ", ".join(i.avoid_flavors))
    if i.exclude:
        bits.append("without: " + ", ".join(i.exclude))
    if i.must_include:
        bits.append("must use: " + ", ".join(i.must_include))
    if i.strength:
        bits.append("strength: " + i.strength)
    if i.alcohol == "none":
        bits.append("alcohol-free")
    elif i.alcohol == "only":
        bits.append("with alcohol")
    if i.vibe:
        bits.append(f"vibe: {i.vibe}")
    return "; ".join(bits) or "no specific constraints"


def understood(reply: Reply) -> str:
    tools = ", ".join(reply.trace)
    return f"{intent_summary(reply.intent)}  [{tools}]" if tools else intent_summary(reply.intent)


def to_text(reply: Reply) -> str:
    out = []
    if reply.added:
        out.append(f"(added to your bar: {', '.join(reply.added)})")
    if reply.kind == "chat":
        return "\n".join(out + [reply.text])
    out += [reply.intro, f"Understood: {understood(reply)}", ""]
    out += [f"! {n}" for n in reply.notes]
    if not reply.matches:
        out.append("No drink fits with what you have.")
    for idx, m in enumerate(reply.matches, 1):
        r = m.recipe
        status = "ready to make" if not m.missing else "missing " + ", ".join(m.missing)
        out += ["", f"{idx}. {r.name}  ({status})",
                f"   {r.category or ''}{' | ' + r.glass if r.glass else ''}"]
        if m.explanation:
            out.append(f"   {m.explanation}")
        out.append("   Ingredients:")
        out += [f"     - {label}{tag}" for label, tag in _ingredient_lines(m)]
        if r.instructions:
            out.append(f"   How: {r.instructions}")
    if reply.unlocks:
        out += ["", "Shopping ideas:"]
        for u in reply.unlocks:
            out.append(f"  - buy {u['ingredient']} -> {u['unlocks_count']} more drink(s) (e.g. {', '.join(u['examples'])})")
    return "\n".join(out)


def to_markdown(reply: Reply) -> str:
    head = [f"*added to your bar: {', '.join(reply.added)}*"] if reply.added else []
    if reply.kind == "chat":
        return "\n\n".join(head + [reply.text])
    out = head + [reply.intro, f"*understood: {understood(reply)}*"]
    out += [f"> {n}" for n in reply.notes]
    if not reply.matches:
        out.append("\n**No drink fits with what you have.** Try allowing one more missing ingredient, or check the shopping ideas.")
    for idx, m in enumerate(reply.matches, 1):
        r = m.recipe
        status = "ready to make" if not m.missing else "🛒 missing " + ", ".join(m.missing)
        out.append(f"\n### {idx}. {'✨ ' if r.source == 'generated' else ''}{r.name} — {status}")
        meta = " · ".join(x for x in (r.category, r.glass) if x)
        if meta:
            out.append(f"*{meta}*")
        if m.explanation:
            out.append(m.explanation)
        out.append("\n**Ingredients**")
        out += [f"- {label}{('  *' + tag.strip(' <-') + '*') if tag else ''}" for label, tag in _ingredient_lines(m)]
        if r.instructions:
            out.append(f"\n**Method:** {r.instructions}")
    return "\n".join(out)


def unlocks_markdown(reply: Reply) -> str:
    if not reply.unlocks:
        return "*Nothing obvious to buy: no drink is exactly one ingredient away.*"
    lines = ["**Shopping ideas**"]
    for u in reply.unlocks:
        lines.append(f"- **{u['ingredient']}** → {u['unlocks_count']} more drink(s), e.g. {', '.join(u['examples'])}")
    return "\n".join(lines)
