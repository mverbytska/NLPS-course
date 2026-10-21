#!/usr/bin/env python3
import argparse
import json
import sys

from cocktail.chat import ALCOHOL_ALIASES, set_exclude, turn
from cocktail.inventory import parse_inventory_text
from cocktail.memory import Profile, Session
from cocktail.pipeline import CocktailPipeline
from cocktail.render import to_text

HELP = """Commands (they match the controls in the web UI):
  /bar rum, lime, mint     set your bar (saved for next time)      /bar        show it
  /add gin, tonic          add to your bar                        /remove gin  take out of your bar
  /alcohol any|none|only   any drink / alcohol-free / with alcohol (also: just say "no alcohol tonight")
  /exclude gin, cream      never include these (empty list: /exclude none)
  /state                   show what I'm keeping in mind
  /missing 0|1|2           how many missing ingredients are OK
  /top N                   how many suggestions
  /new                     start a new conversation (clears standing preferences, keeps your bar)
  /help, /quit
Anything else is a message to Marie: a drink request, a follow-up ("another one, less sweet"),
"I just bought gin", "no alcohol tonight", or a question ("how do I make a negroni?")."""

ALCOHOL_NAMES = {"any": "any drink", "none": "alcohol-free", "only": "with alcohol"}


def answer(pipe, message, session, profile, args) -> None:
    reply, profile.bar = turn(pipe, message, session, profile.bar, args.missing, args.top)
    print(json.dumps(reply.model_dump(), indent=2, ensure_ascii=False) if args.json else to_text(reply))


def show_state(session) -> None:
    st = session.state
    print(f"Alcohol: {ALCOHOL_NAMES[st.alcohol]}")
    print("Never include:", ", ".join(st.exclude) or "-")
    print("Less of:", ", ".join(st.avoid_flavors) or "-")


def repl(pipe, session, profile, args) -> None:
    try:
        import readline
    except ImportError:
        pass
    print("marie bar 🍸  type /help for commands, /quit to leave.")
    print("Your bar:", ", ".join(profile.bar) or "(empty - set it with /bar rum, lime, ...)")
    while True:
        try:
            status = session.state.describe() or "any drink"
            line = input(f"\n[{status}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        try:
            if cmd in ("/quit", "/exit", "/q"):
                return
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/bar":
                if rest:
                    profile.bar = parse_inventory_text(rest)
                    profile.save()
                print("Your bar:", ", ".join(profile.bar) or "(empty)")
            elif cmd == "/add":
                profile.bar = list(dict.fromkeys(profile.bar + parse_inventory_text(rest)))
                profile.save()
                print("Your bar:", ", ".join(profile.bar))
            elif cmd == "/remove":
                gone = {pipe.vocab.canon(x) for x in parse_inventory_text(rest)} | set(parse_inventory_text(rest))
                profile.bar = [b for b in profile.bar if b not in gone and pipe.vocab.canon(b) not in gone]
                profile.save()
                print("Your bar:", ", ".join(profile.bar) or "(empty)")
            elif cmd == "/alcohol" and rest.lower() in ALCOHOL_ALIASES:
                session.state.alcohol = ALCOHOL_ALIASES[rest.lower()]
                print("Alcohol:", ALCOHOL_NAMES[session.state.alcohol])
            elif cmd == "/exclude":
                if rest:
                    set_exclude(pipe, session, "" if rest.lower() in ("none", "clear") else rest)
                print("Never include:", ", ".join(session.state.exclude) or "-")
            elif cmd == "/state":
                show_state(session)
            elif cmd == "/missing" and rest in {"0", "1", "2"}:
                args.missing = int(rest)
                print(f"Allowing up to {rest} missing ingredient(s).")
            elif cmd == "/top" and rest.isdigit() and int(rest) > 0:
                args.top = int(rest)
                print(f"Showing {rest} suggestion(s).")
            elif cmd == "/new":
                session = Session()
                print("New conversation. Standing preferences cleared.")
            elif line.startswith("/"):
                print("Unknown command or missing argument. /help lists them.")
            else:
                answer(pipe, line, session, profile, args)
        except Exception as e:
            print(f"[error] {e}")


def main() -> int:
    p = argparse.ArgumentParser(description="Cocktail suggestions from what's in your bar.")
    p.add_argument("query", nargs="?", help="what you feel like (omit for interactive mode)")
    p.add_argument("--have", help="comma-separated ingredients you own (replaces the saved bar)")
    p.add_argument("--alcohol", choices=["any", "none", "only"], default="any", help="alcohol setting to start with")
    p.add_argument("--exclude", default="", help="comma-separated ingredients to never include")
    p.add_argument("--top", type=int, default=3, help="number of suggestions")
    p.add_argument("--missing", type=int, default=1, choices=[0, 1, 2], help="max missing ingredients")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args()

    try:
        pipe = CocktailPipeline()
    except (RuntimeError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        return 1

    profile, session = Profile.load(), Session()
    session.state.alcohol = args.alcohol
    if args.exclude:
        set_exclude(pipe, session, args.exclude)
    if args.have:
        profile.bar = parse_inventory_text(args.have)
        profile.save()

    if args.query:
        answer(pipe, args.query, session, profile, args)
    else:
        repl(pipe, session, profile, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
