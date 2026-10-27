"""Which target selectors each resolution path actually supports.

A parsed action is ``RuleAction(kind, target, amount, scope, condition)``.  The
honest-status gate in ``tools/build_card_rules.py`` only ever checked ``kind``,
so an action whose *target* no path could resolve was still marked
``implemented`` and silently did nothing.  Real cases found this way:

* ``modify_attack`` + ``all_units``        — 5 cards, "All units get +N" did nothing
* ``move_to_frontline`` + ``source``       — 3 cards, "move this unit" did nothing
* ``move_to_support_line`` + ``random_enemy`` — the random retreat never happened
* ``damage`` + ``selected_enemy``          — the resolver returned no target

An empirical probe (``tools/probe_selector_matrix.py``) showed that MOST kinds
ignore the target entirely: draw / discard / kredit-slot effects act on the
player whatever the target says, and the stat kinds accept every unit selector.
So this module deliberately declares a family only where the target genuinely
gates the outcome — movement — plus two checks that hold for every kind: a
target no path resolves at all, and a ``chosen`` target read by no handler.
Declaring more than that produced false alarms (``pin`` + ``random_enemy`` was
reported broken and is not).

This module is the single declaration of what each path resolves.  The paths
below import these sets and dispatch on them, so a selector cannot be in a set
without the path handling it; a behavioural test
(``tests/test_selector_contract.py``) asserts the reverse, that every declared
selector really resolves to something on a normal board.

Paths:
    UNIT_SELECTORS      simulator.rules.native._resolve_units
    MOVEMENT_SELECTORS  simulator.rules.native.NativeRuleEngine._move_units
    EFFECT_SELECTORS    simulator.effects.resolver.EffectResolver._targets,
                        plus the random-target rewrite native does before it
    PLAYER_SELECTORS    the player-scoped branches (draw, kredit slots, ...)
"""

from __future__ import annotations

# The acting card's own unit.
SOURCE = frozenset({"self", "source"})
# A unit chosen when the card was played (context.target_unit_id).
CHOSEN = frozenset({"selected_target", "target", "selected_friendly", "selected_enemy"})
# Whole groups of units.
GROUPS = frozenset({"friendly_units", "enemy_units", "all_units", "all"})
# Aliases the effect layer also accepts for the two single-side groups.
GROUP_ALIASES = frozenset({"friendly", "enemy"})
# Resolved by picking at random when the action runs.
RANDOM = frozenset({"random_enemy", "random_friendly"})
# A player rather than a unit.
PLAYERS = frozenset({"owner", "friendly_hq", "enemy_hq", "enemy_hand"})
# The subset of PLAYERS that EffectResolver._targets itself resolves.  "enemy_hand"
# is not one of them: it is read directly by the player-scoped kinds (the kredit
# slot handlers check ``action.target == "enemy_hand"`` themselves).
EFFECT_PLAYERS = frozenset({"owner", "friendly_hq", "enemy_hq"})
# Set by a choose_one action earlier in the same rule (NativeRuleEngine._chosen).
# Only the kinds in CHOSEN_LINK_KINDS read it; for any other kind the target is
# ignored and the action quietly acts on the wrong thing.
CHOSEN_LINK = frozenset({"chosen"})
CHOSEN_LINK_KINDS = frozenset({"add_card", "return_to_deck", "discard"})
# "it"/"them" in a later clause of the same card: the units an earlier clause
# deployed, else the unit the card chose.  Only delayed_destroy reads it.
REFERENT_LINK = frozenset({"referent"})
REFERENT_LINK_KINDS = frozenset({"delayed_destroy", "delayed_to_hand"})

UNIT_SELECTORS = SOURCE | CHOSEN | GROUPS
# What a unit-only effect can reach: every unit selector, plus the group aliases
# the effect layer accepts and the random targets native rewrites first.  These
# effects do nothing for a player selector (owner, enemy_hq, ...), which is
# correct — "give your units +1 attack" has nothing to do with a player.
UNIT_EFFECT_SELECTORS = UNIT_SELECTORS | GROUP_ALIASES | RANDOM
MOVEMENT_SELECTORS = SOURCE | CHOSEN | frozenset({"friendly_units", "enemy_units"}) | RANDOM
# What EffectResolver._targets resolves on its own.  RANDOM is excluded on
# purpose: native rewrites a random target into the chosen one BEFORE handing the
# action to the resolver, so the resolver never sees "random_enemy" itself.
EFFECT_SELECTORS = UNIT_SELECTORS | GROUP_ALIASES | EFFECT_PLAYERS
PLAYER_SELECTORS = PLAYERS

FAMILIES: dict[str, frozenset[str]] = {
    "unit": UNIT_SELECTORS,
    "unit_effect": UNIT_EFFECT_SELECTORS,
    "movement": MOVEMENT_SELECTORS,
    "effect": EFFECT_SELECTORS,
    "player": PLAYER_SELECTORS,
}

# Every selector any path can resolve; a target outside this set is resolvable
# by nothing at all and is always a bug.
ALL_SELECTORS = frozenset().union(*FAMILIES.values()) | CHOSEN_LINK | REFERENT_LINK | RANDOM | PLAYERS


def chosen_link_ok(kind: str, target: str) -> bool:
    """Whether a ``chosen`` target is read by this kind."""
    if target in REFERENT_LINK:
        return kind in REFERENT_LINK_KINDS
    if target not in CHOSEN_LINK:
        return True
    return kind in CHOSEN_LINK_KINDS


def supports(family: str, target: str) -> bool:
    """Whether the named resolution path resolves ``target``."""
    selectors = FAMILIES.get(family)
    if selectors is None:
        raise KeyError("Unknown selector family: " + str(family))
    return target in selectors
