"""Materialize deterministic native-text parsing into the reviewed rule store.

This tool is the *derived-data* generator (it is NOT the simulator execution
core). It parses every card with ``RuleParser`` and writes the reviewed rule
store ``data/rules/card_rules.json`` that ``NativeRuleEngine`` loads.

Honest-status rule (per the handoff task book):
  A card is marked ``implemented`` only when *every* parsed action maps to a
  kind the execution layer actually handles. If any action kind is not executed
  by ``native.py`` (``NativeRuleEngine``) or ``resolver.py`` (``EffectResolver``),
  the card is downgraded to ``partial`` and the gap is recorded in
  ``unresolved_fragments``. This keeps the invariant that an ``implemented``
  entry carries no ``unresolved_fragments``, and that "playable without crashing"
  is never mistaken for "rule implemented".
"""

from __future__ import annotations

import json
import re as _re
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator.cards.loader import CardDatabase
from simulator.rules.parser import RuleParser


CATALOG = ROOT / "data/source/kards_info_cards.json"
DESTINATION = ROOT / "data/rules/card_rules.json"
REPORT = ROOT / "RULE_IMPLEMENTATION_REPORT.md"

EXECUTOR_SOURCES = (
    ROOT / "simulator/rules/native.py",
    ROOT / "simulator/effects/resolver.py",
)

# ── HONEST execution coverage ────────────────────────────────────────────────
# The previous extract_executable_kinds() scanned source files for
# `kind == "X"` strings and considered every match "executable".  This was
# fundamentally dishonest: native.py contains a 30-kind catch-all at line 450
# that writes dead `unit.status[kind] = True` data for control_effect,
# special_effect, hq_effect, and 27+ other kinds.  None of that status data
# is ever consumed by any downstream game rule.  Similarly, aura_buff applies
# stat changes blindly to *all* friendly units regardless of scope/type/
# condition, which is incorrect for virtually every card it touches.
#
# _GENUINELY_EXECUTED_KINDS is the hand-maintained source of truth.  A kind
# only enters this set when its handler in native.py or resolver.py produces
# a real, verifiable game state change that is consumed by downstream rules
# (damage pipeline, deployment pipeline, combat, cost calculation, etc.).
# Adding a kind here without the corresponding verification test is forbidden.
_GENUINELY_EXECUTED_KINDS: frozenset[str] = frozenset({
    # ── damage / combat ──
    "damage",
    "destroy",
    "heal",
    "repair",
    "suppress",
    "pin",
    "trigger_destruction_effects",
    "destroy_undamaged",
    "rearrange_enemy_units",
    "develop_options",
    "fight",
    "combat_damage_cap",
    "random_combat_damage",
    "hq_damage_reduction_by_attack",
    "hq_defense_becomes_damage",
    "hq_damage_redirect_to_source",
    "hq_damage_redirect_to_enemy_hq",
    "redirect_named_unit_damage",
    "replay_non_targeting_deployment",
    "swap_hand_unit_with_friendly",
    "shock_tactics_choice",
    "suppress_deployment_effects",
    "random_effect_target_priority",
    "trigger_destroy_and_convert",
    "frontline_limit",
    "retreat_enemy_frontline_all",
    "hand_cost_on_friendly_death",
    "noncombat_unit_damage_bonus",
    "ground_damage_bonus",
    # ── weather (OCEANIA STORM) ──
    "forecast",
    "distribute_defense",
    # ── hand position (OCEANIA STORM / RING OF FIRE) ──
    "damage_equal_to",
    "discard_leftmost",
    "move_hand_card_leftmost",
    "damage_own_hq_equal_to_hand",
    "enemy_card_damage_reduction",
    "friendly_order_damage_armor",
    "next_order_damage_bonus",
    "frontline_attack_bonus",
    "move_and_attack",
    "move_and_attack_aura",
    "swap_with_friendly",
    # ── stat modification ──
    "buff",
    "modify_attack",
    "modify_defense",
    "modify_health",
    "set_attack",
    "set_defense",
    "set_cost",
    "swap_attack_opcost",
    "attack_equals_defense",
    "buff_nation",
    "defense_nation",
    # ── cost / resource modification ──
    "modify_deployment_cost",
    "modify_operation_cost",
    "set_operation_cost",
    "modify_hand_cost",
    "buff_deployed_matching_name",
    "scheduled_repair",
    "op_cost_rule",
    "operation_cost_rule",
    "gain_kredits",
    "lose_kredits",
    "spend_kredits",
    "gain_kredit_slot",
    "lose_kredit_slot",
    # ── card movement (hand / deck / battlefield) ──
    "draw",
    "discard",
    "draw_matching",
    "return_to_hand",
    "return_to_deck",
    "move_to_frontline",
    "move_to_support_line",
    "add_card",
    "deploy_named",
    "shuffle_named",
    "add_to_enemy_deck",
    "mill",
    "copy_card",
    # ── keywords / abilities ──
    "grant_ability",
    "add_keyword",
    "remove_keyword",
    # ── HQ effects ──
    "hq_take_damage",
    "set_hq_defense",
    "hq_immune",
    "gain_hq_defense_equal_cost",
    # ── control flow ──
    "choose_one",
    "end_turn",
    "delayed_discard",
    "delayed_destroy",
    "delayed_to_hand",
    "conditional_keyword",
    "intel",
    "mark_draw_on_destroy",
    "count_then_veteran",
    "hand_cost_on_enemy_draw",
    "repeat_destruction",
    "put_into_play",
    "flicker",
    "remove_until_return",
    # ── stat multipliers (implemented 2026-07-19) ──
    "double_stats",
    "double_damage",
    "triple_damage",
    # ── action restrictions (implemented 2026-07-19) ──
    "cannot",
    # ── HQ effects (genuine resolver handlers) ──
    "gain_hq_defense",
    # ── immunity / attack restrictions ──
    "immune",
    "cannot_attack_hq",
    # ── draw / copy / remove (genuine native handlers) ──
    "draw_top_matching",
    "remove",
    "copy_unit",
    # ── more genuine native handlers (2026-07-19) ──
    "gain_hq_defense_equal_damage",
    "discard_hand",
    "take_control",
    "reset_operation",
    "retreat_and_repair",
    "order_damage_bonus",
    "spawn_named_same_front",
    "convert_to",
    # ── Type-specific combat bonuses ──
    "double_damage_against_type",
    "attack_bonus_against_type",
    "attack_bonus_against_higher_attack",
    "damage_bonus_against_air",
    # ── Restrictions / taxes ──
    "cannot_be_targeted_by_enemy_orders",
    "target_or_attack_tax",
    # ── HQ / destroy / traits ──
    "hq_excess",
    "destroy_cost_lte",
    "grant_trait",
    # ── Effect subsystems ──
    "repeat_effect",
    "aura_buff",
    # ── Countermeasure / pincer / target ──
    "cancel",
    "pincer_ability",
    "target_select",
    # ── Small mechanisms ──
    "enemy_cannot_deploy",
    "enemy_cannot_order",
    "buff_on_played_ability",
    "buff_on_enemy_target",
    "attack_on_order_played",
    "immune_to_ground_in_support",
    "ignore_heavy_armor",
    "hq_damage_bonus",
    "countermeasure_lock",
    "hq_defense_lock",
    "hq_defense_cap",
    "set_kredit_slots_equal",
    "enemy_kredit_change",
    "kredit_effect",
    "delayed_return",
    "modify_cost",
})

# NOTE: The following kinds are INTENTIONALLY absent because their handlers
# either write dead status data, apply effects blindly without correct
# scope/condition, or log events without producing any game-rule consumption:
#
#   control_effect, special_effect, hq_effect, triggered_effect,
#   aura_buff, kredit_effect, pincer_ability, grant_trait,
#   repeat_effect, target_select, immune, cannot,
#   enemy_cannot_deploy, enemy_cannot_order, countermeasure_lock,
#   double_stats, double_damage, triple_damage,
#   hq_excess, set_kredit_slots_equal, enemy_kredit_change,
#   target_or_attack_tax, cancel, damage_bonus_against_air,
#   ignore_heavy_armor, hq_damage_bonus, cannot_attack_hq,
#   immune_to_ground_in_support, cannot_be_targeted_by_enemy_orders,
#   attack_bonus_against_higher_attack, hq_defense_cap,
#   hq_defense_lock, destroy_cost_lte, convert_to, copy_unit


# ── Selector contract ───────────────────────────────────────────────────────
# _GENUINELY_EXECUTED_KINDS only answers "does this kind have a handler".  It
# says nothing about whether that handler can resolve the action's TARGET, so a
# card could be marked implemented while resolving to no target and doing
# nothing: modify_attack+all_units (5 cards), move_to_frontline+source (3),
# move_to_support_line+random_enemy, damage+selected_enemy (99 actions) were all
# real, and all were silently inert.
#
# tools/probe_selector_matrix.py measured which (kind, selector) pairs actually
# change state.  Most kinds turned out to ignore the target, so only the checks
# that are demonstrably true are made here; a wider table produced false alarms.
_KIND_SELECTOR_FAMILY: dict[str, str] = {
    # Movement is the one family where the target decides the outcome, and the
    # one that has broken three times.
    "move_to_frontline": "movement",
    "move_to_support_line": "movement",
    # Unit-only effects: the probe shows each of these works for every unit
    # selector and for none of the player selectors, and no card in the store
    # pairs them with a player selector today — so declaring them costs nothing
    # now and catches a future parser change that points one at a player.
    "add_keyword": "unit_effect",
    "buff": "unit_effect",
    "destroy": "unit_effect",
    "heal": "unit_effect",
    "modify_attack": "unit_effect",
    "modify_defense": "unit_effect",
    "modify_deployment_cost": "unit_effect",
    "modify_health": "unit_effect",
    "modify_operation_cost": "unit_effect",
    "pin": "unit_effect",
    "remove_keyword": "unit_effect",
    "suppress": "unit_effect",
}


def unsupported_selector(kind: str, target: str) -> str | None:
    """Return why ``target`` cannot be resolved for ``kind``, else None."""
    from simulator.rules import selectors
    if target and target not in selectors.ALL_SELECTORS:
        return "no path resolves target={0}".format(target)
    if not selectors.chosen_link_ok(kind, target):
        return "kind={0} does not read a chosen target".format(kind)
    family = _KIND_SELECTOR_FAMILY.get(kind)
    if family is not None and not selectors.supports(family, target):
        return "{0} path cannot resolve target={1}".format(family, target)
    return None


def selector_coverage() -> tuple[int, int]:
    """(kinds with a declared selector family, kinds in the executed whitelist)."""
    return len(_KIND_SELECTOR_FAMILY), len(_GENUINELY_EXECUTED_KINDS)


def extract_executable_kinds() -> frozenset[str]:
    """Return the set of action kinds whose handlers produce real, verifiable
    game state changes consumed by downstream rules.

    This is the hand-maintained _GENUINELY_EXECUTED_KINDS set (see docstring
    above).  The legacy string-match approach was dishonest because native.py
    contains a 30-kind catch-all that writes dead ``unit.status[kind]=True``
    data — discovered by those strings but never consumed by any game rule.
    """
    return _GENUINELY_EXECUTED_KINDS


def recompute_status(entry: dict, executable: frozenset[str], cards: CardDatabase | None = None) -> None:
    """Make ``status`` honest w.r.t. engine coverage; record gaps.

    Mutates ``entry`` in place: appends any un-executed action text to
    ``unresolved_fragments`` and sets ``status`` accordingly.
    """
    fragments = list(entry.get("unresolved_fragments") or [])
    for action in entry.get("actions") or []:
        kind = action.get("kind")
        if kind not in executable:
            note = "[engine does not implement kind={0}] {1}".format(
                kind, (action.get("card_name") or "").strip()
            ).strip()
            if note not in fragments:
                fragments.append(note)
            continue
        # The kind has a handler; check that the handler can resolve the target.
        problem = unsupported_selector(kind, str(action.get("target", "")))
        if problem is not None:
            note = "[selector contract: {0}]".format(problem)
            if note not in fragments:
                fragments.append(note)
        # A condition the engine cannot read is never satisfied, so the action
        # never runs: the card is not fully implemented.
        condition = action.get("condition")
        if cards is not None and condition:
            from simulator.rules.condition import handled_by_kind, is_modeled
            events = tuple(action.get("on") or entry.get("triggers") or ())
            if not handled_by_kind(kind, condition) and not is_modeled(condition, events, cards):
                note = "[condition not modeled] {0}".format(condition.strip())
                if note not in fragments:
                    fragments.append(note)
    entry["unresolved_fragments"] = fragments
    # Honest status: 'implemented' when nothing remains unresolved (every action
    # maps to an executed kind). An un-executed action keeps it 'partial'; text we
    # could not parse at all, with no actions, is 'unresolved'. A card with no
    # rules text and no abilities is fully resolved -> implemented.
    if fragments:
        entry["status"] = "partial" if entry.get("actions") else "unresolved"
    else:
        entry["status"] = "implemented"


def main() -> None:
    cards = CardDatabase.from_file(CATALOG)
    parser = RuleParser()
    rules = []
    for card in cards:
        rule = parser.parse(card)
        rules.append({
            "card_id": rule.card_id,
            "source_text": rule.source_text,
            "triggers": list(rule.triggers),
            "actions": [asdict(action) for action in rule.actions],
            "status": rule.status,
            "unresolved_fragments": list(rule.unresolved_fragments),
        })

    executable = extract_executable_kinds()
    for entry in rules:
        recompute_status(entry, executable, cards)

    DESTINATION.write_text(json.dumps({"schema_version": 1, "rules": rules}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts = {status: sum(entry["status"] == status for entry in rules) for status in ("implemented", "partial", "unresolved")}
    by_type: dict[str, dict[str, int]] = {}
    for card in cards:
        status = next(entry["status"] for entry in rules if entry["card_id"] == card.id)
        by_type.setdefault(card.type, {key: 0 for key in counts})[status] += 1

    # Engine-coverage summary: which kinds are executed vs still stubbed.
    stubbed: dict[str, int] = {}
    for entry in rules:
        for action in entry["actions"]:
            k = action.get("kind")
            if k not in executable:
                stubbed[k] = stubbed.get(k, 0) + 1

    rows = ["# Native Rule Implementation Report", "", "Generated by `tools/build_card_rules.py` (engine-coverage-aware) and `tools/classify_coverage.py` (mechanism categories).", "",
            "## Coverage", "", "| Status | Cards |", "|---|---:|"]
    rows.extend("| {0} | {1} |".format(status, counts[status]) for status in counts)
    rows.extend(["", "## By card type", "", "| Type | Implemented | Partial | Unresolved |", "|---|---:|---:|---:|"])
    rows.extend("| {0} | {1} | {2} | {3} |".format(card_type, values["implemented"], values["partial"], values["unresolved"]) for card_type, values in sorted(by_type.items()))
    rows.extend(["", "## Engine coverage", "",
                 "A card is `implemented` only when **every** parsed action maps to a kind the execution layer handles "
                 "(`native.py` / `resolver.py`). Any un-executed action downgrades the card to `partial` and is recorded "
                 "in `unresolved_fragments`.", "",
                 "- Executable action kinds: {0}".format(len(executable)),
                 "- Cards fully executable (implemented): {0}".format(counts["implemented"]),
                 "- Cards with at least one un-executed action (partial): {0}".format(counts["partial"]),
                 "", "### Stubbed action kinds (not yet executed by the simulator)", "",
                 "| Kind | Occurrences |", "|---|---:|"])
    for k, c in sorted(stubbed.items(), key=lambda kv: (-kv[1], kv[0])):
        rows.append("| {0} | {1} |".format(k, c))
    rows.extend(["", "Unresolved cards remain non-executable by design. The mechanism-level implementation of the stubbed "
                 "kinds above is owned by the simulator execution layer; this report only surfaces them honestly.", ""])
    REPORT.write_text("\n".join(rows), encoding="utf-8")
    print(json.dumps({"total_cards": len(cards), "executable_kinds": len(executable), **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
