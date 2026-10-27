"""Primitive state transitions used by the data-driven effect resolver."""

from __future__ import annotations

from simulator.core.hq import HQResolver
from simulator.core.state import GameState, UnitState
from simulator.core.turn import TurnManager


def damage(state: GameState, target: UnitState | str, amount: int, source_player: str | None = None) -> None:
    if isinstance(target, UnitState):
        redirectors = [
            unit for unit in state.players[target.owner_id].units
            if unit.instance_id != target.instance_id
            and unit.status.get("redirect_damage_from_card_id") == target.card_id
        ]
        if redirectors:
            redirector = redirectors[0]
            state.event_log.append({
                "event": "unit_damage_redirected",
                "from_unit_id": target.instance_id,
                "to_unit_id": redirector.instance_id,
                "amount": amount,
            })
            damage(state, redirector, amount, source_player)
            return
        # Immune units take no damage.
        if target.status.get("immune"):
            state.event_log.append({"event": "damage_ignored_immune", "target_id": target.instance_id})
            return
        target.defense -= amount
        floor = target.status.get("defense_floor")
        if isinstance(floor, int) and target.defense < floor:
            target.defense = floor          # 511th REGIMENT: "cannot go lower than 1"
        state.event_log.append({"event": "damage_dealt", "target_id": target.instance_id, "amount": amount})
        remove_dead_units(state)
        return
    # HQ damage: apply persistent, board-dependent reductions before the HQ's
    # defense modifier.  The board is inspected at damage time so buffs and
    # deaths immediately change cards such as M6 rather than leaving a stale
    # deployment-time count in state.
    player = state.players[target]
    for unit in player.units:
        if unit.status.get("hq_damage_reduced"):
            amount = max(0, amount - 1)
        dynamic_reduction = unit.status.get("hq_damage_reduction_by_attack")
        if isinstance(dynamic_reduction, dict):
            minimum_attack = dynamic_reduction.get("minimum_attack", 4)
            per_unit = dynamic_reduction.get("amount", 1)
            if isinstance(minimum_attack, int) and isinstance(per_unit, int):
                qualifying = sum(candidate.attack >= minimum_attack for candidate in player.units)
                amount = max(0, amount - (qualifying * max(0, per_unit)))
    HQResolver.damage(state, target, amount, source_player)


def heal(state: GameState, target: UnitState | str, amount: int) -> None:
    if isinstance(target, UnitState):
        target.defense += amount
    else:
        HQResolver.heal(state, target, amount)


def destroy(state: GameState, target: UnitState) -> None:
    target.defense = 0
    remove_dead_units(state)


def move(state: GameState, target: UnitState, position: str) -> None:
    if position not in state.battlefield:
        raise ValueError("Unknown battlefield position: {0}".format(position))
    if target.instance_id in state.battlefield.get(target.position, []):
        state.battlefield[target.position].remove(target.instance_id)
    target.position = position
    state.battlefield[position].append(target.instance_id)


def draw(state: GameState, player_id: str, amount: int) -> None:
    for _ in range(max(0, amount)):
        TurnManager.draw_card(state, player_id)


def remove_dead_units(state: GameState) -> None:
    for player_id, player in state.players.items():
        survivors = []
        for unit in player.units:
            if unit.defense > 0:
                survivors.append(unit)
                continue
            if unit.instance_id in state.battlefield.get(unit.position, []):
                state.battlefield[unit.position].remove(unit.instance_id)
            state.graveyard.setdefault(player_id, []).append(unit.card_id)
            from simulator.rules import counters
            counters.note_unit_destroyed(state, player_id, unit.card_id)
            state.event_log.append({"event": "unit_died", "player_id": player_id, "unit_id": unit.instance_id, "card_id": unit.card_id})
        player.units = survivors
    from simulator.rules.intel import refresh
    refresh(state)
