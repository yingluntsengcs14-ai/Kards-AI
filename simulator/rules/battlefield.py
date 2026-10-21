"""Authoritative KARDS line capacity and frontline-control rules."""
from __future__ import annotations
from simulator.core.state import GameState, UnitState

SUPPORT_UNIT_LIMIT = 4
FRONTLINE_UNIT_LIMIT = 5
# A player may never hold more than 9 cards; a 10th card is lost instead.
HAND_CARD_LIMIT = 9


class BattlefieldRules:
    @staticmethod
    def frontline_limit(state: GameState) -> int:
        limits = [unit.status.get("frontline_limit") for player in state.players.values() for unit in player.units]
        active = [value for value in limits if isinstance(value, int) and value > 0]
        return min([FRONTLINE_UNIT_LIMIT, *active])
    @staticmethod
    def frontline_controller(state: GameState) -> str | None:
        owners = {unit.owner_id for p in state.players.values() for unit in p.units if unit.position == "frontline"}
        return next(iter(owners)) if len(owners) == 1 else None

    @staticmethod
    def can_deploy_to_support(state: GameState, player_id: str) -> bool:
        return sum(unit.position == "support_line" for unit in state.players[player_id].units) < SUPPORT_UNIT_LIMIT

    @staticmethod
    def can_move_to_frontline(state: GameState, unit: UnitState) -> bool:
        controller = BattlefieldRules.frontline_controller(state)
        frontline_count = sum(unit.position == "frontline" for p in state.players.values() for unit in p.units)
        return unit.position == "support_line" and frontline_count < BattlefieldRules.frontline_limit(state) and controller in (None, unit.owner_id)
