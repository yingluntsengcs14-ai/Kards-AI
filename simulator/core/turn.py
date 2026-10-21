"""Turn lifecycle and deterministic card draw operations."""

from __future__ import annotations

from simulator.actions.validator import opponent_id
from simulator.core.state import GameState
from simulator.core.hq import HQResolver


MAX_KREDITS = 12


class TurnManager:
    """Owns turn-boundary mutations; actions delegate here instead of inlining them."""

    @staticmethod
    def start_turn(state: GameState, player_id: str, draw_card: bool = True) -> None:
        player = state.players[player_id]
        if player.active_countermeasures:
            player.active_countermeasures.clear()
            state.event_log.append({"event": "countermeasures_expired", "player_id": player_id})
        # Cards may raise the normal 12-slot ceiling.  Keep the ceiling in
        # player status so those effects survive turn transitions.
        limit = player.status.get("kredit_limit", MAX_KREDITS)
        player.resources.max_kredits = min(max(MAX_KREDITS, int(limit)), player.resources.max_kredits + 1)
        player.resources.kredits = player.resources.max_kredits
        for unit in player.units:
            unit.status.pop("attacked_this_turn", None)
            unit.status.pop("attack_count", None)
            unit.status.pop("deployed_this_turn", None)
            unit.status.pop("moved_this_turn", None)
            unit.status.pop("ambush_used_this_round", None)
        # "this turn" counters belong to the turn now starting, for both players:
        # a player's HQ can be damaged during the opponent's turn.
        from simulator.rules import counters
        counters.clear_turn_counters(state)
        # A forecast made on an earlier turn produces its weather card now.
        from simulator.rules import weather
        weather.resolve_pending_forecast(state, player_id)
        if draw_card:
            TurnManager.draw_card(state, player_id)
        state.event_log.append({"event": "turn_started", "player_id": player_id, "turn": state.turn_number})
        # Fire scheduled one-shot effects due at the start of this player's turn
        # (e.g. "return it at the start of your next turn"). Imported lazily to
        # avoid a circular import (native -> effects -> turn).
        from simulator.rules.native import NativeRuleEngine
        NativeRuleEngine.fire_scheduled(state, player_id, "start")

    @staticmethod
    def draw_card(state: GameState, player_id: str) -> str | None:
        player = state.players[player_id]
        if not player.deck:
            player.fatigue_damage += 1
            HQResolver.damage(state, player_id, player.fatigue_damage)
            state.event_log.append({"event": "deck_empty", "player_id": player_id, "fatigue_damage": player.fatigue_damage})
            return None
        card_id = player.deck.pop(0)
        # A new copy of a hand-reactive card must not inherit a cost history
        # from a previous copy that already left the hand.
        bonuses = player.status.get("hand_cost_on_friendly_death")
        if isinstance(bonuses, dict) and card_id not in player.hand:
            bonuses.pop(card_id, None)
        if len(player.hand) >= 9:
            state.graveyard.setdefault(player_id, []).append(card_id)
            state.event_log.append({"event": "card_overdrawn", "player_id": player_id, "card_id": card_id})
            return None
        player.hand.append(card_id)
        from simulator.rules.intel import refresh
        refresh(state)
        state.event_log.append({"event": "card_drawn", "player_id": player_id, "card_id": card_id})
        return card_id

    @staticmethod
    def end_turn(state: GameState, player_id: str) -> None:
        next_player = opponent_id(state, player_id)
        # Revert the ending player's temporary (this_turn / next_turn) effects
        # before the turn number advances, using the current turn as the expiry
        # reference. Imported lazily to avoid a circular import with the rules
        # layer (native -> effects -> turn).
        from simulator.rules.native import NativeRuleEngine
        NativeRuleEngine.revert_temporary(state, player_id, state.turn_number)
        # Fire scheduled one-shot effects due at the end of this player's turn
        # (e.g. "discarded end of turn" / "discard it at the end of your next turn").
        NativeRuleEngine.fire_scheduled(state, player_id, "end")
        # "if Rain was played this turn" only ever asks about the current turn.
        from simulator.rules import weather as weather_rules
        weather_rules.clear_turn_state(state, player_id)
        state.event_log.append({"event": "turn_ended", "player_id": player_id, "turn": state.turn_number})
        state.current_player = next_player
        state.turn_number += 1
        TurnManager.start_turn(state, next_player)
