"""Effects attached to cards while they sit in a hand.

Hands store card ids, so these are kept per card id on ``PlayerState.status``:

* ``hand_cost_reduction`` -- {card id: [[amount, floor], ...]}, one entry per
  discounted copy; the first entry applies when that card is played.
  FRONT AND HOMELAND (cards drawn from the deck), 27. FÜSILIERS (German units
  entering your hand, never below 1).
* ``deck_cost_reduction`` -- {card id: copies}; moved to the hand list when a
  copy is drawn.
* ``on_play_destroy_random`` -- {card id: copies}; CORPS COMMAND's granted text.
"""
from __future__ import annotations

HAND_COST = "hand_cost_reduction"
DECK_COST = "deck_cost_reduction"
ON_PLAY_DESTROY = "on_play_destroy_random"


def card_entered_hand(state, player_id: str, card_id: str, from_deck: bool) -> None:
    from simulator.rules.native import NativeRuleEngine
    player = state.players[player_id]
    if from_deck:
        pending = player.status.get(DECK_COST) or {}
        if pending.get(card_id):
            pending[card_id] -= 1
            if not pending[card_id]:
                pending.pop(card_id)
            player.status.setdefault(HAND_COST, {}).setdefault(card_id, []).append([1, 0])
    engine = NativeRuleEngine._last_engine
    if engine is None or card_id not in engine.cards:
        return
    card = engine.cards.get(card_id)
    fusiliers = sum(1 for unit in player.units if unit.card_id == "27_fsiliers")
    if fusiliers and card.is_unit and card.nation == "Germany":
        for _ in range(fusiliers):
            player.status.setdefault(HAND_COST, {}).setdefault(card_id, []).append([1, 1])


def reduced_cost(player, card, cost: int) -> int:
    entries = (player.status.get(HAND_COST) or {}).get(card.id) or []
    if not entries:
        return cost
    amount, floor = entries[0]
    return max(cost - int(amount), int(floor)) if cost > int(floor) else cost


def card_played(player, card_id: str) -> None:
    entries = (player.status.get(HAND_COST) or {}).get(card_id)
    if entries:
        entries.pop(0)
        if not entries:
            player.status[HAND_COST].pop(card_id)
