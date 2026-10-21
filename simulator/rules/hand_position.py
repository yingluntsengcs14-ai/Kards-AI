"""Hand order: the left-most and right-most card (OCEANIA STORM / RING OF FIRE).

Several cards read or write a card's position in hand: "Discard your left-most
card", "Move it all the way to the left", "Deal 3 damage instead if this card
was left-most in hand".  ``PlayerState.hand`` is already an ordered list, so the
convention is fixed here in one place rather than spread across handlers:

    hand[0]  is the LEFT-most card
    hand[-1] is the RIGHT-most card

``TurnManager.draw_card`` appends, so a freshly drawn card is right-most and the
left-most card is the one held longest.

NOT CONFIRMED IN THE CLIENT: that a drawn card really enters on the right. If the
live game inserts elsewhere, only this module's convention changes.
"""

from __future__ import annotations

LEFTMOST = 0
RIGHTMOST = -1


def is_leftmost(hand_index: int | None, hand_size: int | None) -> bool:
    """Whether the card played from ``hand_index`` was the left-most one."""
    if hand_index is None or not hand_size:
        return False
    return hand_index == 0


def is_rightmost(hand_index: int | None, hand_size: int | None) -> bool:
    """Whether the card played from ``hand_index`` was the right-most one."""
    if hand_index is None or not hand_size:
        return False
    return hand_index == hand_size - 1


def discard_leftmost(state, player_id: str) -> str | None:
    """Discard the left-most card in hand. Returns the discarded card id."""
    player = state.players[player_id]
    if not player.hand:
        state.event_log.append({"event": "hand_empty_nothing_to_discard", "player_id": player_id})
        return None
    card_id = player.hand.pop(LEFTMOST)
    state.graveyard.setdefault(player_id, []).append(card_id)
    state.event_log.append({"event": "leftmost_card_discarded", "player_id": player_id,
                            "card_id": card_id})
    return card_id


def move_to_leftmost(state, player_id: str, card_id: str) -> bool:
    """Move one copy of ``card_id`` to the left-most hand position."""
    player = state.players[player_id]
    if card_id not in player.hand:
        state.event_log.append({"event": "hand_card_not_found", "player_id": player_id,
                                "card_id": card_id})
        return False
    player.hand.remove(card_id)
    player.hand.insert(LEFTMOST, card_id)
    state.event_log.append({"event": "hand_card_moved_leftmost", "player_id": player_id,
                            "card_id": card_id})
    return True
