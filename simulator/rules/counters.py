"""Battle-long and turn-long counters that card text refers to.

Some cards name a quantity the board alone cannot answer: "the kredit slots you
have lost this battle", "Polish units destroyed this battle", "damage to your HQ
this turn", "kredits spent operating this turn".  Those cards used to deal 0
damage (or ignore a cost reduction) because nothing counted.

Counters live on ``PlayerState.status``, so they serialize and branch with the
state like everything else the AI searches over.

Two lifetimes:

* **battle** — kept for the whole game (kredit slots lost, destroyed cards).
* **turn**   — cleared for BOTH players when a turn starts, because "this turn"
  means the turn now being played, and a player's HQ can be damaged during the
  opponent's turn.
"""

from __future__ import annotations

KREDIT_SLOTS_LOST = "kredit_slots_lost_this_battle"
DESTROYED_CARD_IDS = "destroyed_card_ids_this_battle"
HQ_DAMAGE_THIS_TURN = "hq_damage_this_turn"
OPERATING_SPEND_THIS_TURN = "kredits_spent_operating_this_turn"
ORDERS_THIS_TURN = "orders_given_this_turn"
UNITS_DESTROYED_THIS_TURN = "units_destroyed_this_turn"
DEPLOYED_BY_TURN = "units_deployed_by_turn"   # battle: {turn number: [card ids]}

# Cleared at the start of every turn.
TURN_KEYS = (HQ_DAMAGE_THIS_TURN, OPERATING_SPEND_THIS_TURN, ORDERS_THIS_TURN, UNITS_DESTROYED_THIS_TURN)


def _bump(state, player_id: str, key: str, amount: int) -> int:
    player = state.players.get(player_id)
    if player is None or amount <= 0:
        return 0
    player.status[key] = int(player.status.get(key, 0)) + int(amount)
    return player.status[key]


def _read(state, player_id: str, key: str) -> int:
    player = state.players.get(player_id)
    if player is None:
        return 0
    value = player.status.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


# ── kredit slots lost (battle) ──────────────────────────────────────────────

def note_kredit_slots_lost(state, player_id: str, amount: int) -> None:
    if _bump(state, player_id, KREDIT_SLOTS_LOST, amount):
        state.event_log.append({"event": "kredit_slots_lost_counted", "player_id": player_id,
                                "amount": int(amount),
                                "total": _read(state, player_id, KREDIT_SLOTS_LOST)})


def kredit_slots_lost(state, player_id: str) -> int:
    return _read(state, player_id, KREDIT_SLOTS_LOST)


# ── destroyed units (battle) ────────────────────────────────────────────────

def note_unit_destroyed(state, player_id: str, card_id: str) -> None:
    """Record a destroyed unit's card id so its nation can be read back later.

    The card id is stored rather than a per-nation tally because the destruction
    site has no card database to resolve the nation with.
    """
    player = state.players.get(player_id)
    if player is None or not card_id:
        return
    destroyed = player.status.setdefault(DESTROYED_CARD_IDS, [])
    destroyed.append(card_id)
    _bump(state, player_id, UNITS_DESTROYED_THIS_TURN, 1)


def units_destroyed_this_turn(state) -> int:
    """Units destroyed this turn, both sides (24th UŁAN)."""
    return sum(_read(state, player_id, UNITS_DESTROYED_THIS_TURN) for player_id in state.players)


def destroyed_card_ids(state, player_id: str | None = None) -> tuple[str, ...]:
    """Destroyed card ids for one player, or for both when player_id is None."""
    ids: list[str] = []
    for candidate, player in state.players.items():
        if player_id is not None and candidate != player_id:
            continue
        stored = player.status.get(DESTROYED_CARD_IDS)
        if isinstance(stored, list):
            ids.extend(str(card_id) for card_id in stored)
    return tuple(ids)


def units_destroyed_of_nation(state, nation: str, cards, player_id: str | None = None) -> int:
    """How many units of ``nation`` were destroyed this battle.

    Counts both players by default: "Polish units destroyed this battle" does not
    say whose.  NOT CONFIRMED IN THE CLIENT — if it only counts your own, pass
    the caster's player_id.
    """
    from simulator.rules.nations import matches, normalize
    if not normalize(nation):
        return 0
    total = 0
    for card_id in destroyed_card_ids(state, player_id):
        if card_id in cards and matches(cards.get(card_id).nation, nation):
            total += 1
    return total


# ── HQ damage (turn) ────────────────────────────────────────────────────────

def note_hq_damage(state, player_id: str, amount: int) -> None:
    _bump(state, player_id, HQ_DAMAGE_THIS_TURN, amount)


def hq_damage_this_turn(state, player_id: str) -> int:
    return _read(state, player_id, HQ_DAMAGE_THIS_TURN)


# ── operating spend (turn) ──────────────────────────────────────────────────

def note_operating_spend(state, player_id: str, amount: int) -> None:
    """Kredits spent operating (attacking or moving) during this turn."""
    _bump(state, player_id, OPERATING_SPEND_THIS_TURN, amount)


def operating_spend_this_turn(state, player_id: str) -> int:
    return _read(state, player_id, OPERATING_SPEND_THIS_TURN)


# ── orders given (turn) ─────────────────────────────────────────────────────

def note_order_given(state, player_id: str) -> int:
    """Count an order as it is played, before its effects: "if this is the
    second order you give this turn" counts the order being played."""
    return _bump(state, player_id, ORDERS_THIS_TURN, 1)


def orders_given_this_turn(state, player_id: str) -> int:
    return _read(state, player_id, ORDERS_THIS_TURN)


# ── units deployed (battle, by turn) ────────────────────────────────────────

def note_unit_deployed(state, player_id: str, card_id: str) -> None:
    player = state.players.get(player_id)
    if player is None or not card_id:
        return
    by_turn = player.status.setdefault(DEPLOYED_BY_TURN, {})
    by_turn.setdefault(str(state.turn_number), []).append(card_id)
    # Only "this turn" and "last turn" are ever asked; keep the record small.
    for key in [key for key in by_turn if int(key) < state.turn_number - 4]:
        by_turn.pop(key)


def units_deployed(state, player_id: str, turns_ago: int = 0) -> tuple[str, ...]:
    """Card ids a player deployed on their own turn ``turns_ago`` turns back
    (0 = this turn, 1 = their previous turn, two game turns earlier)."""
    player = state.players.get(player_id)
    if player is None:
        return ()
    by_turn = player.status.get(DEPLOYED_BY_TURN) or {}
    return tuple(by_turn.get(str(state.turn_number - 2 * turns_ago), ()))


# ── lifetime ───────────────────────────────────────────────────────────────

def clear_turn_counters(state) -> None:
    """Reset the turn-scoped counters for both players."""
    for player in state.players.values():
        for key in TURN_KEYS:
            player.status.pop(key, None)
