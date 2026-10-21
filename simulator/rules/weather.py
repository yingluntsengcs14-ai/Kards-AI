"""KARDS weather rules (OCEANIA STORM): Forecast tokens and the nine weather cards.

Model
-----
A card that reads "Forecast" creates one of three forecast tokens, which
announces the weather category that follows:

    BLUE SKY ("It will be Sunny.")   -> sunny
    MIST     ("Rain is on its way.") -> rain
    GALE     ("A Storm is coming.")  -> storm

Each category has three weather cards, costing 2 / 4 / 6 kredits:

    sunny: HEATWAVE, JUNGLE FEVER, SCORCHING SUN
    rain:  DELUGE, TORRENTIAL RAIN, MONSOON RAIN
    storm: THUNDERSTORM, TROPICAL STORM, CYCLONE

Seventeen OCEANIA STORM cards read this state, in two shapes: an event ("When
you play a Sunny card…", "Draw a card when you Forecast") and a per-turn
question ("if Sunny or Rain was played this turn", "Costs 2 less if Rain was
played this turn").  Both are served from here: the event emitters live in the
action layer, the per-turn record in ``played_this_turn``.

CONFIRMED IN THE CLIENT (2026-10-20, by the project owner)
----------------------------------------------------------
* **The player picks the category** when forecasting, so Forecast is a
  three-way choice in the action space (see ``ActionGenerationSpec.forecast``).
* **The cost tier is random** among that category's three cards (2 / 4 / 6
  kredits).  See ``pick_weather_card``.
* **The forecast token never enters hand.**  Only the weather card does, at the
  start of the forecasting player's next turn.  The pending forecast therefore
  lives in ``player.status["pending_forecast"]`` and is not a playable card.

Strategic shape of the three categories, for anyone reading the AI's play later:
sunny is the board-filling / aggression package, rain is the suppression,
healing and card-draw package, and storm is the board-clearing package.
"""

from __future__ import annotations

import random

# Forecast token -> weather category.
FORECAST_TOKENS: dict[str, str] = {
    "blue_sky": "sunny",
    "mist": "rain",
    "gale": "storm",
}

# Weather category -> its cards, cheapest tier first.
WEATHER_TIERS: dict[str, tuple[str, ...]] = {
    "sunny": ("heatwave", "jungle_fever", "scorching_sun"),
    "rain": ("deluge", "torrential_rain", "monsoon_rain"),
    "storm": ("thunderstorm", "tropical_storm", "cyclone"),
}

# Weather card -> weather category.
WEATHER_CARDS: dict[str, str] = {
    card_id: category for category, cards in WEATHER_TIERS.items() for card_id in cards
}

CATEGORIES = tuple(WEATHER_TIERS)

# Weather category -> its forecast token.  The player picks the category when
# forecasting (confirmed in the live client, 2026-10-20), so the action layer
# needs this direction of the mapping.
TOKEN_FOR_CATEGORY: dict[str, str] = {category: token for token, category in FORECAST_TOKENS.items()}

_PENDING_KEY = "pending_forecast"
_PLAYED_KEY = "weather_played_this_turn"
_HAND_LIMIT = 9


def is_weather_card(card_id: str) -> bool:
    """Whether ``card_id`` is one of the nine weather cards."""
    return card_id in WEATHER_CARDS


def is_forecast_token(card_id: str) -> bool:
    """Whether ``card_id`` is one of the three forecast tokens."""
    return card_id in FORECAST_TOKENS


def category_of(card_id: str) -> str | None:
    """Weather category of a weather card or a forecast token, else None."""
    return WEATHER_CARDS.get(card_id) or FORECAST_TOKENS.get(card_id)


def _rng(state) -> random.Random:
    """Deterministic RNG, seeded the way the rest of the simulator seeds it."""
    return random.Random((getattr(state, "rng_seed", 0) or 0) + len(state.event_log))


def forecast(state, player_id: str, token_id: str | None = None) -> str:
    """Create a forecast token for ``player_id`` and return its card id.

    The token replaces any forecast that has not resolved yet: a player has one
    forecast at a time.  ``token_id`` forces a specific token, which keeps tests
    independent of the random draw.
    """
    if token_id is None:
        token_id = _rng(state).choice(tuple(FORECAST_TOKENS))
    if token_id not in FORECAST_TOKENS:
        raise ValueError("Unknown forecast token: " + str(token_id))
    player = state.players[player_id]
    previous = player.status.get(_PENDING_KEY)
    player.status[_PENDING_KEY] = token_id
    state.event_log.append({"event": "forecast_made", "player_id": player_id,
                            "card_id": token_id, "weather": FORECAST_TOKENS[token_id],
                            "replaced": previous})
    return token_id


def pending_forecast(state, player_id: str) -> str | None:
    """The forecast token waiting to resolve for ``player_id``, if any."""
    return state.players[player_id].status.get(_PENDING_KEY)


def pick_weather_card(state, player_id: str, category: str) -> str:
    """Choose which of a category's three cards a forecast produces.

    The cost tier is picked at random among the category's 2 / 4 / 6 kredit
    cards (confirmed in the live client by the project owner, 2026-10-20).
    The draw uses the simulator's usual seeding so self-play stays reproducible.
    """
    return _rng(state).choice(WEATHER_TIERS[category])


def resolve_pending_forecast(state, player_id: str) -> str | None:
    """Turn a pending forecast into its weather card in hand. Returns the card id.

    Called at the start of the forecasting player's turn (confirmed in the live
    client).  Only the weather card reaches hand; the forecast token itself never
    does.  A full hand loses the card, matching the ordinary 9-card limit.
    """
    player = state.players[player_id]
    token_id = player.status.pop(_PENDING_KEY, None)
    if token_id is None:
        return None
    category = FORECAST_TOKENS[token_id]
    card_id = pick_weather_card(state, player_id, category)
    if len(player.hand) < _HAND_LIMIT:
        player.hand.append(card_id)
        overdrawn = False
    else:
        state.graveyard.setdefault(player_id, []).append(card_id)
        overdrawn = True
    state.event_log.append({"event": "forecast_resolved", "player_id": player_id,
                            "from_card_id": token_id, "card_id": card_id,
                            "weather": category, "overdrawn": overdrawn})
    return card_id


def note_weather_played(state, player_id: str, card_id: str) -> str | None:
    """Record that ``player_id`` played a weather card this turn."""
    category = WEATHER_CARDS.get(card_id)
    if category is None:
        return None
    played = state.players[player_id].status.setdefault(_PLAYED_KEY, [])
    played.append(category)
    state.event_log.append({"event": "weather_played", "player_id": player_id,
                            "card_id": card_id, "weather": category})
    return category


def played_this_turn(state, player_id: str) -> tuple[str, ...]:
    """Weather categories ``player_id`` has played during the current turn."""
    return tuple(state.players[player_id].status.get(_PLAYED_KEY, ()))


def was_played_this_turn(state, player_id: str, *categories: str) -> bool:
    """Whether any of ``categories`` was played by ``player_id`` this turn."""
    played = set(played_this_turn(state, player_id))
    return any(category in played for category in categories)


def clear_turn_state(state, player_id: str) -> None:
    """Forget this turn's weather record; called at the player's turn end."""
    state.players[player_id].status.pop(_PLAYED_KEY, None)
