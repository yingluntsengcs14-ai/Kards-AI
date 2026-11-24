"""Fixed-shape, player-relative observation tensor encoder (layout v2).

Layout v2 (2026-11-24), all player-relative (``own`` = the player to encode):

* global block (``GLOBAL_DIM``): turn, whose turn, who holds the frontline;
  per player HQ / kredits / hand, deck and active-countermeasure counts,
  board size, the public pending Forecast (sunny / rain / storm) and whether
  weather was played this turn; main and ally nation one-hots.
* ``MAX_UNITS_PER_PLAYER`` unit slots per side (``UNIT_DIM`` each): static
  card features, current attack/defense, damaged, frontline, and runtime
  status (pinned, suppressed, veteran, just deployed, moved, attacks made,
  runtime keywords such as granted Blitz).
* ``MAX_HAND`` own hand slots (``HAND_DIM`` each), including the real cost
  after cost modifiers.
* ``CARD_SLOTS`` card-vocabulary indices at the very end (units, hand, own
  active countermeasures), stored as integer-valued floats; ``KARDSNet``
  embeds them. Enemy hand cards and enemy countermeasures are hidden, so only
  their counts appear.

v1 (362 dims) encoded a card as 8 floats incl. a 16-bit hash and had no
countermeasure, weather, deck-size or unit-status information.
"""
from __future__ import annotations

import torch

from ai.card_features import DEFENSE_INDEX, KEYWORD_INDEX, KREDITS_INDEX, KEYWORDS, NATIONS, STATIC_DIM, card_index, static_features
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, UnitState
from simulator.rules.keywords import KeywordEngine

OBSERVATION_VERSION = 2
MAX_HAND = 12
MAX_UNITS_PER_PLAYER = 8
MAX_OWN_COUNTERMEASURES = 4
_FORECASTS = ("blue_sky", "mist", "gale")  # sunny / rain / storm tokens
PLAYER_DIM = 4 + 4 + len(_FORECASTS) + 1
GLOBAL_DIM = 2 + 2 + 2 * PLAYER_DIM + 4 * len(NATIONS)
UNIT_DIM = 1 + STATIC_DIM + 2 + 2 + 6 + len(KEYWORDS)
HAND_DIM = 1 + STATIC_DIM + 1
CARD_SLOTS = 2 * MAX_UNITS_PER_PLAYER + MAX_HAND + MAX_OWN_COUNTERMEASURES
NUMERIC_DIM = GLOBAL_DIM + 2 * MAX_UNITS_PER_PLAYER * UNIT_DIM + MAX_HAND * HAND_DIM
STATE_DIM = NUMERIC_DIM + CARD_SLOTS
# Kept for older imports; v2 no longer has a single per-card width.
CARD_FEATURE_DIM = STATIC_DIM
_RUNTIME_KEYWORD_KEYS = ("suppressed", "added_abilities", "removed_abilities", "aura_abilities", "keyword_if_attack")


class ObservationEncoder:
    """Encodes only information observable by ``player_id``; no enemy cards leak."""
    def __init__(self, cards: CardDatabase) -> None:
        self.cards = cards

    def encode(self, state: GameState, player_id: str) -> torch.Tensor:
        enemy_id = "p2" if player_id == "p1" else "p1"
        own, enemy = state.players[player_id], state.players[enemy_id]
        values: list[float] = [min(state.turn_number, 30) / 30.0, float(state.current_player == player_id),
                               float(any(unit.position == "frontline" for unit in own.units)),
                               float(any(unit.position == "frontline" for unit in enemy.units))]
        for player in (own, enemy):
            forecast = player.status.get("pending_forecast")
            values += [player.hq.current_health / 20.0, player.hq.defense_modifier / 30.0,
                       player.resources.kredits / 12.0, player.resources.max_kredits / 12.0,
                       len(player.hand) / 10.0, len(player.deck) / 40.0,
                       len(player.active_countermeasures) / 3.0, len(player.units) / 8.0]
            values += [float(forecast == token) for token in _FORECASTS]
            values.append(float(bool(player.status.get("weather_played_this_turn"))))
        for player in (own, enemy):
            for nation in (player.nation, player.ally_nation):
                values += [float(nation == name) for name in NATIONS]
        ids: list[float] = []
        for player in (own, enemy):
            units = list(player.units)[:MAX_UNITS_PER_PLAYER]
            for unit in units:
                values += self._unit_features(unit)
                ids.append(float(card_index(unit.card_id)))
            missing = MAX_UNITS_PER_PLAYER - len(units)
            values += [0.0] * (UNIT_DIM * missing); ids += [0.0] * missing
        hand = own.hand[:MAX_HAND]
        for card_id in hand:
            values += self._hand_features(state, player_id, card_id)
            ids.append(float(card_index(card_id)))
        values += [0.0] * (HAND_DIM * (MAX_HAND - len(hand))); ids += [0.0] * (MAX_HAND - len(hand))
        counters = [entry.get("card_id") for entry in own.active_countermeasures][:MAX_OWN_COUNTERMEASURES]
        ids += [float(card_index(card_id)) for card_id in counters] + [0.0] * (MAX_OWN_COUNTERMEASURES - len(counters))
        values += ids
        if len(values) != STATE_DIM:
            raise AssertionError("Observation encoder shape drift")
        return torch.tensor(values, dtype=torch.float32)

    def _unit_features(self, unit: UnitState) -> list[float]:
        status = unit.status
        static = static_features(unit.card_id)
        base_defense = static[DEFENSE_INDEX]  # printed defense / 20
        values = [1.0, *static,
                  max(-1.0, min(2.0, unit.attack / 20.0)), max(-1.0, min(2.0, unit.defense / 20.0)),
                  float(unit.defense / 20.0 < base_defense), float(unit.position == "frontline"),
                  float(bool(status.get("pinned"))), float(bool(status.get("suppressed"))),
                  float(bool(status.get("veteran"))), float(bool(status.get("deployed_this_turn"))),
                  float(bool(status.get("moved_this_turn"))), min(1.0, status.get("attack_count", 0) / 2.0)]
        # Runtime keywords differ from the printed ones only when something
        # granted, removed or suppressed abilities; skip the (hot) keyword
        # engine otherwise -- it was ~10% of self-play time.
        if not any(key in status for key in _RUNTIME_KEYWORD_KEYS) or unit.card_id not in self.cards:
            return values + list(static[KEYWORD_INDEX:KEYWORD_INDEX + len(KEYWORDS)])
        card = self.cards.get(unit.card_id)
        return values + [float(KeywordEngine.has(card, keyword, unit)) for keyword in KEYWORDS]

    def _hand_features(self, state: GameState, player_id: str, card_id: str) -> list[float]:
        static = static_features(card_id)
        cost = static[KREDITS_INDEX] * 12.0  # printed kredits
        if card_id in self.cards:
            try:
                from simulator.actions.action import card_play_cost
                cost = card_play_cost(state, player_id, self.cards.get(card_id), self.cards)
            except Exception:  # cost modifiers are best-effort context, never fatal
                pass
        return [1.0, *static, max(0.0, cost) / 12.0]
