"""Fixed candidate-action tensors and legal-action masks."""
from __future__ import annotations

from dataclasses import fields
import hashlib
from typing import Sequence

import torch

from simulator.actions.action import Action, AttackAction, MoveUnitAction, MulliganAction, PassAction, PlayCardAction, UseAbilityAction

# Layout (28 dims):
#   0..5   action type one-hot
#   6      unknown action type
#   7      reserved
#   8..21  seven identity slots x (hash, length) - one per field of the widest
#          action, so NO field is dropped
#   22..24 weather choice one-hot: sunny / rain / storm
#   25..26 keyword choice one-hot: blitz / shock
#   27     the selected card came from hand
#
# Why 28 and not the original 16: the old layout encoded only the FIRST FIVE
# fields of the action, and PlayCardAction has seven:
#   player_id, card_id, position, target_unit_id, selected_card_id,
#   selected_option, selected_card_zone
#                 ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^  both dropped
# So any two actions differing only in the chosen OPTION encoded to identical
# feature vectors.  The MCTS tree still told them apart (children are keyed by
# action index), but the network's prior and the training target could not, so
# every "choose one of N" decision was unlearnable:
#   * Forecast - sunny / rain / storm (10 OCEANIA STORM cards)
#   * SHOCK TACTICS - blitz / shock
#   * Develop options
# The old layout also had a collision: an unrecognized action type wrote to
# index 15, which was already the length feature of the fifth identity slot.
ACTION_FEATURE_DIM = 28
# KARDS orders can combine many selectable cards and battlefield targets.  The
# original 512-row transport limit rejected legal Japanese deck positions;
# 1024 covers the full currently bundled card catalogue without pruning.
MAX_ACTIONS = 1024
_TYPE = {PassAction: 0, PlayCardAction: 1, AttackAction: 2, MoveUnitAction: 3, UseAbilityAction: 4, MulliganAction: 5}
_UNKNOWN_TYPE_SLOT = 6
_IDENTITY_BASE = 8
# Every field of the widest action; keep in step with PlayCardAction.
_IDENTITY_SLOTS = 7
_OPTION_BASE = 22
# Closed option vocabularies get an explicit one-hot on top of the identity
# hash, so the three Forecast choices are linearly separable instead of being
# three arbitrary hash values the network has to memorize.
_WEATHER_OPTIONS = ("sunny", "rain", "storm")
_KEYWORD_OPTIONS = ("blitz", "shock")
_ZONE_SLOT = 27


def action_key(action: Action) -> tuple:
    """Stable, serializable identity used to map MCTS/network slots to actions."""
    return (type(action).__name__,) + tuple(getattr(action, item.name) for item in fields(action))


class ActionEncoder:
    def __init__(self, max_actions: int = MAX_ACTIONS) -> None:
        self.max_actions = max_actions
        self._feature_cache: dict[tuple, torch.Tensor] = {}

    def encode(self, action: Action) -> torch.Tensor:
        key = action_key(action)
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached
        x = torch.zeros(ACTION_FEATURE_DIM, dtype=torch.float32)
        x[_TYPE.get(type(action), _UNKNOWN_TYPE_SLOT)] = 1.0
        # Compact identifiers retain action/card/target distinctions without
        # making a network output depend on a global, ever-changing action ID.
        values = key[1:]
        for index, value in enumerate(values[:_IDENTITY_SLOTS]):
            if value is None:
                continue
            text = str(value)
            digest = hashlib.blake2b(text.encode(), digest_size=2).digest()
            x[_IDENTITY_BASE + index * 2] = int.from_bytes(digest, "big") / 65535.0
            x[_IDENTITY_BASE + index * 2 + 1] = min(1.0, len(text) / 32.0)
        # A choice the player makes when playing the card is a decision the
        # network has to be able to see, not just an identity difference.
        option = str(getattr(action, "selected_option", None) or "").lower()
        if option in _WEATHER_OPTIONS:
            x[_OPTION_BASE + _WEATHER_OPTIONS.index(option)] = 1.0
        elif option in _KEYWORD_OPTIONS:
            x[_OPTION_BASE + len(_WEATHER_OPTIONS) + _KEYWORD_OPTIONS.index(option)] = 1.0
        if str(getattr(action, "selected_card_zone", None) or "").lower() == "hand":
            x[_ZONE_SLOT] = 1.0
        # Encoded features are immutable; keeping a bounded per-encoder cache
        # avoids hashing the same MCTS candidates at every expanded node.
        if len(self._feature_cache) < 8192:
            self._feature_cache[key] = x
        return x

    def encode_legal_actions(self, actions: Sequence[Action], *, pad_to_max: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode legal candidates, optionally omitting mask-only padding for inference."""
        if not actions:
            raise ValueError("A non-terminal state must expose at least one legal action")
        # Tree search and replay keep only the meaningful legal prefix, so
        # they can safely represent an arbitrarily large legal action set.
        # The fixed width is only required by legacy padded consumers.
        if pad_to_max and len(actions) > self.max_actions:
            raise ValueError("Legal action count exceeds configured MAX_ACTIONS")
        size = self.max_actions if pad_to_max else len(actions)
        features = torch.zeros((size, ACTION_FEATURE_DIM), dtype=torch.float32)
        mask = torch.zeros(size, dtype=torch.bool)
        for index, action in enumerate(actions):
            features[index] = self.encode(action)
            mask[index] = True
        return features, mask

    @staticmethod
    def action_at(actions: Sequence[Action], index: int) -> Action:
        if index < 0 or index >= len(actions):
            raise ValueError("Network selected an illegal padded action slot")
        return actions[index]
