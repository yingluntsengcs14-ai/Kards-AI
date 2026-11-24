"""PyTorch policy/value network scored against legal candidate actions."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from ai.action_encoder import ACTION_CARD_SLOTS, ACTION_FEATURE_DIM
from ai.card_features import CARD_VOCAB
from ai.observation import CARD_SLOTS, STATE_DIM


class KARDSNet(nn.Module):
    """Policy/value network.

    v2 (2026-11-24): the last ``state_card_slots`` state columns and the last
    ``action_card_slots`` action columns are card-vocabulary indices; they go
    through one shared ``nn.Embedding`` (the same card means the same thing in
    hand, on the board or in an action) and are concatenated with the numeric
    features. The state trunk is three layers deep. Checkpoints written
    before v2 have no ``*_card_slots`` in their architecture and are rebuilt
    with the old two-layer, embedding-free layout (``layout_version=1``).
    """
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_FEATURE_DIM, hidden_dim: int = 128,
                 state_card_slots: int = CARD_SLOTS, action_card_slots: int = ACTION_CARD_SLOTS,
                 card_vocab: int = CARD_VOCAB, card_embed_dim: int = 16, layout_version: int = 2) -> None:
        super().__init__()
        if layout_version == 1:
            state_card_slots = action_card_slots = card_embed_dim = 0
        self.architecture = {"state_dim": state_dim, "action_dim": action_dim, "hidden_dim": hidden_dim,
                             "state_card_slots": state_card_slots, "action_card_slots": action_card_slots,
                             "card_vocab": card_vocab, "card_embed_dim": card_embed_dim, "layout_version": layout_version}
        self.state_card_slots, self.action_card_slots, self.card_vocab = state_card_slots, action_card_slots, card_vocab
        state_in = state_dim - state_card_slots + state_card_slots * card_embed_dim
        action_in = action_dim - action_card_slots + action_card_slots * card_embed_dim
        if layout_version == 1:
            self.card_embedding = None
            self.state_encoder = nn.Sequential(nn.Linear(state_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        else:
            self.card_embedding = nn.Embedding(card_vocab, card_embed_dim, padding_idx=0)
            self.state_encoder = nn.Sequential(nn.Linear(state_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                               nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_in, hidden_dim), nn.ReLU())
        self.policy_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1), nn.Tanh())

    def _embed_tail(self, x: torch.Tensor, slots: int) -> torch.Tensor:
        """Replace the trailing ``slots`` index columns by their embeddings."""
        if not slots or self.card_embedding is None:
            return x
        ids = x[..., -slots:].float().round().long().clamp_(0, self.card_vocab - 1)
        embedded = self.card_embedding(ids).flatten(-2).to(x.dtype)
        return torch.cat((x[..., :-slots], embedded), dim=-1)

    @staticmethod
    def _batched_states(states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 1: states = states.unsqueeze(0)
        return states

    @staticmethod
    def _batched_actions(action_features: torch.Tensor, legal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if action_features.ndim == 2: action_features = action_features.unsqueeze(0)
        if legal_mask.ndim == 1: legal_mask = legal_mask.unsqueeze(0)
        return action_features, legal_mask

    def _policy_from_state_hidden(self, state_hidden: torch.Tensor, action_features: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        action_hidden = self.action_encoder(self._embed_tail(action_features, self.action_card_slots))
        expanded = state_hidden.unsqueeze(1).expand(-1, action_features.shape[1], -1)
        logits = self.policy_head(torch.cat((expanded, action_hidden), dim=-1)).squeeze(-1)
        # ``-1e9`` overflows under CUDA autocast float16.  The finite minimum
        # for the active dtype is sufficient for masked softmax and keeps the
        # real mixed-precision training path valid.
        return logits.masked_fill(~legal_mask.bool(), torch.finfo(logits.dtype).min)

    def forward(self, states: torch.Tensor, action_features: torch.Tensor, legal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        states = self._batched_states(states)
        action_features, legal_mask = self._batched_actions(action_features, legal_mask)
        state_hidden = self.state_encoder(self._embed_tail(states, self.state_card_slots))
        return self._policy_from_state_hidden(state_hidden, action_features, legal_mask), self.value_head(state_hidden).squeeze(-1)

    def save_checkpoint(self, path: str | Path, **metadata: object) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "architecture": self.architecture, "metadata": metadata}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> "KARDSNet":
        payload = torch.load(path, map_location=device, weights_only=False)
        architecture = dict(payload.get("architecture", {}))
        architecture.setdefault("layout_version", 2 if "state_card_slots" in architecture else 1)
        model = cls(**architecture); model.load_state_dict(payload["state_dict"]); model.to(device); return model
