"""Match-based evaluation metrics for AI agents."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from ai.agents import BaseAgent
from ai.observation import ObservationEncoder
from ai.replay_buffer import ReplayBuffer
from ai.selfplay import SelfPlayRunner
from simulator.cards.loader import CardDatabase

@dataclass(frozen=True)
class EvaluationReport:
    games: int; wins: int; losses: int; draws: int; win_rate: float; average_turns: float = 0.0; hq_damage: float = 0.0; cards_played: float = 0.0

class Evaluator:
    def __init__(self, cards: CardDatabase, encoder: ObservationEncoder, seed: int = 0) -> None: self.cards, self.encoder, self.seed = cards, encoder, seed
    def evaluate(self, candidate: BaseAgent, opponent: BaseAgent, games: int = 20,
                 *, deck_pool: list[tuple[str, str | None]] | None = None) -> EvaluationReport:
        """Evaluate on the supplied legal deck distribution, with a fixed seed.

        ``deck_pool`` must match the formal self-play distribution when used
        for Champion promotion.  Falling back to the legacy default is kept
        only for callers that do not provide a training configuration.
        """
        # Seat-balanced (2026-11-23): the candidate used to always be p1, and
        # p1 wins ~58% of self-play games, so a model "beat" an equal copy of
        # itself. Half the games from each seat, same seed for both halves so
        # deck draws and shuffles are mirrored.
        first, second = (games + 1) // 2, games // 2
        a = SelfPlayRunner(self.cards, self.encoder, ReplayBuffer(), seed=self.seed).run(first, candidate, opponent, deck_pool=deck_pool)
        wins, losses, draws = a.p1_wins, a.p2_wins, a.draws
        turns, hq, played = a.average_turns * first, a.average_hq_damage * first, a.average_cards_played * first
        if second:
            b = SelfPlayRunner(self.cards, self.encoder, ReplayBuffer(), seed=self.seed).run(second, opponent, candidate, deck_pool=deck_pool)
            wins, losses, draws = wins + b.p2_wins, losses + b.p1_wins, draws + b.draws
            turns, hq, played = turns + b.average_turns * second, hq + b.average_hq_damage * second, played + b.average_cards_played * second
        return EvaluationReport(games, wins, losses, draws, wins / games, turns / games, hq / games, played / games)

    @staticmethod
    def save_report(report: EvaluationReport, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
