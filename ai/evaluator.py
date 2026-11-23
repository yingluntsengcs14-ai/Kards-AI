"""Match-based evaluation metrics for AI agents."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from itertools import combinations
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
    # Paired meta-deck evaluation only: {deck name: {games, wins, win_rate}}
    # from the candidate's side, and which protocol produced the numbers.
    per_deck: dict = field(default_factory=dict)
    protocol: str = "deck_pool"

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

    def evaluate_meta(self, candidate: BaseAgent, opponent: BaseAgent, meta_decks: list, games_per_pair: int = 2) -> EvaluationReport:
        """Paired ("duplicate") evaluation on the tournament meta decks.

        2026-11-24: random deck_pool decks are far weaker than the 17 meta
        decks self-play actually trains on, so they measured the wrong thing.
        Every unordered pair of distinct decks (i, j) is played so that deck
        strength and first-move advantage cancel out:

        * games_per_pair=2: candidate plays i as p1 vs j, then j as p2 vs i.
        * games_per_pair=4: additionally j as p1 vs i, and i as p2 vs j.

        Both halves reuse the same seed and schedule, so shuffles mirror.
        """
        if games_per_pair not in (2, 4):
            raise ValueError("games_per_pair must be 2 or 4")
        pairs = list(combinations(meta_decks, 2))
        schedule = pairs + ([(b, a) for a, b in pairs] if games_per_pair == 4 else [])
        per_deck: dict[str, dict] = {deck.name: {"games": 0, "wins": 0} for deck in meta_decks}
        wins = losses = draws = 0; turns = hq = played = 0.0
        for candidate_seat in ("p1", "p2"):
            runner = SelfPlayRunner(self.cards, self.encoder, ReplayBuffer(), seed=self.seed)
            agents = (candidate, opponent) if candidate_seat == "p1" else (opponent, candidate)
            report = runner.run(len(schedule), *agents, deck_schedule=schedule)
            for record in runner.game_records:
                deck = record["decks"][0 if candidate_seat == "p1" else 1]
                per_deck[deck]["games"] += 1
                if record["winner"] == candidate_seat: wins += 1; per_deck[deck]["wins"] += 1
                elif record["winner"] == "draw": draws += 1
                else: losses += 1
            completed = report.p1_wins + report.p2_wins + report.draws
            turns += report.average_turns * completed; hq += report.average_hq_damage * completed; played += report.average_cards_played * completed
        games = max(1, wins + losses + draws)
        for item in per_deck.values():
            item["win_rate"] = item["wins"] / item["games"] if item["games"] else 0.0
        return EvaluationReport(wins + losses + draws, wins, losses, draws, wins / games, turns / games, hq / games, played / games,
                                per_deck=per_deck, protocol=f"meta_paired_{games_per_pair}")

    @staticmethod
    def save_report(report: EvaluationReport, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
