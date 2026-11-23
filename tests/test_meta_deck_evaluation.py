# -*- coding: utf-8 -*-
"""2026-11-24: evaluation moved from randomly built deck_pool decks to paired
games over the 17 meta decks. Checks the pairing schedule, per-deck
attribution from the candidate's seat, and the anchor hand-over when the
evaluation protocol changes."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai import evaluator as evaluator_module  # noqa: E402
from ai.selfplay import SelfPlayReport  # noqa: E402
from auto_train import _checkpoint_step, switch_evaluation_protocol  # noqa: E402
from simulator.cards.deck_code import Deck  # noqa: E402

DECKS = [Deck(name, "Germany", None, ()) for name in ("A", "B", "C")]


def fake_runner(calls):
    def run(self_, episodes, p1, p2, deck_schedule=None, **kwargs):
        calls.append(([(a.name, b.name) for a, b in deck_schedule], p1))
        # p1 always wins, so a fair protocol must come out at exactly 50%.
        self_.game_records = [{"winner": "p1", "decks": [a.name, b.name]} for a, b in deck_schedule]
        n = len(deck_schedule)
        return SelfPlayReport(n, 0, n, 0, 0, 10.0, 5.0, 3.0)
    return run


class MetaDeckEvaluationTests(unittest.TestCase):
    def _evaluate(self, games_per_pair):
        calls = []
        candidate, opponent = object(), object()
        with mock.patch.object(evaluator_module.SelfPlayRunner, "run", fake_runner(calls)):
            report = evaluator_module.Evaluator(None, None, seed=1).evaluate_meta(candidate, opponent, DECKS, games_per_pair)
        return calls, report, candidate, opponent

    def test_two_per_pair_swaps_deck_and_seat_together(self) -> None:
        calls, report, candidate, opponent = self._evaluate(2)
        pairs = [("A", "B"), ("A", "C"), ("B", "C")]
        self.assertEqual(calls, [(pairs, candidate), (pairs, opponent)])
        self.assertEqual((report.games, report.wins, report.win_rate), (6, 3, 0.5))
        self.assertEqual(report.protocol, "meta_paired_2")
        # Every deck is played by the candidate once per opponent deck.
        self.assertEqual({k: v["games"] for k, v in report.per_deck.items()}, {"A": 2, "B": 2, "C": 2})

    def test_four_per_pair_covers_both_decks_from_both_seats(self) -> None:
        calls, report, candidate, _ = self._evaluate(4)
        schedule = calls[0][0]
        self.assertEqual(len(schedule), 6)
        self.assertEqual(set(schedule), {("A", "B"), ("A", "C"), ("B", "C"), ("B", "A"), ("C", "A"), ("C", "B")})
        self.assertEqual((report.games, report.win_rate), (12, 0.5))
        # Seat bias cancels per deck too: p1 always wins, each deck is 50%.
        self.assertTrue(all(v["games"] == 4 and v["win_rate"] == 0.5 for v in report.per_deck.values()))

    def test_protocol_switch_sets_old_anchor_aside(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            (runs / "best_external_metrics.json").write_text(json.dumps({"score": 0.8}), encoding="utf-8")
            (runs / "best_external.pt").write_bytes(b"old")
            self.assertEqual(switch_evaluation_protocol(runs, "meta_paired_2"), "deck_pool")
            self.assertFalse((runs / "best_external_metrics.json").exists())
            self.assertFalse((runs / "best_external.pt").exists())
            self.assertEqual((runs / "best_external_deck_pool.pt").read_bytes(), b"old")
            self.assertTrue((runs / "best_external_metrics_deck_pool.json").exists())
            # Same protocol again: nothing moves.
            (runs / "best_external_metrics.json").write_text(json.dumps({"protocol": "meta_paired_2"}), encoding="utf-8")
            self.assertIsNone(switch_evaluation_protocol(runs, "meta_paired_2"))
            self.assertTrue((runs / "best_external_metrics.json").exists())

    def test_checkpoint_rotation_orders_by_step_not_name(self) -> None:
        names = [Path(f"checkpoint_step_{n}.pt") for n in (26000, 5000, 9500, 12000)]
        self.assertEqual([p.name for p in sorted(names, key=_checkpoint_step)][:-2],
                         ["checkpoint_step_5000.pt", "checkpoint_step_9500.pt"])


if __name__ == "__main__":
    unittest.main()
