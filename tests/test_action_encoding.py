# -*- coding: utf-8 -*-
"""動作特徵編碼：玩家的選擇必須被網路看見。

`ActionEncoder.encode` 原本只編碼動作的前五個欄位，而 `PlayCardAction` 有七個：

    player_id, card_id, position, target_unit_id, selected_card_id,
                                                  selected_option, selected_card_zone
                                                  ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^ 被丟掉

所以**只差在選項的動作，特徵向量完全相同**。MCTS 樹還能區分（子節點用動作索引），
但網路的 policy prior 與訓練目標分不出來，等於「N 選一」的決策完全學不到：
Forecast 的晴／雨／風暴（10 張卡）、SHOCK TACTICS 的 blitz／shock、Develop 的選項。
"""

from pathlib import Path
import unittest

from simulator.actions.action import AttackAction, PassAction, PlayCardAction
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.testing import build_random_deck

# The AI layer needs torch; the rules layer does not.  Skip instead of failing
# to import, so running the suite without torch still gives a clean signal.
try:
    import torch

    from ai.action_encoder import ACTION_FEATURE_DIM, ActionEncoder, action_key

    HAS_TORCH = True
except ModuleNotFoundError:                 # pragma: no cover - env without torch
    HAS_TORCH = False
    ACTION_FEATURE_DIM = 0

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"


@unittest.skipUnless(HAS_TORCH, "AI 層需要 torch")
class ActionOptionEncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = ActionEncoder()

    def _forecast_actions(self):
        return [PlayCardAction("p1", "overcast", "support_line", None, None, option)
                for option in ("sunny", "rain", "storm")]

    def test_forecast_choices_encode_differently(self) -> None:
        """晴／雨／風暴三選一必須產生三個不同的特徵向量。"""
        vectors = [self.encoder.encode(action) for action in self._forecast_actions()]
        for first in range(len(vectors)):
            for second in range(first + 1, len(vectors)):
                self.assertFalse(torch.equal(vectors[first], vectors[second]),
                                 "第 {0} 與第 {1} 個 Forecast 動作編碼相同".format(first, second))

    def test_weather_choice_has_a_dedicated_one_hot(self) -> None:
        """三種天氣各有專屬維度，而且互斥 —— 不是只靠雜湊值區分。"""
        onehots = [self.encoder.encode(action)[22:25] for action in self._forecast_actions()]
        for index, onehot in enumerate(onehots):
            self.assertEqual(onehot.sum().item(), 1.0)
            self.assertEqual(onehot[index].item(), 1.0)

    def test_keyword_choice_is_encoded(self) -> None:
        """SHOCK TACTICS 的 blitz／shock（上游既有的同類問題）。"""
        blitz = self.encoder.encode(PlayCardAction("p1", "shock_tactics", "support_line", "u1", None, "blitz"))
        shock = self.encoder.encode(PlayCardAction("p1", "shock_tactics", "support_line", "u1", None, "shock"))
        self.assertFalse(torch.equal(blitz, shock))
        self.assertEqual(blitz[25].item(), 1.0)
        self.assertEqual(shock[26].item(), 1.0)

    def test_selected_card_zone_is_encoded(self) -> None:
        """選中的卡來自手牌或場上，也是被丟掉的欄位之一。"""
        from_hand = self.encoder.encode(
            PlayCardAction("p1", "out_with_the_old", "support_line", None, "yak_9", None, "hand"))
        from_board = self.encoder.encode(
            PlayCardAction("p1", "out_with_the_old", "support_line", None, "yak_9", None, "battlefield"))
        self.assertFalse(torch.equal(from_hand, from_board))
        self.assertEqual(from_hand[27].item(), 1.0)
        self.assertEqual(from_board[27].item(), 0.0)

    def test_an_unknown_option_still_differs_by_identity(self) -> None:
        """不在固定詞彙裡的選項（Develop 的卡名）靠身分雜湊區分。"""
        first = self.encoder.encode(PlayCardAction("p1", "develop_card", "support_line", None, None, "radar"))
        second = self.encoder.encode(PlayCardAction("p1", "develop_card", "support_line", None, None, "garrison"))
        self.assertFalse(torch.equal(first, second))

    def test_every_action_field_has_an_identity_slot(self) -> None:
        """身分槽的數量必須夠放最寬動作的所有欄位。"""
        from dataclasses import fields
        self.assertGreaterEqual(7, len(fields(PlayCardAction)))
        self.assertEqual(len(action_key(PlayCardAction("p1", "radar"))) - 1, len(fields(PlayCardAction)))

    def test_unknown_action_type_does_not_collide_with_an_identity_slot(self) -> None:
        """未知的動作型別有自己的維度（舊版寫到 index 15，和第五個身分槽相撞）。"""
        class WeirdAction(PassAction):
            pass
        vector = self.encoder.encode(WeirdAction("p1"))
        self.assertEqual(vector[6].item(), 1.0)

    def test_encoding_is_stable(self) -> None:
        action = PlayCardAction("p1", "overcast", "support_line", None, None, "rain")
        self.assertTrue(torch.equal(self.encoder.encode(action), self.encoder.encode(action)))


@unittest.skipUnless(HAS_TORCH, "AI 層需要 torch")
class LegalActionEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)

    def test_width_matches_the_declared_dim(self) -> None:
        encoder = ActionEncoder()
        features, mask = encoder.encode_legal_actions(
            [PassAction("p1"), AttackAction("p1", "u1")], pad_to_max=False)
        self.assertEqual(features.shape, (2, ACTION_FEATURE_DIM))
        self.assertTrue(bool(mask.all()))

    def test_distinct_legal_actions_get_distinct_features(self) -> None:
        """真實盤面上，任兩個不同的合法動作不該編碼成同一個向量。

        這是 selected_option 被丟掉時最直接的後果：同一張卡的 N 個選項會變成
        N 個一模一樣的候選。
        """
        simulator = Simulator(cards=self.cards, record_replay=False)
        deck = build_random_deck(self.cards, seed=7, main_nation="Germany")
        simulator.reset(list(deck), list(deck), nations=("Germany", "Germany"), seed=7)
        # The opening turn only affords Pass; advance until the board offers a
        # real choice.
        from simulator.actions.action import PassAction
        actions = simulator.get_available_actions()
        for _ in range(12):
            if len(actions) > 3:
                break
            simulator.step(PassAction(simulator.state.current_player))
            actions = simulator.get_available_actions()
        encoder = ActionEncoder()
        self.assertGreater(len(actions), 3, "測試盤面沒有產生足夠的合法動作")
        seen: dict[tuple, object] = {}
        for action in actions:
            signature = tuple(round(value, 6) for value in encoder.encode(action).tolist())
            previous = seen.get(signature)
            self.assertIsNone(previous,
                              "這兩個動作編碼相同：{0} / {1}".format(previous, action))
            seen[signature] = action


if __name__ == "__main__":
    unittest.main()
