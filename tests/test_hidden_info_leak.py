# -*- coding: utf-8 -*-
"""隱藏資訊：決定化與洩漏測試。

`ai/mcts.py` 直接在真實 `GameState` 上搜尋，所以每個節點都讀得到對手手牌與
牌庫順序 —— 那是**完美資訊搜尋**，實力無法遷移到真實對局。

`ai/pimc.py` 不改搜尋本身，改餵給它的輸入：決定化出一個「與觀察者所見一致」
的世界，對每個世界跑一次原本的 MCTS，再跨世界投票。

這個檔案驗兩件事：
1. 決定化**不會改變**觀察者看得到的任何東西（可觀察投影不變）。
2. 決定化**確實重抽**了觀察者看不到的東西（否則等於沒做）。
3. 代理人的決策不依賴隱藏資訊 —— 兩個只差在隱藏資訊的狀態要選出同一個動作。
"""

from pathlib import Path
import random
import unittest

from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.testing import build_random_deck

try:
    from ai.observation import ObservationEncoder
    from ai.pimc import KNOWN_ENEMY_HAND, PIMCAgent, determinize, observable_signature

    HAS_TORCH = True
except ModuleNotFoundError:                 # pragma: no cover - env without torch
    HAS_TORCH = False

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"


def _started_game(cards: CardDatabase, seed: int = 11) -> Simulator:
    simulator = Simulator(cards=cards, record_replay=False)
    deck = build_random_deck(cards, seed=seed, main_nation="Germany")
    simulator.reset(list(deck), list(deck), nations=("Germany", "Germany"), seed=seed)
    return simulator


@unittest.skipUnless(HAS_TORCH, "AI 層需要 torch")
class DeterminizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)

    def test_observable_projection_is_unchanged(self) -> None:
        """觀察者看得到的一切都不能被改動 —— 這就是「沒有洩漏」的定義。"""
        state = _started_game(self.cards).state
        before = observable_signature(state, "p1")
        for seed in range(6):
            world = determinize(state, "p1", self.cards, random.Random(seed))
            self.assertEqual(observable_signature(world, "p1"), before, "seed={0}".format(seed))

    def test_hidden_information_is_actually_resampled(self) -> None:
        """對手手牌必須真的被重抽，否則決定化是空操作。"""
        state = _started_game(self.cards).state
        worlds = [determinize(state, "p1", self.cards, random.Random(seed)) for seed in range(8)]
        hands = {tuple(world.players["p2"].hand) for world in worlds}
        self.assertGreater(len(hands), 1, "八個世界的對手手牌完全一樣")

    def test_the_original_state_is_not_mutated(self) -> None:
        state = _started_game(self.cards).state
        enemy_hand = list(state.players["p2"].hand)
        own_deck = list(state.players["p1"].deck)
        determinize(state, "p1", self.cards, random.Random(3))
        self.assertEqual(state.players["p2"].hand, enemy_hand)
        self.assertEqual(state.players["p1"].deck, own_deck)

    def test_own_deck_order_is_resampled_but_contents_are_not(self) -> None:
        """自己的牌庫內容是已知的，順序不是。"""
        state = _started_game(self.cards).state
        world = determinize(state, "p1", self.cards, random.Random(5))
        self.assertEqual(sorted(world.players["p1"].deck), sorted(state.players["p1"].deck))

    def test_intel_revealed_cards_stay_in_the_enemy_hand(self) -> None:
        """Intel 揭露過的敵方手牌是已知資訊，不該被洗掉。"""
        state = _started_game(self.cards).state
        revealed = state.players["p2"].hand[0]
        state.players["p1"].status[KNOWN_ENEMY_HAND] = [revealed]
        for seed in range(6):
            world = determinize(state, "p1", self.cards, random.Random(seed))
            self.assertIn(revealed, world.players["p2"].hand, "seed={0}".format(seed))

    def test_enemy_zone_sizes_are_preserved(self) -> None:
        state = _started_game(self.cards).state
        world = determinize(state, "p1", self.cards, random.Random(7))
        self.assertEqual(len(world.players["p2"].hand), len(state.players["p2"].hand))
        self.assertEqual(len(world.players["p2"].deck), len(state.players["p2"].deck))

    def test_enemy_card_multiset_is_preserved(self) -> None:
        """重抽只能重新分配，不能無中生有或弄丟卡。"""
        state = _started_game(self.cards).state
        world = determinize(state, "p1", self.cards, random.Random(9))
        before = sorted(state.players["p2"].hand + state.players["p2"].deck)
        after = sorted(world.players["p2"].hand + world.players["p2"].deck)
        self.assertEqual(after, before)

    def test_face_down_countermeasure_keeps_its_cost(self) -> None:
        """面朝下的反制卡可以換成別張，但花費必須一致（花掉的指揮點是公開的）。"""
        cards = self.cards
        countermeasures = [card for card in cards if card.type == "countermeasure"]
        by_cost: dict[int, list[str]] = {}
        for card in countermeasures:
            by_cost.setdefault(card.kredits, []).append(card.id)
        cost, ids = next((cost, ids) for cost, ids in sorted(by_cost.items()) if len(ids) >= 3)
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Germany", deck=["garrison"] * 5, hand=["garrison"],
                                  resources=ResourceState(5, 5)),
                "p2": PlayerState("p2", "Germany", deck=list(ids[1:]), hand=[ids[0]],
                                  resources=ResourceState(5, 5),
                                  active_countermeasures=[{"card_id": ids[0], "activated_turn": 1}]),
            },
            graveyard={"p1": [], "p2": []}, rng_seed=4,
        )
        seen = set()
        for seed in range(10):
            world = determinize(state, "p1", cards, random.Random(seed))
            entry = world.players["p2"].active_countermeasures[0]
            seen.add(entry["card_id"])
            self.assertEqual(cards.get(entry["card_id"]).kredits, cost)
            self.assertEqual(cards.get(entry["card_id"]).type, "countermeasure")
        self.assertGreater(len(seen), 1, "反制卡的身分從未被重抽")


@unittest.skipUnless(HAS_TORCH, "AI 層需要 torch")
class PIMCAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)
        cls.encoder = ObservationEncoder(cls.cards)

    def _agent(self, seed: int = 0, worlds: int = 4) -> "PIMCAgent":
        return PIMCAgent(None, self.encoder, worlds=worlds, simulations=8, seed=seed)

    @staticmethod
    def _reshuffle_opponent(state, observer: str, seed: int):
        """重新切分對手的手牌／牌庫（聯集不變），其餘完全不動。"""
        variant = state.clone()
        opponent = next(pid for pid in variant.players if pid != observer)
        pool = list(variant.players[opponent].hand) + list(variant.players[opponent].deck)
        random.Random(seed).shuffle(pool)
        hand_size = len(variant.players[opponent].hand)
        variant.players[opponent].hand = pool[:hand_size]
        variant.players[opponent].deck = pool[hand_size:]
        return variant, opponent

    def _rich_position(self, seed: int = 11):
        """推進到有多個合法動作的盤面。"""
        simulator = _started_game(self.cards, seed)
        from simulator.actions.action import PassAction
        actions = simulator.get_available_actions()
        for _ in range(12):
            if len(actions) > 3:
                break
            simulator.step(PassAction(simulator.state.current_player))
            actions = simulator.get_available_actions()
        return simulator, actions

    def test_agent_picks_a_legal_action(self) -> None:
        simulator, actions = self._rich_position()
        agent = self._agent()
        agent.set_state(simulator.state, simulator.state.current_player, self.cards)
        chosen = agent.select_action({}, actions)
        self.assertIn(chosen, actions)

    def test_policy_is_a_distribution_over_the_legal_actions(self) -> None:
        """`last_policy` 會被 self-play 當訓練目標，必須對齊合法動作並且總和為 1。"""
        simulator, actions = self._rich_position()
        agent = self._agent()
        agent.set_state(simulator.state, simulator.state.current_player, self.cards)
        agent.select_action({}, actions)
        self.assertEqual(len(agent.last_policy), len(actions))
        self.assertAlmostEqual(sum(agent.last_policy), 1.0, places=5)

    def test_every_world_was_usable(self) -> None:
        """決定化出來的世界的合法動作應與觀察者一致，不該被丟棄。"""
        simulator, actions = self._rich_position()
        agent = self._agent(worlds=4)
        agent.set_state(simulator.state, simulator.state.current_player, self.cards)
        agent.select_action({}, actions)
        self.assertEqual(agent.stats.skipped_worlds, 0, "有世界的合法動作對不上")
        self.assertEqual(agent.stats.worlds, 4)

    def test_decision_does_not_depend_on_hidden_information(self) -> None:
        """核心洩漏測試：只改動隱藏資訊，決策必須不變。

        做法是把**對手**的手牌與牌庫重新切分（聯集相同）—— 觀察者看不出差別，
        所以決策也不該有差別。注意一定要動「對手」而不是行動方自己：行動方的
        手牌是公開資訊，改了它決策本來就該變。

        完美資訊搜尋在同樣的比較下會分歧 —— 用
        `tools/measure_hidden_info_leak.py` 可以重現那個數字（需要有網路，
        因為 model=None 時 MCTS 的 value 恆為 0，對任何輸入都不敏感）。
        """
        checked = 0
        for seed in (11, 13, 17, 19, 23, 29, 31, 37):
            simulator, actions = self._rich_position(seed)
            if len(actions) <= 3:
                continue
            observer = simulator.state.current_player
            variant, opponent = self._reshuffle_opponent(simulator.state, observer, seed * 7)
            if simulator.state.players[opponent].hand == variant.players[opponent].hand:
                continue
            checked += 1

            first = self._agent(seed=5)
            first.set_state(simulator.state, observer, self.cards)
            choice_a = first.select_action({}, actions)

            second = self._agent(seed=5)
            second.set_state(variant, observer, self.cards)
            choice_b = second.select_action({}, actions)

            self.assertEqual(choice_a, choice_b, "seed={0} 的決策受隱藏資訊影響".format(seed))
            self.assertEqual(first.last_policy, second.last_policy, "seed={0}".format(seed))
        self.assertGreater(checked, 3, "沒有檢查到足夠的盤面")

    def test_sampling_depends_only_on_the_belief_not_the_true_split(self) -> None:
        """同一個 seed、只差隱藏切分的兩個狀態，必須抽出同一個世界。

        這條擋住一個很細的洩漏：`random.shuffle` 的結果取決於輸入順序，而輸入
        順序就是隱藏的真相。所以抽樣前要先把牌排序正規化。
        """
        for seed in (11, 17, 23):
            simulator, actions = self._rich_position(seed)
            observer = simulator.state.current_player
            variant, opponent = self._reshuffle_opponent(simulator.state, observer, seed * 7)
            world_a = determinize(simulator.state, observer, self.cards, random.Random(1))
            world_b = determinize(variant, observer, self.cards, random.Random(1))
            self.assertEqual(world_a.to_dict(), world_b.to_dict(), "seed={0}".format(seed))

    def test_a_single_legal_action_skips_the_search(self) -> None:
        simulator = _started_game(self.cards)
        actions = simulator.get_available_actions()
        if len(actions) != 1:
            self.skipTest("開局盤面不是只有一個合法動作")
        agent = self._agent()
        agent.set_state(simulator.state, simulator.state.current_player, self.cards)
        self.assertEqual(agent.select_action({}, actions), actions[0])
        self.assertEqual(agent.stats.worlds, 0)


if __name__ == "__main__":
    unittest.main()
