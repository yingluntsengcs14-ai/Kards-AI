# -*- coding: utf-8 -*-
"""天氣機制測試（大洋洲風暴）。

涵蓋：Forecast 產生預報 → 回合開始解算成天氣卡 → 打出天氣卡被記錄成
「本回合打過晴／雨／風暴」→ 回合結束清除。

預報轉成哪一張天氣卡（哪個費用階層、幾回合後）官方 API 沒寫，目前是
weather.pick_weather_card 的暫定規則，相關測試只驗「類型正確、確實進手牌」，
不把暫定的階層選擇當成正確規則釘死。
"""

from pathlib import Path
import unittest

from simulator.actions.action import PlayCardAction
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.core.game import Simulator
from simulator.core.turn import TurnManager
from simulator.effects.resolver import EffectContext
from simulator.rules import weather
from simulator.rules.native import NativeRuleEngine

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"


class WeatherTablesTests(unittest.TestCase):
    """卡牌對照表要和官方卡面一致。"""

    def test_three_forecast_tokens_map_to_three_categories(self) -> None:
        self.assertEqual(weather.FORECAST_TOKENS,
                         {"blue_sky": "sunny", "mist": "rain", "gale": "storm"})

    def test_nine_weather_cards_in_three_tiers(self) -> None:
        self.assertEqual(len(weather.WEATHER_CARDS), 9)
        for category, tiers in weather.WEATHER_TIERS.items():
            self.assertEqual(len(tiers), 3, category)
            for card_id in tiers:
                self.assertEqual(weather.category_of(card_id), category)

    def test_tiers_are_ordered_cheapest_first(self) -> None:
        cards = CardDatabase.from_file(CATALOG)
        for category, tiers in weather.WEATHER_TIERS.items():
            costs = [cards.get(card_id).kredits for card_id in tiers]
            self.assertEqual(costs, [2, 4, 6], category)

    def test_forecast_token_is_not_a_weather_card(self) -> None:
        self.assertTrue(weather.is_forecast_token("mist"))
        self.assertFalse(weather.is_weather_card("mist"))
        self.assertTrue(weather.is_weather_card("deluge"))


class WeatherStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)
        cls.engine = NativeRuleEngine(cls.cards)

    def _state(self, p1_hand=(), p1_units=(), kredits=6) -> GameState:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "USA", deck=["garrison"] * 10, hand=list(p1_hand),
                                  resources=ResourceState(kredits, kredits),
                                  units=[UnitState(uid, cid, atk, dfn, "p1", "support_line")
                                         for uid, cid, atk, dfn in p1_units]),
                "p2": PlayerState("p2", "Japan", deck=["garrison"] * 10,
                                  resources=ResourceState(kredits, kredits)),
            },
            graveyard={"p1": [], "p2": []},
        )
        for unit in state.players["p1"].units:
            state.battlefield["support_line"].append(unit.instance_id)
        return state

    # --- Forecast ---

    def test_forecast_records_pending_forecast(self) -> None:
        """Forecast 會留下一個待解算的預報。"""
        state = self._state()
        self.assertIsNone(weather.pending_forecast(state, "p1"))
        weather.forecast(state, "p1", "gale")
        self.assertEqual(weather.pending_forecast(state, "p1"), "gale")

    def test_second_forecast_replaces_the_first(self) -> None:
        """一位玩家同時只會有一個預報。"""
        state = self._state()
        weather.forecast(state, "p1", "gale")
        weather.forecast(state, "p1", "blue_sky")
        self.assertEqual(weather.pending_forecast(state, "p1"), "blue_sky")

    def test_forecast_is_per_player(self) -> None:
        """我的預報不會變成對手的預報。"""
        state = self._state()
        weather.forecast(state, "p1", "mist")
        self.assertIsNone(weather.pending_forecast(state, "p2"))

    def test_unknown_forecast_token_is_rejected(self) -> None:
        state = self._state()
        with self.assertRaises(ValueError):
            weather.forecast(state, "p1", "sunshower")

    # --- 預報解算 ---

    def test_pending_forecast_resolves_into_a_card_of_that_category(self) -> None:
        """預報在回合開始解算成同類型的天氣卡並進手牌。"""
        state = self._state()
        weather.forecast(state, "p1", "mist")
        card_id = weather.resolve_pending_forecast(state, "p1")
        self.assertEqual(weather.category_of(card_id), "rain")
        self.assertIn(card_id, state.players["p1"].hand)
        self.assertIsNone(weather.pending_forecast(state, "p1"))

    def test_turn_start_resolves_the_forecast(self) -> None:
        """TurnManager.start_turn 會解算預報。"""
        state = self._state()
        weather.forecast(state, "p1", "blue_sky")
        TurnManager.start_turn(state, "p1", draw_card=False)
        self.assertTrue(any(weather.category_of(card_id) == "sunny"
                            for card_id in state.players["p1"].hand),
                        "回合開始沒有把預報解算成晴天卡")

    def test_resolution_is_a_no_op_without_a_forecast(self) -> None:
        state = self._state()
        self.assertIsNone(weather.resolve_pending_forecast(state, "p1"))
        self.assertEqual(state.players["p1"].hand, [])

    def test_full_hand_loses_the_weather_card(self) -> None:
        """手牌已滿 9 張時，解算出的天氣卡進墳場而不是變第 10 張。"""
        state = self._state(p1_hand=tuple(["garrison"] * 9))
        weather.forecast(state, "p1", "gale")
        card_id = weather.resolve_pending_forecast(state, "p1")
        self.assertEqual(len(state.players["p1"].hand), 9)
        self.assertIn(card_id, state.graveyard["p1"])

    # --- 本回合打過的天氣 ---

    def test_playing_a_weather_card_records_its_category(self) -> None:
        """打出雷暴（風暴卡）會被記成本回合打過風暴。"""
        state = self._state(p1_hand=("thunderstorm",))
        PlayCardAction("p1", "thunderstorm").execute(state, self.cards)
        self.assertEqual(weather.played_this_turn(state, "p1"), ("storm",))
        self.assertTrue(weather.was_played_this_turn(state, "p1", "storm"))
        self.assertFalse(weather.was_played_this_turn(state, "p1", "sunny", "rain"))

    def test_playing_a_non_weather_order_records_nothing(self) -> None:
        state = self._state(p1_hand=("radar",))
        PlayCardAction("p1", "radar").execute(state, self.cards)
        self.assertEqual(weather.played_this_turn(state, "p1"), ())

    def test_weather_record_is_per_player(self) -> None:
        state = self._state(p1_hand=("thunderstorm",))
        PlayCardAction("p1", "thunderstorm").execute(state, self.cards)
        self.assertEqual(weather.played_this_turn(state, "p2"), ())

    def test_weather_played_event_is_emitted(self) -> None:
        """打出天氣卡要發出 on_weather_played 事件（給「打出雨卡時…」這類卡聽）。"""
        state = self._state(p1_hand=("deluge",))
        PlayCardAction("p1", "deluge").execute(state, self.cards)
        self.assertTrue(any(entry.get("event") == "weather_played" and entry.get("weather") == "rain"
                            for entry in state.event_log),
                        "沒有記錄 weather_played")
        self.assertTrue(any(entry.get("event") == "on_weather_played" for entry in state.event_log),
                        "沒有廣播 on_weather_played")

    def test_turn_end_clears_the_weather_record(self) -> None:
        """「本回合打過雨」只問當前回合，回合結束要清掉。"""
        state = self._state(p1_hand=("deluge",))
        PlayCardAction("p1", "deluge").execute(state, self.cards)
        TurnManager.end_turn(state, "p1")
        self.assertEqual(weather.played_this_turn(state, "p1"), ())

    # --- 真實卡牌 ---

    def test_overcast_forecasts_when_played(self) -> None:
        """陰雲密佈（Forecast. 全部總部 +2 防禦…）打出後要留下預報。"""
        # 卡面的「全部總部 +2 防禦」被解析成需要指定目標，所以要給一個單位。
        state = self._state(p1_hand=("overcast",), p1_units=(("u1", "garrison", 0, 3),), kredits=6)
        PlayCardAction("p1", "overcast", target_unit_id="u1").execute(state, self.cards)
        self.assertIsNotNone(weather.pending_forecast(state, "p1"),
                             "陰雲密佈沒有產生預報")

    def test_deployment_forecast_unit_forecasts_on_deploy(self) -> None:
        """橫須賀 E14Y（部署：Forecast）部署時要留下預報。"""
        state = self._state(p1_units=(("u1", "yokosuka_e14y", 3, 3),))
        self.engine.execute("yokosuka_e14y", "on_deploy", state,
                            EffectContext("p1", "yokosuka_e14y", "u1"))
        self.assertIsNotNone(weather.pending_forecast(state, "p1"))

    def test_forecast_emits_on_forecast_event(self) -> None:
        """Forecast 要廣播 on_forecast（給「當你 Forecast 時…」這類卡聽）。"""
        state = self._state(p1_units=(("u1", "yokosuka_e14y", 3, 3),))
        self.engine.execute("yokosuka_e14y", "on_deploy", state,
                            EffectContext("p1", "yokosuka_e14y", "u1"))
        self.assertTrue(any(entry.get("event") == "on_forecast" for entry in state.event_log))

    # --- 玩家選類型、階層隨機（2026-10-20 遊戲內確認） ---

    def test_forecast_uses_the_chosen_category(self) -> None:
        """Forecast 時玩家選的類型要被採用。"""
        state = self._state(p1_hand=("overcast",), p1_units=(("u1", "garrison", 0, 3),))
        PlayCardAction("p1", "overcast", "support_line", "u1", None, "rain").execute(state, self.cards)
        self.assertEqual(weather.pending_forecast(state, "p1"), "mist")

    def test_forecast_card_offers_one_action_per_category(self) -> None:
        """有 Forecast 的卡要產生三個合法動作，一個類型一個。"""
        sim = Simulator(cards=self.cards, record_replay=False)
        sim.state = self._state(p1_hand=("overcast",), p1_units=(("u1", "garrison", 0, 3),))
        options = {action.selected_option for action in sim.get_available_actions()
                   if isinstance(action, PlayCardAction) and action.card_id == "overcast"}
        self.assertEqual(options, set(weather.CATEGORIES))

    def test_tier_is_random_but_stays_in_the_chosen_category(self) -> None:
        """階層隨機，但一定是所選類型的三張之一。"""
        drawn = set()
        for seed in range(12):
            state = self._state()
            state.rng_seed = seed
            weather.forecast(state, "p1", "mist")
            drawn.add(weather.resolve_pending_forecast(state, "p1"))
        self.assertTrue(drawn <= set(weather.WEATHER_TIERS["rain"]),
                        "解算出了不屬於雨類型的卡：{0}".format(drawn))
        self.assertGreater(len(drawn), 1, "階層看起來沒有隨機")

    def test_weather_cards_are_not_deck_eligible(self) -> None:
        """天氣卡與預報卡都是 OnlySpawnable，不能放進牌組。"""
        for card_id in list(weather.WEATHER_CARDS) + list(weather.FORECAST_TOKENS):
            self.assertTrue(self.cards.get(card_id).is_token, card_id)


if __name__ == "__main__":
    unittest.main()
