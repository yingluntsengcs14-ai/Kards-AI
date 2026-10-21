# -*- coding: utf-8 -*-
"""天氣條件測試：讀寫天氣狀態的卡只能在正確的事件觸發。

這些卡原本全部落在 condition.py 的「沒建模的條件一律視為成立」預設行為上，
所以一部署就生效。這個檔案把正確行為釘住：部署當下不該生效，只有在
Forecast／打出對應天氣卡時才生效。
"""

from pathlib import Path
import unittest

from simulator.actions.action import PlayCardAction
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects.resolver import EffectContext
from simulator.rules import weather
from simulator.rules.condition import evaluate as condition_evaluate
from simulator.rules.native import NativeRuleEngine

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data/source/kards_info_cards.json"


class _Context:
    """condition.evaluate 只用到 player_id 與 metadata。"""

    def __init__(self, player_id="p1", **metadata):
        self.player_id = player_id
        self.metadata = metadata


class WeatherConditionParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)
        cls.engine = NativeRuleEngine(cls.cards)

    def test_weather_cards_get_a_weather_trigger(self) -> None:
        """讀天氣的卡要掛上 on_forecast / on_weather_played 觸發器。"""
        expected = {
            "22nd_pioneers": "on_forecast",
            "kawanishi_h8k": "on_forecast",
            "162nd_regiment": "on_weather_played",
            "79th_infantry_regiment": "on_weather_played",
            "16th_infantry_brigade": "on_weather_played",
            "16th_infantry_brigade_vet": "on_weather_played",
            "6th_marine_regiment": "on_weather_played",
        }
        for card_id, trigger in expected.items():
            self.assertIn(trigger, self.engine.rule_for(card_id).triggers, card_id)

    def test_order_keeps_on_play_alongside_a_weather_trigger(self) -> None:
        """指令卡的 on_play 不能被天氣觸發器擠掉，否則主效果永遠不執行。

        用合成卡面測規則本身，而不是綁定某一張卡 —— 「指令卡沒有其他觸發器才補
        on_play」這個預設值會被新加的天氣觸發器蓋掉，是通用的回歸風險。
        """
        from dataclasses import replace
        from simulator.rules.parser import RuleParser
        synthetic = replace(self.cards.get("radar"),
                            text="Costs 2 less if Rain was played this turn. Destroy a random enemy unit.")
        rule = RuleParser().parse(synthetic)
        self.assertIn("on_play", rule.triggers)
        self.assertIn("on_weather_played", rule.triggers)
        self.assertTrue(any(action.kind == "destroy" for action in rule.actions))

    def test_weather_conditions_are_kept_not_dropped(self) -> None:
        """「after you play a Rain card」這種尾句不能被丟掉。"""
        rule = self.engine.rule_for("16th_infantry_brigade_vet")
        self.assertTrue(all(action.condition for action in rule.actions),
                        "條件被丟掉了，效果會在部署時就生效")


class WeatherConditionEvaluationTests(unittest.TestCase):
    """condition.evaluate 對三種天氣句型的判斷。"""

    def test_forecast_condition_requires_the_forecast_event(self) -> None:
        context = _Context(event_player_id="p1")
        self.assertTrue(condition_evaluate("when you Forecast.", "on_forecast", context, None))
        self.assertFalse(condition_evaluate("when you Forecast.", "on_deploy", context, None))

    def test_forecast_condition_ignores_the_opponents_forecast(self) -> None:
        context = _Context(player_id="p1", event_player_id="p2")
        self.assertFalse(condition_evaluate("when you Forecast.", "on_forecast", context, None))

    def test_weather_play_condition_matches_only_the_named_categories(self) -> None:
        clause = "after you play a Rain or a Storm card."
        for played, expected in (("rain", True), ("storm", True), ("sunny", False)):
            context = _Context(event_player_id="p1", weather=played)
            self.assertEqual(
                condition_evaluate(clause, "on_weather_played", context, None), expected, played)

    def test_weather_play_condition_requires_the_weather_event(self) -> None:
        context = _Context(event_player_id="p1", weather="rain")
        self.assertFalse(condition_evaluate(
            "when you play a Rain card.", "on_deploy", context, None))

    def test_per_turn_condition_reads_the_turn_record(self) -> None:
        """「本回合打過雨」要查本回合記錄，不是無條件成立。"""
        state = GameState(
            current_player="p1",
            players={"p1": PlayerState("p1", "USA"), "p2": PlayerState("p2", "Japan")},
            graveyard={"p1": [], "p2": []},
        )
        context = _Context()
        clause = "if Rain was played this turn."
        self.assertFalse(condition_evaluate(clause, "on_play", context, None, state))
        weather.note_weather_played(state, "p1", "deluge")
        self.assertTrue(condition_evaluate(clause, "on_play", context, None, state))

    def test_per_turn_condition_distinguishes_categories(self) -> None:
        state = GameState(
            current_player="p1",
            players={"p1": PlayerState("p1", "USA"), "p2": PlayerState("p2", "Japan")},
            graveyard={"p1": [], "p2": []},
        )
        weather.note_weather_played(state, "p1", "heatwave")  # Sunny
        context = _Context()
        self.assertTrue(condition_evaluate(
            "if Sunny or Rain was played this turn.", "on_play", context, None, state))
        self.assertFalse(condition_evaluate(
            "if Rain was played this turn.", "on_play", context, None, state))

    def test_non_weather_play_conditions_are_untouched(self) -> None:
        """非天氣的「when you play a X card」不能被天氣分支攔走。"""
        self.assertEqual((), __import__(
            "simulator.rules.condition", fromlist=["_weather_categories"]
        )._weather_categories("navy"))


class WeatherCardBehaviourTests(unittest.TestCase):
    """整局行為：部署不該生效，打出天氣卡才生效。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(CATALOG)
        cls.engine = NativeRuleEngine(cls.cards)

    def _state(self, p1_units=(), p1_hand=()) -> GameState:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Anzac", deck=["garrison"] * 10, hand=list(p1_hand),
                                  resources=ResourceState(12, 12),
                                  units=[UnitState(uid, cid, atk, dfn, "p1", "support_line")
                                         for uid, cid, atk, dfn in p1_units]),
                "p2": PlayerState("p2", "Japan", deck=["garrison"] * 10,
                                  resources=ResourceState(12, 12)),
            },
            graveyard={"p1": [], "p2": []},
        )
        for unit in state.players["p1"].units:
            state.battlefield["support_line"].append(unit.instance_id)
        return state

    def test_16th_brigade_does_not_become_veteran_on_deploy(self) -> None:
        """16 步兵旅（打出雨或風暴卡後成為老兵）部署時不該變老兵。"""
        state = self._state(p1_units=(("u1", "16th_infantry_brigade", 4, 4),))
        unit = state.players["p1"].units[0]
        self.engine.execute("16th_infantry_brigade", "on_deploy", state,
                            EffectContext("p1", "16th_infantry_brigade", "u1"))
        self.assertEqual(unit.card_id, "16th_infantry_brigade")

    def test_16th_brigade_becomes_veteran_after_a_rain_card(self) -> None:
        """打出暴雨（雨卡）後，16 步兵旅要換成老兵版。"""
        state = self._state(p1_units=(("u1", "16th_infantry_brigade", 4, 4),),
                            p1_hand=("deluge",))
        unit = state.players["p1"].units[0]
        PlayCardAction("p1", "deluge").execute(state, self.cards)
        self.assertEqual(unit.card_id, "16th_infantry_brigade_vet",
                         "打出雨卡後沒有變成老兵版")

    def test_16th_brigade_ignores_a_sunny_card(self) -> None:
        """晴天卡不符合「雨或風暴」的條件。"""
        state = self._state(p1_units=(("u1", "16th_infantry_brigade", 4, 4),),
                            p1_hand=("heatwave",))
        unit = state.players["p1"].units[0]
        PlayCardAction("p1", "heatwave").execute(state, self.cards)
        self.assertEqual(unit.card_id, "16th_infantry_brigade")

    def test_79th_regiment_buffs_only_after_a_sunny_card(self) -> None:
        """79 步兵團（打出晴天卡時給單位 +2 攻擊）要等晴天卡才生效。"""
        state = self._state(p1_units=(("u1", "79th_infantry_regiment", 3, 4),
                                      ("u2", "garrison", 0, 3)),
                            p1_hand=("heatwave", "deluge"))
        ally = state.players["p1"].units[1]
        self.engine.execute("79th_infantry_regiment", "on_deploy", state,
                            EffectContext("p1", "79th_infantry_regiment", "u1"))
        self.assertEqual(ally.attack, 0, "部署當下就加了攻擊力")
        PlayCardAction("p1", "deluge").execute(state, self.cards)
        self.assertEqual(ally.attack, 0, "雨卡不該觸發晴天條件")
        PlayCardAction("p1", "heatwave").execute(state, self.cards)
        self.assertGreater(ally.attack, 0, "打出晴天卡後沒有加攻擊力")

    def test_22nd_pioneers_buffs_only_on_forecast(self) -> None:
        """22 工兵（當你 Forecast 時友方 +1+1）要等 Forecast 才生效。"""
        state = self._state(p1_units=(("u1", "22nd_pioneers", 3, 3),
                                      ("u2", "garrison", 0, 3)))
        ally = state.players["p1"].units[1]
        self.engine.execute("22nd_pioneers", "on_deploy", state,
                            EffectContext("p1", "22nd_pioneers", "u1"))
        self.assertEqual((ally.attack, ally.defense), (0, 3), "部署當下就加了數值")
        self.engine.execute("yokosuka_e14y", "on_deploy", state,
                            EffectContext("p1", "yokosuka_e14y", "u1"))
        self.assertEqual((ally.attack, ally.defense), (1, 4), "Forecast 後沒有加 +1+1")


if __name__ == "__main__":
    unittest.main()
