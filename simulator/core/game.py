"""Gym-like facade around the state and action layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import random

from simulator.actions.action import Action, AttackAction, MoveUnitAction, MulliganAction, PassAction, PlayCardAction
from simulator.replay import Replay
from simulator.actions.validator import opponent_id
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, GameStatus, PlayerState
from simulator.core.turn import TurnManager
from simulator.rules.native import _hand_unit_matches, develop_offers, engine_for, hand_choice_matches
from simulator.rules.develop import offers as develop_offers_for
from simulator.rules.deck import DeckValidator
from simulator.rules import weather as weather_rules


@dataclass
class Simulator:
    """Headless training interface. Effects and handlers are connected in later phases."""

    cards: CardDatabase
    state: GameState | None = None
    replay: Replay | None = None
    # Replay snapshots serialize the complete state after every action.  They
    # are useful for debugging, but self-play can safely opt out because it
    # stores its own training examples.
    record_replay: bool = True
    _action_cache: list[Action] | None = field(default=None, init=False, repr=False)

    def reset(self, player_one_deck: Iterable[str], player_two_deck: Iterable[str], nations: tuple[str | None, str | None] = (None, None), seed: int = 0, auto_mulligan: bool = True, ally_nations: tuple[str | None, str | None] = (None, None)) -> GameState:
        self._action_cache = None
        deck_one, deck_two = list(player_one_deck), list(player_two_deck)
        nation_one = nations[0] or self._infer_nation(deck_one)
        nation_two = nations[1] or self._infer_nation(deck_two)
        self._validate_deck(deck_one, nation_one, ally_nations[0])
        self._validate_deck(deck_two, nation_two, ally_nations[1])
        self.state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", nation_one, ally_nations[0], deck_one),
                "p2": PlayerState("p2", nation_two, ally_nations[1], deck_two),
            },
            graveyard={"p1": [], "p2": []}, rng_seed=seed,
        )
        rng = random.Random(seed)
        rng.shuffle(self.state.players["p1"].deck)
        rng.shuffle(self.state.players["p2"].deck)
        # 2nd WEST AFRICA asks whether a card "started" in the deck.
        for player in self.state.players.values():
            player.status["starting_deck"] = sorted(player.deck)
        for _ in range(4): TurnManager.draw_card(self.state, "p1")
        for _ in range(5): TurnManager.draw_card(self.state, "p2")
        self.state.mulligan_pending = ["p1", "p2"]
        if auto_mulligan:
            MulliganAction("p1").execute(self.state, self.cards)
            MulliganAction("p2").execute(self.state, self.cards)
            self.state.current_player = "p1"
            TurnManager.start_turn(self.state, "p1", draw_card=False)
        self.replay = Replay(self.state.to_dict()) if self.record_replay else None
        return self.get_state()

    def step(self, action: Action) -> tuple[GameState, float, bool]:
        reward, terminal = self.step_fast(action)
        return self.get_state(), reward, terminal

    def step_fast(self, action: Action) -> tuple[float, bool]:
        """Execute an action without creating the unused public state clone.

        The public ``step`` API intentionally remains unchanged.  This method
        is for internal high-volume self-play callers that already hold the
        simulator-owned state and only need reward/termination information.
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step()")
        self._action_cache = None
        opponent = opponent_id(self.state, action.player_id)
        before = self.state.players[opponent].hq.current_health
        action.execute(self.state, self.cards)
        if self.replay:
            self.replay.record(action, self.state)
        reward = float(before - self.state.players[opponent].hq.current_health)
        terminal = self.is_terminal()
        if terminal and self.state.game_status != GameStatus.DRAW:
            reward += 100.0
        return reward, terminal

    def get_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("Call reset() before get_state()")
        return self.state.clone()

    def get_available_actions(self) -> list[Action]:
        if self.state is None or self.is_terminal():
            return []
        if self._action_cache is not None:
            # Never expose the cached container: callers may reorder/filter
            # legal candidates, while a state has not changed underneath it.
            return list(self._action_cache)
        player_id = self.state.current_player
        if self.state.mulligan_pending:
            if player_id not in self.state.mulligan_pending:
                return []
            self._action_cache = [MulliganAction(player_id, subset) for subset in _subsets(tuple(self.state.players[player_id].hand))]
            return list(self._action_cache)
        player = self.state.players[player_id]
        actions: list[Action] = [PassAction(player_id)]
        rule_engine = engine_for(self.cards)
        hand_ids = tuple(player.hand)
        unit_hand = [(card_id, "hand") for card_id in hand_ids if self.cards.get(card_id).is_unit]
        convert_choices = [(card_id, "hand") for card_id in hand_ids] + [(card_id, "deck") for card_id in player.deck]
        all_units: list[str] | None = None
        has_high_attack = any(unit.attack >= 4 for unit in player.units)
        # Two copies of a card in hand give the same actions; listing them twice
        # gave the search duplicate branches with identical features.
        for card_id in dict.fromkeys(hand_ids):
            positions = ("support_line",)  # KARDS units deploy to the support line.
            spec = rule_engine.action_spec_for(card_id)
            if spec.needs_target:
                if all_units is None:
                    all_units = [unit.instance_id for candidate in self.state.players.values() for unit in candidate.units]
                targets = [None, *all_units] if spec.may_target_enemy_hq else all_units
            else:
                targets = (None,)
            selected_cards = (
                [(candidate_id, zone) for candidate_id, zone in unit_hand if candidate_id != card_id]
                if spec.swap_hand_unit
                else [(candidate_id, zone) for candidate_id, zone in dict.fromkeys(unit_hand)
                      if _hand_unit_matches(self.cards.get(candidate_id), *spec.put_into_play)]
                if spec.put_into_play
                else ([(candidate_id, "hand") for candidate_id in dict.fromkeys(hand_ids)
                       if hand_choice_matches(spec.hand_choice, self.cards, candidate_id, hand_ids, card_id)]
                      or [(None, None)])
                if spec.hand_choice
                else ([(candidate_id, "deck") for candidate_id in develop_offers(self.state, player_id, spec.develop_from_deck)]
                      or [(None, None)])
                if spec.develop_from_deck
                else ([(candidate_id, "offer") for candidate_id in develop_offers_for(self.state, player_id, self.cards, spec.develop_phrase)]
                      or [(None, None)])
                if spec.develop_phrase
                else [(candidate_id, zone) for candidate_id, zone in convert_choices if zone != "hand" or candidate_id != card_id]
                if spec.convert_to
                else [(None, None)]
            )
            selected_options = (
                [None] if has_high_attack else ["blitz", "shock"]
                if spec.shock_tactics
                # Forecast is a three-way player choice: sunny / rain / storm.
                else list(weather_rules.CATEGORIES)
                if spec.forecast
                # Develop A, B or C: the player picks one; no "no choice" action.
                else (list(spec.develop_options) or [None])
            )
            for position in positions:
                for target_unit_id in targets:
                    for selected_card_id, selected_card_zone in selected_cards:
                        for selected_option in selected_options:
                            action = PlayCardAction(player_id, card_id, position, target_unit_id, selected_card_id, selected_option, selected_card_zone)
                            if _is_valid(action, self.state, self.cards):
                                actions.append(action)
        enemy_id = opponent_id(self.state, player_id)
        for unit in player.units:
            move = MoveUnitAction(player_id, unit.instance_id)
            if _is_valid(move, self.state, self.cards): actions.append(move)
            for target in [None] + [enemy.instance_id for enemy in self.state.players[enemy_id].units]:
                action = AttackAction(player_id, unit.instance_id, target)
                if _is_valid(action, self.state, self.cards):
                    actions.append(action)
        self._action_cache = actions
        return list(self._action_cache)

    def is_terminal(self) -> bool:
        return self.state is not None and self.state.game_status != GameStatus.IN_PROGRESS

    def _validate_deck(self, deck: list[str], nation: str | None, ally_nation: str | None = None) -> None:
        if nation is None:
            raise ValueError("Deck nation could not be inferred")
        result = DeckValidator(self.cards).validate(deck, nation, ally_nation)
        if not result.valid:
            raise ValueError("Illegal deck: " + "; ".join(result.errors))

    def _infer_nation(self, deck: list[str]) -> str | None:
        for card_id in deck:
            if card_id in self.cards and self.cards.get(card_id).nation != "Neutral":
                return self.cards.get(card_id).nation
        return None

    def get_observation(self, player_id: str) -> dict:
        """Player-view API that hides the opponent's hand IDs."""
        if self.state is None:
            raise RuntimeError("Call reset() before get_observation()")
        if player_id not in self.state.players:
            raise ValueError("Unknown player")
        player = self.state.players[player_id]
        enemy = self.state.players[opponent_id(self.state, player_id)]
        opponent_view = _observation_player(enemy, False)
        # Countermeasure activation is deliberately hidden: exposing remaining
        # kredits would reveal the activation payment to the opponent.
        opponent_view["kredits"] = None
        opponent_view["known_hand"] = list(player.status.get("known_enemy_hand", ()))
        return {"turn": self.state.turn_number, "current": self.state.current_player == player_id,
                "self": _observation_player(player, True), "opponent": opponent_view,
                "frontline": list(self.state.battlefield["frontline"]), "support_line": list(self.state.battlefield["support_line"]),
                "terminal": self.is_terminal()}


def _observation_player(player: PlayerState, reveal_hand: bool) -> dict:
    return {"hq_health": player.hq.current_health, "hq_defense": player.hq.defense_modifier,
            "kredits": player.resources.kredits, "max_kredits": player.resources.max_kredits,
            "deck_count": len(player.deck), "hand": list(player.hand) if reveal_hand else None,
            "hand_count": len(player.hand), "units": [{"id": unit.instance_id, "card_id": None if unit.status.get("covert") and not reveal_hand else unit.card_id, "attack": None if unit.status.get("covert") and not reveal_hand else unit.attack, "defense": None if unit.status.get("covert") and not reveal_hand else unit.defense, "position": unit.position} for unit in player.units]}


def _is_valid(action: Action, state: GameState, cards: CardDatabase) -> bool:
    try:
        action.validate(state, cards)
    except ValueError:
        return False
    return True


def _subsets(values: tuple[str, ...]) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = [()]
    for value in values:
        result += [prefix + (value,) for prefix in tuple(result)]
    return result
