"""Action commands. State is mutated only after each action validates successfully."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random

from simulator.actions.validator import ActionValidationError, find_unit, opponent_id, require_active_player
from simulator.cards.loader import CardDatabase
from simulator.cards.availability import is_card_available
from simulator.core.state import GameState, UnitState, GameStatus
from simulator.core.turn import TurnManager
from simulator.effects.resolver import EffectContext, EffectResolver
from simulator.rules.battlefield import BattlefieldRules
from simulator.rules.keywords import KeywordEngine
from simulator.rules.abilities import AbilityEngine
from simulator.rules.native import engine_for
from simulator.rules import weather as weather_rules
from simulator.rules import counters
from simulator.countermeasures import CountermeasureResolver
from simulator.rules import intel as intel_rules


class Action(ABC):
    player_id: str

    @abstractmethod
    def validate(self, state: GameState, cards: CardDatabase) -> None:
        """Raise ActionValidationError without changing state if this action is illegal."""

    @abstractmethod
    def execute(self, state: GameState, cards: CardDatabase) -> GameState:
        """Apply an already-valid action and return the same state object."""


@dataclass(frozen=True)
class PlayCardAction(Action):
    player_id: str
    card_id: str
    position: str = "support_line"
    target_unit_id: str | None = None
    selected_card_id: str | None = None
    selected_option: str | None = None
    selected_card_zone: str | None = None

    def validate(self, state: GameState, cards: CardDatabase) -> None:
        require_active_player(state, self.player_id)
        player = state.players[self.player_id]
        if self.card_id not in player.hand:
            raise ActionValidationError("Card is not in the player's hand")
        if self.card_id not in cards:
            raise ActionValidationError("Card is absent from the card database")
        if not is_card_available(self.card_id):
            raise ActionValidationError("Card is retired and cannot be played")
        card = cards.get(self.card_id)
        # Enemy "cannot deploy" / "cannot order" aura: check opposing units.
        if card.type in ("infantry", "fighter", "tank", "bomber", "artillery"):
            for enemy_unit in state.players[opponent_id(state, self.player_id)].units:
                if enemy_unit.status.get("enemy_cannot_deploy"):
                    raise ActionValidationError("Enemy unit prevents deploying")
        if card.type == "order":
            for enemy_unit in state.players[opponent_id(state, self.player_id)].units:
                if enemy_unit.status.get("enemy_cannot_order"):
                    raise ActionValidationError("Enemy unit prevents orders")
        active = any(entry.get("card_id") == card.id for entry in player.active_countermeasures)
        if card.type == "countermeasure" and active:
            return
        rule = engine_for(cards).rule_for(card.id)
        if any(action.kind == "swap_hand_unit_with_friendly" for action in rule.actions):
            if self.selected_card_id is None or self.selected_card_id not in player.hand:
                raise ActionValidationError("This card requires a unit selected from hand")
            selected_card = cards.get(self.selected_card_id) if self.selected_card_id in cards else None
            if selected_card is None or not selected_card.is_unit:
                raise ActionValidationError("Selected hand card must be a unit")
        put = next((action for action in rule.actions if action.kind == "put_into_play"), None)
        if put is not None:
            from simulator.rules.native import _hand_unit_matches
            family, _, nation = (put.scope or "").partition(":")
            chosen = cards.get(self.selected_card_id) if self.selected_card_id in cards else None
            if self.selected_card_id not in player.hand or chosen is None or not _hand_unit_matches(chosen, family, nation):
                raise ActionValidationError("This card puts a matching unit from hand into play")
            if self.selected_card_id == self.card_id and player.hand.count(self.card_id) < 2:
                raise ActionValidationError("The chosen unit must be another card in hand")
            if not BattlefieldRules.can_deploy_to_support(state, self.player_id):
                raise ActionValidationError("Support line is full")
        if any(action.kind == "convert_to" for action in rule.actions) and self.selected_card_id is not None:
            selected_zone = self.selected_card_zone or "hand"
            if selected_zone not in {"hand", "deck"}:
                raise ActionValidationError("Convert target zone must be hand or deck")
            if self.selected_card_id not in getattr(player, selected_zone):
                raise ActionValidationError("Selected conversion card is absent from its zone")
        if any(action.kind == "shock_tactics_choice" for action in rule.actions):
            if self.target_unit_id is None:
                raise ActionValidationError("This card requires a ground unit target")
            target = find_unit(state, self.target_unit_id)
            if cards.get(target.card_id).type not in {"infantry", "tank", "artillery"}:
                raise ActionValidationError("This card requires a ground unit target")
            has_high_attack = any(unit.attack >= 4 for unit in player.units)
            if not has_high_attack and self.selected_option not in {"blitz", "shock"}:
                raise ActionValidationError("This card requires Blitz or Shock selection")
        if any(action.kind == "cannot" and action.card_name == "be_deployed_after_unit" for action in rule.actions) \
                and counters.units_deployed(state, self.player_id, 0):
            raise ActionValidationError("Cannot be deployed after deploying a unit this turn")
        aims_at_hq = any(action.scope == "or_enemy_hq" for action in rule.actions) and self.target_unit_id is None
        if card.type != "countermeasure" and rule.needs_target and not aims_at_hq:
            if self.target_unit_id is None:
                raise ActionValidationError("This card requires a unit target")
            target = find_unit(state, self.target_unit_id)
            target_roles = {action.target for action in rule.actions if action.target.startswith("selected_")}
            if "selected_enemy" in target_roles and target.owner_id == self.player_id:
                raise ActionValidationError("This card requires an enemy unit target")
            if "selected_friendly" in target_roles and target.owner_id != self.player_id:
                raise ActionValidationError("This card requires a friendly unit target")
            if card.type == "order" and target.owner_id != self.player_id and target.status.get("cannot_be_targeted_by_enemy_orders"):
                raise ActionValidationError("Target cannot be targeted by enemy orders")
            if card.type == "order" and target.owner_id != self.player_id and target.status.get("covert"):
                if not any(action.card_name == "can_target_covert" for action in rule.actions):
                    raise ActionValidationError("Covert unit cannot be targeted by enemy orders")
            target_categories = {action.card_name for action in rule.actions if action.card_name in {"air", "ground", "frontline"}}
            if "air" in target_categories and cards.get(target.card_id).type not in {"fighter", "bomber"}:
                raise ActionValidationError("This card requires an air unit target")
            if "ground" in target_categories and cards.get(target.card_id).type not in {"infantry", "tank", "artillery"}:
                raise ActionValidationError("This card requires a ground unit target")
            if "frontline" in target_categories and target.position != "frontline":
                raise ActionValidationError("This card requires a frontline unit target")
            if any(action.kind == "retreat_and_repair" for action in rule.actions):
                owner = state.players[target.owner_id]
                if sum(unit.position == "support_line" for unit in owner.units) >= 4:
                    raise ActionValidationError("Target owner's support line is full")
            held = next((action for action in rule.actions if action.kind == "remove_until_return"), None)
            if held is not None:
                # SECOND FRONT: "Remove a German ground unit".
                from simulator.rules.native import _hand_unit_matches
                family, _, nation = (held.scope or "").partition(":")
                if not _hand_unit_matches(cards.get(target.card_id), family, nation):
                    raise ActionValidationError("Target is not a unit this card can remove")
            take = next((action for action in rule.actions if action.kind == "take_control"), None)
            if take is not None:
                # "Take control of an enemy unit with cost 3 or less": the cap
                # was never checked, and a full support line has no room.
                if take.amount and cards.get(target.card_id).kredits > take.amount:
                    raise ActionValidationError("Target exceeds this card's take-control cost limit")
                if sum(unit.position == "support_line" for unit in player.units) >= 4:
                    raise ActionValidationError("No room in the support line for the taken unit")
            max_destroy_cost = next((action.amount for action in rule.actions if action.kind == "destroy_cost_lte"), None)
            if max_destroy_cost is not None and cards.get(target.card_id).kredits > max_destroy_cost:
                raise ActionValidationError("Target exceeds this card's destroy cost limit")
            if any(action.kind == "destroy_undamaged" for action in rule.actions):
                printed_defense = cards.get(target.card_id).defense
                if printed_defense is None or target.defense != printed_defense:
                    raise ActionValidationError("This card requires an undamaged unit target")
            if any(action.kind == "trigger_destroy_and_convert" for action in rule.actions):
                target_card = cards.get(target.card_id)
                if target_card.nation != "Soviet" or target_card.type not in {"fighter", "bomber"}:
                    raise ActionValidationError("This card requires a Soviet air unit target")
                if target_card.defense is None or target.defense >= target_card.defense:
                    raise ActionValidationError("This card requires a damaged unit target")
        elif self.target_unit_id is not None:
            raise ActionValidationError("This card does not take a unit target")
        target_tax = _target_tax(state, self.target_unit_id, self.player_id)
        if player.resources.kredits < card_play_cost(state, self.player_id, card, cards) + target_tax:
            raise ActionValidationError("Insufficient kredits")
        if card.is_unit and self.position != "support_line":
            raise ActionValidationError("Units must deploy to the support line")
        if card.is_unit and not BattlefieldRules.can_deploy_to_support(state, self.player_id):
            raise ActionValidationError("Support line is full")

    def execute(self, state: GameState, cards: CardDatabase) -> GameState:
        self.validate(state, cards)
        player = state.players[self.player_id]
        card = cards.get(self.card_id)
        # Price the card exactly as validate() did: with the card still in hand.
        # Pricing it again after it left the hand (or charging the printed cost
        # for a countermeasure whose cost a modifier had changed) is how
        # kredits went below 0.
        cost = card_play_cost(state, self.player_id, card, cards) + _target_tax(state, self.target_unit_id, self.player_id)
        if card.type == "countermeasure":
            existing = next((entry for entry in player.active_countermeasures if entry.get("card_id") == card.id), None)
            if existing:
                player.active_countermeasures.remove(existing)
                player.resources.kredits += int(existing.get("paid", card.kredits))
                state.event_log.append({"event": "countermeasure_deactivated", "player_id": self.player_id, "card_id": card.id})
                return state
            player.resources.kredits -= cost
            player.active_countermeasures.append({"card_id": card.id, "activated_turn": state.turn_number, "paid": cost})
            state.event_log.append({"event": "countermeasure_activated", "player_id": self.player_id, "card_id": card.id})
            return state
        # Capture the hand position BEFORE the card leaves hand: cards that read
        # "if this card was left-most in hand" are resolved after the removal.
        hand_index = player.hand.index(self.card_id)
        hand_size = len(player.hand)
        player.hand.remove(self.card_id)
        intel_rules.refresh(state)
        player.resources.kredits -= cost
        reductions = player.status.get("hand_cost_on_enemy_draw")
        if isinstance(reductions, dict) and self.card_id not in player.hand:
            reductions.pop(self.card_id, None)   # the last copy left the hand
        if card.type == "order":
            counters.note_order_given(state, self.player_id)
        elif card.is_unit:
            counters.note_unit_deployed(state, self.player_id, card.id)
        intel = KeywordEngine.numeric(card, "intel")
        if intel:
            known = intel_rules.reveal(state, self.player_id, opponent_id(state, self.player_id), intel)
            state.event_log.append({"event": "intel_revealed", "player_id": self.player_id, "count": len(known)})
        if card.is_unit:
            instance_id = _next_unit_id(state, self.card_id)
            unit = UnitState(instance_id, card.id, card.attack or 0, card.defense or 0, self.player_id, self.position)
            unit.status["deployed_this_turn"] = True
            if KeywordEngine.has(card, "covert", unit):
                unit.status["covert"] = True
            player.units.append(unit)
            state.battlefield[self.position].append(instance_id)
            apply_op_cost_rules(state, self.player_id, unit, cards)
            state.event_log.append({"event": "unit_deployed", "player_id": self.player_id, "card_id": self.card_id, "unit_id": instance_id})
            metadata = {key: value for key, value in (("selected_card_id", self.selected_card_id), ("selected_option", self.selected_option), ("selected_card_zone", self.selected_card_zone)) if value is not None}
            metadata.update({"hand_index": hand_index, "hand_size": hand_size})
            context = EffectContext(self.player_id, self.card_id, instance_id, self.target_unit_id, metadata=metadata)
            if self.target_unit_id is not None:
                engine_for(cards).emit("on_targeted_by_enemy_effect", state, EffectContext(self.player_id, card.id, instance_id, self.target_unit_id, event="on_targeted_by_enemy_effect"))
            has_deployment_effect = "deployment:" in (card.text or "").lower()
            deployment_suppressed = False
            is_explicit_deployment_effect = has_deployment_effect
            deployment_cancelled = CountermeasureResolver.intercept(
                state, cards, self.player_id, "on_deploy",
                EffectContext(self.player_id, self.card_id, instance_id, self.target_unit_id,
                              metadata={"has_deployment_effect": has_deployment_effect}),
            )
            if deployment_cancelled:
                state.event_log.append({"event": "deployment_effect_cancelled", "card_id": self.card_id, "unit_id": instance_id})
            else:
                # ``on_deploy`` is the deployed card's own Deployment trigger.
                # It is not a broadcast event: broadcasting it would re-run every
                # unit's Deployment text whenever any later unit enters play.
                deployment_suppressed = any(
                    candidate.status.get("suppress_deployment_effects")
                    for owner in state.players.values() for candidate in owner.units
                    if candidate.instance_id != instance_id
                )
                if deployment_suppressed and is_explicit_deployment_effect:
                    state.event_log.append({"event": "deployment_effect_suppressed", "card_id": card.id, "unit_id": instance_id})
                else:
                    engine_for(cards).execute(card.id, "on_deploy", state, context)
                # Broadcast only explicit battlefield listeners such as
                # "When the enemy deploys a unit".  Own Deployment rules were
                # already executed above and must not be replayed.
                engine_for(cards).emit(
                    "on_deploy", state,
                    EffectContext(
                        self.player_id, card.id, instance_id, instance_id,
                        event="on_deploy",
                        metadata={"played_card_id": card.id, "broadcast_listeners": True},
                    ),
                )
                # Persistent named deployment auras (for example, a source
                # which says its named units get stats when deployed) are
                # evaluated after the new unit's own Deployment effect has
                # registered itself.  The routine scans runtime units only;
                # card definitions remain immutable catalog data.
                engine_for(cards).apply_deployment_name_auras(state, unit)
            engine_for(cards).emit("on_friendly_card_played", state, EffectContext(self.player_id, card.id, instance_id, event="on_friendly_card_played", metadata={"played_card_id": card.id}))
            if not (deployment_suppressed and is_explicit_deployment_effect):
                EffectResolver(cards).emit("on_deploy", state, context)
        else:
            if CountermeasureResolver.intercept(state, cards, self.player_id, "on_command_played", EffectContext(self.player_id, self.card_id, target_unit_id=self.target_unit_id)):
                state.graveyard.setdefault(self.player_id, []).append(self.card_id)
                state.event_log.append({"event": "action_cancelled", "action": "play_order", "card_id": self.card_id})
                return state
            state.graveyard.setdefault(self.player_id, []).append(self.card_id)
            state.event_log.append({"event": "command_played", "player_id": self.player_id, "card_id": self.card_id})
            # Record the weather BEFORE the card resolves, so an effect that
            # reads "if Rain was played this turn" already sees this card.
            weather_category = weather_rules.note_weather_played(state, self.player_id, self.card_id)
            metadata = {key: value for key, value in (("selected_card_id", self.selected_card_id), ("selected_option", self.selected_option), ("selected_card_zone", self.selected_card_zone)) if value is not None}
            metadata.update({"hand_index": hand_index, "hand_size": hand_size})
            context = EffectContext(self.player_id, self.card_id, target_unit_id=self.target_unit_id, metadata=metadata)
            if self.target_unit_id is not None:
                engine_for(cards).emit("on_targeted_by_enemy_effect", state, EffectContext(self.player_id, card.id, target_unit_id=self.target_unit_id, event="on_targeted_by_enemy_effect"))
            engine_for(cards).execute(card.id, "on_play", state, context)
            if weather_category is not None:
                state.event_log.append({"event": "on_weather_played", "player_id": self.player_id,
                                        "weather": weather_category})
                engine_for(cards).emit("on_weather_played", state, EffectContext(
                    self.player_id, card.id, event="on_weather_played",
                    metadata={"played_card_id": card.id, "weather": weather_category}))
            engine_for(cards).emit("on_command_played", state, context)
            engine_for(cards).emit("on_friendly_card_played", state, EffectContext(self.player_id, card.id, event="on_friendly_card_played", metadata={"played_card_id": card.id}))
            EffectResolver(cards).emit("on_play", state, context)
        return state


@dataclass(frozen=True)
class MulliganAction(Action):
    player_id: str
    card_ids: tuple[str, ...] = ()

    def validate(self, state: GameState, cards: CardDatabase) -> None:
        if self.player_id not in state.mulligan_pending:
            raise ActionValidationError("Player is not awaiting a mulligan")
        hand = state.players[self.player_id].hand
        if any(card_id not in hand for card_id in self.card_ids):
            raise ActionValidationError("Mulligan card is not in hand")

    def execute(self, state: GameState, cards: CardDatabase) -> GameState:
        self.validate(state, cards)
        player = state.players[self.player_id]
        for card_id in self.card_ids:
            player.hand.remove(card_id)
            player.deck.append(card_id)
        random.Random((state.rng_seed or 0) + len(state.event_log)).shuffle(player.deck)
        for _ in self.card_ids:
            TurnManager.draw_card(state, self.player_id)
        state.mulligan_pending.remove(self.player_id)
        if state.mulligan_pending:
            state.current_player = state.mulligan_pending[0]
        state.event_log.append({"event": "mulligan_completed", "player_id": self.player_id, "replaced": len(self.card_ids)})
        return state


@dataclass(frozen=True)
class AttackAction(Action):
    player_id: str
    attacker_id: str
    target_unit_id: str | None = None

    def validate(self, state: GameState, cards: CardDatabase) -> None:
        require_active_player(state, self.player_id)
        attacker = find_unit(state, self.attacker_id)
        if attacker.owner_id != self.player_id:
            raise ActionValidationError("Attacker is not controlled by player")
        card = cards.get(attacker.card_id)
        if attacker.status.get("attack_count", 0) >= KeywordEngine.attack_limit(card, attacker):
            raise ActionValidationError("Unit has already attacked this turn")
        if attacker.status.get("suppressed"):
            raise ActionValidationError("Suppressed unit cannot attack")
        if attacker.status.get("pinned"):
            raise ActionValidationError("Pinned unit cannot attack")
        cannot_reason = attacker.status.get("cannot")
        if cannot_reason in ("attack", True):
            raise ActionValidationError("Unit cannot attack")
        if not KeywordEngine.can_attack_from_position(card, attacker.position):
            raise ActionValidationError("Unit cannot attack from this line")
        if attacker.status.get("deployed_this_turn") and not KeywordEngine.can_attack_on_deploy(card, attacker):
            raise ActionValidationError("Unit cannot attack on its deployment turn")
        if attacker.status.get("moved_this_turn") and not _can_move_and_attack(state, attacker, card, cards):
            raise ActionValidationError("Unit cannot move and attack in the same turn")
        if self.target_unit_id is not None:
            target = find_unit(state, self.target_unit_id)
            if target.owner_id == self.player_id:
                raise ActionValidationError("Unit cannot attack a friendly target")
            can_target_covert = attacker.status.get("can_target_covert") or attacker.status.get("target_select") == "can_target_covert"
            if target.status.get("covert") and not can_target_covert:
                raise ActionValidationError("Covert unit cannot be attacked")
            if AbilityEngine.has_smokescreen(cards.get(target.card_id), target):
                raise ActionValidationError("Unit with Smokescreen cannot be attacked")
            if target.position == "support_line" and target.status.get("immune_to_ground_in_support") and cards.get(attacker.card_id).type in {"infantry", "tank"}:
                raise ActionValidationError("Target cannot be attacked by ground units in support line")
            if KeywordEngine.is_guarded(state, target, card, cards):
                raise ActionValidationError("Target is guarded")
        else:
            if attacker.status.get("cannot_attack_hq"):
                raise ActionValidationError("Unit cannot attack enemy HQ")
            if KeywordEngine.is_hq_guarded(state, opponent_id(state, self.player_id), card, cards):
                raise ActionValidationError("Enemy HQ is guarded")
        if state.players[self.player_id].resources.kredits < _operation_cost(attacker, card) + _target_tax(state, self.target_unit_id, self.player_id):
            raise ActionValidationError("Insufficient kredits for operation")

    def execute(self, state: GameState, cards: CardDatabase) -> GameState:
        self.validate(state, cards)
        attacker = find_unit(state, self.attacker_id)
        if AbilityEngine.has_smokescreen(cards.get(attacker.card_id), attacker):
            AbilityEngine.consume(attacker, "smokescreen")
        target_owner = opponent_id(state, self.player_id)
        attacker.status["attack_count"] = attacker.status.get("attack_count", 0) + 1
        if CountermeasureResolver.intercept(state, cards, self.player_id, "on_attack", EffectContext(self.player_id, attacker.card_id, attacker.instance_id, self.target_unit_id)):
            state.event_log.append({"event": "action_cancelled", "action": "attack", "unit_id": attacker.instance_id})
            return state
        operating_cost = _operation_cost(attacker, cards.get(attacker.card_id)) + _target_tax(state, self.target_unit_id, self.player_id)
        state.players[self.player_id].resources.kredits -= operating_cost
        counters.note_operating_spend(state, self.player_id, operating_cost)
        resolver = EffectResolver(cards)
        resolver.emit("before_attack", state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id, self.target_unit_id))
        resolver.emit("on_attack", state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id, self.target_unit_id))
        _emit_operated(state, cards, self.player_id, attacker)
        engine_for(cards).emit("on_attack", state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id, self.target_unit_id))
        if self.target_unit_id is None:
            hq_health_before = state.players[target_owner].hq.current_health
            # HQ damage bonus: some units deal extra damage when attacking HQs.
            hq_damage = attacker.attack
            hq_bonus = attacker.status.get("hq_damage_bonus")
            if isinstance(hq_bonus, int):
                hq_damage += hq_bonus
            resolver.resolve({"type": "damage", "target": "enemy_hq", "value": {"amount": hq_damage}}, state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id, metadata={"combat_damage": True}))
            hq_damage = hq_health_before - state.players[target_owner].hq.current_health
            if hq_damage > 0:
                engine_for(cards).emit("on_enemy_hq_damaged", state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id, event="on_enemy_hq_damaged", metadata={"damage_amount": hq_damage}))
            state.event_log.append({"event": "hq_attacked", "attacker_id": attacker.instance_id, "player_id": target_owner, "damage": attacker.attack})
            resolver.emit("after_attack", state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id))
            return state

        # Attack listeners may legally remove or return the defender before
        # combat damage is dealt.  Re-check after the FIFO trigger window;
        # combat then ends without trying to damage an entity that left play.
        try:
            target = find_unit(state, self.target_unit_id)
        except ActionValidationError:
            state.event_log.append({"event": "attack_target_no_longer_exists", "attacker_id": attacker.instance_id,
                                    "target_id": self.target_unit_id})
            resolver.emit("after_attack", state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id))
            return state
        target.status.pop("covert", None)
        engine_for(cards).emit("on_targeted_by_enemy_attack", state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id, target.instance_id, event="on_targeted_by_enemy_attack"))
        try:
            target = find_unit(state, self.target_unit_id)
        except ActionValidationError:
            state.event_log.append({"event": "attack_target_no_longer_exists", "attacker_id": attacker.instance_id,
                                    "target_id": self.target_unit_id})
            resolver.emit("after_attack", state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id))
            return state
        if AbilityEngine.has_ambush(cards.get(target.card_id), target) and not target.status.get("ambush_used_this_round"):
            target.status["ambush_used_this_round"] = True
            resolver.resolve({"type": "damage", "target": "self", "value": {"amount": target.attack}}, state, EffectContext(target.owner_id, target.card_id, target.instance_id, attacker.instance_id))
            if attacker.instance_id not in {unit.instance_id for player in state.players.values() for unit in player.units}:
                state.event_log.append({"event": "ambush_destroyed_attacker", "attacker_id": attacker.instance_id, "target_id": target.instance_id})
                return state
        attack_value = attacker.attack
        higher_attack_bonus = attacker.status.get("attack_bonus_against_higher_attack")
        if isinstance(higher_attack_bonus, int) and target.attack > attacker.attack:
            attack_value += higher_attack_bonus
        type_attack_bonus = attacker.status.get("attack_bonus_against_type")
        if isinstance(type_attack_bonus, dict) and KeywordEngine.matches_type(cards.get(target.card_id), target, str(type_attack_bonus.get("type", ""))):
            amount = type_attack_bonus.get("amount")
            if isinstance(amount, int):
                attack_value += amount
        damage = max(0, attack_value - KeywordEngine.heavy_armor(cards.get(target.card_id), target))
        # Some units ignore heavy armor entirely.
        if attacker.status.get("ignore_heavy_armor"):
            damage = attack_value
        air_damage_bonus = attacker.status.get("damage_bonus_against_air")
        if isinstance(air_damage_bonus, int) and cards.get(target.card_id).type in {"fighter", "bomber"}:
            damage += air_damage_bonus
        if KeywordEngine.matches_type(cards.get(target.card_id), target, str(attacker.status.get("double_damage_against_type", ""))):
            damage *= 2
        if attacker.status.get("double_damage"):
            damage *= 2
        if attacker.status.get("triple_damage") == cards.get(target.card_id).type:
            damage *= 3
        random_bonus = attacker.status.get("random_combat_damage")
        if isinstance(random_bonus, int) and random_bonus > 0:
            damage += random.Random((state.rng_seed or 0) + len(state.event_log)).randint(0, random_bonus)
        damage = _cap_combat_damage(target, damage)
        death_start = len(state.event_log)
        resolver.resolve({"type": "damage", "target": "selected_target", "value": {"amount": damage}}, state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id, target.instance_id, metadata={"combat_damage": True}))
        engine_for(cards).emit("on_damage_dealt", state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id, target.instance_id, metadata={"damage_amount": damage}))
        if damage > 0:
            engine_for(cards).emit("on_combat_damage_dealt", state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id, target.instance_id, metadata={"damage_amount": damage}))
        engine_for(cards).emit_damage_since(state, death_start)
        engine_for(cards).emit_deaths_since(state, death_start)
        living = {unit.instance_id for player in state.players.values() for unit in player.units}
        if target.instance_id not in living and attacker.instance_id in living:
            # 77th REGIMENT, 7. SCHÜTZEN: "after this unit destroys a unit".
            engine_for(cards).emit("on_destroys_unit", state, EffectContext(
                self.player_id, attacker.card_id, attacker.instance_id, event="on_destroys_unit",
                metadata={"destroyed_card_id": target.card_id}))
        # hq_excess: deal excess damage to enemy HQ after killing a unit
        destroyed = target.instance_id not in {u.instance_id for p in state.players.values() for u in p.units}
        if attacker.status.get("hq_excess") and destroyed:
            hq_dmg = damage - (cards.get(target.card_id).defense or 0)
            if hq_dmg > 0:
                resolver.resolve({"type": "damage", "target": "enemy_hq", "value": {"amount": hq_dmg}}, state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id, metadata={"combat_damage": True}))
        live_unit_ids = {unit.instance_id for player in state.players.values() for unit in player.units}
        shock = KeywordEngine.has(cards.get(attacker.card_id), "shock", attacker)
        if shock:
            AbilityEngine.consume(attacker, "shock")
        if not shock and attacker.instance_id in live_unit_ids and target.instance_id in live_unit_ids:
            death_start = len(state.event_log)
            # The counter-attack is damage dealt by the DEFENDER, so its own
            # side's damage auras apply (e.g. 110. PANZERGRENADIER).
            return_damage = _cap_combat_damage(attacker, target.attack + resolver.unit_damage_bonus(state, target))
            resolver.resolve({"type": "damage", "target": "self", "value": {"amount": return_damage}}, state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id, metadata={"combat_damage": True}))
            engine_for(cards).emit_damage_since(state, death_start)
            engine_for(cards).emit_deaths_since(state, death_start)
        if attacker.instance_id in {unit.instance_id for player in state.players.values() for unit in player.units}:
            engine_for(cards).emit("on_survives_combat", state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id, target.instance_id))
        state.event_log.append({"event": "unit_attacked", "attacker_id": attacker.instance_id, "target_id": target.instance_id})
        resolver.emit("after_attack", state, EffectContext(self.player_id, attacker.card_id, attacker.instance_id, target.instance_id))
        return state


@dataclass(frozen=True)
class PassAction(Action):
    player_id: str

    def validate(self, state: GameState, cards: CardDatabase) -> None:
        require_active_player(state, self.player_id)

    def execute(self, state: GameState, cards: CardDatabase) -> GameState:
        self.validate(state, cards)
        resolver = EffectResolver(cards)
        resolver.emit("on_turn_end", state, EffectContext(self.player_id))
        engine_for(cards).emit("on_turn_end", state, EffectContext(self.player_id))
        resolver.emit("on_friendly_turn_end", state, EffectContext(self.player_id))
        draw_start = len(state.event_log)
        TurnManager.end_turn(state, self.player_id)
        engine_for(cards).emit_draws_since(state, draw_start)
        resolver.emit("on_turn_start", state, EffectContext(state.current_player))
        engine_for(cards).emit("on_turn_start", state, EffectContext(state.current_player))
        resolver.emit("on_friendly_turn_start", state, EffectContext(state.current_player))
        return state


@dataclass(frozen=True)
class MoveUnitAction(Action):
    player_id: str
    unit_id: str

    def validate(self, state: GameState, cards: CardDatabase) -> None:
        require_active_player(state, self.player_id)
        unit = find_unit(state, self.unit_id)
        if unit.owner_id != self.player_id or not BattlefieldRules.can_move_to_frontline(state, unit):
            raise ActionValidationError("Unit cannot move to the frontline")
        if unit.status.get("pinned"):
            raise ActionValidationError("Pinned unit cannot move")
        card = cards.get(unit.card_id)
        if unit.status.get("deployed_this_turn") and not KeywordEngine.can_attack_on_deploy(card, unit):
            raise ActionValidationError("Unit cannot move on its deployment turn")
        if state.players[self.player_id].resources.kredits < _operation_cost(unit, cards.get(unit.card_id)):
            raise ActionValidationError("Insufficient kredits for operation")

    def execute(self, state: GameState, cards: CardDatabase) -> GameState:
        self.validate(state, cards)
        # Smokescreen is consumed when the move is actually executed, not during
        # validation.  This keeps validate() pure (no side effects) so AI tree
        # search and get_available_actions() can safely enumerate moves.
        unit = find_unit(state, self.unit_id)
        if AbilityEngine.has_smokescreen(cards.get(unit.card_id), unit):
            AbilityEngine.consume(unit, "smokescreen")
        unit = find_unit(state, self.unit_id)
        move_cost = _operation_cost(unit, cards.get(unit.card_id))
        state.players[self.player_id].resources.kredits -= move_cost
        counters.note_operating_spend(state, self.player_id, move_cost)
        state.battlefield["support_line"].remove(unit.instance_id)
        state.battlefield["frontline"].append(unit.instance_id)
        unit.position = "frontline"
        unit.status["moved_this_turn"] = True
        bonus = unit.status.get("frontline_attack_bonus")
        if isinstance(bonus, int) and not unit.status.get("frontline_attack_bonus_applied"):
            unit.attack += bonus
            unit.status["frontline_attack_bonus_applied"] = True
        state.event_log.append({"event": "unit_moved_to_frontline", "player_id": self.player_id, "unit_id": unit.instance_id})
        _emit_operated(state, cards, self.player_id, unit)
        # The parser normalizes move-into-frontline listeners onto the shared
        # deployment event family.  Mark the causal metadata so only those
        # listeners run, rather than replaying ordinary Deployment text.
        engine_for(cards).emit(
            "on_deploy", state,
            EffectContext(
                self.player_id, unit.card_id, unit.instance_id, unit.instance_id,
                event="on_deploy",
                metadata={"played_card_id": unit.card_id, "broadcast_listeners": True, "moved_to_frontline": True},
            ),
        )
        return state


@dataclass(frozen=True)
class UseAbilityAction(Action):
    player_id: str
    unit_id: str
    ability_id: str

    def validate(self, state: GameState, cards: CardDatabase) -> None:
        require_active_player(state, self.player_id)
        unit = find_unit(state, self.unit_id)
        if unit.owner_id != self.player_id:
            raise ActionValidationError("Unit is not controlled by player")
        raise ActionValidationError("Activated abilities are not implemented until Phase 3")

    def execute(self, state: GameState, cards: CardDatabase) -> GameState:
        self.validate(state, cards)
        return state


def _emit_operated(state: GameState, cards: CardDatabase, player_id: str, unit: UnitState) -> None:
    """Operating = attacking or moving (3. PANZERGRENADIER, 30. INFANTRY REGIMENT)."""
    engine_for(cards).emit("on_friendly_operate", state, EffectContext(
        player_id, unit.card_id, unit.instance_id, event="on_friendly_operate",
        metadata={"played_card_id": unit.card_id}))


def _next_unit_id(state: GameState, card_id: str) -> str:
    existing = sum(len(player.units) for player in state.players.values())
    return "{0}-{1}".format(card_id, existing + 1)


def _operation_cost(unit: UnitState, card) -> int:
    if unit.status.get("covert"):
        return 1
    set_val = next((m["value"] for m in unit.modifiers if m.get("type") == "set_operation_cost"), None)
    if set_val is not None:
        return max(0, set_val)
    return max(0, (card.operationCost or 0) + sum(mod.get("amount", 0) for mod in unit.modifiers if mod.get("type") in {"modify_cost", "modify_operation_cost"}))


def card_play_cost(state: GameState, player_id: str, card, cards: CardDatabase) -> int:
    """Final kredit cost of playing ``card`` from hand, incl. cost modifiers."""
    player = state.players[player_id]
    total = card.kredits or 0
    hand_death_bonuses = player.status.get("hand_cost_on_friendly_death", {})
    if isinstance(hand_death_bonuses, dict):
        bonus = hand_death_bonuses.get(card.id, 0)
        if isinstance(bonus, int):
            total += max(0, bonus)
    for mod in player.cost_modifiers:
        if not _cost_mod_applies(mod, card, cards):
            continue
        if mod.get("set_cost") is not None:
            total = mod["set_cost"]
        else:
            total += mod.get("amount", 0)
    # Cards that re-price themselves from live state (DETAILED RECON,
    # SURPRISE ENGAGEMENT); see simulator.rules.dynamic_costs.
    from simulator.rules import dynamic_costs
    total -= dynamic_costs.reduction_for(state, player_id, card)
    enemy_draws = player.status.get("hand_cost_on_enemy_draw", {})
    if isinstance(enemy_draws, dict):
        total -= int(enemy_draws.get(card.id, 0) or 0)
    total = max(0, total)
    min_cost = max(
        (mod.get("min_cost", 0) for mod in player.cost_modifiers if _cost_mod_applies(mod, card, cards) and mod.get("min_cost")),
        default=0,
    )
    return max(total, min_cost)


def _cost_mod_applies(mod: dict, card, cards: CardDatabase) -> bool:
    scope = mod.get("scope", "all")
    fv = mod.get("filter_value", "")
    if scope == "all":
        return True
    if scope == "order":
        return card.type == "order"
    if scope == "exclude_nation":
        return (card.nation or "") != fv
    if scope == "ability":
        abilities = cards.get(card.id).abilities if card.id in cards else []
        return any(fv.lower() in ability.lower() for ability in abilities)
    if scope == "name":
        # Catalog text commonly uses a plural family name ("Spitfires")
        # while individual card names use its singular form.  Name-scoped
        # auras must not silently fall through to every card.
        normalized = fv.lower().strip().rstrip("s")
        return bool(normalized) and normalized in card.name.lower()
    return True


def apply_op_cost_rules(state: GameState, player_id: str, unit: UnitState, cards: CardDatabase) -> None:
    """Apply persistent operation-cost rules when a unit is deployed."""
    card = cards.get(unit.card_id)
    if card is None:
        return
    for rule in state.players[player_id].op_cost_rules:
        if not _op_rule_matches(rule, card):
            continue
        _apply_op_cost_rule_to_unit(unit, rule)


def _apply_op_cost_rule_to_unit(unit: UnitState, rule: dict) -> None:
    """Attach one operation-cost rule to a unit exactly once.

    Rules are materialized on units so the combat action can calculate its
    operation cost without reaching back into GameState.  Provenance and
    expiry are retained, allowing the native rule engine to remove the exact
    modifier when a temporary effect expires or its source leaves play.
    """
    rule_id = rule.get("rule_id")
    if rule_id and any(mod.get("op_rule_id") == rule_id for mod in unit.modifiers):
        return
    common = {
        "op_rule_id": rule_id,
        "expires_turn": rule.get("expires_turn"),
        "source_unit_id": rule.get("source_unit_id"),
    }
    if rule.get("set_cost") is not None:
        unit.modifiers.append({"type": "set_operation_cost", "value": rule["set_cost"], **common})
    else:
        unit.modifiers.append({"type": "modify_operation_cost", "amount": rule["amount"], **common})


def _op_rule_matches(rule: dict, card) -> bool:
    scope = rule.get("scope", "all")
    fv = (rule.get("filter_value") or "").lower()
    if scope == "all":
        return True
    if scope == "type":
        if fv == "air":
            return card.type in {"fighter", "bomber"}
        if fv == "ground":
            return card.type in {"infantry", "tank", "artillery"}
        return card.type == fv
    if scope in {"infantry", "fighter", "tank", "bomber", "artillery"}:
        return card.type == scope
    if scope == "air":
        return card.type in {"fighter", "bomber"}
    if scope == "ground":
        return card.type in {"infantry", "tank", "artillery"}
    if scope == "ability":
        return any(fv in ability.lower() for ability in card.abilities)
    if scope == "name":
        return fv in card.name.lower()
    return True


def _target_tax(state: GameState, target_unit_id: str | None, acting_player_id: str) -> int:
    """Return a target's verified additional targeting/attack cost."""
    if target_unit_id is None:
        return 0
    target = find_unit(state, target_unit_id)
    if target.owner_id == acting_player_id:
        return 0
    tax = target.status.get("target_or_attack_tax", 0)
    return tax if isinstance(tax, int) and tax > 0 else 0


def _cap_combat_damage(unit: UnitState, amount: int) -> int:
    """Apply a unit's persistent cap to incoming combat damage only.

    Rule effects such as 2nd California explicitly mention *combat* damage;
    order and triggered-effect damage must therefore continue through the
    ordinary damage resolver unchanged.
    """
    cap = unit.status.get("combat_damage_cap")
    if isinstance(cap, int) and cap >= 0:
        return min(amount, cap)
    return amount


def _can_move_and_attack(state: GameState, unit: UnitState, card, cards: CardDatabase) -> bool:
    """Whether a unit may take both move and attack operations this turn.

    KARDS Blitz allows a newly deployed unit to operate.  The official Blitz
    rule grants the move-and-attack combination to tanks; other unit types use
    one operation unless a card's explicit continuous text grants the combo.
    """
    if card.type == "tank" and KeywordEngine.can_attack_on_deploy(card, unit):
        return True
    if unit.status.get("can_move_and_attack"):
        return True
    owner = state.players[unit.owner_id]
    for source in owner.units:
        scopes = source.status.get("move_and_attack_auras", ())
        for scope in scopes:
            if scope == "alpine" and KeywordEngine.has(card, "alpine", unit):
                return True
    return False
