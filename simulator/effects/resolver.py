"""FIFO event dispatcher and interpreter for the KARDS effect DSL."""

from __future__ import annotations

from collections import deque
import random
from dataclasses import dataclass, field
from typing import Any, Mapping

from simulator.actions.validator import ActionValidationError, find_unit, opponent_id
from simulator.cards.loader import CardDatabase
from simulator.core.hq import HQResolver
from simulator.core.state import GameState, UnitState
from simulator.effects import basic_effects

# Ground (land) unit types for damage auras such as 110. PANZERGRENADIER.
_GROUND_TYPES = frozenset({"infantry", "tank", "artillery"})


@dataclass(frozen=True)
class EffectContext:
    player_id: str
    source_card_id: str | None = None
    source_unit_id: str | None = None
    target_unit_id: str | None = None
    event: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EffectResolver:
    """Executes rule-layer effect actions against runtime game state.

    Static cards are no longer expected to embed an ``effects`` field. Native
    kards.info text/abilities enter through ``RuleParser`` and handlers may pass
    explicit effect actions to this resolver.
    """

    def __init__(self, cards: CardDatabase) -> None:
        self.cards = cards

    def resolve(self, effect: Mapping[str, Any], state: GameState, context: EffectContext) -> None:
        """Resolve one effect document, supporting stable and legacy source shapes."""
        if state.game_status.value != "in_progress":
            return
        # Keep the public primitive layer consistent with native unit abilities:
        # an enemy unit ability may not affect a Covert target unless a caller
        # has explicitly revealed it first.
        if context.source_unit_id and context.target_unit_id:
            # An event can outlive a unit it refers to: for example, combat
            # damage can destroy the target before a later FIFO listener is
            # resolved.  Such a targeted effect has no legal target and must
            # fizzle rather than making the whole game invalid.
            try:
                source = find_unit(state, context.source_unit_id)
                target = find_unit(state, context.target_unit_id)
            except ActionValidationError:
                state.event_log.append({"event": "effect_target_no_longer_exists", "target_unit_id": context.target_unit_id})
                return
            if source is not None and target is not None and source.owner_id != target.owner_id and target.status.get("covert"):
                state.event_log.append({"event": "covert_effect_ignored", "target_unit_id": target.instance_id})
                return
        if not self._conditions_met(effect.get("conditions", effect.get("condition", [])), state, context):
            return
        for action in effect.get("actions", [effect]):
            if isinstance(action, Mapping):
                self._resolve_action(action, state, context)
                if state.game_status.value != "in_progress":
                    break
        from simulator.rules.intel import refresh as refresh_intel
        refresh_intel(state)

    def emit(self, event: str, state: GameState, context: EffectContext) -> None:
        """Drain event-triggered effects in FIFO order for deterministic simulations."""
        queue = deque([(event, context)])
        while queue:
            event_name, event_context = queue.popleft()
            state.event_log.append({"event": event_name, "player_id": event_context.player_id})
            # Effects are supplied by the rule/handler layer, not serialized in
            # cards. Keeping this queue makes event ordering stable for future
            # native-text parsers and reusable HandlerRegistry implementations.

    def _resolve_action(self, action: Mapping[str, Any], state: GameState, context: EffectContext) -> None:
        kind = str(action.get("type", action.get("action", "")))
        amount = _amount(action.get("value", action.get("parameters", {})))
        targets = self._targets(str(action.get("target", "self")), state, context)
        if kind == "damage":
            amount += self._order_damage_bonus(state, context)
            if context.source_card_id in self.cards and self.cards.get(context.source_card_id).type == "order":
                amount += int(state.players[context.player_id].status.pop("next_order_damage_bonus", 0) or 0)
            amount += self._unit_damage_bonus(state, context)
            amount = max(0, amount - self._enemy_card_damage_reduction(state, context))
            for target in targets:
                final_amount = amount
                if isinstance(target, UnitState) and self._friendly_order_damage_uses_armor(state, context, target):
                    from simulator.rules.keywords import KeywordEngine
                    final_amount = max(0, final_amount - KeywordEngine.heavy_armor(self.cards.get(target.card_id), target))
                basic_effects.damage(state, target, final_amount, context.player_id)
        elif kind == "heal":
            for target in targets:
                basic_effects.heal(state, target, amount)
        elif kind in {"buff", "debuff"}:
            values = action.get("value", {})
            attack = _amount(values.get("attack", values.get("amount", 0)))
            health = _amount(values.get("health", values.get("defense", values.get("amount", 0))))
            if kind == "debuff":
                attack, health = -abs(attack), -abs(health)
            for target in _units(targets):
                target.attack += attack
                target.defense += health
            basic_effects.remove_dead_units(state)
        elif kind in {"modify_attack", "modify_defense", "modify_health"}:
            for target in _units(targets):
                if kind == "modify_attack":
                    target.attack += amount
                else:
                    target.defense += amount
            basic_effects.remove_dead_units(state)
        elif kind in {"modify_cost", "modify_deployment_cost", "modify_operation_cost"}:
            for target in _units(targets):
                target.modifiers.append({"type": kind, "amount": amount})
        elif kind == "draw":
            basic_effects.draw(state, context.player_id, amount)
        elif kind == "discard":
            player = state.players[context.player_id]
            for _ in range(min(amount, len(player.hand))):
                card_id = player.hand.pop(0)
                state.graveyard.setdefault(context.player_id, []).append(card_id)
                state.event_log.append({"event": "card_discarded", "player_id": context.player_id, "card_id": card_id})
            from simulator.rules.intel import refresh
            refresh(state)
        elif kind == "destroy":
            for target in _units(targets):
                basic_effects.destroy(state, target)
        elif kind in {"move", "move_to_frontline", "move_to_support_line"}:
            position = "frontline" if kind == "move_to_frontline" else "support_line" if kind == "move_to_support_line" else str(action.get("position", "support_line"))
            for target in _units(targets):
                basic_effects.move(state, target, position)
        elif kind in {"spawn", "spawn_unit"}:
            self._spawn(state, context.player_id, str(action.get("card_reference", "")), str(action.get("position", "support_line")))
        elif kind == "suppress":
            # Deferred to avoid resolver -> rules package -> native -> resolver
            # import recursion during simulator startup.
            from simulator.rules import suppression
            for target in _units(targets):
                suppression.apply(target, self.cards.get(target.card_id))
        elif kind == "pin":
            from simulator.rules.pin import apply as apply_pin
            for target in _units(targets):
                apply_pin(target)
        elif kind in {"add_keyword", "remove_keyword"}:
            keyword = str(action.get("keyword", ""))
            for target in _units(targets):
                keywords = target.status.setdefault("keywords", [])
                if kind == "add_keyword" and keyword and keyword not in keywords: keywords.append(keyword)
                if kind == "remove_keyword" and keyword in keywords: keywords.remove(keyword)
        elif kind == "transform":
            card_id = str(action.get("card_reference", ""))
            for target in _units(targets):
                if card_id in self.cards and self.cards.get(card_id).is_unit:
                    card=self.cards.get(card_id); target.card_id, target.attack, target.defense=card_id, card.attack or 0, card.defense or 0
        elif kind == "copy_card":
            card_id=str(action.get("card_reference", context.source_card_id or ""))
            if card_id in self.cards: state.players[context.player_id].hand.append(card_id)
        elif kind in {"random", "choice"}:
            options=list(action.get("options", ()))
            if options:
                index=action.get("choice_index", 0)
                selected=options[int(index)] if kind == "choice" else random.Random(state.rng_seed).choice(options)
                if isinstance(selected, Mapping): self._resolve_action(selected, state, context)
        elif kind == "gain_hq_defense":
            HQResolver.modify_defense(state, context.player_id, amount)
        elif kind in {"", "keyword"}:
            return  # Source placeholders intentionally have no executable primitive.
        else:
            state.event_log.append({"event": "unsupported_effect", "type": kind, "source_card_id": context.source_card_id})

    def _spawn(self, state: GameState, player_id: str, card_id: str, position: str) -> None:
        if card_id not in self.cards or not self.cards.get(card_id).is_unit:
            state.event_log.append({"event": "spawn_failed", "card_id": card_id})
            return
        card = self.cards.get(card_id)
        instance_id = "{0}-spawn-{1}".format(card_id, sum(len(player.units) for player in state.players.values()) + 1)
        unit = UnitState(instance_id, card_id, card.attack or 0, card.defense or 0, player_id, position)
        state.players[player_id].units.append(unit)
        state.battlefield[position].append(instance_id)

    def _order_damage_bonus(self, state: GameState, context: EffectContext) -> int:
        """Sum verified in-play bonuses that modify damage dealt by Orders."""
        if context.source_card_id not in self.cards or self.cards.get(context.source_card_id).type != "order":
            return 0
        return sum(
            bonus for unit in state.players[context.player_id].units
            if isinstance((bonus := unit.status.get("order_damage_bonus")), int) and bonus > 0
        )

    def unit_damage_bonus(self, state: GameState, unit: UnitState | None) -> int:
        """Bonus that friendly auras add to damage dealt BY ``unit``.

        110. PANZERGRENADIER ("Your ground units with 4 or more attack deal +3
        damage") boosts every damage that the unit itself deals: attacks,
        counter-attacks and deployment effects alike (confirmed in client).
        Damage dealt by an Order has no source unit, so it is never boosted.
        """
        if unit is None:
            return 0
        card = self.cards.get(unit.card_id)
        total = 0
        for other in state.players[unit.owner_id].units:
            ground = other.status.get("ground_damage_bonus")
            if (isinstance(ground, dict) and card.type in _GROUND_TYPES
                    and unit.attack >= int(ground.get("minimum_attack", 4) or 4)):
                total += int(ground.get("amount", 0))
        return total

    def _unit_damage_bonus(self, state: GameState, context: EffectContext) -> int:
        """Apply in-play unit damage auras to damage dealt by a source unit."""
        if not context.source_unit_id:
            return 0
        try:
            source = find_unit(state, context.source_unit_id)
        except ActionValidationError:
            return 0
        total = self.unit_damage_bonus(state, source)
        # "noncombat" auras stay restricted to non-combat unit effects.
        if not context.metadata.get("combat_damage"):
            for unit in state.players[context.player_id].units:
                rule = unit.status.get("noncombat_unit_damage_bonus")
                if isinstance(rule, dict) and isinstance(rule.get("amount"), int):
                    total += rule["amount"]
        return total

    @staticmethod
    def _enemy_card_damage_reduction(state: GameState, context: EffectContext) -> int:
        """Return in-play reduction imposed by the defending player's units."""
        defender_id = opponent_id(state, context.player_id)
        return sum(
            value for unit in state.players[defender_id].units
            if isinstance((value := unit.status.get("enemy_card_damage_reduction")), int) and value > 0
        )

    def _friendly_order_damage_uses_armor(self, state: GameState, context: EffectContext, target: UnitState) -> bool:
        return (
            context.source_card_id in self.cards
            and self.cards.get(context.source_card_id).type == "order"
            and target.owner_id == context.player_id
            and any(unit.status.get("friendly_order_damage_armor") for unit in state.players[context.player_id].units)
        )

    def _targets(self, selector: str, state: GameState, context: EffectContext) -> list[UnitState | str]:
        # The selector vocabulary is declared in simulator.rules.selectors
        # (EFFECT_SELECTORS covers this method).  Imported lazily because
        # simulator.rules.__init__ pulls in the native engine, which imports
        # this module — the same cycle turn.py and native.py work around.
        from simulator.rules import selectors
        opponent = opponent_id(state, context.player_id)
        if selector in selectors.SOURCE and context.source_unit_id:
            return [find_unit(state, context.source_unit_id)]
        # selected_enemy / selected_friendly are the same chosen unit as
        # selected_target; without them the resolver returned no target and the
        # effect silently did nothing (163 rule actions use these two names).
        if selector in selectors.CHOSEN and context.target_unit_id:
            return [find_unit(state, context.target_unit_id)]
        if selector == "enemy_hq":
            return [opponent]
        if selector in {"owner", "friendly_hq"}:
            return [context.player_id]
        if selector in {"friendly", "friendly_units"}:
            return list(state.players[context.player_id].units)
        if selector in {"all_units", "all"}:
            return list(state.players[context.player_id].units) + list(state.players[opponent].units)
        if selector in {"enemy", "enemy_units"}:
            return list(state.players[opponent].units)
        return []

    @staticmethod
    def _conditions_met(conditions: Any, state: GameState, context: EffectContext) -> bool:
        if not conditions or conditions == {}:
            return True
        if isinstance(conditions, Mapping):
            conditions = [conditions]
        for condition in conditions:
            if not isinstance(condition, Mapping):
                continue
            kind = condition.get("type")
            if kind == "friendly_unit_exists" and not state.players[context.player_id].units:
                return False
            if kind == "enemy_unit_exists" and not state.players[opponent_id(state, context.player_id)].units:
                return False
            if kind == "has_unit_type":
                if not any(self_unit.card_id in self.cards and self.cards.get(self_unit.card_id).type == condition.get("unit_type") for self_unit in state.players[context.player_id].units): return False
            if kind == "has_keyword" and context.source_unit_id:
                unit=find_unit(state, context.source_unit_id)
                if condition.get("keyword") not in unit.status.get("keywords", []): return False
            if kind == "compare_resources":
                if state.players[context.player_id].resources.kredits < int(condition.get("amount", 0)): return False
        return True


def _amount(value: Any) -> int:
    if isinstance(value, Mapping):
        value = value.get("amount", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _units(targets: list[UnitState | str]) -> list[UnitState]:
    return [target for target in targets if isinstance(target, UnitState)]
