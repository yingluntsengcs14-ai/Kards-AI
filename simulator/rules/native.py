"""Native rule execution layer bridging card text ASTs to EffectResolver."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import random

from simulator.cards.loader import CardDatabase
from simulator.effects.resolver import EffectContext, EffectResolver
from simulator.effects import basic_effects
from simulator.actions.validator import ActionValidationError, find_unit, opponent_id
from simulator.core.hq import HQResolver
from simulator.core.state import UnitState
from simulator.rules.battlefield import BattlefieldRules, FRONTLINE_UNIT_LIMIT, HAND_CARD_LIMIT, SUPPORT_UNIT_LIMIT
from simulator.rules.parser import CardRule, RuleAction, RuleParser
from simulator.rules.store import CardRuleStore
from simulator.rules.condition import evaluate as condition_evaluate
from simulator.rules.keywords import KeywordEngine
from simulator.rules import selectors, suppression

# Lowercased nation names used by scoped combat filters (card.nation is
# capitalized, e.g. "Germany"; RuleAction.card_name carries the lowercased form).
_NATION_LOWER = {"britain", "germany", "usa", "soviet", "japan", "france", "italy", "poland", "finland"}

# simulator.rules.condition treats text it does not model as satisfied, so a
# conditional "Becomes Veteran" would swap the card on deploy.  Only the
# turn-boundary clauses that condition.evaluate models exactly are trusted to
# drive the card swap; any other clause leaves the promotion pending.
_VETERAN_MODELED_CONDITIONS = (
    "at the start of your turn",
    "at the end of your turn",
    "at the end of turn",
    # Weather conditions are modeled in simulator.rules.condition and gated on
    # the on_weather_played event, so they can be trusted to drive the swap.
    "you play a rain",
    "you play a storm",
    "you play a sunny",
)


@dataclass(frozen=True)
class ActionGenerationSpec:
    """Static action-shape facts compiled with each card's rule AST."""

    needs_target: bool
    swap_hand_unit: bool
    convert_to: bool
    shock_tactics: bool
    develop_options: tuple[str, ...]
    # True when the card forecasts: the player picks sunny / rain / storm, so
    # the action layer must offer one action per category.
    forecast: bool = False


def _action_generation_spec(rule: CardRule) -> ActionGenerationSpec:
    actions = rule.actions
    return ActionGenerationSpec(
        needs_target=rule.needs_target,
        swap_hand_unit=any(action.kind == "swap_hand_unit_with_friendly" for action in actions),
        convert_to=any(action.kind == "convert_to" for action in actions),
        shock_tactics=any(action.kind == "shock_tactics_choice" for action in actions),
        forecast=any(action.kind == "forecast" for action in actions),
        develop_options=tuple(
            option.strip()
            for action in actions if action.kind == "develop_options"
            for option in (action.card_name or "").split("|") if option.strip()
        ),
    )


def _expiry_turn(duration: str | None, current_turn: int) -> int | None:
    """Turn number at which a temporary effect reverts (None = permanent)."""
    if duration == "this_turn":
        return current_turn          # revert at end of the current turn
    if duration == "next_turn":
        return current_turn + 1      # revert at end of the player's next turn
    return None


def _resolve_units(target: str, state, context) -> list:
    """Resolve a target selector to a list of UnitState objects.

    The selectors handled here are declared in simulator.rules.selectors as
    UNIT_SELECTORS, which the rule-store generator checks against.
    """
    opp = opponent_id(state, context.player_id)
    if target in selectors.SOURCE and context.source_unit_id:
        unit = find_unit(state, context.source_unit_id)
        return [unit] if unit is not None else []
    if target in selectors.CHOSEN and context.target_unit_id:
        unit = find_unit(state, context.target_unit_id)
        return [unit] if unit is not None else []
    if target == "friendly_units":
        return list(state.players[context.player_id].units)
    if target == "enemy_units":
        return list(state.players[opp].units)
    if target in {"all_units", "all"}:   # selectors.GROUPS
        # "All units get +2 attack" covers both sides.  Without this the
        # selector resolved to nothing and the effect silently did nothing.
        return list(state.players[context.player_id].units) + list(state.players[opp].units)
    return []


def _unit_scope_matches(unit: UnitState, action: RuleAction, cards) -> bool:
    """Return True if a unit satisfies a copy_unit scope filter."""
    scope = action.scope or ""
    fv = (action.card_name or "").lower()
    card = cards.get(unit.card_id)
    if card is None:
        return False
    if scope in {"air", "ground", "tank", "infantry", "artillery", "bomber", "fighter"}:
        if scope == "air":
            return card.type in {"fighter", "bomber"}
        if scope == "ground":
            return card.type in {"infantry", "tank", "artillery"}
        return KeywordEngine.matches_type(card, unit, scope)
    if scope == "tank_or_infantry":
        return KeywordEngine.matches_type(card, unit, "tank") or KeywordEngine.matches_type(card, unit, "infantry")
    if scope == "guard":
        abilities = [a.lower() for a in (card.abilities or ())]
        added = [a.lower() for a in (unit.status.get("added_abilities") or [])]
        return "guard" in abilities or "guard" in added
    if scope == "veteran":
        return bool(unit.status.get("veteran"))
    if fv:
        return fv in (card.abilities or ())
    return True


class NativeRuleEngine:
    """Compile and execute only verified parser actions for native card text."""

    def __init__(self, cards: CardDatabase, parser: RuleParser | None = None, store: CardRuleStore | None = None) -> None:
        self.cards = cards
        self.parser = parser or RuleParser()
        self.store = store or CardRuleStore.from_file(default_rule_path())
        self._rules = {
            card.id: self._augment_event_triggers(self.store.get(card.id) or self.parser.parse(card))
            for card in cards
        }
        self._action_specs = {card_id: _action_generation_spec(rule) for card_id, rule in self._rules.items()}
        self._by_name = {card.name.upper(): card.id for card in cards}
        # Transient reference to the card selected by a choose_one action during
        # a single execute() run. Consumed by subsequent actions with
        # target="chosen" (Pattern B/C linkage). Reset per execute().
        self._chosen: tuple[str, str] | None = None

    @staticmethod
    def _augment_event_triggers(rule: CardRule) -> CardRule:
        """Correct legacy rule-store trigger labels from unambiguous text.

        Reviewed rules predate the dedicated friendly-card-play event for
        Navy-card listeners.  The action AST remains unchanged; only the
        event route is added, keeping card data and rule-store schema stable.
        """
        if "when you play a navy card" not in rule.source_text.lower():
            return rule
        if "on_friendly_card_played" in rule.triggers:
            return rule
        return replace(rule, triggers=rule.triggers + ("on_friendly_card_played",))

    def rule_for(self, card_id: str) -> CardRule:
        return self._rules[card_id]

    def action_spec_for(self, card_id: str) -> ActionGenerationSpec:
        """Return cached static facts used by the high-frequency action API."""
        return self._action_specs[card_id]

    def execute(self, card_id: str, event: str, state, context: EffectContext) -> bool:
        if context.source_unit_id:
            # ``emit`` deliberately iterates a stable listener snapshot.  A
            # preceding listener may remove a later source unit, in which
            # case that later trigger no longer exists and simply fizzles.
            try:
                source = find_unit(state, context.source_unit_id)
            except ActionValidationError:
                state.event_log.append({"event": "native_rule_source_no_longer_exists", "card_id": card_id,
                                        "trigger": event, "source_unit_id": context.source_unit_id})
                return False
            if source is not None and source.status.get("suppressed"):
                return False
        rule = self.rule_for(card_id)
        if context.target_unit_id and context.source_unit_id:
            try:
                target = find_unit(state, context.target_unit_id)
                source = find_unit(state, context.source_unit_id)
            except ActionValidationError:
                target = source = None
            if target is not None and source is not None and target.owner_id != source.owner_id and target.status.get("covert"):
                if not any(action.card_name == "can_target_covert" for action in rule.actions):
                    return False
        if event not in rule.triggers:
            return False
        # "Once per turn" belongs to the card rule, not necessarily to an
        # individual parsed action clause.  Guard only after confirming that
        # at least one action's event condition is eligible; this avoids a
        # unit consuming its allowance merely because it was deployed.
        once_key = None
        if "once per turn" in rule.source_text.lower() and context.source_unit_id:
            eligible = any(
                action.condition is None
                or condition_evaluate(action.condition, event, context, self.cards, state)
                for action in rule.actions
            )
            source = find_unit(state, context.source_unit_id)
            if eligible and source is not None:
                once_key = "{0}|{1}".format(card_id, event)
                used = source.status.setdefault("once_per_turn_rules", {})
                if used.get(once_key) == state.turn_number:
                    state.event_log.append({"event": "native_rule_once_per_turn_used", "card_id": card_id})
                    return False
                used[once_key] = state.turn_number
        executed = False
        self._current_event = event
        self._chosen = None
        # Snapshot the targeted unit's card_id so delayed effects that fire
        # *after* a paired "Remove X" action can still resolve their referent.
        ctx = context
        if context.target_unit_id:
            try:
                tu = find_unit(state, context.target_unit_id)
                if tu is not None:
                    ctx = replace(context, metadata={**context.metadata, "target_card_id": tu.card_id, "target_owner_id": tu.owner_id})
            except Exception:
                pass
        if "reveal target covert" in rule.source_text.lower() and context.target_unit_id:
            try:
                target = find_unit(state, context.target_unit_id)
            except ActionValidationError:
                target = None
            if target is not None:
                target.status.pop("covert", None)
        for action in rule.actions:
            try:
                self._execute_action(action, state, ctx)
            except ActionValidationError as error:
                # A multi-clause card can remove its selected unit before a
                # later clause runs.  The later clause then has no target and
                # legally fizzles; it must not abort the enclosing game.
                stale_ids = {candidate for candidate in (ctx.source_unit_id, ctx.target_unit_id) if candidate}
                if stale_ids and any(candidate in str(error) for candidate in stale_ids):
                    state.event_log.append({"event": "native_rule_stale_reference", "card_id": card_id,
                                            "trigger": event, "detail": str(error)})
                    continue
                raise
            executed = True
            if state.game_status.value != "in_progress":
                break
        # Native rules contain several hand mutations (return, exchange,
        # generated cards). Reconcile Intel once the atomic rule resolves.
        from simulator.rules.intel import refresh as refresh_intel
        refresh_intel(state)
        state.event_log.append({"event": "native_rule_executed", "card_id": card_id, "trigger": event, "status": rule.status})
        return executed

    def emit(self, event: str, state, context: EffectContext) -> int:
        """Run matching triggers on a stable snapshot of battlefield units."""
        fired = 0
        units = [unit for player in state.players.values() for unit in tuple(player.units)]
        event_metadata = {
            **context.metadata,
            "event_player_id": context.player_id,
            "event_source_unit_id": context.source_unit_id,
        }
        if context.target_unit_id:
            try:
                target = find_unit(state, context.target_unit_id)
                if target is not None:
                    event_metadata["event_target_owner_id"] = target.owner_id
            except Exception:
                # A preceding effect may have removed the target (for example
                # an order followed by a command-played event).  The event is
                # still valid; it simply has no living target owner.
                pass
        for unit in units:
            if state.game_status.value != "in_progress":
                break
            if unit.card_id not in self.cards:
                continue
            rule = self.rule_for(unit.card_id)
            if event in {"on_turn_start", "on_turn_end"} and unit.owner_id != context.player_id:
                continue
            if event == "on_card_drawn" and unit.owner_id != context.player_id:
                continue
            if event == "on_friendly_card_played" and unit.owner_id != context.player_id:
                continue
            if event in {"on_targeted_by_enemy_effect", "on_targeted_by_enemy_attack"} and context.target_unit_id and unit.instance_id != context.target_unit_id:
                continue
            if event == "on_attack" and context.source_unit_id and unit.instance_id != context.source_unit_id:
                if not self._is_battlefield_listener(rule, event):
                    continue
            if event == "on_deploy":
                # A unit's own Deployment text is executed directly by
                # PlayCardAction.  This broadcast is only for cards which
                # explicitly listen for another unit entering play.
                if context.metadata.get("broadcast_listeners"):
                    if not self._is_battlefield_listener(rule, event):
                        continue
                elif context.source_unit_id and unit.instance_id != context.source_unit_id:
                    continue
            if event in {"on_damage_dealt", "on_combat_damage_dealt", "on_enemy_hq_damaged", "on_survives_combat"} and context.source_unit_id and unit.instance_id != context.source_unit_id:
                continue
            if event == "on_damage" and context.target_unit_id and unit.instance_id != context.target_unit_id:
                continue
            listener_target_id = context.target_unit_id
            # For "When an enemy unit attacks, Retreat it", the pronoun
            # refers to the attacker rather than the defended unit.  Friendly
            # attacked listeners, in contrast, intentionally retain the
            # defender as their selected target.
            if event == "on_attack" and "enemy unit attacks" in rule.source_text.lower():
                listener_target_id = context.source_unit_id
            unit_context = EffectContext(
                unit.owner_id, unit.card_id, unit.instance_id,
                listener_target_id, event, event_metadata,
            )
            if self.execute(unit.card_id, event, state, unit_context):
                fired += 1
        state.event_log.append({"event": "native_event_emitted", "trigger": event, "fired": fired})
        return fired

    @staticmethod
    def _is_battlefield_listener(rule: CardRule, event: str) -> bool:
        """Whether text describes an event caused by another battlefield unit.

        The parser shares event names between a unit's own triggers and global
        listeners.  Keeping this decision here prevents a deployment/attack
        broadcast from replaying every existing unit's own text.
        """
        text = rule.source_text.lower()
        if event == "on_attack":
            return any(phrase in text for phrase in (
                "friendly unit is attacked", "enemy unit attacks", "enemy attacks",
                "when a unit attacks", "when another unit attacks",
            ))
        if event == "on_deploy":
            return any(phrase in text for phrase in (
                "enemy deploys", "enemy unit is deployed", "when a unit is deployed",
                "when you deploy", "when you deploy or add",
                "moves into the frontline", "moves to the frontline",
            ))
        return False

    def emit_deaths_since(self, state, start_index: int) -> int:
        """Dispatch destruction rules recorded by primitive state transitions."""
        deaths = [entry for entry in state.event_log[start_index:] if entry.get("event") == "unit_died"]
        fired = 0
        for death in deaths:
            card_id = death.get("card_id")
            player_id = death.get("player_id")
            unit_id = death.get("unit_id")
            if isinstance(card_id, str) and isinstance(player_id, str) and card_id in self.cards:
                # Hand-reactive cards observe only deaths that occur while a
                # copy is actually in their owner's hand.  The status map is
                # keyed by card ID because identical catalog copies have no
                # distinct identity in the current state representation.
                player = state.players[player_id]
                for hand_card_id in set(player.hand):
                    hand_rule = self.rule_for(hand_card_id) if hand_card_id in self.cards else None
                    if hand_rule and any(action.kind == "hand_cost_on_friendly_death" for action in hand_rule.actions):
                        bonuses = player.status.setdefault("hand_cost_on_friendly_death", {})
                        bonuses[hand_card_id] = int(bonuses.get(hand_card_id, 0)) + 1
                        state.event_log.append({"event": "hand_unit_cost_increased_on_death", "player_id": player_id,
                                                "card_id": hand_card_id, "amount": bonuses[hand_card_id]})
                if self.execute(card_id, "on_destroy", state, EffectContext(player_id, card_id, event="on_destroy")):
                    fired += 1
                # The destroyed card's own Destruction effect above is not a
                # broadcast.  Remaining units may still listen to a friendly
                # or enemy unit being destroyed.
                destroyed_context = EffectContext(
                    player_id, card_id, unit_id, unit_id, event="on_destroy",
                    metadata={"event_player_id": player_id, "event_source_unit_id": unit_id},
                )
                for listener_player in state.players.values():
                    for listener in tuple(listener_player.units):
                        if listener.card_id not in self.cards:
                            continue
                        listener_rule = self.rule_for(listener.card_id)
                        listener_text = listener_rule.source_text.lower()
                        if not ("friendly unit is destroyed" in listener_text or "enemy unit is destroyed" in listener_text):
                            continue
                        listener_context = EffectContext(
                            listener.owner_id, listener.card_id, listener.instance_id,
                            unit_id, event="on_destroy", metadata=destroyed_context.metadata,
                        )
                        if self.execute(listener.card_id, "on_destroy", state, listener_context):
                            fired += 1
                if unit_id:
                    self._cleanup_continuous_effects(state, player_id, unit_id)
        return fired

    def apply_deployment_name_auras(self, state, unit: UnitState) -> int:
        """Apply generic in-play name-scoped deployment auras to ``unit``.

        An aura is stored on its source unit rather than copied onto the card
        database.  This makes it naturally disappear with its source and lets
        future cards reuse the same rule shape ("your <name> get +X+Y when
        deployed") without adding card-specific action code.
        """
        deployed_card = self.cards.get(unit.card_id)
        if deployed_card is None:
            return 0
        applied = 0
        owner = state.players[unit.owner_id]
        for source in tuple(owner.units):
            for aura in tuple(source.status.get("deployment_name_auras", ())):
                name = str(aura.get("name", "")).strip().lower().rstrip("s")
                if not name or name not in deployed_card.name.lower():
                    continue
                attack = aura.get("attack", 0)
                defense = aura.get("defense", 0)
                if not isinstance(attack, int) or not isinstance(defense, int):
                    continue
                unit.attack += attack
                unit.defense += defense
                applied += 1
                state.event_log.append({
                    "event": "deployment_name_aura_applied",
                    "source_unit_id": source.instance_id,
                    "unit_id": unit.instance_id,
                    "name": aura.get("name"),
                    "attack": attack,
                    "defense": defense,
                })
        return applied

    def emit_draws_since(self, state, start_index: int) -> int:
        """Dispatch draw triggers for newly logged successful draw events."""
        fired = 0
        for entry in state.event_log[start_index:]:
            if entry.get("event") != "card_drawn":
                continue
            player_id = entry.get("player_id")
            if isinstance(player_id, str):
                fired += self.emit("on_card_drawn", state, EffectContext(player_id, event="on_card_drawn"))
        return fired

    def emit_damage_since(self, state, start_index: int) -> int:
        """Dispatch damage triggers for surviving units that took damage."""
        fired = 0
        for entry in state.event_log[start_index:]:
            if entry.get("event") != "damage_dealt":
                continue
            target_id = entry.get("target_id")
            if not isinstance(target_id, str):
                continue
            if any(unit.instance_id == target_id for player in state.players.values() for unit in player.units):
                fired += self.emit("on_damage", state, EffectContext("", target_unit_id=target_id, event="on_damage", metadata={"damage_amount": entry.get("amount", 0)}))
        return fired

    @staticmethod
    def _cleanup_continuous_effects(state, player_id: str, unit_id: str) -> None:
        """Remove cost_modifier / op_cost_rule entries tied to a unit that left play."""
        if not unit_id:
            return
        player = state.players[player_id]
        before = len(player.cost_modifiers), len(player.op_cost_rules)
        player.cost_modifiers = [
            m for m in player.cost_modifiers
            if m.get("source_unit_id") != unit_id
        ]
        player.op_cost_rules = [
            r for r in player.op_cost_rules
            if r.get("source_unit_id") != unit_id
        ]
        for candidate_player in state.players.values():
            for unit in candidate_player.units:
                unit.modifiers = [
                    modifier for modifier in unit.modifiers
                    if modifier.get("source_unit_id") != unit_id
                ]
        after = len(player.cost_modifiers), len(player.op_cost_rules)
        removed_cm = before[0] - after[0]
        removed_op = before[1] - after[1]
        if removed_cm or removed_op:
            state.event_log.append({
                "event": "continuous_effects_cleaned_up",
                "player_id": player_id,
                "unit_id": unit_id,
                "cost_modifier_count": removed_cm,
                "op_cost_rule_count": removed_op,
            })

    def _execute_action(self, action: RuleAction, state, context: EffectContext) -> None:
        # Conditional actions are skipped when their condition is not satisfied
        # for the current event/context (M8 condition engine).
        if action.condition is not None and not condition_evaluate(action.condition, self._current_event, context, self.cards, state):
            state.event_log.append({"event": "native_rule_condition_false", "kind": action.kind, "condition": action.condition})
            return
        # Delayed one-shot effects (M10): schedule a discard/return-to-hand to a
        # future turn phase instead of executing immediately. The trigger phase
        # is carried in action.duration ("end_of_turn" / "start_of_next_turn" /
        # "end_of_next_turn"); "end of your next turn" skips one end_turn (wait=1).
        if action.kind in ("delayed_discard", "delayed_return"):
            self._schedule_delayed(action, state, context)
            return
        if action.kind == "scheduled_repair":
            state.players[context.player_id].scheduled.append({
                "trigger": action.duration or "end_of_turn",
                "kind": "repair_all_friendly",
                "owner_id": context.player_id,
                "repairs": [
                    {"unit_id": unit.instance_id, "defense": self.cards.get(unit.card_id).defense}
                    for unit in state.players[context.player_id].units
                    if self.cards.get(unit.card_id) is not None
                ],
            })
            state.event_log.append({"event": "rule_scheduled", "trigger": action.duration or "end_of_turn", "kind": "repair_all_friendly", "player_id": context.player_id})
            return
        # Temporary (duration-bounded) stat/status effects are applied and their
        # reversal is registered so TurnManager can revert them at the right turn.
        if action.duration and action.kind in ("buff", "modify_attack", "modify_defense", "set_attack", "suppress", "grant_ability"):
            self._apply_temporary(action, state, context)
            return
        # Scoped (multi-target) combat/attribute actions carry a type/nation/ability
        # filter in `scope`/`card_name`; the generic EffectResolver cannot filter by
        # those, so the engine resolves and applies them directly here.
        if action.kind in ("damage", "buff", "modify_attack", "modify_defense", "suppress", "destroy", "heal") and (action.scope or action.card_name):
            self._apply_scoped_combat(action, state, context)
            return
        if action.kind == "reset_operation" and context.target_unit_id:
            find_unit(state, context.target_unit_id).status.pop("attack_count", None)
            return
        if action.kind == "retreat_and_repair" and context.target_unit_id:
            unit = find_unit(state, context.target_unit_id)
            owner = state.players[unit.owner_id]
            if unit.position != "frontline" or sum(candidate.position == "support_line" for candidate in owner.units) >= 4:
                state.event_log.append({"event": "native_rule_retreat_failed", "unit_id": unit.instance_id})
                return
            state.battlefield["frontline"].remove(unit.instance_id)
            state.battlefield["support_line"].append(unit.instance_id)
            unit.position = "support_line"
            unit.defense = self.cards.get(unit.card_id).defense or unit.defense
            state.event_log.append({"event": "unit_retreated_and_repaired", "unit_id": unit.instance_id})
            return
        if action.kind == "destroy_undamaged" and context.target_unit_id:
            unit = find_unit(state, context.target_unit_id)
            card = self.cards.get(unit.card_id) if unit is not None else None
            if unit is None or card is None or unit.defense != card.defense:
                state.event_log.append({"event": "native_rule_undamaged_destroy_failed", "unit_id": context.target_unit_id})
                return
            start = len(state.event_log)
            basic_effects.destroy(state, unit)
            self.emit_deaths_since(state, start)
            return
        if action.kind == "trigger_destruction_effects" and context.target_unit_id:
            target = find_unit(state, context.target_unit_id)
            if target is None:
                return
            # Resolve the target's own destruction text under the caster's
            # control, matching the card's "as if it was yours" wording.
            self.execute(target.card_id, "on_destroy", state,
                         EffectContext(context.player_id, target.card_id,
                                       target.instance_id, target.instance_id,
                                       event="on_destroy",
                                       metadata={"triggered_as_friendly": True,
                                                 "original_owner": target.owner_id}))
            state.event_log.append({"event": "destruction_effects_triggered",
                                    "unit_id": target.instance_id,
                                    "as_player": context.player_id})
            return
        if action.kind == "rearrange_enemy_units":
            enemy_id = opponent_id(state, context.player_id)
            enemy = state.players[enemy_id]
            units = [u for u in enemy.units]
            random.Random((state.rng_seed or 0) + len(state.event_log)).shuffle(units)
            # Rearrangement changes battlefield ordering without moving units
            # between support/frontline lines or exceeding line capacity.
            for position in ("support_line", "frontline"):
                ids = [u.instance_id for u in units if u.position == position]
                state.battlefield[position] = ids
            state.event_log.append({"event": "enemy_units_rearranged", "player_id": context.player_id})
            return
        if action.kind == "develop_options":
            options = [name.strip() for name in (action.card_name or "").split("|") if name.strip()]
            if not options:
                return
            requested = context.metadata.get("selected_option")
            chosen_name = requested if requested in options else random.Random((state.rng_seed or 0) + len(state.event_log)).choice(options)
            card_id = self._by_name.get(chosen_name.upper())
            if card_id is None:
                state.event_log.append({"event": "develop_failed", "name": chosen_name})
                return
            card = self.cards.get(card_id)
            if card.is_unit:
                self._deploy_named_card(state, context, card_id)
            else:
                self._add_to_hand(state, context.player_id, card_id)
            state.event_log.append({"event": "develop_selected", "card_id": card_id})
            return
        if action.kind == "grant_ability":
            ability = (action.card_name or "").strip()
            if not ability:
                return
            # Resolve target units: source, specific target, or all friendly.
            if action.target == "friendly_units":
                targets = state.players[context.player_id].units
            elif context.target_unit_id:
                unit = find_unit(state, context.target_unit_id)
                targets = [unit] if unit and not (action.scope and not _unit_scope_matches(unit, action, self.cards)) else []
            elif context.source_unit_id:
                unit = find_unit(state, context.source_unit_id)
                targets = [unit] if unit else []
            elif action.target == "selected_friendly":
                # best-effort: grant to a friendly frontline unit
                for u in state.players[context.player_id].units:
                    if u.position == "frontline":
                        targets = [u]
                        break
                else:
                    return
            else:
                return
            for unit_obj in targets:
                granted = unit_obj.status.setdefault("added_abilities", [])
                if ability not in granted:
                    granted.append(ability)
                if ability.lower() == "veteran":
                    unit_obj.status["veteran"] = True
                    # A conditional promotion ("Becomes Veteran when you have
                    # destroyed 4 enemy units") must not swap the card on deploy;
                    # see _VETERAN_MODELED_CONDITIONS.  The condition itself was
                    # already evaluated before this action ran.
                    condition_text = (action.condition or "").lower()
                    if action.condition is None or any(
                            phrase in condition_text for phrase in _VETERAN_MODELED_CONDITIONS):
                        self._promote_to_veteran(state, unit_obj)
                    else:
                        state.event_log.append({"event": "veteran_promotion_pending",
                                                "unit_id": unit_obj.instance_id,
                                                "condition": action.condition})
            state.event_log.append({"event": "ability_granted", "ability": ability, "target": action.target})
            return
        if action.kind == "buff_on_enemy_target" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            unit.attack += action.attack
            unit.defense += action.defense
            return
        if action.kind == "attack_on_order_played" and context.source_unit_id:
            played_card_id = context.metadata.get("played_card_id")
            if isinstance(played_card_id, str) and played_card_id in self.cards and self.cards.get(played_card_id).type == "order":
                find_unit(state, context.source_unit_id).attack += action.amount
            return
        if action.kind == "buff_on_played_ability" and context.source_unit_id:
            played_card_id = context.metadata.get("played_card_id")
            if not isinstance(played_card_id, str) or played_card_id not in self.cards:
                return
            abilities = self.cards.get(played_card_id).abilities
            required = action.card_name or ""
            if required == "intel":
                matches = any(ability.startswith("intel") for ability in abilities)
            else:
                matches = required in abilities
            if matches:
                unit = find_unit(state, context.source_unit_id)
                unit.attack += action.attack
                unit.defense += action.defense
            return
        if action.kind == "spawn_named_same_front" and context.source_unit_id:
            source = find_unit(state, context.source_unit_id)
            card_id = self._by_name.get((action.card_name or "").upper())
            player = state.players[source.owner_id]
            if not card_id or not self.cards.get(card_id).is_unit:
                state.event_log.append({"event": "native_rule_unresolved_card_name", "name": action.card_name, "source_card_id": context.source_card_id})
                return
            if source.position == "support_line" and not BattlefieldRules.can_deploy_to_support(state, source.owner_id):
                state.event_log.append({"event": "rule_spawn_failed", "reason": "support_line_full", "source_unit_id": source.instance_id})
                return
            if source.position == "frontline":
                frontline_count = sum(unit.position == "frontline" for owner in state.players.values() for unit in owner.units)
                if frontline_count >= FRONTLINE_UNIT_LIMIT or BattlefieldRules.frontline_controller(state) not in (None, source.owner_id):
                    state.event_log.append({"event": "rule_spawn_failed", "reason": "frontline_full_or_uncontrolled", "source_unit_id": source.instance_id})
                    return
            card = self.cards.get(card_id)
            instance_id = "{0}-rule-{1}".format(card_id, sum(len(candidate.units) for candidate in state.players.values()) + 1)
            player.units.append(UnitState(instance_id, card_id, card.attack or 0, card.defense or 0, source.owner_id, source.position))
            state.battlefield[source.position].append(instance_id)
            state.event_log.append({"event": "rule_unit_spawned", "card_id": card_id, "player_id": source.owner_id, "position": source.position})
            return
        if action.kind == "order_damage_bonus" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            unit.status[action.kind] = action.amount
            return
        # ── Stat multipliers ────────────────────────────────────────────────
        if action.kind == "double_stats":
            targets = _resolve_units(action.target, state, context)
            for unit in targets:
                unit.attack *= 2
                unit.defense *= 2
            state.event_log.append({"event": "stats_doubled", "target_count": len(targets)})
            return
        if action.kind == "double_damage":
            # Mark the source to deal double combat damage.  Consumed by
            # the combat resolution pipeline when calculating final damage.
            if context.source_unit_id:
                unit = find_unit(state, context.source_unit_id)
                if unit:
                    unit.status["double_damage"] = True
            state.event_log.append({"event": "double_damage_activated", "source_unit_id": context.source_unit_id})
            return
        if action.kind == "triple_damage":
            if context.source_unit_id:
                unit = find_unit(state, context.source_unit_id)
                if unit:
                    unit.status["triple_damage"] = True
            state.event_log.append({"event": "triple_damage_activated", "source_unit_id": context.source_unit_id})
            return
        if action.kind == "cannot" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                restriction = (action.card_name or "").lower()
                # Sanitize: parser may use underscores for multi-word restrictions
                restriction = restriction.replace("_", " ")
                # Order matters: check multi-word restrictions before single-word
                if "retreat" in restriction and "suppressed" in restriction:
                    unit.status["cannot_retreat"] = True
                    unit.status["cannot_be_suppressed"] = True
                elif "attack" in restriction and ("unit" in restriction or "units" in restriction):
                    unit.status["cannot_attack_units"] = True
                elif "attack" in restriction and "move" in restriction:
                    unit.status["cannot_attack"] = True
                    unit.status["cannot_move"] = True
                elif "pinned" in restriction and "suppressed" in restriction:
                    unit.status["cannot_be_pinned"] = True
                    unit.status["cannot_be_suppressed"] = True
                elif "retreat" in restriction:
                    unit.status["cannot_retreat"] = True
                elif "suppressed" in restriction:
                    unit.status["cannot_be_suppressed"] = True
                elif "pinned" in restriction:
                    unit.status["cannot_be_pinned"] = True
                elif "attack" in restriction:
                    unit.status["cannot_attack"] = True
                elif "deployed" in restriction:
                    unit.status["deploy_restriction"] = restriction
                elif "targeted" in restriction or "be_targeted" in restriction:
                    unit.status["cannot_be_targeted"] = True
                else:
                    unit.status["cannot"] = restriction
            state.event_log.append({"event": "cannot_applied", "source_unit_id": context.source_unit_id,
                                    "restriction": action.card_name})
            return
        # ── Immune / cannot_attack_hq (consumed by damage pipeline / attack validation) ──
        if action.kind == "immune" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["immune"] = True
            state.event_log.append({"event": "immune_applied", "source_unit_id": context.source_unit_id})
            return
        if action.kind == "cannot_attack_hq" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["cannot_attack_hq"] = True
            state.event_log.append({"event": "cannot_attack_hq_applied", "source_unit_id": context.source_unit_id})
            return
        if action.kind == "combat_damage_cap" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["combat_damage_cap"] = max(0, action.amount)
            state.event_log.append({
                "event": "combat_damage_cap_applied",
                "source_unit_id": context.source_unit_id,
                "amount": action.amount,
            })
            return
        if action.kind == "random_combat_damage" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["random_combat_damage"] = max(0, action.amount)
            state.event_log.append({
                "event": "random_combat_damage_applied",
                "source_unit_id": context.source_unit_id,
                "maximum_bonus": action.amount,
            })
            return
        if action.kind == "hq_damage_reduction_by_attack" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["hq_damage_reduction_by_attack"] = {
                    "amount": max(0, action.amount),
                    "minimum_attack": max(0, action.min_cost),
                }
            state.event_log.append({
                "event": "hq_damage_reduction_by_attack_applied",
                "source_unit_id": context.source_unit_id,
                "amount": action.amount,
                "minimum_attack": action.min_cost,
            })
            return
        if action.kind in {"noncombat_unit_damage_bonus", "ground_damage_bonus"} and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status[action.kind] = {"amount": action.amount, "minimum_attack": action.min_cost}
            return
        if action.kind == "enemy_card_damage_reduction" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["enemy_card_damage_reduction"] = max(0, action.amount)
            return
        if action.kind == "friendly_order_damage_armor" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["friendly_order_damage_armor"] = True
            return
        if action.kind == "hq_defense_becomes_damage" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit is not None:
                unit.status["hq_defense_becomes_damage"] = True
            return
        if action.kind == "hq_damage_redirect_to_source" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit is not None:
                unit.status["hq_damage_redirect_to_source"] = True
            return
        if action.kind == "hq_damage_redirect_to_enemy_hq" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit is not None:
                unit.status["hq_damage_redirect_to_enemy_hq"] = True
            return
        if action.kind == "redirect_named_unit_damage" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            protected_card_id = self._by_name.get((action.card_name or "").upper())
            if unit is not None and protected_card_id is not None:
                unit.status["redirect_damage_from_card_id"] = protected_card_id
            return
        if action.kind == "replay_non_targeting_deployment":
            candidates = _resolve_units(action.target, state, context)
            replayed = 0
            for unit in candidates:
                # Replaying the source would allow a self-referential replay
                # card to recurse forever. Non-targeting deployment text on
                # other friendly units is the actual reusable effect here.
                if unit.instance_id == context.source_unit_id:
                    continue
                rule = self.rule_for(unit.card_id)
                if "on_deploy" not in rule.triggers or rule.needs_target:
                    continue
                self.execute(unit.card_id, "on_deploy", state,
                             EffectContext(unit.owner_id, unit.card_id, unit.instance_id,
                                           event="on_deploy", metadata={"replayed_deployment": True}))
                replayed += 1
            state.event_log.append({"event": "non_targeting_deployment_replayed", "count": replayed, "player_id": context.player_id})
            return
        if action.kind == "swap_hand_unit_with_friendly" and context.target_unit_id:
            selected_card_id = context.metadata.get("selected_card_id")
            target = find_unit(state, context.target_unit_id)
            if not isinstance(selected_card_id, str) or target is None or target.owner_id != context.player_id:
                state.event_log.append({"event": "hand_battlefield_swap_failed", "reason": "invalid_selection"})
                return
            owner = state.players[context.player_id]
            selected_card = self.cards.get(selected_card_id) if selected_card_id in self.cards else None
            if selected_card is None or not selected_card.is_unit or selected_card_id not in owner.hand:
                state.event_log.append({"event": "hand_battlefield_swap_failed", "reason": "invalid_hand_unit"})
                return
            owner.hand.remove(selected_card_id)
            owner.units.remove(target)
            owner.hand.append(target.card_id)
            position = target.position
            instance_id = "{0}-swap-{1}".format(selected_card_id, len(state.event_log) + 1)
            replacement = UnitState(instance_id, selected_card_id, selected_card.attack or 0,
                                    selected_card.defense or 0, context.player_id, position)
            owner.units.append(replacement)
            line = state.battlefield[position]
            line[line.index(target.instance_id)] = replacement.instance_id
            state.event_log.append({"event": "hand_battlefield_unit_swapped", "player_id": context.player_id,
                                    "hand_card_id": selected_card_id, "returned_card_id": target.card_id,
                                    "unit_id": replacement.instance_id})
            return
        if action.kind == "shock_tactics_choice" and context.target_unit_id:
            target = find_unit(state, context.target_unit_id)
            target_card = self.cards.get(target.card_id) if target is not None else None
            if target is None or target_card is None or target_card.type not in {"infantry", "tank", "artillery"}:
                state.event_log.append({"event": "shock_tactics_failed", "reason": "invalid_target"})
                return
            if any(unit.attack >= 4 for unit in state.players[context.player_id].units):
                abilities = ("blitz", "shock")
            else:
                option = context.metadata.get("selected_option")
                if option not in {"blitz", "shock"}:
                    state.event_log.append({"event": "shock_tactics_failed", "reason": "choice_required"})
                    return
                abilities = (option,)
            granted = target.status.setdefault("added_abilities", [])
            for ability in abilities:
                if ability not in granted:
                    granted.append(ability)
            state.event_log.append({"event": "shock_tactics_applied", "unit_id": target.instance_id, "abilities": list(abilities)})
            return
        if action.kind == "suppress_deployment_effects" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit is not None:
                unit.status["suppress_deployment_effects"] = True
            return
        if action.kind == "random_effect_target_priority" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit is not None:
                unit.status["random_effect_target_priority"] = True
            return
        if action.kind == "trigger_destroy_and_convert" and context.target_unit_id:
            target = find_unit(state, context.target_unit_id)
            target_card = self.cards.get(target.card_id) if target is not None else None
            yak_id = self._by_name.get((action.card_name or "YAK 9").upper())
            yak = self.cards.get(yak_id) if yak_id else None
            if (target is None or target_card is None or yak is None
                    or target_card.nation != "Soviet" or target_card.type not in {"fighter", "bomber"}
                    or target.defense >= (target_card.defense or 0)):
                state.event_log.append({"event": "destroy_convert_failed", "unit_id": context.target_unit_id})
                return
            # The destruction rule belongs to the target's owner.  It is
            # deliberately invoked before replacement, while the original
            # card identity is still available to its trigger resolver.
            self.execute(target.card_id, "on_destroy", state,
                         EffectContext(target.owner_id, target.card_id, target.instance_id,
                                       event="on_destroy", metadata={"triggered_by": context.source_card_id}))
            target.card_id = yak.id
            target.attack = yak.attack or 0
            target.defense = yak.defense or 0
            target.modifiers.clear()
            target.status.clear()
            state.event_log.append({"event": "unit_converted", "unit_id": target.instance_id,
                                    "from_card_id": target_card.id, "to_card_id": yak.id})
            return
        if action.kind == "retreat_enemy_frontline_all":
            enemy_id = opponent_id(state, context.player_id)
            enemy = state.players[enemy_id]
            frontline_units = [unit for unit in enemy.units if unit.position == "frontline"]
            free_slots = max(0, 4 - sum(unit.position == "support_line" for unit in enemy.units))
            # Return only as many support units as are needed to make room.
            # A full hand discards the displaced support unit instead.
            needed = max(0, len(frontline_units) - free_slots)
            supports = [unit for unit in enemy.units if unit.position == "support_line"]
            for unit in reversed(supports[:needed]):
                if unit.instance_id in state.battlefield["support_line"]:
                    state.battlefield["support_line"].remove(unit.instance_id)
                enemy.units.remove(unit)
                if len(enemy.hand) < 9:
                    enemy.hand.append(unit.card_id)
                    event = "black_prince_support_returned"
                else:
                    state.graveyard.setdefault(enemy_id, []).append(unit.card_id)
                    event = "black_prince_support_discarded"
                self._cleanup_continuous_effects(state, enemy_id, unit.instance_id)
                state.event_log.append({"event": event, "unit_id": unit.instance_id, "card_id": unit.card_id})
            # Retreat frontline units in stable front-line order.  Any unit
            # still without a support slot is discarded as specified.
            for unit in tuple(frontline_units):
                if unit.instance_id in state.battlefield["frontline"]:
                    state.battlefield["frontline"].remove(unit.instance_id)
                if sum(candidate.position == "support_line" for candidate in enemy.units) < 4:
                    unit.position = "support_line"
                    state.battlefield["support_line"].append(unit.instance_id)
                    state.event_log.append({"event": "black_prince_frontline_retreated", "unit_id": unit.instance_id})
                else:
                    enemy.units.remove(unit)
                    state.graveyard.setdefault(enemy_id, []).append(unit.card_id)
                    self._cleanup_continuous_effects(state, enemy_id, unit.instance_id)
                    state.event_log.append({"event": "black_prince_frontline_discarded", "unit_id": unit.instance_id, "card_id": unit.card_id})
            return
        if action.kind == "damage_equal_to":
            # The amount is a named quantity (see simulator.rules.references);
            # RuleAction has a fixed shape, so the reference travels in
            # card_name and an optional nation filter in scope.
            from simulator.rules import references
            amount = references.evaluate(action.card_name or "", state, context, self.cards,
                                         action.scope or None)
            if amount is None:
                state.event_log.append({"event": "native_rule_reference_unresolved",
                                        "reference": action.card_name,
                                        "source_card_id": context.source_card_id})
                return
            EffectResolver(self.cards).resolve(
                {"type": "damage", "target": action.target, "value": {"amount": max(0, int(amount))}},
                state, context)
            state.event_log.append({"event": "damage_equal_to", "reference": action.card_name,
                                    "amount": int(amount), "target": action.target})
            return
        if action.kind == "discard_leftmost":
            from simulator.rules import hand_position
            hand_position.discard_leftmost(state, context.player_id)
            return
        if action.kind == "move_hand_card_leftmost":
            from simulator.rules import hand_position
            chosen = context.metadata.get("selected_card_id")
            if not chosen:
                state.event_log.append({"event": "native_rule_no_hand_card_chosen",
                                        "source_card_id": context.source_card_id})
                return
            hand_position.move_to_leftmost(state, context.player_id, str(chosen))
            return
        if action.kind == "distribute_defense":
            # DELUGE: "Distribute 6 defense between random friendly targets."
            # Points are handed out one at a time so the split is uneven, as the
            # wording implies; the HQ counts as a friendly target.
            owner = context.player_id
            targets = list(state.players[owner].units)
            rng = random.Random((state.rng_seed or 0) + len(state.event_log))
            granted: dict[str, int] = {}
            for _ in range(max(0, action.amount)):
                choice = rng.randrange(len(targets) + 1)
                if choice == len(targets):
                    HQResolver.modify_defense(state, owner, 1)
                    granted["hq"] = granted.get("hq", 0) + 1
                else:
                    unit = targets[choice]
                    unit.defense += 1
                    granted[unit.instance_id] = granted.get(unit.instance_id, 0) + 1
            state.event_log.append({"event": "weather_defense_distributed", "player_id": owner,
                                    "amount": action.amount, "granted": granted})
            return
        if action.kind == "damage_own_hq_equal_to_hand":
            # CYCLONE: "deal damage to your HQ equal to cards in your hand."
            owner = context.player_id
            amount = len(state.players[owner].hand)
            if amount:
                HQResolver.damage(state, owner, amount)
            state.event_log.append({"event": "weather_self_hq_damage", "player_id": owner,
                                    "amount": amount})
            return
        if action.kind == "forecast":
            from simulator.rules import weather
            # The player chooses the weather category; the cost tier is random.
            # Without a choice (a direct engine call, or an agent that did not
            # pick) the category is drawn at random.
            chosen = str(context.metadata.get("selected_option") or "")
            weather.forecast(state, context.player_id, weather.TOKEN_FOR_CATEGORY.get(chosen))
            # "Your other units get +1+1 when you Forecast", "Draw a card when
            # you Forecast" listen for this event.  emit() does not log the
            # broadcast itself, so the record is written here.
            state.event_log.append({"event": "on_forecast", "player_id": context.player_id})
            self.emit("on_forecast", state, EffectContext(
                context.player_id, context.source_card_id, context.source_unit_id,
                event="on_forecast"))
            return
        if action.kind == "frontline_attack_bonus" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["frontline_attack_bonus"] = action.amount
            return
        if action.kind == "frontline_limit" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["frontline_limit"] = action.amount
            return
        if action.kind == "move_and_attack" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit is not None:
                unit.status["can_move_and_attack"] = True
            return
        if action.kind == "move_and_attack_aura" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit is not None:
                scopes = unit.status.setdefault("move_and_attack_auras", [])
                if action.scope and action.scope not in scopes:
                    scopes.append(action.scope)
            return
        if action.kind == "next_order_damage_bonus":
            player = state.players[context.player_id]
            player.status["next_order_damage_bonus"] = max(0, action.amount)
            player.temporary_effects.append({"turn": state.turn_number, "reverts": [{"player_status": "next_order_damage_bonus"}]})
            return
        if action.kind == "swap_with_friendly" and context.source_unit_id and context.target_unit_id:
            source = find_unit(state, context.source_unit_id)
            target = find_unit(state, context.target_unit_id)
            if not source or not target or source.owner_id != target.owner_id or source.position == target.position:
                return
            source_position, target_position = source.position, target.position
            state.battlefield[source_position].remove(source.instance_id)
            state.battlefield[target_position].remove(target.instance_id)
            state.battlefield[source_position].append(target.instance_id)
            state.battlefield[target_position].append(source.instance_id)
            source.position, target.position = target_position, source_position
            bonus = source.status.get("frontline_attack_bonus")
            if source.position == "frontline" and isinstance(bonus, int) and not source.status.get("frontline_attack_bonus_applied"):
                source.attack += bonus
                source.status["frontline_attack_bonus_applied"] = True
            state.event_log.append({"event": "friendly_units_swapped", "source_unit_id": source.instance_id, "target_unit_id": target.instance_id})
            return
        # ── Type-specific combat bonuses ─────────────────────────────────────
        if action.kind == "double_damage_against_type" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["double_damage_against_type"] = action.scope or action.card_name or "tank"
            state.event_log.append({"event": "double_damage_against_type", "source_unit_id": context.source_unit_id})
            return
        if action.kind == "attack_bonus_against_type" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["attack_bonus_against_type"] = {"type": action.card_name, "amount": action.amount}
            state.event_log.append({"event": "attack_bonus_against_type", "source_unit_id": context.source_unit_id})
            return
        if action.kind == "attack_bonus_against_higher_attack" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["attack_bonus_against_higher_attack"] = True
            state.event_log.append({"event": "attack_bonus_against_higher_attack", "source_unit_id": context.source_unit_id})
            return
        if action.kind == "damage_bonus_against_air" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["damage_bonus_against_air"] = True
            state.event_log.append({"event": "damage_bonus_against_air", "source_unit_id": context.source_unit_id})
            return
        # ── Targeting / action restrictions ──────────────────────────────────
        if action.kind == "cannot_be_targeted_by_enemy_orders" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["cannot_be_targeted_by_enemy_orders"] = True
            state.event_log.append({"event": "cannot_be_targeted", "source_unit_id": context.source_unit_id})
            return
        if action.kind == "target_or_attack_tax" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["target_or_attack_tax"] = action.amount or 2
            state.event_log.append({"event": "target_or_attack_tax", "source_unit_id": context.source_unit_id, "amount": action.amount})
            return
        # ── HQ excess damage ─────────────────────────────────────────────────
        if action.kind == "hq_excess" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["hq_excess"] = True
            state.event_log.append({"event": "hq_excess_active", "source_unit_id": context.source_unit_id})
            return
        # ── Destroy with cost filter ─────────────────────────────────────────
        if action.kind == "destroy_cost_lte" and context.target_unit_id:
            unit = find_unit(state, context.target_unit_id)
            card = self.cards.get(unit.card_id)
            max_cost = action.amount
            if card and (card.kredits or 0) <= max_cost:
                basic_effects_run = __import__('simulator.effects.basic_effects', fromlist=['destroy'])
                basic_effects_run.destroy(state, unit)
            state.event_log.append({"event": "destroy_cost_check", "target_unit_id": context.target_unit_id, "max_cost": max_cost})
            return
        # ── Trait granting (navy type, also tank, etc.) ─────────────────────
        if action.kind == "grant_trait" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                trait = (action.card_name or "").lower()
                if "tank" in trait:
                    unit.status["also_tank"] = True
                elif "navy" in trait:
                    unit.status["navy_type"] = True
                else:
                    unit.status["granted_trait"] = trait
            state.event_log.append({"event": "trait_granted", "source_unit_id": context.source_unit_id, "trait": action.card_name})
            return
        # ── Repeat effect ────────────────────────────────────────────────────
        if action.kind == "repeat_effect" and context.source_card_id:
            repeat_key = f"{context.source_card_id}_{self._current_event}"
            already_repeated = any(
                e.get("event") == "effect_repeated" and e.get("source_card_id") == context.source_card_id
                for e in state.event_log
            )
            if not already_repeated:
                state.event_log.append({"event": "effect_repeated", "source_card_id": context.source_card_id})
                self.execute(context.source_card_id, self._current_event, state, context)
            return
        # ── Aura buff ────────────────────────────────────────────────────────
        if action.kind == "aura_buff":
            scope = (action.scope or "").lower()
            buffed = 0
            for unit in state.players[context.player_id].units:
                if scope:
                    card = self.cards.get(unit.card_id)
                    if not card:
                        continue
                    if scope in ("tank", "infantry", "artillery", "fighter", "bomber"):
                        if card.type != scope:
                            continue
                    elif scope == "air":
                        if card.type not in ("fighter", "bomber"):
                            continue
                    elif scope == "ground":
                        if card.type not in ("infantry", "tank", "artillery"):
                            continue
                    elif scope == "support_line" and unit.position != "support_line":
                        continue
                    elif scope == "frontline" and unit.position != "frontline":
                        continue
                if action.attack:
                    unit.attack += action.attack
                if action.defense:
                    unit.defense += action.defense
                if action.amount:
                    unit.attack += action.amount
                if context.source_unit_id:
                    unit.status.setdefault("aura_sources", []).append(context.source_unit_id)
                buffed += 1
            state.event_log.append({"event": "aura_buff_applied", "player_id": context.player_id, "units_buffed": buffed, "scope": scope})
            return
        # ── Cancel (countermeasure) ───────────────────────────────────────────
        # Sets a cancellation flag that the action validation layer consumes.
        if action.kind == "cancel":
            if context.source_card_id:
                state.pending_cancels.append({
                    "source_card_id": context.source_card_id,
                    "player_id": context.player_id,
                    "min_cost": getattr(action, "min_cost", 0),
                    "cancel_targets_friendly": "friendly" in (action.target or ""),
                })
            state.event_log.append({"event": "cancel_primed", "source_card_id": context.source_card_id})
            return
        # ── Pincer ability ────────────────────────────────────────────────────
        # Applies pincer-linked effects to the source unit (and pincer partner
        # where applicable).  Pincer partners share an adjacent front position.
        if action.kind == "pincer_ability" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                inner = (action.card_name or "").strip().lower()
                if "-1 operation cost" in inner:
                    unit.modifiers.append({"type": "set_operation_cost", "value": max(0, (self.cards.get(unit.card_id).operationCost or 0) - 1)})
                elif "+1 heavy armor" in inner:
                    unit.defense += 1
                elif "cannot be pinned" in inner:
                    unit.status["cannot_be_pinned"] = True
                elif "ambush" in inner:
                    unit.status.setdefault("added_abilities", []).append("ambush")
                elif "blitz" in inner:
                    unit.status.setdefault("added_abilities", []).append("blitz")
                elif "destroyed" in inner and "deck" in inner:
                    unit.status["pincer_return_to_deck"] = True
                elif "damage" in inner and "partner" in inner:
                    unit.status["pincer_damage_redirect"] = True
                elif "target or attack" in inner:
                    unit.status["target_or_attack_tax"] = 2
                else:
                    unit.status["pincer_ability"] = inner
            state.event_log.append({"event": "pincer_applied", "source_unit_id": context.source_unit_id, "effect": action.card_name})
            return
        # ── Target selection ──────────────────────────────────────────────────
        # Marks a unit for special target selection rules (e.g., "random effects
        # always choose this unit").
        if action.kind == "target_select" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["target_select"] = action.card_name or "always_chosen"
                if action.card_name == "can_target_covert":
                    unit.status["can_target_covert"] = True
            state.event_log.append({"event": "target_select_applied", "source_unit_id": context.source_unit_id})
            return
        # ── Enemy action restrictions ────────────────────────────────────────
        if action.kind == "enemy_cannot_deploy":
            opp = opponent_id(state, context.player_id)
            state.players[opp].status["enemy_cannot_deploy"] = True
            state.event_log.append({"event": "enemy_cannot_deploy", "player_id": opp})
            return
        if action.kind == "enemy_cannot_order":
            opp = opponent_id(state, context.player_id)
            state.players[opp].status["enemy_cannot_order"] = True
            state.event_log.append({"event": "enemy_cannot_order", "player_id": opp})
            return
        # ── Buff variants ────────────────────────────────────────────────────
        if action.kind == "buff_on_played_ability":
            if context.source_unit_id:
                unit = find_unit(state, context.source_unit_id)
                if unit:
                    unit.status["buff_on_played_ability"] = action.card_name or True
            state.event_log.append({"event": "buff_on_played_ability", "source_unit_id": context.source_unit_id})
            return
        if action.kind == "buff_on_enemy_target" and context.target_unit_id:
            unit = find_unit(state, context.target_unit_id)
            if unit and action.attack:
                unit.attack += action.attack
            if unit and action.defense:
                unit.defense += action.defense
            state.event_log.append({"event": "buff_on_enemy_target", "target_unit_id": context.target_unit_id})
            return
        if action.kind == "attack_on_order_played" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["attack_on_order_played"] = True
            state.event_log.append({"event": "attack_on_order_played", "source_unit_id": context.source_unit_id})
            return
        # ── Miscellaneous combat/HQ flags ─────────────────────────────────────
        if action.kind == "immune_to_ground_in_support" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["immune_to_ground_in_support"] = True
            state.event_log.append({"event": "immune_to_ground", "source_unit_id": context.source_unit_id})
            return
        if action.kind == "ignore_heavy_armor" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["ignore_heavy_armor"] = True
            state.event_log.append({"event": "ignore_heavy_armor", "source_unit_id": context.source_unit_id})
            return
        if action.kind == "hq_damage_bonus" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["hq_damage_bonus"] = True
            state.event_log.append({"event": "hq_damage_bonus", "source_unit_id": context.source_unit_id})
            return
        if action.kind == "countermeasure_lock":
            source = find_unit(state, context.source_unit_id) if context.source_unit_id else None
            if source:
                source.status["countermeasure_lock"] = True
                state.event_log.append({"event": "countermeasure_lock", "source_unit_id": source.instance_id})
            return
        if action.kind == "hq_defense_lock" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["hq_defense_lock"] = True
            state.event_log.append({"event": "hq_defense_lock", "source_unit_id": context.source_unit_id})
            return
        if action.kind == "hq_defense_cap" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["hq_defense_cap"] = True
            state.event_log.append({"event": "hq_defense_cap", "source_unit_id": context.source_unit_id})
            return
        # ── Resource modification ─────────────────────────────────────────────
        if action.kind == "set_kredit_slots_equal":
            mine = state.players[context.player_id].resources
            opp = state.players[opponent_id(state, context.player_id)].resources
            gained = max(0, opp.max_kredits - mine.max_kredits)
            mine.max_kredits = max(mine.max_kredits, opp.max_kredits)
            mine.kredits = min(mine.max_kredits, mine.kredits + gained)
            state.event_log.append({"event": "kredit_slots_equalized", "player_id": context.player_id})
            return
        if action.kind == "enemy_kredit_change":
            opp = opponent_id(state, context.player_id)
            state.players[opp].resources.kredits = max(0, state.players[opp].resources.kredits + action.amount)
            state.event_log.append({"event": "enemy_kredits_changed", "player_id": opp, "amount": action.amount})
            return
        if action.kind == "kredit_effect" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.status["kredit_effect"] = action.card_name or True
            state.event_log.append({"event": "kredit_effect", "source_unit_id": context.source_unit_id})
            return
        # ── Delayed return ────────────────────────────────────────────────────
        if action.kind == "delayed_return":
            if context.source_unit_id:
                unit = find_unit(state, context.source_unit_id)
                if unit:
                    unit.status["delayed_return"] = action.duration or "next_turn"
            state.event_log.append({"event": "delayed_return_scheduled", "source_unit_id": context.source_unit_id})
            return
        # ── Modify deployment cost ────────────────────────────────────────────
        if action.kind == "modify_cost" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            if unit:
                unit.modifiers.append({"type": "modify_cost", "amount": action.amount})
            state.event_log.append({"event": "cost_modified", "source_unit_id": context.source_unit_id})
            return
        if action.kind == "attack_bonus_against_type" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            unit.status[action.kind] = {"type": action.card_name, "amount": action.amount}
            return
        if action.kind == "gain_hq_defense_equal_damage":
            amount = context.metadata.get("damage_amount", 0)
            if isinstance(amount, int) and amount > 0:
                HQResolver.modify_defense(state, context.player_id, amount)
            return
        # ── Engine-stubbed kinds ────────────────────────────────────────────
        # These action kinds are produced by the parser's residual fallback
        # (_resolve_residual / _classify_residual) but the engine has NO
        # implementation that produces real, consumable game state changes.
        # They are caught here solely to prevent runtime crashes; the card
        # will be correctly marked as "partial" by the honest coverage tools.
        #
        # ANY kind added here MUST also appear in:
        #   1. _CATCH_ALL_KINDS in simulator/rules/parser.py
        #   2. The "INTENTIONALLY absent" comment in tools/build_card_rules.py
        #
        # When a real implementation is written, MOVE the kind to its own
        # handler block ABOVE this section and remove it from both lists.
        if action.kind in {
            "control_effect",
            "hq_effect",
            "special_effect",
            "gain_kredit_slot_on_deploy",
        }:
            state.event_log.append({
                "event": "engine_stubbed_action",
                "kind": action.kind,
                "source_card_id": context.source_card_id,
                "card_name": action.card_name,
            })
            return
        if action.kind == "fight" and context.source_unit_id and context.target_unit_id:
            source = find_unit(state, context.source_unit_id)
            target = find_unit(state, context.target_unit_id)
            resolver = EffectResolver(self.cards)
            start = len(state.event_log)
            resolver.resolve({"type": "damage", "target": "selected_target", "value": {"amount": source.attack}}, state, EffectContext(context.player_id, source.card_id, source.instance_id, target.instance_id))
            self.emit_damage_since(state, start)
            self.emit_deaths_since(state, start)
            if source.instance_id in {unit.instance_id for player in state.players.values() for unit in player.units} and target.instance_id in {unit.instance_id for player in state.players.values() for unit in player.units}:
                start = len(state.event_log)
                resolver.resolve({"type": "damage", "target": "self", "value": {"amount": target.attack}}, state, EffectContext(target.owner_id, target.card_id, target.instance_id, source.instance_id))
                self.emit_damage_since(state, start)
                self.emit_deaths_since(state, start)
            return
        if action.kind == "destroy_cost_lte":
            action = RuleAction("destroy", action.target)
        if action.kind == "choose_one":
            self._handle_choose_one(action, state, context)
            return
        if action.kind == "end_turn":
            from simulator.core.turn import TurnManager
            TurnManager.end_turn(state, context.player_id)
            return
        if action.kind == "add_card":
            if action.card_name == "chosen_copy":
                # Add a copy of the card selected by a preceding choose_one.
                if self._chosen is None:
                    state.event_log.append({"event": "choose_one_no_selection", "source_card_id": context.source_card_id})
                    return
                self._add_to_hand(state, context.player_id, self._chosen[0])
                return
            card_id = self._by_name.get((action.card_name or "").upper())
            if card_id:
                target_player = opponent_id(state, context.player_id) if action.target == "enemy_hand" else context.player_id
                state.players[target_player].hand.append(card_id)
                state.event_log.append({"event": "card_added_to_hand", "player_id": target_player, "card_id": card_id})
            else:
                state.event_log.append({"event": "native_rule_unresolved_card_name", "name": action.card_name, "source_card_id": context.source_card_id})
            return
        if action.kind == "draw_matching" and action.card_name in {"unit", "order"}:
            player = state.players[context.player_id]
            candidates = [card_id for card_id in player.deck if card_id in self.cards and (self.cards.get(card_id).is_unit if action.card_name == "unit" else self.cards.get(card_id).type == "order")]
            if candidates:
                card_id = random.Random((state.rng_seed or 0) + len(state.event_log)).choice(candidates)
                player.deck.remove(card_id)
                if len(player.hand) < 9:
                    player.hand.append(card_id)
                    state.event_log.append({"event": "card_drawn", "player_id": context.player_id, "card_id": card_id})
                else:
                    state.graveyard.setdefault(context.player_id, []).append(card_id)
                    state.event_log.append({"event": "card_overdrawn", "player_id": context.player_id, "card_id": card_id})
            else:
                state.event_log.append({"event": "native_rule_no_matching_card", "selector": action.card_name})
            return
        if action.kind in {"buff_nation", "defense_nation"}:
            for unit in state.players[context.player_id].units:
                if unit.card_id in self.cards and self.cards.get(unit.card_id).nation == action.card_name:
                    if action.kind == "buff_nation":
                        unit.attack += action.attack
                        unit.defense += action.defense
                    else:
                        unit.defense += action.amount
            return
        if action.kind == "deploy_named":
            card_id = self._by_name.get((action.card_name or "").upper())
            player = state.players[context.player_id]
            if card_id and self.cards.get(card_id).is_unit:
                count = action.amount if action.amount and action.amount > 0 else 1
                deployed = 0
                for _ in range(count):
                    if sum(unit.position == "support_line" for unit in player.units) >= 4:
                        state.event_log.append({"event": "native_rule_support_line_full", "name": action.card_name})
                        break
                    card = self.cards.get(card_id)
                    instance_id = "{0}-rule-{1}".format(card_id, sum(len(candidate.units) for candidate in state.players.values()) + 1)
                    player.units.append(UnitState(instance_id, card_id, card.attack or 0, card.defense or 0, context.player_id))
                    state.battlefield["support_line"].append(instance_id)
                    deployed += 1
                if deployed:
                    state.event_log.append({"event": "rule_unit_deployed", "card_id": card_id, "player_id": context.player_id, "count": deployed})
                else:
                    state.event_log.append({"event": "native_rule_deploy_failed", "name": action.card_name})
            else:
                state.event_log.append({"event": "native_rule_deploy_failed", "name": action.card_name})
            return
        if action.kind == "add_to_enemy_deck":
            card_id = self._by_name.get((action.card_name or "").upper())
            if card_id:
                opponent = opponent_id(state, context.player_id)
                deck = state.players[opponent].deck
                deck.append(card_id)
                random.Random((state.rng_seed or 0) + len(state.event_log)).shuffle(deck)
                state.event_log.append({"event": "card_added_to_enemy_deck", "player_id": opponent, "card_id": card_id})
            return
        if action.kind == "shuffle_named":
            card_id = self._by_name.get((action.card_name or "").upper())
            if card_id:
                deck = state.players[context.player_id].deck
                deck.append(card_id)
                random.Random((state.rng_seed or 0) + len(state.event_log)).shuffle(deck)
                state.event_log.append({"event": "card_shuffled_into_deck", "player_id": context.player_id, "card_id": card_id})
            else:
                state.event_log.append({"event": "native_rule_unresolved_card_name", "name": action.card_name, "source_card_id": context.source_card_id})
            return
        if action.kind == "gain_kredits":
            player = state.players[context.player_id]
            player.resources.kredits = min(player.resources.max_kredits, player.resources.kredits + action.amount)
            return
        if action.kind == "gain_kredit_slot":
            owner = opponent_id(state, context.player_id) if action.target == "enemy_hand" else context.player_id
            player = state.players[owner]
            added = min(action.amount, 12 - player.resources.max_kredits)
            player.resources.max_kredits += added
            player.resources.kredits = min(player.resources.max_kredits, player.resources.kredits + added)
            state.event_log.append({"event": "kredit_slot_gained", "player_id": owner, "amount": added})
            return
        if action.kind == "lose_kredit_slot":
            owner = opponent_id(state, context.player_id) if action.target == "enemy_hand" else context.player_id
            player = state.players[owner]
            lost = min(action.amount, player.resources.max_kredits)
            player.resources.max_kredits -= lost
            from simulator.rules import counters
            counters.note_kredit_slots_lost(state, owner, lost)
            player.resources.kredits = min(player.resources.kredits, player.resources.max_kredits)
            state.event_log.append({"event": "kredit_slot_lost", "player_id": owner, "amount": lost})
            return
        if action.kind == "lose_kredits":
            owner = opponent_id(state, context.player_id) if action.target == "enemy_hand" else context.player_id
            player = state.players[owner]
            lost = min(action.amount, player.resources.kredits)
            player.resources.kredits -= lost
            state.event_log.append({"event": "kredits_lost", "player_id": owner, "amount": lost})
            return
        if action.kind == "spend_kredits":
            owner = context.player_id
            player = state.players[owner]
            spent = min(action.amount, player.resources.kredits)
            player.resources.kredits -= spent
            state.event_log.append({"event": "kredits_spent", "player_id": owner, "amount": spent})
            return
        if action.kind == "swap_attack_opcost" and context.source_unit_id:
            unit = find_unit(state, context.source_unit_id)
            card = self.cards.get(unit.card_id)
            base_op = (card.operationCost or 0) if card else 0
            for mod in unit.modifiers:
                if mod.get("type") == "set_operation_cost":
                    base_op = mod.get("value", base_op)
            unit.attack, new_op = base_op, unit.attack
            unit.modifiers.append({"type": "set_operation_cost", "value": new_op})
            state.event_log.append({"event": "attack_opcost_swapped", "unit_id": unit.instance_id})
            return
        if action.kind == "modify_hand_cost":
            owner = opponent_id(state, context.player_id) if action.target == "enemy_hand" else context.player_id
            state.players[owner].cost_modifiers.append({
                "kind": "hand_cost",
                "amount": action.amount,
                "set_cost": action.set_cost,
                "scope": action.scope or "all",
                "filter_value": action.card_name or "",
                "min_cost": action.min_cost,
                "expires_turn": _expiry_turn(action.duration, state.turn_number),
                "source_unit_id": context.source_unit_id,
            })
            state.event_log.append({"event": "hand_cost_modified", "player_id": owner, "amount": action.amount, "set_cost": action.set_cost})
            return
        if action.kind == "buff_deployed_matching_name" and context.source_unit_id:
            source = find_unit(state, context.source_unit_id)
            if source is not None:
                source.status.setdefault("deployment_name_auras", []).append({
                    "name": action.card_name or "",
                    "attack": action.attack,
                    "defense": action.defense,
                })
            return
        if action.kind in {"op_cost_rule", "operation_cost_rule"}:
            owner = opponent_id(state, context.player_id) if action.target == "enemy_hand" else context.player_id
            player = state.players[owner]
            rule_id = "op-rule-{0}-{1}".format(context.source_unit_id or context.source_card_id or "rule", len(player.op_cost_rules) + 1)
            rule = {
                "rule_id": rule_id,
                "amount": action.amount,
                "set_cost": action.set_cost,
                "scope": action.scope or "all",
                "filter_value": action.card_name or "",
                "expires_turn": _expiry_turn(action.duration, state.turn_number),
                "source_unit_id": context.source_unit_id,
            }
            player.op_cost_rules.append(rule)
            # A rule such as "your units operate for 1 less this turn" applies
            # immediately to units already on the battlefield as well as those
            # deployed later this turn.
            from simulator.actions.action import _apply_op_cost_rule_to_unit, _op_rule_matches
            for unit in player.units:
                card = self.cards.get(unit.card_id)
                if card is not None and _op_rule_matches(rule, card):
                    _apply_op_cost_rule_to_unit(unit, rule)
            state.event_log.append({"event": "op_cost_rule_added", "player_id": owner, "amount": action.amount})
            return
        if action.kind == "set_operation_cost":
            for unit in _resolve_units(action.target, state, context):
                unit.modifiers.append({"type": "set_operation_cost", "value": action.amount})
            state.event_log.append({"event": "operation_cost_set", "target": action.target, "value": action.amount})
            return
        if action.kind == "hq_take_damage":
            owner = opponent_id(state, context.player_id) if action.target == "enemy_hand" else context.player_id
            HQResolver.damage(state, owner, action.amount, context.player_id)
            return
        if action.kind == "set_hq_defense":
            owner = opponent_id(state, context.player_id) if action.target == "enemy_hand" else context.player_id
            state.players[owner].hq.defense_modifier = action.amount
            state.event_log.append({"event": "hq_defense_set", "player_id": owner, "amount": action.amount})
            return
        if action.kind == "hq_immune":
            owner = opponent_id(state, context.player_id) if action.target == "enemy_hand" else context.player_id
            state.players[owner].hq.immune_until_end_of_turn = True
            state.event_log.append({"event": "hq_immune_set", "player_id": owner})
            return
        if action.kind == "gain_hq_defense_equal_cost":
            card = self.cards.get(context.source_card_id) if context.source_card_id else None
            value = (card.operationCost or card.kredits or 0) if card else 0
            HQResolver.modify_defense(state, context.player_id, value)
            return
        if action.kind == "discard":
            if action.target == "chosen":
                # Discard the card selected by a preceding choose_one (Pattern B).
                if self._chosen is None:
                    state.event_log.append({"event": "choose_one_no_selection", "source_card_id": context.source_card_id})
                    return
                cid, owner = self._chosen
                player = state.players[owner]
                if cid in player.hand:
                    player.hand.remove(cid)
                    state.graveyard.setdefault(owner, []).append(cid)
                    state.event_log.append({"event": "card_discarded", "player_id": owner, "card_id": cid})
                return
            owner = opponent_id(state, context.player_id) if action.target == "enemy_hand" else context.player_id
            player = state.players[owner]
            for _ in range(min(action.amount, len(player.hand))):
                card_id = player.hand.pop(0)
                state.graveyard.setdefault(owner, []).append(card_id)
                state.event_log.append({"event": "card_discarded", "player_id": owner, "card_id": card_id})
            return
        if action.kind == "discard_hand":
            owner = opponent_id(state, context.player_id) if action.target == "enemy_hand" else context.player_id
            player = state.players[owner]
            count = 0
            while player.hand:
                cid = player.hand.pop(0)
                state.graveyard.setdefault(owner, []).append(cid)
                count += 1
            state.event_log.append({"event": "hand_discarded", "player_id": owner, "count": count})
            return
        if action.kind == "mill":
            owner = opponent_id(state, context.player_id) if action.target == "enemy_hand" else context.player_id
            player = state.players[owner]
            removed = 0
            for _ in range(min(action.amount, len(player.deck))):
                cid = player.deck.pop(0)
                state.removed_cards.append(cid)
                removed += 1
            state.event_log.append({"event": "cards_milled", "player_id": owner, "count": removed})
            return
        if action.kind == "copy_unit":
            self._copy_unit(action, state, context)
            return
        if action.kind == "draw_top_matching":
            self._draw_top_matching(action, state, context)
            return
        if action.kind == "return_to_hand" and context.target_unit_id:
            unit = find_unit(state, context.target_unit_id)
            owner = state.players[unit.owner_id]
            if unit.instance_id in state.battlefield.get(unit.position, []):
                state.battlefield[unit.position].remove(unit.instance_id)
            owner.units.remove(unit)
            owner.hand.append(unit.card_id)
            state.event_log.append({"event": "unit_returned_to_hand", "unit_id": unit.instance_id})
            return
        if action.kind == "attack_equals_defense" and context.target_unit_id:
            find_unit(state, context.target_unit_id).attack = find_unit(state, context.target_unit_id).defense
            return
        if action.kind in ("set_defense", "set_attack") and action.target in (
                "friendly_units", "enemy_units", "all_units", "all"):
            # "The defense of each enemy unit is set to 1" — a group form of the
            # single-target set below.
            for unit in _resolve_units(action.target, state, context):
                if action.kind == "set_defense":
                    unit.defense = action.amount
                else:
                    unit.attack = action.amount
            basic_effects.remove_dead_units(state)
            state.event_log.append({"event": "rule_" + action.kind + "_group",
                                    "target": action.target, "amount": action.amount})
            return
        if action.kind == "set_defense" and context.target_unit_id:
            find_unit(state, context.target_unit_id).defense = action.amount
            return
        if action.kind == "set_attack" and context.target_unit_id:
            find_unit(state, context.target_unit_id).attack = action.amount
            return
        if action.kind == "remove" and context.target_unit_id:
            unit = find_unit(state, context.target_unit_id)
            owner = state.players[unit.owner_id]
            if unit.instance_id in state.battlefield.get(unit.position, []):
                state.battlefield[unit.position].remove(unit.instance_id)
            owner.units.remove(unit)
            state.removed_cards.append(unit.card_id)
            state.event_log.append({"event": "unit_removed", "unit_id": unit.instance_id})
            self._cleanup_continuous_effects(state, unit.owner_id, unit.instance_id)
            return
        if action.kind == "return_to_deck" and action.target == "chosen":
            # Move the card selected by a preceding choose_one to the top of the
            # owner's deck (Pattern C: choose a card in hand, put on top).
            if self._chosen is None:
                state.event_log.append({"event": "choose_one_no_selection", "source_card_id": context.source_card_id})
                return
            cid, owner = self._chosen
            player = state.players[owner]
            if cid in player.hand:
                player.hand.remove(cid)
                player.deck.insert(0, cid)
                state.event_log.append({"event": "card_to_deck_top", "player_id": owner, "card_id": cid})
            return
        if action.kind == "return_to_deck" and context.target_unit_id:
            unit = find_unit(state, context.target_unit_id)
            owner = state.players[unit.owner_id]
            if unit.instance_id in state.battlefield.get(unit.position, []):
                state.battlefield[unit.position].remove(unit.instance_id)
            owner.units.remove(unit)
            owner.deck.insert(0, unit.card_id)
            state.event_log.append({"event": "unit_returned_to_deck", "unit_id": unit.instance_id})
            return
        if action.kind == "convert_to":
            # "Convert X into <card>": materialize the named card. Units are
            # deployed to the support line; everything else (e.g. the PLAN order)
            # is added to the owner's hand. Descriptor names that do not resolve
            # to a real card (e.g. "target unit", "it") are left unresolved so
            # prior no-op behavior is preserved.
            card_id = self._by_name.get((action.card_name or "").upper())
            if card_id:
                card = self.cards.get(card_id)
                target = None
                if context.target_unit_id:
                    try:
                        target = find_unit(state, context.target_unit_id)
                    except ActionValidationError:
                        pass
                if card is not None and card.is_unit and target is not None:
                    # Convert changes the existing battlefield object instead
                    # of creating a second unit. Converted state lasts only
                    # for this battle because UnitState never leaves GameState.
                    target.card_id = card_id
                    target.attack = card.attack or 0
                    target.defense = card.defense or 0
                    target.modifiers.clear()
                    target.status.clear()
                    state.event_log.append({"event": "unit_converted", "unit_id": target.instance_id, "card_id": card_id})
                elif context.metadata.get("selected_card_id"):
                    # Hand/deck conversions retain the selected card's zone and
                    # replace only that card instance (first matching catalog
                    # id, because deck/hand still store card ids).
                    selected_id = context.metadata["selected_card_id"]
                    owner = state.players[context.player_id]
                    selected_zone = context.metadata.get("selected_card_zone", "hand")
                    if selected_zone == "hand" and selected_id in owner.hand:
                        owner.hand[owner.hand.index(selected_id)] = card_id
                        state.event_log.append({"event": "hand_card_converted", "from_card_id": selected_id, "card_id": card_id})
                    elif selected_zone == "deck" and selected_id in owner.deck:
                        owner.deck[owner.deck.index(selected_id)] = card_id
                        state.event_log.append({"event": "deck_card_converted", "from_card_id": selected_id, "card_id": card_id})
                    else:
                        state.event_log.append({"event": "convert_target_missing", "card_id": selected_id})
                elif card is not None and card.is_unit:
                    self._deploy_named_card(state, context, card_id)
                else:
                    self._add_to_hand(state, context.player_id, card_id)
                state.event_log.append({"event": "converted_to", "card_id": card_id, "player_id": context.player_id})
            else:
                state.event_log.append({"event": "native_rule_unresolved_card_name", "name": action.card_name, "source_card_id": context.source_card_id})
            return
        if action.kind == "take_control" and context.target_unit_id:
            unit = find_unit(state, context.target_unit_id)
            old_owner = state.players[unit.owner_id]
            new_owner = state.players[context.player_id]
            if unit.owner_id != context.player_id:
                old_owner.units.remove(unit)
                unit.owner_id = context.player_id
                new_owner.units.append(unit)
                state.event_log.append({"event": "unit_control_changed", "unit_id": unit.instance_id, "player_id": context.player_id})
            return
        if action.kind in ("move_to_frontline", "move_to_support_line"):
            self._move_units("frontline" if action.kind == "move_to_frontline" else "support_line", action, state, context)
            return
        if action.kind == "repair" and context.target_unit_id:
            unit = find_unit(state, context.target_unit_id)
            if unit and unit.status.get("cannot") in ("be_repaired", True):
                state.event_log.append({"event": "unit_cannot_be_repaired", "unit_id": unit.instance_id})
                return
            card = self.cards.get(unit.card_id)
            if card is not None:
                unit.defense = card.defense or unit.defense
                state.event_log.append({"event": "unit_repaired", "unit_id": unit.instance_id})
            return
        if action.target in selectors.RANDOM:
            # A random target is rewritten into the chosen one before the effect
            # layer runs, because EffectResolver._targets has no random branch.
            # Both sides are handled so "random_friendly" cannot become a
            # silently inert selector the way "random_enemy" once was.
            owner = (opponent_id(state, context.player_id)
                     if action.target == "random_enemy" else context.player_id)
            candidates = list(state.players[owner].units)
            if not candidates:
                state.event_log.append({"event": "native_rule_no_random_target", "source_card_id": context.source_card_id})
                return
            priority = [unit for unit in candidates if unit.status.get("random_effect_target_priority")]
            target = priority[0] if priority else random.Random((state.rng_seed or 0) + len(state.event_log)).choice(candidates)
            if priority:
                state.event_log.append({"event": "random_target_priority_applied", "unit_id": target.instance_id})
            context = EffectContext(context.player_id, context.source_card_id, context.source_unit_id, target.instance_id, context.event, context.metadata)
            selector = "selected_target"
        else:
            selector = "selected_target" if action.target.startswith("selected_") else action.target
        effect = {"type": action.kind, "target": selector, "value": {"amount": action.amount, "attack": action.attack, "defense": action.defense}}
        start = len(state.event_log)
        EffectResolver(self.cards).resolve(effect, state, context)
        self.emit_damage_since(state, start)
        self.emit_deaths_since(state, start)

    # --- Delayed-effect scheduling (M10) ---
    def _schedule_delayed(self, action: RuleAction, state, context: EffectContext) -> None:
        trigger = action.duration or "end_of_turn"
        owner = context.player_id
        # Prefer the explicit target (e.g. a unit just removed by a paired
        # "Remove X. Return it" action) so delayed_return brings back the right
        # unit; fall back to the source unit (self-referential "discard it").
        unit_id = context.target_unit_id or context.source_unit_id
        try:
            unit = find_unit(state, unit_id) if unit_id else None
        except Exception:
            unit = None
        card_id = unit.card_id if unit is not None else context.metadata.get("target_card_id")
        card_owner = unit.owner_id if unit is not None else context.metadata.get("target_owner_id")
        kind = "discard" if action.kind == "delayed_discard" else "return_to_hand"
        wait = 1 if trigger == "end_of_next_turn" else 0
        state.players[owner].scheduled.append({
            "trigger": trigger,
            "unit_id": unit_id,
            "card_id": card_id,
            "card_owner_id": card_owner,
            "kind": kind,
            "owner_id": owner,
            "source_card_id": context.source_card_id,
            "wait": wait,
        })
        state.event_log.append({"event": "rule_scheduled", "trigger": trigger, "kind": kind, "player_id": owner})

    @staticmethod
    def fire_scheduled(state, player_id: str, phase: str) -> None:
        """Fire the owner's scheduled one-shot actions due at this turn phase.

        phase == "start" (called from TurnManager.start_turn) covers
        "start_of_next_turn"; phase == "end" (called from TurnManager.end_turn)
        covers "end_of_turn" immediately and "end_of_next_turn" after its wait.
        """
        player = state.players[player_id]
        remaining: list[dict] = []
        for entry in player.scheduled:
            if entry["trigger"] == "start_of_next_turn":
                if phase == "start":
                    NativeRuleEngine._apply_delayed(entry, state)
                else:
                    remaining.append(entry)
            elif entry["trigger"] in ("end_of_turn", "end_of_next_turn"):
                if phase == "end":
                    if entry.get("wait", 0) > 0:
                        entry["wait"] -= 1
                        remaining.append(entry)
                    else:
                        NativeRuleEngine._apply_delayed(entry, state)
                else:
                    remaining.append(entry)
            else:
                remaining.append(entry)
        player.scheduled = remaining
        from simulator.rules.intel import refresh as refresh_intel
        refresh_intel(state)

    @staticmethod
    def _apply_delayed(entry: dict, state) -> None:
        # The affected unit returns to / is discarded by its *owning* player
        # (e.g. an enemy unit removed by "Remove X. Return it" goes back to the
        # enemy's hand), not the player who scheduled the effect.
        owner = state.players.get(entry.get("card_owner_id") or entry.get("owner_id"))
        if owner is None:
            return
        if entry["kind"] == "repair_all_friendly":
            repaired = 0
            for repair in entry.get("repairs", []):
                try:
                    unit = find_unit(state, repair.get("unit_id"))
                except Exception:
                    unit = None
                if unit is None or unit.owner_id != owner.player_id:
                    continue
                if unit.status.get("cannot") in ("be_repaired", True):
                    continue
                base_defense = repair.get("defense")
                if isinstance(base_defense, int):
                    unit.defense = base_defense
                    repaired += 1
            state.event_log.append({"event": "scheduled_units_repaired", "player_id": owner.player_id, "count": repaired})
            return
        try:
            unit = find_unit(state, entry["unit_id"]) if entry.get("unit_id") else None
        except Exception:
            unit = None
        if entry["kind"] == "discard":
            if unit is not None and unit in owner.units:
                owner.units.remove(unit)
                state.graveyard.setdefault(owner.player_id, []).append(unit.card_id)
                state.event_log.append({"event": "delayed_discard", "unit_id": unit.instance_id, "card_id": unit.card_id})
        elif entry["kind"] == "return_to_hand":
            if unit is not None and unit in owner.units:
                owner.units.remove(unit)
                owner.hand.append(unit.card_id)
                state.event_log.append({"event": "delayed_return", "unit_id": unit.instance_id, "card_id": unit.card_id})
            elif entry.get("card_id") and entry["card_id"] not in owner.hand:
                # Unit was removed from the battlefield earlier (e.g. by a paired
                # "Remove X" action); returning it means its card comes back to hand.
                owner.hand.append(entry["card_id"])
                state.event_log.append({"event": "delayed_return", "card_id": entry["card_id"]})

    def _handle_choose_one(self, action: RuleAction, state, context: EffectContext) -> None:
        """Select a card per a choose_one action and record it in `self._chosen`.

        Patterns:
          scope=="unit"      -> reveal a random nation unit and add it to hand
                                (KARDS "Choose 1 of N random elite <nation> units").
          scope=="enemy_hand"-> select 1 of the (up to N revealed) enemy hand cards.
          scope=="hand"      -> select a card from the owner's hand.
        Selection is seeded for determinism; an AI policy would override the pick.
        """
        if action.scope == "unit":
            nation = action.card_name
            pool = [c for c in self.cards if c.is_unit and (nation is None or c.nation == nation)]
            if not pool:
                state.event_log.append({"event": "choose_one_no_pool", "nation": nation})
                return
            rng = random.Random((state.rng_seed or 0) + len(state.event_log))
            chosen = rng.choice(pool)
            self._add_to_hand(state, context.player_id, chosen.id)
            self._chosen = (chosen.id, context.player_id)
            state.event_log.append({"event": "choose_one_unit_added", "card_id": chosen.id, "nation": nation})
            return
        if action.scope == "enemy_hand":
            enemy = opponent_id(state, context.player_id)
            hand = state.players[enemy].hand
            if not hand:
                state.event_log.append({"event": "choose_one_enemy_hand_empty"})
                return
            reveal = hand[: action.amount] if action.amount else hand
            rng = random.Random((state.rng_seed or 0) + len(state.event_log))
            chosen = rng.choice(reveal)
            self._chosen = (chosen, enemy)
            state.event_log.append({"event": "choose_one_enemy_card", "card_id": chosen, "from": enemy})
            return
        if action.scope == "hand":
            hand = state.players[context.player_id].hand
            if not hand:
                state.event_log.append({"event": "choose_one_hand_empty"})
                return
            rng = random.Random((state.rng_seed or 0) + len(state.event_log))
            chosen = rng.choice(hand)
            self._chosen = (chosen, context.player_id)
            state.event_log.append({"event": "choose_one_hand_card", "card_id": chosen})
            return
        state.event_log.append({"event": "choose_one_no_scope", "scope": action.scope})

    def _add_to_hand(self, state, player_id: str, card_id: str) -> None:
        """Add a card to a player's hand, routing overflow to the graveyard."""
        player = state.players[player_id]
        if len(player.hand) < 9:
            bonuses = player.status.get("hand_cost_on_friendly_death")
            if isinstance(bonuses, dict) and card_id not in player.hand:
                bonuses.pop(card_id, None)
            player.hand.append(card_id)
            state.event_log.append({"event": "card_added_to_hand", "player_id": player_id, "card_id": card_id})
        else:
            state.graveyard.setdefault(player_id, []).append(card_id)
            state.event_log.append({"event": "card_overdrawn", "player_id": player_id, "card_id": card_id})

    def _deploy_named_card(self, state, context: EffectContext, card_id: str) -> None:
        """Deploy a named unit card to the owner's support line (cap at 4)."""
        card = self.cards.get(card_id)
        if card is None or not card.is_unit:
            return
        player = state.players[context.player_id]
        if sum(unit.position == "support_line" for unit in player.units) >= 4:
            state.event_log.append({"event": "rule_spawn_failed", "reason": "support_line_full", "card_id": card_id})
            return
        instance_id = "{0}-rule-{1}".format(card_id, sum(len(p.units) for p in state.players.values()) + 1)
        player.units.append(UnitState(instance_id, card_id, card.attack or 0, card.defense or 0, context.player_id))
        state.battlefield["support_line"].append(instance_id)
        state.event_log.append({"event": "rule_unit_deployed", "card_id": card_id, "player_id": context.player_id})

    def _resolve_units_scoped(self, target: str, action: RuleAction, state, context) -> list:
        """Resolve a target selector then filter by type/nation/ability scope."""
        base = _resolve_units(target, state, context)
        scope = action.scope or ""
        fv = (action.card_name or "").lower()
        if not scope and not fv:
            return base
        out = []
        for unit in base:
            card = self.cards.get(unit.card_id)
            if card is None:
                continue
            ok = True
            if scope in {"air", "ground", "tank", "infantry", "artillery", "bomber", "fighter"}:
                if scope == "air":
                    ok = ok and card.type in {"fighter", "bomber"}
                elif scope == "ground":
                    ok = ok and card.type in {"infantry", "tank", "artillery"}
                else:
                    ok = ok and card.type == scope
            elif scope == "tank_or_infantry":
                ok = ok and card.type in {"tank", "infantry"}
            elif scope == "guard":
                abilities = [a.lower() for a in (card.abilities or ())]
                added = [a.lower() for a in (unit.status.get("added_abilities") or [])]
                ok = ok and ("guard" in abilities or "guard" in added)
            elif scope == "veteran":
                ok = ok and bool(unit.status.get("veteran"))
            if fv:
                if fv in _NATION_LOWER:
                    ok = ok and (card.nation or "").lower() == fv
                else:
                    ok = ok and fv in (a.lower() for a in (card.abilities or ()))
            if ok:
                out.append(unit)
        return out

    def _order_damage_bonus(self, state, context: EffectContext) -> int:
        """Sum verified in-play bonuses that modify damage dealt by Orders."""
        if context.source_card_id not in self.cards or self.cards.get(context.source_card_id).type != "order":
            return 0
        return sum(
            bonus for unit in state.players[context.player_id].units
            if isinstance((bonus := unit.status.get("order_damage_bonus")), int) and bonus > 0
        )

    def _apply_scoped_combat(self, action: RuleAction, state, context: EffectContext) -> None:
        """Apply a scoped combat/attribute action to resolved units (or HQ)."""
        opp = opponent_id(state, context.player_id)
        if action.target == "enemy_hq":
            if action.kind == "damage":
                HQResolver.damage(state, opp, action.amount + self._order_damage_bonus(state, context), context.player_id)
            elif action.kind == "heal":
                HQResolver.heal(state, opp, action.amount)
            return
        units = self._resolve_units_scoped(action.target, action, state, context)
        start = len(state.event_log)
        bonus = self._order_damage_bonus(state, context) if action.kind == "damage" else 0
        for unit in units:
            if action.kind == "damage":
                basic_effects.damage(state, unit, action.amount + bonus, context.player_id)
            elif action.kind == "heal":
                basic_effects.heal(state, unit, action.amount)
            elif action.kind == "buff":
                unit.attack += action.attack
                unit.defense += action.defense
            elif action.kind == "modify_attack":
                unit.attack += action.amount
            elif action.kind == "modify_defense":
                unit.defense += action.amount
            elif action.kind == "suppress":
                suppression.apply(unit, self.cards.get(unit.card_id))
            elif action.kind == "pin":
                from simulator.rules.pin import apply as apply_pin
                apply_pin(unit)
            elif action.kind == "destroy":
                basic_effects.destroy(state, unit)
        self.emit_damage_since(state, start)
        self.emit_deaths_since(state, start)

    def _apply_temporary(self, action: RuleAction, state, context: EffectContext) -> None:
        """Apply a temporary stat/status effect and register its reversal.

        Mirrors _apply_scoped_combat for the reversible kinds, but records a
        revert-op per affected unit on the owner's temporary_effects list so the
        effect is undone at the turn boundary computed from action.duration.
        """
        if action.target == "enemy_hq":
            # HQ temporary effects are not yet reversible; apply permanently.
            self._apply_scoped_combat(action, state, context)
            return
        units = (
            self._resolve_units_scoped(action.target, action, state, context)
            if (action.scope or action.card_name) else _resolve_units(action.target, state, context)
        )
        reverts: list[dict] = []
        start = len(state.event_log)
        for unit in units:
            if action.kind == "grant_ability":
                if action.scope and not _unit_scope_matches(unit, action, self.cards):
                    continue
                granted = unit.status.setdefault("added_abilities", [])
                if action.card_name and action.card_name not in granted:
                    granted.append(action.card_name)
                reverts.append({"unit_id": unit.instance_id, "remove_ability": action.card_name})
                continue
            if action.kind == "suppress":
                if suppression.apply(unit, self.cards.get(unit.card_id)):
                    reverts.append({"unit_id": unit.instance_id, "attr": "suppressed", "delta": 1})
                continue
            if action.kind == "pin":
                from simulator.rules.pin import apply as apply_pin
                if apply_pin(unit):
                    reverts.append({"unit_id": unit.instance_id, "attr": "pinned", "delta": 1})
                continue
            if action.kind == "buff":
                reverts.append({"unit_id": unit.instance_id, "attr": "attack", "delta": action.attack})
                reverts.append({"unit_id": unit.instance_id, "attr": "defense", "delta": action.defense})
                unit.attack += action.attack
                unit.defense += action.defense
            elif action.kind == "modify_attack":
                reverts.append({"unit_id": unit.instance_id, "attr": "attack", "delta": action.amount})
                unit.attack += action.amount
            elif action.kind == "modify_defense":
                reverts.append({"unit_id": unit.instance_id, "attr": "defense", "delta": action.amount})
                unit.defense += action.amount
            elif action.kind == "set_attack":
                reverts.append({"unit_id": unit.instance_id, "restore_attack": unit.attack})
                unit.attack = 0
        if reverts:
            expiry = _expiry_turn(action.duration, state.turn_number)
            if expiry is not None:
                state.players[context.player_id].temporary_effects.append({"turn": expiry, "reverts": reverts})
                state.event_log.append({"event": "temporary_effect_applied", "player_id": context.player_id, "expires_turn": expiry, "kind": action.kind})
        self.emit_damage_since(state, start)
        self.emit_deaths_since(state, start)

    @staticmethod
    def revert_temporary(state, player_id: str, current_turn: int) -> None:
        """Revert the player's temporary effects whose expiry turn has passed.

        Called by TurnManager at the end of the player's turn. 'this_turn'
        effects (expiry == current_turn) and 'next_turn' effects (expiry ==
        current_turn, i.e. the player's next turn just ended) are both reverted.
        """
        player = state.players[player_id]
        remaining: list[dict] = []
        for entry in player.temporary_effects:
            if entry["turn"] is not None and entry["turn"] <= current_turn:
                for rev in entry["reverts"]:
                    if "player_status" in rev:
                        player.status.pop(rev["player_status"], None)
                        continue
                    try:
                        unit = find_unit(state, rev["unit_id"])
                    except ActionValidationError:
                        # A unit may die or be removed before its temporary
                        # effect expires. There is then no runtime state left
                        # to restore, so expiration must remain a no-op.
                        continue
                    if "remove_ability" in rev:
                        added = unit.status.get("added_abilities")
                        if added and rev["remove_ability"] in added:
                            added.remove(rev["remove_ability"])
                    elif "restore_attack" in rev:
                        unit.attack = rev["restore_attack"]
                    elif rev.get("attr") == "suppressed":
                        suppression.restore(unit)
                    elif rev.get("attr") == "pinned":
                        unit.status.pop("pinned", None)
                    else:
                        if rev["attr"] == "attack":
                            unit.attack -= rev["delta"]
                        elif rev["attr"] == "defense":
                            unit.defense -= rev["delta"]
                state.event_log.append({"event": "temporary_effect_reverted", "player_id": player_id})
            else:
                remaining.append(entry)
        player.temporary_effects = remaining
        # Purge expired cost modifiers / op-cost rules.
        player.cost_modifiers = [m for m in player.cost_modifiers if m.get("expires_turn") is None or m["expires_turn"] > current_turn]
        player.op_cost_rules = [r for r in player.op_cost_rules if r.get("expires_turn") is None or r["expires_turn"] > current_turn]
        for candidate_player in state.players.values():
            for unit in candidate_player.units:
                unit.modifiers = [
                    modifier for modifier in unit.modifiers
                    if modifier.get("expires_turn") is None or modifier["expires_turn"] > current_turn
                ]
        # Reset per-turn HQ flags.
        player.hq.immune_until_end_of_turn = False

    def _displace_unit(self, state, unit: UnitState, returned_event: str, discarded_event: str) -> str:
        """Take a unit out of play without destroying it (no death triggers).

        The unit's card returns to its owner's hand, or is put into the graveyard
        when that hand is already at ``HAND_CARD_LIMIT``.
        """
        owner = state.players[unit.owner_id]
        if unit.instance_id in state.battlefield.get(unit.position, []):
            state.battlefield[unit.position].remove(unit.instance_id)
        if unit in owner.units:
            owner.units.remove(unit)
        if len(owner.hand) < HAND_CARD_LIMIT:
            owner.hand.append(unit.card_id)
            event = returned_event
        else:
            state.graveyard.setdefault(unit.owner_id, []).append(unit.card_id)
            event = discarded_event
        self._cleanup_continuous_effects(state, unit.owner_id, unit.instance_id)
        state.event_log.append({"event": event, "unit_id": unit.instance_id, "card_id": unit.card_id})
        return event

    def _veteran_index(self) -> dict:
        """Map each card id to its Veteran counterpart via kards.info "VeteranOf:"."""
        index = getattr(self, "_veteran_ids", None)
        if index is None:
            index = {}
            for card in self.cards:
                for attribute in card.unknownAttributes:
                    # The veteran card points back ("VeteranOf:<base>"), and the
                    # base card points forward ("BecomesVeteran:<veteran>").
                    if attribute.startswith("VeteranOf:"):
                        index[attribute[len("VeteranOf:"):].strip()] = card.id
                    elif attribute.startswith("BecomesVeteran:"):
                        index[card.id] = attribute[len("BecomesVeteran:"):].strip()
            self._veteran_ids = index
        return index

    def _promote_to_veteran(self, state, unit: UnitState) -> None:
        """Swap a unit for its Veteran card.

        Becoming a Veteran replaces the unit with a brand-new card: accumulated
        damage, stat buffs and operation-cost modifiers are all lost.  Per-turn
        bookkeeping (already attacked / deployed / moved this turn) is kept so the
        promotion cannot be used to gain an extra action.
        """
        veteran_id = self._veteran_index().get(unit.card_id)
        if veteran_id is None or veteran_id not in self.cards:
            state.event_log.append({"event": "veteran_card_missing", "card_id": unit.card_id,
                                    "unit_id": unit.instance_id})
            return
        card = self.cards.get(veteran_id)
        preserved = {key: unit.status[key] for key in ("attack_count", "deployed_this_turn", "moved_this_turn")
                     if key in unit.status}
        previous = unit.card_id
        unit.card_id = veteran_id
        unit.attack = card.attack or 0
        unit.defense = card.defense or 0
        unit.modifiers.clear()
        unit.status.clear()
        unit.status.update(preserved)
        unit.status["veteran"] = True
        state.event_log.append({"event": "unit_became_veteran", "unit_id": unit.instance_id,
                                "from_card_id": previous, "to_card_id": veteran_id})

    def _move_units(self, position: str, action: RuleAction, state, context) -> None:
        """Move resolved units (single target or a scoped group) to a position."""
        if action.target in ("friendly_units", "enemy_units"):
            units = self._resolve_units_scoped(action.target, action, state, context)
        elif action.target in selectors.CHOSEN and context.target_unit_id:
            unit = find_unit(state, context.target_unit_id)
            units = [unit] if unit is not None else []
        elif action.target in selectors.SOURCE and context.source_unit_id:
            # "Move this unit into the frontline" — without this the selector
            # resolved to nothing and the move silently did not happen.
            try:
                units = [find_unit(state, context.source_unit_id)]
            except ActionValidationError:
                units = []
        elif action.target in selectors.RANDOM:
            # The generic random-target rewrite runs AFTER this branch, so a
            # movement action with a random target would otherwise be dropped.
            owner = opponent_id(state, context.player_id) if action.target == "random_enemy" else context.player_id
            candidates = list(state.players[owner].units)
            units = ([random.Random((state.rng_seed or 0) + len(state.event_log)).choice(candidates)]
                     if candidates else [])
        else:
            units = []
        for unit in units:
            if position == "support_line" and unit.status.get("cannot") in ("retreat", True):
                state.event_log.append({"event": "unit_cannot_retreat", "unit_id": unit.instance_id})
                continue
            if position == "frontline" and not BattlefieldRules.can_move_to_frontline(state, unit):
                state.event_log.append({"event": "native_rule_frontline_full_or_uncontrolled", "unit_id": unit.instance_id})
                continue
            owner = state.players[unit.owner_id]
            if position == "support_line" and sum(candidate.position == "support_line" for candidate in owner.units) >= SUPPORT_UNIT_LIMIT:
                # KARDS retreat overflow: with no free support slot the unit goes
                # back to its owner's hand, and is removed from the game when the
                # hand already holds the 9-card maximum.  It is not destroyed, so
                # no death trigger fires.
                self._displace_unit(state, unit, "native_rule_support_line_full_returned",
                                    "native_rule_support_line_full_discarded")
                continue
            if unit.instance_id in state.battlefield.get(unit.position, []):
                state.battlefield[unit.position].remove(unit.instance_id)
            unit.position = position
            state.battlefield[position].append(unit.instance_id)

    def _copy_unit(self, action: RuleAction, state, context: EffectContext) -> None:
        owner = context.player_id
        source = None
        if action.target in ("source", "self") and context.source_unit_id:
            source = find_unit(state, context.source_unit_id)
        if source is None and context.target_unit_id and action.target.startswith("selected_"):
            source = find_unit(state, context.target_unit_id)
        if source is None and action.target in ("friendly_units", "owner"):
            candidates = [u for u in state.players[owner].units if _unit_scope_matches(u, action, self.cards)]
            source = candidates[0] if candidates else None
        if source is None:
            return
        card = self.cards.get(source.card_id)
        if card is None or not card.is_unit:
            return
        player = state.players[source.owner_id]
        if sum(u.position == "support_line" for u in player.units) >= 4:
            state.event_log.append({"event": "copy_failed", "reason": "support_line_full"})
            return
        instance_id = "{0}-copy-{1}".format(card.id, sum(len(p.units) for p in state.players.values()) + 1)
        player.units.append(UnitState(instance_id, card.id, card.attack or 0, card.defense or 0, source.owner_id, "support_line"))
        state.battlefield["support_line"].append(instance_id)
        state.event_log.append({"event": "unit_copied", "card_id": card.id, "player_id": source.owner_id})

    def _draw_top_matching(self, action: RuleAction, state, context: EffectContext) -> None:
        owner = opponent_id(state, context.player_id) if action.target == "enemy_hand" else context.player_id
        player = state.players[owner]
        if not player.deck:
            return
        top = player.deck[0]
        c = self.cards.get(top)
        if c is None:
            return
        scope = action.scope or ""
        fv = (action.card_name or "").lower()
        ok = True
        if scope == "unit" and not c.is_unit:
            ok = False
        if scope == "order" and c.type != "order":
            ok = False
        if scope in {"air", "ground"}:
            if scope == "air" and c.type not in {"fighter", "bomber"}:
                ok = False
            if scope == "ground" and c.type not in {"infantry", "tank", "artillery"}:
                ok = False
        if fv:
            if fv.startswith("non_"):
                if (c.nation or "").lower() == fv[4:]:
                    ok = False
            elif (c.nation or "").lower() != fv and fv not in (c.abilities or ()):
                ok = False
        if action.amount and c.operationCost != action.amount:
            ok = False
        if not ok:
            return
        player.deck.pop(0)
        if len(player.hand) < 9:
            player.hand.append(top)
            state.event_log.append({"event": "card_drawn", "player_id": owner, "card_id": top})
        else:
            state.graveyard.setdefault(owner, []).append(top)
            state.event_log.append({"event": "card_overdrawn", "player_id": owner, "card_id": top})

    def coverage(self) -> dict[str, int]:
        result = {"implemented": 0, "partial": 0, "unresolved": 0}
        for rule in self._rules.values():
            result[rule.status] += 1
        return result


def default_rule_path() -> Path:
    """Reserved stable location for reviewed card-rule overrides and AST exports."""
    return Path(__file__).parents[2] / "data/rules/card_rules.json"


_ENGINES: dict[int, NativeRuleEngine] = {}


def engine_for(cards: CardDatabase) -> NativeRuleEngine:
    """Reuse compiled ASTs for a catalog during high-volume AI stepping."""
    key = id(cards)
    if key not in _ENGINES:
        _ENGINES[key] = NativeRuleEngine(cards)
    return _ENGINES[key]
