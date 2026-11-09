"""Activated-countermeasure dispatcher for native card rules."""
from simulator.effects.resolver import EffectContext
from simulator.actions.validator import opponent_id
from simulator.rules.native import engine_for
from simulator.rules.condition import evaluate as condition_evaluate


class CountermeasureResolver:
    @staticmethod
    def intercept(state, cards, acting_player_id, event, context):
        """Try to counter the current action with the opponent's active countermeasures.

        Returns True if the action was cancelled (at least one countermeasure
        fired and had a `cancel` action with its conditions satisfied).

        Conditions (e.g. 'target must be friendly', 'order cost >= 4') are
        evaluated BEFORE the countermeasure is consumed.  If the condition
        fails, the countermeasure stays active and the action proceeds normally.
        """
        enemy = opponent_id(state, acting_player_id)
        # C6N SAIUN-style effects are sourced by an in-play unit.  Do not use
        # a sticky player flag: the lock must disappear when that unit leaves.
        if any(unit.status.get("countermeasure_lock") for player in state.players.values() for unit in player.units):
            state.event_log.append({"event": "countermeasure_trigger_blocked", "player_id": enemy, "global": True})
            return False
        cancelled = False
        # Iterate over a snapshot so modifications from execute() do not skew
        # the loop.
        for counter in list(state.players[enemy].active_countermeasures):
            card_id = counter.get("card_id")
            if not isinstance(card_id, str) or card_id not in cards:
                continue
            rule = engine_for(cards).rule_for(card_id)
            if event not in rule.triggers:
                continue
            # Covert units cannot be affected by countermeasures unless the
            # countermeasure explicitly says it can target/affect Covert.
            covert_target_id = context.target_unit_id or context.source_unit_id
            if covert_target_id:
                try:
                    target = next(unit for player in state.players.values() for unit in player.units if unit.instance_id == covert_target_id)
                except StopIteration:
                    target = None
                if target is not None and target.status.get("covert") and target.owner_id != enemy:
                    if not any(action.card_name == "can_target_covert" for action in rule.actions):
                        continue

            # Check that at least one action would actually fire (conditions
            # satisfied) BEFORE consuming the countermeasure.
            should_activate = False
            intercepted_target_id = context.target_unit_id
            # "When a friendly unit is attacked" (TACTICAL RETREAT) is read by
            # condition.py off metadata["event_target_owner_id"] -- the same
            # key native.emit() fills in for broadcast-style triggers. Without
            # it this countermeasure's condition is permanently false, so it
            # can never actually activate (found 2026-11-04 while chasing the
            # E9 crash; fixed 2026-11-09).
            intercepted_target_owner_id = None
            if intercepted_target_id:
                try:
                    intercepted_target_owner_id = next(
                        unit.owner_id for player in state.players.values() for unit in player.units
                        if unit.instance_id == intercepted_target_id
                    )
                except StopIteration:
                    pass
            counter_context = EffectContext(
                enemy, card_id,
                # The countermeasure targets the intercepted action's target,
                # not its source.  Attack-triggered countermeasures instead
                # operate on the attacker when no defending unit was selected.
                target_unit_id=intercepted_target_id or context.source_unit_id,
                event=event,
                metadata={
                    "intercepted_card_id": context.source_card_id,
                    "intercepted_card_cost": (
                        cards.get(context.source_card_id).kredits
                        if context.source_card_id in cards else None
                    ),
                    "intercepted_player_id": acting_player_id,
                    "has_deployment_effect": bool(context.metadata.get("has_deployment_effect")),
                    "event_target_owner_id": intercepted_target_owner_id,
                },
            )
            for action in rule.actions:
                if action.card_name == "deployment_effect" and not counter_context.metadata.get("has_deployment_effect"):
                    continue
                if action.min_cost:
                    cost = counter_context.metadata.get("intercepted_card_cost")
                    if not isinstance(cost, int) or cost < action.min_cost:
                        continue
                if action.condition is None:
                    should_activate = True
                    break
                if action.condition == "target_friendly":
                    if intercepted_target_id is None:
                        continue
                    try:
                        target = next(
                            unit for player in state.players.values() for unit in player.units
                            if unit.instance_id == intercepted_target_id
                        )
                    except StopIteration:
                        continue
                    if target.owner_id == enemy:
                        should_activate = True
                        break
                    continue
                if condition_evaluate(
                    action.condition, event, counter_context, cards, state,
                ):
                    should_activate = True
                    break

            if not should_activate:
                continue

            # Condition passed: activate the countermeasure.
            engine_for(cards).execute(card_id, event, state, counter_context)
            state.players[enemy].active_countermeasures.remove(counter)
            if card_id in state.players[enemy].hand:
                state.players[enemy].hand.remove(card_id)
            from simulator.rules.intel import refresh as refresh_intel
            refresh_intel(state)
            state.graveyard.setdefault(enemy, []).append(card_id)
            state.event_log.append({
                "event": "countermeasure_triggered",
                "player_id": enemy,
                "counter_id": card_id,
                "trigger": event,
            })
            # STURMBRIGADE RHODOS ("your countermeasure triggers") reacts to
            # its owner's own countermeasure; MIG-3 ("an enemy countermeasure
            # triggers") reacts to the opponent's -- ``enemy`` here is whoever
            # just activated the countermeasure (the original action's target),
            # so from their own hand's perspective it was "your" trigger, and
            # from the acting player's hand it was "an enemy" trigger.
            from simulator.rules.native import engine_for as _engine_for
            engine = _engine_for(cards)
            engine.bump_hand_cost_reaction(state, enemy, "hand_cost_on_own_countermeasure")
            engine.bump_hand_cost_reaction(state, acting_player_id, "hand_cost_on_enemy_countermeasure")
            if any(action.kind == "cancel" for action in rule.actions):
                cancelled = True

        return cancelled
