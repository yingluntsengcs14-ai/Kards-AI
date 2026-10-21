"""Evaluate parsed ``RuleAction.condition`` strings against a firing event.

The parser stores normalized condition clauses (e.g. ``when it damages the enemy
hq.``, ``if enemy has 3 or more units.``). The native engine uses this to decide
whether a conditional action should actually execute for the current event/context.

Modeled phrases are strictly gated to the matching event + (where applicable) the
played card's type/ability. Unrecognized conditions are retained as satisfied
for backward-compatible execution until the parser can distinguish persistent
text from unresolved event predicates.
"""

from __future__ import annotations

import re

from simulator.cards.loader import CardDatabase


def evaluate(
    condition: str | None,
    event: str,
    context,
    cards: CardDatabase,
    state=None,
) -> bool:
    """Return True if the action should execute for ``event``/``context``."""
    if not condition:
        return True
    c = condition.lower()
    metadata = getattr(context, "metadata", {}) or {}
    event_player_id = metadata.get("event_player_id", getattr(context, "player_id", ""))
    listener_player_id = getattr(context, "player_id", "")
    event_target_owner_id = metadata.get("event_target_owner_id")
    # Battlefield listeners receive the same event as the acting card.  These
    # clauses distinguish the listener's owner from the player that caused the
    # event; without this, a card reading "when an enemy unit attacks" either
    # never fires or incorrectly fires for its controller's own attacks.
    # ── Hand position (OCEANIA STORM / RING OF FIRE) ────────────────────────
    # "if this card was left-most in hand" / "unless ... left-most".  The played
    # card's position is captured by PlayCardAction before it leaves hand.
    if "left-most in hand" in c or "right-most in hand" in c:
        from simulator.rules import hand_position
        index, size = metadata.get("hand_index"), metadata.get("hand_size")
        at_edge = (hand_position.is_leftmost(index, size) if "left-most" in c
                   else hand_position.is_rightmost(index, size))
        return not at_edge if c.strip().startswith("unless") else at_edge
    # FEROCIOUS ASSAULT: "Lose a kredit slot if its attack was greater than your
    # kredit slots."  "its" is the targeted unit, which this action is ordered
    # before the destroy so that the unit is still on the battlefield here.
    if "attack was greater than your kredit slots" in c:
        if state is None or not getattr(context, "target_unit_id", None):
            return False
        from simulator.actions.validator import ActionValidationError, find_unit
        try:
            target = find_unit(state, context.target_unit_id)
        except ActionValidationError:
            return False
        return target.attack > state.players[listener_player_id].resources.max_kredits
    # ── Weather (OCEANIA STORM) ─────────────────────────────────────────────
    # Three shapes appear on cards: the Forecast event, the "you played a
    # weather card" event, and a per-turn question about what was played.
    # Each is gated on the matching event so the effect cannot fire on deploy.
    if "you forecast" in c:
        return event == "on_forecast" and (not event_player_id or event_player_id == listener_player_id)
    weather_event = re.search(r"\b(?:when|after)\s+you\s+play\s+a\s+(.+?)\s*card", c)
    if weather_event and _weather_categories(weather_event.group(1)):
        if event != "on_weather_played":
            return False
        if event_player_id and event_player_id != listener_player_id:
            return False
        return metadata.get("weather") in _weather_categories(weather_event.group(1))
    weather_turn = re.search(
        r"\b(?:if\s+)?(?:you\s+played\s+a\s+)?([a-z ,]*?)\s*(?:card\s+)?was\s+played\s+this\s+turn", c)
    weather_turn = weather_turn or re.search(r"\bif\s+you\s+played\s+a\s+(.+?)\s*card\s+this\s+turn", c)
    if weather_turn:
        categories = _weather_categories(weather_turn.group(1))
        if categories:
            if state is None:
                return False
            from simulator.rules import weather as weather_rules
            return weather_rules.was_played_this_turn(state, listener_player_id, *categories)
    if "at the start of your turn" in c:
        return event == "on_turn_start"
    if "at the end of your turn" in c or "at the end of turn" in c:
        return event == "on_turn_end"
    if "when deployed" in c:
        return event == "on_deploy"
    if "when this unit attacks" in c or "after each time this unit attacks" in c:
        return event == "on_attack" and (
            not metadata.get("event_source_unit_id")
            or metadata.get("event_source_unit_id") == getattr(context, "source_unit_id", None)
        )
    if "friendly unit is attacked" in c:
        return event == "on_attack" and event_target_owner_id == listener_player_id
    if "enemy unit attacks" in c or "enemy attacks" in c:
        return event == "on_attack" and bool(event_player_id) and event_player_id != listener_player_id
    if "when a unit attacks" in c or "when another unit attacks" in c:
        return event == "on_attack"
    if "enemy deploys" in c or "enemy unit is deployed" in c:
        if event != "on_deploy" or not event_player_id or event_player_id == listener_player_id:
            return False
        if "tank unit" in c:
            deployed = _played_card(context, cards)
            return deployed is not None and deployed.type == "tank"
        return True
    if "when a unit is deployed" in c:
        return event == "on_deploy"
    if "when you deploy" in c:
        return event == "on_deploy" and event_player_id == listener_player_id
    if "when you play a navy card" in c:
        played = _played_card(context, cards)
        return (
            event == "on_friendly_card_played"
            and played is not None
            and ("navy" in played.name.lower() or _has_ability(played, "navy"))
        )
    if "moves into the frontline" in c or "moves to the frontline" in c:
        if event != "on_deploy" or not metadata.get("moved_to_frontline"):
            return False
        if "enemy unit" in c:
            return bool(event_player_id) and event_player_id != listener_player_id
        if "friendly unit" in c:
            return event_player_id == listener_player_id
        return True
    if "friendly unit is destroyed" in c:
        return event == "on_destroy" and event_player_id == listener_player_id
    if "enemy unit is destroyed" in c:
        return event == "on_destroy" and bool(event_player_id) and event_player_id != listener_player_id
    # Event-gated self-buffs.
    if "enemy hq" in c:
        return event == "on_enemy_hq_damaged"
    if "survives combat" in c:
        return event == "on_survives_combat"
    if "is dealt damage" in c or "takes damage" in c or "is damaged" in c or "dealt damage" in c:
        return event == "on_damage"
    # Card-play triggers filtered by the played card's type / ability.
    played = _played_card(context, cards)
    if "give an order" in c or "play an order" in c or "gives an order" in c:
        return event == "on_friendly_card_played" and played is not None and played.type == "order"
    if "card with intel" in c:
        return event == "on_friendly_card_played" and _has_ability(played, "intel")
    if "card with bond" in c:
        return event == "on_friendly_card_played" and _has_ability(played, "bond")
    if state is not None and context.player_id in state.players:
        player = state.players[context.player_id]
        opponent = next((candidate for pid, candidate in state.players.items() if pid != context.player_id), None)
        enemy_count = len(opponent.units) if opponent is not None else 0
        own_count = len(player.units)
        match = re.search(r"enemy has (\d+) or more units?", c)
        if match:
            return enemy_count >= int(match.group(1))
        match = re.search(r"you control (\d+) or more units?", c)
        if match:
            return own_count >= int(match.group(1))
        match = re.search(r"you control (?:another )?(?:unit )?with (\d+) or more attack", c)
        if match:
            source_id = getattr(context, "source_unit_id", None)
            units = [unit for unit in player.units if "another" not in c or unit.instance_id != source_id]
            return any(unit.attack >= int(match.group(1)) for unit in units)
        if re.search(r"you control (?:an? )?infantry", c):
            return any(cards.get(unit.card_id).type == "infantry" for unit in player.units if unit.card_id in cards)
        if "you control the frontline" in c:
            frontline = state.battlefield.get("frontline", [])
            return bool(frontline) and all(
                any(unit.instance_id == unit_id and unit.owner_id == context.player_id for unit in player.units)
                for unit_id in frontline
            )
        match = re.search(r"your hq has (\d+) or less defense", c)
        if match:
            return player.hq.current_health <= int(match.group(1))
        if "you have no cards in hand" in c:
            return not player.hand
        if "navy card on top of your deck" in c:
            if not player.deck or player.deck[0] not in cards:
                return False
            top = cards.get(player.deck[0])
            return "navy" in top.name.lower() or any("navy" in ability.lower() for ability in top.abilities)
    # State-based / other conditions: retained leniently for existing parsed
    # persistent effects. Training readiness audits report this as a blocker.
    return True


def _weather_categories(phrase: str) -> tuple[str, ...]:
    """Weather categories named in a clause ("a Rain or a Storm" -> rain, storm).

    Returns () when the clause names no weather, which keeps non-weather
    "when you play a X card" clauses on their existing code path.
    """
    lower = (phrase or "").lower()
    names = {"sunny": "sunny", "rain": "rain", "storm": "storm"}
    return tuple(category for word, category in names.items() if re.search(r"\b" + word + r"\b", lower))


def _played_card(context, cards: CardDatabase):
    cid = context.metadata.get("played_card_id") if context and context.metadata else None
    return cards.get(cid) if cid else None


def _has_ability(card, ability: str) -> bool:
    if card is None:
        return False
    abilities = card.abilities
    if not isinstance(abilities, (list, tuple)):
        return False
    return any(ability in (a.lower() if isinstance(a, str) else "") for a in abilities)
