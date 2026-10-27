"""Evaluate parsed ``RuleAction.condition`` strings against a firing event.

The parser stores normalized condition clauses (e.g. ``when it damages the enemy
hq.``, ``if enemy has 3 or more units.``). The native engine uses this to decide
whether a conditional action should actually execute for the current event/context.

Modeled phrases are strictly gated to the matching event + (where applicable) the
played card's type/ability. An unrecognized condition is not satisfied; see
:func:`evaluate`.
"""

from __future__ import annotations

import re

from simulator.cards.loader import CardDatabase


class _Unmodeled:
    """Returned by _evaluate when no branch understands the condition."""

    def __repr__(self) -> str:
        return "UNMODELED"


UNMODELED = _Unmodeled()


def evaluate(
    condition: str | None,
    event: str,
    context,
    cards: CardDatabase,
    state=None,
) -> bool:
    """Return True if the action should execute for ``event``/``context``.

    A condition no branch understands is NOT satisfied (changed 2026-10-27).
    It used to count as satisfied, so 371 cards fired effects whose condition
    was never checked -- about a third of all condition checks in simulated
    games -- while their rules still read "implemented".  The miss is logged
    as ``native_rule_condition_unmodeled`` and tools/build_card_rules.py marks
    such a card partial (see :func:`is_modeled`).
    """
    result = _evaluate(condition, event, context, cards, state)
    if result is UNMODELED:
        if state is not None:
            state.event_log.append({"event": "native_rule_condition_unmodeled", "condition": condition,
                                    "source_card_id": getattr(context, "source_card_id", None)})
        return False
    return bool(result)


def _phrase_events(c: str) -> tuple[str, ...]:
    """Events a trigger phrase names ("when you draw a card" -> on_card_drawn).

    Many conditions only restate the card's trigger.  The parser's trigger
    table already maps those phrases, so the condition holds exactly when the
    firing event is one of them.
    """
    from types import SimpleNamespace
    from simulator.rules.parser import _triggers
    if not re.match(r"^(?:when|whenever|after|each time|at the (?:start|end) of)\b", c.strip()):
        return ()
    return tuple(t for t in _triggers(SimpleNamespace(type="infantry", id=""), c) if t != "on_deploy")


def is_modeled(condition: str, events, cards: CardDatabase) -> bool:
    """Whether some branch understands ``condition`` for any of ``events``.

    Used by tools/build_card_rules.py to keep a card's status honest.  The
    probe state is a minimal board; a branch that raises on it still counts as
    understanding the phrase.
    """
    from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
    from simulator.effects.resolver import EffectContext
    state = GameState(current_player="p1", players={
        "p1": PlayerState("p1", "Germany", deck=["garrison"], hand=["garrison"], resources=ResourceState(5, 5),
                          units=[UnitState("u", "garrison", 1, 3, "p1", "support_line")]),
        "p2": PlayerState("p2", "Soviet", deck=["garrison"], resources=ResourceState(5, 5),
                          units=[UnitState("e", "garrison", 1, 3, "p2", "frontline")]),
    }, battlefield={"frontline": ["e"], "support_line": ["u"]})
    for event in (events or ("on_play",)):
        context = EffectContext("p1", "garrison", "u", "e", event=event, metadata={"played_card_id": "garrison"})
        try:
            if _evaluate(condition, event, context, cards, state) is not UNMODELED:
                return True
        except Exception:
            return True
    return False


def _evaluate(
    condition: str | None,
    event: str,
    context,
    cards: CardDatabase,
    state=None,
):
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
    # ── Board and history predicates (2026-10-27, tournament decks) ─────────
    board = _board_predicate(c, context, cards, state)
    if board is not None:
        return board
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
        controls = _controls_predicate(c, context, cards, state, player, opponent)
        if controls is not None:
            return controls
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
    # A condition that only restates the trigger holds on that trigger.
    events = _phrase_events(c)
    if events:
        return event in events
    return UNMODELED


# Words that may describe a unit in "if you control a German tank with 4 or
# more attack".  Any other word makes the predicate unreadable, which keeps it
# UNMODELED instead of quietly ignoring part of the condition.
_UNIT_TYPE_WORDS = {
    "unit": None, "units": None,
    "infantry": {"infantry"}, "tank": {"tank"}, "tanks": {"tank"}, "artillery": {"artillery"},
    "fighter": {"fighter"}, "fighters": {"fighter"}, "bomber": {"bomber"}, "bombers": {"bomber"},
    "air": {"fighter", "bomber"}, "ground": {"infantry", "tank", "artillery"},
}


def _controls_predicate(c, context, cards, state, player, opponent):
    """``if you control a/an/another <descriptor> [with N or more attack|defense]``
    and ``if the enemy controls ...``; None when the clause is not of that shape."""
    from simulator.rules import nations
    match = re.search(
        r"\b(you control|the enemy controls|your enemy controls|enemy controls)\s+(a|an|another|any)\s+"
        r"([a-z -]+?)(?:\s+with\s+(\d+)\s+or\s+more\s+(attack|defense))?\s*[.,;]?\s*$", c)
    if not match:
        return None
    side = player if match.group(1) == "you control" else opponent
    if side is None:
        return False
    words = match.group(3).split()
    wanted_types: set | None = None
    wanted_nation = None
    for word in words:
        if word in _UNIT_TYPE_WORDS:
            if _UNIT_TYPE_WORDS[word] is not None:
                wanted_types = _UNIT_TYPE_WORDS[word] if wanted_types is None else wanted_types & _UNIT_TYPE_WORDS[word]
        elif word in nations.NATION_BY_ADJECTIVE:
            wanted_nation = word
        else:
            return None       # an unknown descriptor: not readable here
    source_id = getattr(context, "source_unit_id", None)
    threshold = int(match.group(4)) if match.group(4) else None
    for unit in side.units:
        if match.group(2) == "another" and unit.instance_id == source_id:
            continue
        card = cards.get(unit.card_id) if unit.card_id in cards else None
        if card is None:
            continue
        if wanted_types is not None and card.type not in wanted_types:
            continue
        if wanted_nation and not nations.matches(card.nation, wanted_nation):
            continue
        if threshold is not None and (unit.attack if match.group(5) == "attack" else unit.defense) < threshold:
            continue
        return True
    return False


_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def _unit_matches_words(card, words) -> bool | None:
    """Whether ``card`` fits descriptor words ("british infantry"); None if a
    word is not understood."""
    from simulator.rules import nations
    for word in words:
        if word in _UNIT_TYPE_WORDS:
            wanted = _UNIT_TYPE_WORDS[word]
            if wanted is not None and card.type not in wanted:
                return False
        elif word in nations.NATION_BY_ADJECTIVE:
            if not nations.matches(card.nation, word):
                return False
        else:
            return None
    return True


def _board_predicate(c, context, cards, state):
    """Conditions the tournament decks rely on; None when ``c`` is none of them."""
    text = c.strip().rstrip(".").strip()
    if text.startswith("unless "):
        inner = _board_predicate("if " + text[len("unless "):], context, cards, state)
        return None if inner is None else not inner
    if state is None or getattr(context, "player_id", None) not in state.players:
        return None
    from simulator.rules import counters
    me = context.player_id
    player = state.players[me]
    opponent = next((candidate for pid, candidate in state.players.items() if pid != me), None)
    metadata = getattr(context, "metadata", {}) or {}
    # "if you have 0 kredits" / "5 or more (unspent) kredits"
    match = re.fullmatch(r"if you have (\d+)( or more| or less)? (?:unspent )?kredits?", text)
    if match:
        value, mode = int(match.group(1)), match.group(2) or ""
        kredits = player.resources.kredits
        return kredits >= value if "more" in mode else kredits <= value if "less" in mode else kredits == value
    # unit counts
    enemy_units = len(opponent.units) if opponent else 0
    if re.fullmatch(r"if you have equal or (?:less|fewer) units(?: on the battlefield)?(?: than the enemy)?", text):
        return len(player.units) <= enemy_units
    if re.fullmatch(r"if you (?:control|have) more units than the enemy", text):
        return len(player.units) > enemy_units
    if re.fullmatch(r"if you (?:control|have) fewer units than the enemy", text):
        return len(player.units) < enemy_units
    # battle counters
    match = re.fullmatch(r"if you have lost (\d+) or more kredit slots this battle", text)
    if match:
        return counters.kredit_slots_lost(state, me) >= int(match.group(1))
    # orders this turn ("this" order already counted when it was played)
    match = re.fullmatch(r"if this is the (first|second|third|fourth) order you (?:give|play) this turn", text)
    if match:
        return counters.orders_given_this_turn(state, me) == _ORDINALS[match.group(1)]
    if re.fullmatch(r"if you have (?:given|played) an order this turn", text):
        return counters.orders_given_this_turn(state, me) >= 1
    # deployment history: "if you deployed a British infantry last turn"
    match = re.fullmatch(r"if you (?:have )?deployed an? ([a-z -]+?) (this|last) turn", text)
    if match:
        words = [word for word in match.group(1).split() if word != "unit"]
        deployed = counters.units_deployed(state, me, 0 if match.group(2) == "this" else 1)
        if match.group(2) == "this" and getattr(context, "source_unit_id", None):
            # The unit whose own Deployment asks this has already been counted.
            deployed = deployed[:-1]
        results = [_unit_matches_words(cards.get(card_id), words) for card_id in deployed if card_id in cards]
        from simulator.rules.nations import NATION_BY_ADJECTIVE
        if any(result is None for result in results) or (not results and any(
                word not in _UNIT_TYPE_WORDS and word not in NATION_BY_ADJECTIVE for word in words)):
            return None
        return any(results)
    # "if you have/control a <descriptor> in the frontline / support line"
    match = re.fullmatch(r"if you (?:have|control) an? ([a-z -]+?) in (?:the |your )?(frontline|support line)", text)
    if match:
        position = "frontline" if match.group(2) == "frontline" else "support_line"
        words = [word for word in match.group(1).split() if word != "unit"]
        found = False
        for unit in player.units:
            if unit.position != position or unit.card_id not in cards:
                continue
            fits = _unit_matches_words(cards.get(unit.card_id), words)
            if fits is None:
                return None
            found = found or fits
        return found
    match = re.fullmatch(r"if you have an? ([a-z -]+?)", text)
    if match:
        return _controls_predicate("if you control a " + match.group(1), context, cards, state, player, opponent)
    # "it" = the unit this card targeted, snapshotted before the effect ran.
    target_owner = metadata.get("target_owner_id")
    if re.fullmatch(r"if it (?:was|is) friendly", text):
        return target_owner == me
    if re.fullmatch(r"if it (?:was|is) (?:an )?enemy(?: unit)?", text):
        return target_owner is not None and target_owner != me
    if re.fullmatch(r"if you do", text):
        return bool(getattr(context, "target_unit_id", None))
    if re.fullmatch(r"if it is the hq", text):
        return not getattr(context, "target_unit_id", None)
    match = re.fullmatch(r"if (?:it is |it's )?an? (tank|infantry|artillery|fighter|bomber|air unit|ground unit)", text)
    if match:
        target_card = metadata.get("target_card_id")
        if not target_card or target_card not in cards:
            return False
        return cards.get(target_card).type in (_UNIT_TYPE_WORDS.get(match.group(1).split()[0]) or {match.group(1)})
    return None


# (action kind, condition pattern) pairs where the condition describes when an
# installed ability acts, and the kind's own handler checks that moment.  The
# condition must not gate installing the ability (TYPE 89 CHI-RO, KATYUSHA).
_HANDLER_OWNED = (
    ("hq_defense_becomes_damage", r"^when a hq gains defense"),
    ("random_combat_damage", r"^when .+ attacks"),
)


def handled_by_kind(kind: str, condition: str | None) -> bool:
    """Whether ``kind``'s handler itself implements ``condition``."""
    c = (condition or "").strip().lower()
    return any(kind == owned_kind and re.search(pattern, c) for owned_kind, pattern in _HANDLER_OWNED)


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
