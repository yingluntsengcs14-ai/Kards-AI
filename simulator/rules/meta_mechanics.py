"""Mechanics found missing by the meta-deck card audit (2026-11-01).

``tools/card_probe.py`` and ``tools/listener_probe.py`` played every card of
the 17 tournament decks on a fixed board.  About 45 cards did something other
than their text; the rule kinds below are what they needed.  Each kind is
dispatched from ``NativeRuleEngine._execute_action`` through :func:`run`.

Continuous auras (TYPE 93, 909th ROSLAVL, MATILDA FROG, 3rd YOKOSUKA) are
recomputed by :func:`refresh_auras` after every action: the attack an aura
gives is kept in ``unit.status["aura_attack"]`` so it can be taken back when
the source leaves or the unit stops qualifying.
"""
from __future__ import annotations

import random

from simulator.actions.validator import opponent_id
from simulator.rules.abilities import AbilityEngine

AIR = {"fighter", "bomber"}
GROUND = {"infantry", "tank", "artillery"}
# Keywords that count as "combat keywords" (44th SILESIA, MONASTYR).
COMBAT_KEYWORDS = ("blitz", "shock", "fury", "guard", "ambush", "smokescreen", "heavyarmor", "alpine")

KINDS = {
    "draw_count", "draw_to_enemy_hand_size", "draw_random_units_of_cost", "buff_per", "slots_per_frontline",
    "destroy_same_name", "destroy_all_filtered", "units_op_cost", "frontline_this_turn", "resistance_to_enemy_deck",
    "order_to_deck_top_free", "pending_buff_top_unit", "buff_same_name_everywhere", "grant_destruction",
    "trigger_destruction", "hq_turn_start_effect", "extra_draw_next_turn", "deploy_last_destroyed_ground",
    "buff_random_friendly", "retreat_all_enemy", "aura",
}


def unit_card(cards, unit):
    return cards.get(unit.card_id) if unit.card_id in cards else None


def combat_keywords(cards, unit) -> int:
    card = unit_card(cards, unit)
    if card is None:
        return 0
    return sum(1 for keyword in COMBAT_KEYWORDS if AbilityEngine.has(card, unit, keyword)
               or any(str(a).lower().startswith(keyword) for a in card.abilities if keyword == "heavyarmor"))


def count(name: str, state, context, cards) -> int:
    me = context.player_id
    enemy = opponent_id(state, me)
    units = state.players[me].units
    if name == "enemy_unit_count":
        return len(state.players[enemy].units)
    if name == "unit_type_count":
        return len({unit_card(cards, u).type for u in units if unit_card(cards, u)})
    if name == "units_two_combat_keywords":
        return sum(1 for u in units if combat_keywords(cards, u) >= 2)
    if name == "other_friendly_air_count":
        return sum(1 for u in units if u.instance_id != context.source_unit_id
                   and (unit_card(cards, u).type if unit_card(cards, u) else "") in AIR)
    if name == "damaged_unit_count":
        return sum(1 for p in state.players.values() for u in p.units
                   if unit_card(cards, u) and unit_card(cards, u).defense is not None
                   and u.defense < unit_card(cards, u).defense)
    if name == "frontline_unit_count":
        return sum(1 for p in state.players.values() for u in p.units if u.position == "frontline")
    raise ValueError(name)


def _draw(state, engine, player_id, times):
    from simulator.core.turn import TurnManager
    start = len(state.event_log)
    for _ in range(max(0, times)):
        TurnManager.draw_card(state, player_id)
    engine.emit_draws_since(state, start)


def _find(state, unit_id):
    return next((u for p in state.players.values() for u in p.units if u.instance_id == unit_id), None)


def run(engine, action, state, context) -> None:
    cards = engine.cards
    me = context.player_id
    enemy = opponent_id(state, me)
    player = state.players[me]
    kind = action.kind

    if kind == "draw_count":
        _draw(state, engine, me, count(action.card_name, state, context, cards))
        return
    if kind == "draw_to_enemy_hand_size":
        # ENIGMA: "Draw cards until you have as many cards in your hand as the enemy."
        _draw(state, engine, me, len(state.players[enemy].hand) - len(player.hand))
        return
    if kind == "draw_random_units_of_cost":
        # Ki-46 DINAH III: "Draw two random 1-cost units from your deck."
        rng = random.Random((state.rng_seed or 0) + len(state.event_log))
        for _ in range(max(1, action.amount)):
            pool = [i for i, cid in enumerate(player.deck)
                    if cid in cards and cards.get(cid).is_unit and cards.get(cid).kredits == int(action.card_name or 0)]
            if not pool:
                break
            from simulator.rules.battlefield import HAND_CARD_LIMIT
            card_id = player.deck.pop(rng.choice(pool))
            if len(player.hand) < HAND_CARD_LIMIT:
                player.hand.append(card_id)
                state.event_log.append({"event": "card_drawn", "player_id": me, "card_id": card_id})
        return
    if kind == "buff_per":
        amount = count(action.card_name, state, context, cards)
        target = _find(state, context.source_unit_id if action.target == "source" else context.target_unit_id)
        if target is not None and amount:
            target.attack += action.attack * amount
            target.defense += action.defense * amount
            state.event_log.append({"event": "buff_per", "unit_id": target.instance_id, "times": amount})
        return
    if kind == "slots_per_frontline":
        # 15th ENGINEERS: slots for each unit in the frontline; your units gain that much attack.
        amount = count("frontline_unit_count", state, context, cards)
        added = min(amount, 12 - player.resources.max_kredits)
        player.resources.max_kredits += added
        for unit in player.units:
            unit.attack += amount
        state.event_log.append({"event": "kredit_slot_gained", "player_id": me, "amount": added})
        return
    if kind == "destroy_same_name":
        # OVERPOWER: the target and every other unit with its name.
        target = _find(state, context.target_unit_id)
        name = unit_card(cards, target).name if target is not None and unit_card(cards, target) else None
        if not name:
            return
        from simulator.effects import basic_effects
        start = len(state.event_log)
        for p in state.players.values():
            for unit in list(p.units):
                card = unit_card(cards, unit)
                if card is not None and card.name == name:
                    basic_effects.destroy(state, unit)
        engine.emit_deaths_since(state, start)
        return
    if kind == "destroy_all_filtered":
        from simulator.effects import basic_effects
        what, _, value = (action.scope or "").partition(":")
        start = len(state.event_log)
        doomed = []
        for p in state.players.values():
            for unit in p.units:
                if what == "max_attack" and unit.attack > int(value):
                    continue
                if what == "max_combat_keywords" and combat_keywords(cards, unit) > int(value):
                    continue
                doomed.append(unit)
        for unit in doomed:
            basic_effects.destroy(state, unit)
        engine.emit_deaths_since(state, start)
        return
    if kind == "units_op_cost":
        # LOTTA SVÄRD: "Your units get ... -1 operation cost."
        for unit in player.units:
            unit.modifiers.append({"type": "modify_operation_cost", "amount": action.amount})
        return
    if kind == "frontline_this_turn":
        # BLITZKRIEG: frontline units +3 attack and operate for one less until end of turn.
        for unit in player.units:
            if unit.position != "frontline":
                continue
            unit.attack += action.attack
            unit.modifiers.append({"type": "modify_operation_cost", "amount": action.amount, "this_turn": True})
            player.temporary_effects.append({"turn": state.turn_number, "reverts": [
                {"unit_id": unit.instance_id, "attr": "attack", "delta": -action.attack}]})
        return
    if kind == "resistance_to_enemy_deck":
        amount = action.amount
        if action.card_name == "excess":
            # ARMING THE RESISTANCE: deal the damage, count what the unit could not absorb.
            target = _find(state, context.target_unit_id)
            if target is None:
                return
            from simulator.effects import basic_effects
            amount = max(0, action.amount - target.defense)
            start = len(state.event_log)
            basic_effects.damage(state, target, action.amount, me)
            engine.emit_damage_since(state, start)
            engine.emit_deaths_since(state, start)
        deck = state.players[enemy].deck
        for _ in range(max(0, amount)):
            deck.append("resistance_b")
        random.Random((state.rng_seed or 0) + len(state.event_log)).shuffle(deck)
        state.event_log.append({"event": "card_added_to_enemy_deck", "player_id": enemy, "card_id": "resistance_b",
                                "amount": amount})
        return
    if kind == "order_to_deck_top_free":
        # GORDON HIGHLANDERS: chosen order costs 0 and goes on top of your deck.
        chosen = context.metadata.get("selected_card_id")
        if chosen in player.hand and cards.get(chosen).type == "order":
            player.hand.remove(chosen)
            player.deck.insert(0, chosen)
            player.status.setdefault("free_next_copy", {})
            player.status["free_next_copy"][chosen] = int(player.status["free_next_copy"].get(chosen, 0)) + 1
            state.event_log.append({"event": "order_to_deck_top_free", "card_id": chosen})
        return
    if kind == "pending_buff_top_unit":
        # USS THE SULLIVANS: the top unit of your deck gets +1+1.
        top = next((cid for cid in player.deck if cid in cards and cards.get(cid).is_unit), None)
        if top:
            player.status.setdefault("pending_buffs", {}).setdefault(top, []).append([action.attack, action.defense])
        return
    if kind == "buff_same_name_everywhere":
        # 116th: the chosen infantry and every card with its name in hand and deck.
        target = _find(state, context.target_unit_id)
        if target is None:
            return
        target.attack += action.attack
        target.defense += action.defense
        copies = player.hand.count(target.card_id) + player.deck.count(target.card_id)
        if copies:
            player.status.setdefault("pending_buffs", {}).setdefault(target.card_id, []).extend(
                [[action.attack, action.defense]] * copies)
        return
    if kind == "grant_destruction":
        # SPECIAL ATTACK (all friendly units) / JUNGLE WARFARE (the chosen unit).
        units = player.units if action.target == "friendly_units" else [u for u in [_find(state, context.target_unit_id)] if u]
        for unit in units:
            unit.status.setdefault("granted_destruction", []).append(action.card_name)
        return
    if kind == "trigger_destruction":
        target = _find(state, context.target_unit_id)
        if target is not None:
            fire_destruction(engine, state, target, trigger_own=True)
        return
    if kind == "hq_turn_start_effect":
        player.status.setdefault("hq_turn_start", []).append(action.card_name)
        return
    if kind == "extra_draw_next_turn":
        player.status["extra_draw_next_turn"] = int(player.status.get("extra_draw_next_turn", 0)) + max(1, action.amount)
        return
    if kind == "deploy_last_destroyed_ground":
        # IL-10: "Add your last destroyed ground unit to your support line."
        from simulator.rules import counters
        ground = [cid for cid in counters.destroyed_card_ids(state, me)
                  if cid in cards and cards.get(cid).type in GROUND and cid != context.source_card_id]
        if ground:
            engine._execute_action(type(action)("deploy_named", "owner", amount=1, card_name=cards.get(ground[-1]).name),
                                   state, context)
        return
    if kind == "buff_random_friendly":
        # 507th PIR: a random friendly infantry gets +2 attack.
        pool = [u for u in player.units if unit_card(cards, u) and unit_card(cards, u).type == (action.scope or "infantry")]
        if pool:
            unit = random.Random((state.rng_seed or 0) + len(state.event_log)).choice(pool)
            unit.attack += action.attack
        return
    if kind == "retreat_all_enemy":
        engine._execute_action(type(action)("retreat_enemy_frontline_all", "enemy_units"), state, context)
        return
    if kind == "aura":
        return      # read by refresh_auras


DESTRUCTIONS = {
    # SPECIAL ATTACK
    "hq1_slot1": lambda engine, state, owner: (
        __import__("simulator.core.hq", fromlist=["HQResolver"]).HQResolver.damage(
            state, opponent_id(state, owner), 1, owner),
        _gain_slot(state, owner)),
    # JUNGLE WARFARE
    "random_enemy2": lambda engine, state, owner: _random_enemy_damage(engine, state, owner, 2),
}


def _gain_slot(state, owner):
    resources = state.players[owner].resources
    if resources.max_kredits < 12:
        resources.max_kredits += 1


def _random_enemy_damage(engine, state, owner, amount):
    from simulator.core.hq import HQResolver
    from simulator.effects import basic_effects
    enemy = opponent_id(state, owner)
    targets = [None] + list(state.players[enemy].units)
    pick = random.Random((state.rng_seed or 0) + len(state.event_log)).choice(targets)
    if pick is None:
        HQResolver.damage(state, enemy, amount, owner)
    else:
        start = len(state.event_log)
        basic_effects.damage(state, pick, amount, owner)
        engine.emit_deaths_since(state, start)


def fire_destruction(engine, state, unit, trigger_own=False, owner=None):
    """Granted Destruction effects of ``unit`` (and its own, for JUNGLE WARFARE)."""
    owner = owner or unit.owner_id
    for name in list(unit.status.get("granted_destruction", ())):
        handler = DESTRUCTIONS.get(name)
        if handler:
            handler(engine, state, owner)
            state.event_log.append({"event": "granted_destruction_fired", "unit_id": unit.instance_id, "effect": name})
    if trigger_own:
        from simulator.effects.resolver import EffectContext
        engine.execute(unit.card_id, "on_destroy", state,
                       EffectContext(owner, unit.card_id, unit.instance_id, event="on_destroy"))


# ── continuous auras ───────────────────────────────────────────────────────
def _line_order(state, unit):
    return [uid for uid in state.battlefield.get(unit.position, [])
            if any(u.instance_id == uid and u.owner_id == unit.owner_id for u in state.players[unit.owner_id].units)]


def _wanted_auras(state, cards):
    attack = {}
    keywords = {}
    for p in state.players.values():
        for source in p.units:
            if source.status.get("suppressed"):
                continue
            cid = source.card_id
            if cid == "type_93":                      # Your other units have +1 attack.
                for u in p.units:
                    if u.instance_id != source.instance_id:
                        attack[u.instance_id] = attack.get(u.instance_id, 0) + 1
            elif cid == "909th_roslavl":              # Your infantry have Blitz.
                for u in p.units:
                    if unit_card(cards, u) and unit_card(cards, u).type == "infantry":
                        keywords.setdefault(u.instance_id, set()).add("blitz")
            elif cid == "matilda_frog":               # Units to the left of this unit have +2 attack.
                order = _line_order(state, source)
                if source.instance_id in order:
                    for uid in order[:order.index(source.instance_id)]:
                        attack[uid] = attack.get(uid, 0) + 2
            elif cid == "3rd_yokosuka":               # +1 attack for each adjacent Japanese unit.
                order = _line_order(state, source)
                if source.instance_id in order:
                    i = order.index(source.instance_id)
                    near = [uid for uid in (order[i - 1] if i else None, order[i + 1] if i + 1 < len(order) else None) if uid]
                    n = sum(1 for uid in near if (unit_card(cards, _find(state, uid)).nation or "") == "Japan")
                    if n:
                        attack[source.instance_id] = attack.get(source.instance_id, 0) + n
    return attack, keywords


def refresh_auras(state, cards) -> None:
    attack, keywords = _wanted_auras(state, cards)
    for p in state.players.values():
        for unit in p.units:
            have = int(unit.status.get("aura_attack", 0) or 0)
            want = attack.get(unit.instance_id, 0)
            if have != want:
                unit.attack = max(0, unit.attack + want - have)
                if want:
                    unit.status["aura_attack"] = want
                else:
                    unit.status.pop("aura_attack", None)
            granted = sorted(keywords.get(unit.instance_id, ()))
            if granted:
                unit.status["aura_abilities"] = granted
            else:
                unit.status.pop("aura_abilities", None)


def apply_pending_buffs(state, player_id, unit) -> None:
    """USS THE SULLIVANS / 116th: a buff waiting for this card to be deployed."""
    pending = state.players[player_id].status.get("pending_buffs") or {}
    queue = pending.get(unit.card_id)
    if queue:
        attack, defense = queue.pop(0)
        unit.attack += attack
        unit.defense += defense
        if not queue:
            pending.pop(unit.card_id, None)


def resistance_drawn(state, player_id) -> None:
    """RESISTANCE: "Play for free when drawn. Lose 1 kredit. Take 1 HQ damage.
    Pin a friendly unit. Draw a card."  It plays itself the moment it is drawn."""
    from simulator.core.hq import HQResolver
    from simulator.core.turn import TurnManager
    from simulator.rules.pin import apply as apply_pin
    player = state.players[player_id]
    if "resistance_b" not in player.hand:
        return
    player.hand.remove("resistance_b")
    state.graveyard.setdefault(player_id, []).append("resistance_b")
    player.resources.kredits = max(0, player.resources.kredits - 1)
    HQResolver.damage(state, player_id, 1)
    if player.units:
        apply_pin(random.Random((state.rng_seed or 0) + len(state.event_log)).choice(player.units))
    state.event_log.append({"event": "resistance_played_on_draw", "player_id": player_id})
    if state.game_status.value == "in_progress":
        TurnManager.draw_card(state, player_id)
