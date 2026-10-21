"""Dynamic amounts: "Deal damage to X equal to <something about the board>".

Twenty catalog cards name a quantity instead of a number ("equal to cards in the
enemy hand", "equal to the total attack of your air units").  The parser used to
turn these into ``damage`` with ``amount=0`` and the note "equal to referenced
value", so every one of them dealt **zero** damage while being marked
implemented.  This module turns the named quantity into a number.

A reference is resolved against the live state, so nothing has to be cached.
References that need a battle- or turn-long counter the simulator does not keep
yet (kredit slots lost this battle, units of a nation destroyed this battle,
damage to your HQ this turn, kredits spent operating this turn) are deliberately
absent: their cards stay ``partial`` instead of silently dealing 0.
"""

from __future__ import annotations

AIR_TYPES = frozenset({"fighter", "bomber"})
GROUND_TYPES = frozenset({"infantry", "tank", "artillery"})

# reference name -> what it counts, for the rule store and for error messages.
REFERENCES = (
    "target_cost",                      # its cost (the targeted card's kredits)
    "target_attack",                    # its attack
    "friendly_unit_count",              # units you control
    "friendly_air_count",               # friendly air units
    "other_friendly_air_count",         # other friendly air units
    "friendly_air_total_attack",        # total attack of your air units
    "friendly_bomber_count_plus_one",   # 1 plus the bombers you control
    "friendly_unit_type_count_plus_one",  # 1 plus the unit types you control
    "enemy_hand_count",                 # cards in the enemy hand
    "known_enemy_hand_count",           # known cards in the enemy hand
    "unspent_kredits",                  # your unspent kredits
    "frontline_nation_tank_total_cost",  # total cost of your <nation> frontline tanks
    "non_nation_friendly_unit_count",   # friendly units not of a nation
    # Counter-backed (see simulator.rules.counters).
    "kredit_slots_lost_this_battle",
    "kredit_slots_lost_this_battle_plus_one",
    "nation_units_destroyed_this_battle",
    "hq_damage_this_turn",
    "kredits_spent_operating_this_turn",
)


from simulator.rules.nations import normalize as _normalize_nation


def _units(state, player_id):
    return list(state.players[player_id].units)


def _card(cards, unit):
    return cards.get(unit.card_id) if unit.card_id in cards else None


def evaluate(reference: str, state, context, cards, nation: str | None = None) -> int | None:
    """Resolve a reference to an amount, or None when it cannot be resolved."""
    from simulator.actions.validator import ActionValidationError, find_unit, opponent_id

    player_id = context.player_id
    enemy_id = opponent_id(state, player_id)

    def target_unit():
        if not getattr(context, "target_unit_id", None):
            return None
        try:
            return find_unit(state, context.target_unit_id)
        except ActionValidationError:
            return None

    if reference == "target_cost":
        unit = target_unit()
        card = _card(cards, unit) if unit else None
        return card.kredits if card else None
    if reference == "target_attack":
        unit = target_unit()
        return unit.attack if unit else None
    if reference == "friendly_unit_count":
        return len(_units(state, player_id))
    if reference in ("friendly_air_count", "other_friendly_air_count"):
        air = [unit for unit in _units(state, player_id)
               if (_card(cards, unit).type if _card(cards, unit) else "") in AIR_TYPES]
        if reference == "other_friendly_air_count":
            air = [unit for unit in air if unit.instance_id != getattr(context, "source_unit_id", None)]
        return len(air)
    if reference == "friendly_air_total_attack":
        return sum(unit.attack for unit in _units(state, player_id)
                   if (_card(cards, unit).type if _card(cards, unit) else "") in AIR_TYPES)
    if reference == "friendly_bomber_count_plus_one":
        return 1 + sum(1 for unit in _units(state, player_id)
                       if (_card(cards, unit).type if _card(cards, unit) else "") == "bomber")
    if reference == "friendly_unit_type_count_plus_one":
        types = {_card(cards, unit).type for unit in _units(state, player_id) if _card(cards, unit)}
        return 1 + len(types)
    if reference == "enemy_hand_count":
        return len(state.players[enemy_id].hand)
    if reference == "known_enemy_hand_count":
        from simulator.rules import intel
        known = getattr(intel, "known_cards", None)
        if callable(known):
            return len(known(state, player_id, enemy_id) or ())
        # Fall back to the intel record kept on the player's status.
        revealed = state.players[player_id].status.get("known_enemy_cards")
        return len(revealed) if isinstance(revealed, (list, tuple)) else None
    if reference == "unspent_kredits":
        return state.players[player_id].resources.kredits
    if reference == "frontline_nation_tank_total_cost":
        total = 0
        for unit in _units(state, player_id):
            card = _card(cards, unit)
            wanted = _normalize_nation(nation) if nation else None
            if (card and card.type == "tank" and unit.position == "frontline"
                    and (wanted is None or (card.nation or "").lower() == wanted)):
                total += card.kredits
        return total
    if reference in ("kredit_slots_lost_this_battle", "kredit_slots_lost_this_battle_plus_one"):
        from simulator.rules import counters
        lost = counters.kredit_slots_lost(state, player_id)
        return lost + 1 if reference.endswith("_plus_one") else lost
    if reference == "nation_units_destroyed_this_battle":
        if nation is None:
            return None
        from simulator.rules import counters
        return counters.units_destroyed_of_nation(state, _normalize_nation(nation), cards)
    if reference == "hq_damage_this_turn":
        from simulator.rules import counters
        return counters.hq_damage_this_turn(state, player_id)
    if reference == "kredits_spent_operating_this_turn":
        from simulator.rules import counters
        return counters.operating_spend_this_turn(state, player_id)
    if reference == "non_nation_friendly_unit_count":
        if nation is None:
            return None
        wanted = _normalize_nation(nation)
        return sum(1 for unit in _units(state, player_id)
                   if _card(cards, unit) and (_card(cards, unit).nation or "").lower() != wanted)
    return None
