"""Cards whose OWN kredit cost depends on live state.

An ordinary cost change is a ``cost_modifier`` entry on the player: some other
card puts it there, and it applies to a class of cards.  Two cards instead
re-price *themselves* from the current state, which a stored modifier cannot
express, because the price has to be read at the moment the card is played:

    DETAILED RECON      "Cost is reduced by the amount spent operating this turn."
    SURPRISE ENGAGEMENT "Costs 2 less if Rain was played this turn."

Before this registry both were wrong in the same way: the reduction was parsed
into an effect that ran *after* the card was played, so SURPRISE ENGAGEMENT made
every OTHER card in hand 2 cheaper and its own cost never moved.

The registry is keyed by card id on purpose: each entry is one card's printed
rule, and a reader can check it against the card. ``card_play_cost`` applies it.
"""

from __future__ import annotations


def _operating_spend(state, player_id: str) -> int:
    from simulator.rules import counters
    return counters.operating_spend_this_turn(state, player_id)


def _two_if_rain_played(state, player_id: str) -> int:
    from simulator.rules import weather
    return 2 if weather.was_played_this_turn(state, player_id, "rain") else 0


REDUCTIONS = {
    "detailed_recon": _operating_spend,
    "surprise_engagement": _two_if_rain_played,
}


def reduction_for(state, player_id: str, card) -> int:
    """Kredits to take off ``card``'s own cost, 0 when it has no dynamic rule."""
    rule = REDUCTIONS.get(getattr(card, "id", ""))
    if rule is None:
        return 0
    try:
        return max(0, int(rule(state, player_id)))
    except Exception:                      # noqa: BLE001 - costing must never crash a turn
        return 0
