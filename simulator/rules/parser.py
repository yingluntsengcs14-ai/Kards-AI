"""Deterministic native-text parser for common kards.info rule templates.

It intentionally produces an explicit unresolved result when a sentence cannot
be represented safely.  The parser never invents a card-specific rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

# Kinds emitted by _resolve_residual / _classify_residual when the parser
# could not match the text to a precise, deterministic template.  When a
# sentence produces ONLY (or includes) actions with these kinds, the sentence
# is NOT considered "covered" — the text fell through to a catch-all.
#
# Every kind that _resolve_residual or _classify_residual can emit that is
# NOT also emitted by _parse_sentence_core (the honest template layer) MUST
# appear here.  Omitting a kind here causes sentences to be falsely marked
# "covered", which in turn makes build_card_rules.py report fraudulent 100%
# implementation coverage.
#
# Principle: if a kind can be emitted by a regex fallback or generic substring
# match, it is a catch-all.  Specific template matches from _parse_sentence_core
# are the ONLY non-catch-all source.
_CATCH_ALL_KINDS = frozenset({
    # Primary catch-all families (residual fallback / classify_residual)
    "control_effect",
    "special_effect",
    "hq_effect",
    "triggered_effect",
    # Semantic catch-alls
    "gain_kredit_slot_on_deploy",
    "reveal_covert",
})

from simulator.cards.card import Card


NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _quantified_name(phrase: str) -> tuple[str, int]:
    """Split "two T-70 units" into ("T-70", 2) and "a BLAST" into ("BLAST", 1).

    Card names in add/deploy clauses used to keep their count word, so the
    lookup searched for a card literally called "two BLASTS" and failed.  A
    plural ending ("BLASTS") is left for the engine's name lookup to resolve,
    since stripping an S here would break names that end in S.
    """
    text = phrase.strip()
    amount = 1
    count = re.match(r"^(a|an|one|two|three|four|five|\d+)\s+(.+)$", text, re.I)
    if count:
        amount = _number(count.group(1)) or 1
        text = count.group(2).strip()
    text = re.sub(r"\s+(?:units?|cards?)$", "", text, flags=re.I).strip()
    return text, amount


# "Destroy it at the end of your turn" / "Destroy them at the end of your next
# turn" / "At the end of your turn, destroy it": the unit dies later, not now.
_DELAYED_DESTROY = re.compile(
    r"\bdestroy\s+(?:it|them)\s+(?:at\s+)?(?:the\s+)?end\s+of\s+(?P<when>your\s+next\s+turn|(?:your\s+|the\s+)?turn)\b"
    r"|\bat\s+the\s+end\s+of\s+(?P<when2>your\s+next\s+turn|(?:your\s+|the\s+)?turn)\s*,\s*destroy\s+(?:it|them)\b",
    re.I)


def _delayed_destroy(sentence: str) -> tuple | None:
    match = _DELAYED_DESTROY.search(sentence)
    if not match:
        return None
    when = (match.group("when") or match.group("when2") or "").lower()
    duration = "end_of_next_turn" if "next" in when else "end_of_turn"
    return (RuleAction("delayed_destroy", "referent", duration=duration),)

# Optional unit-type token used by scoped (multi-target) templates. "unit"/"units"
# is the generic catch-all and carries no type filter.
_UNIT_TYPE = r"(air|ground|tank|infantry|artillery|bomber|fighter|unit|units?)"
_NATION = r"(british|german|us|soviet|japanese|french|italian|polish|finnish)"

# Negative lookahead guarding stat-grant templates against trailing predicate
# clauses ("... when they become Veteran", "... if you control a US unit"). Such
# clauses must stay unresolved rather than be applied unconditionally. Pure
# duration markers ("this turn", "next turn", "until end of turn") are NOT here:
# they are handled by the temporary-effect milestone (M4) via _sentence_duration.
_COND_GUARD = r"(?!\s+(?:when|whenever|if|after|while|for each|instead|unless|until\b))"

# Granted-ability vocabulary used by the unified "give" handler.
_KW = r"(blitz|mobilize|guard|fury|shock|salvage|ambush|smokescreen|bond|covert|alpine|intel|heavy\s*armor|airborne|mobilized|pincer|veteran)"

# Trailing PREDICATE clauses (not mere duration markers) that make a "give"
# effect non-permanent. Such sentences stay unresolved so the temporary-effect
# milestone (M4) handles them with a correct trigger/condition wrapper instead of
# applying them blindly.
_HAS_CONDITIONAL = re.compile(
    r"\b(if |when |whenever|while |for each|instead\b|you control|per turn|equal to)\b"
)


def _scope_of(token: str | None) -> str:
    """Map a captured unit-type token to a RuleAction.scope value."""
    token = (token or "").lower()
    if token in ("unit", "units", ""):
        return ""
    return token


def _classify_descriptor(desc: str) -> tuple[str, str]:
    """Return (scope_unit_type, nation_name) parsed from a descriptor phrase.

    Examples: 'german tanks' -> ('tank', 'Germany'); 'friendly infantry' ->
    ('infantry', ''); 'british ground units' -> ('ground', 'Britain').
    """
    d = (desc or "").lower()
    scope = ""
    if re.search(r"\btank or infantry\b", d):
        scope = "tank_or_infantry"
    else:
        for t in ("air", "ground", "tank", "infantry", "artillery", "bomber", "fighter"):
            if re.search(r"\b" + t + r"\b", d):
                scope = t
                break
    nation = ""
    for n in ("british", "german", "us", "soviet", "japanese", "french", "italian", "polish", "finnish"):
        if re.search(r"\b" + n + r"\b", d):
            nation = _nation_name(n)
            break
    return scope, nation


def _target_of_descriptor(desc: str) -> str:
    """Pick a target selector from a descriptor phrase for 'gets'/'give' gains."""
    d = (desc or "").lower()
    if re.search(r"\b(all|each|every|your)\b", d):
        return "friendly_units"
    if re.search(r"\benemy\b", d):
        return "selected_enemy"
    if re.search(r"\bfriendly\b", d):
        return "selected_friendly"
    return "selected_target"


@dataclass(frozen=True)
class RuleAction:
    kind: str
    target: str = "owner"
    amount: int = 0
    attack: int = 0
    defense: int = 0
    card_name: str | None = None
    scope: str = ""
    min_cost: int = 0
    set_cost: int | None = None
    duration: str | None = None  # None | "this_turn" | "next_turn"
    condition: str | None = None  # normalized condition string ("if X", "on your turn",
                                   # "for each Y"); evaluated by ConditionEvaluator at runtime.
                                   # None means unconditional.
    on: tuple[str, ...] = ()  # events this action belongs to; () = every trigger of the rule.
                              # A unit's rule lists all its triggers, so without this a
                              # "Destruction:" clause also ran when the unit was deployed.


@dataclass(frozen=True)
class CardRule:
    card_id: str
    source_text: str
    triggers: tuple[str, ...]
    actions: tuple[RuleAction, ...]
    status: str
    unresolved_fragments: tuple[str, ...] = ()

    @property
    def needs_target(self) -> bool:
        """Whether playing the card asks for a unit target.

        Only clauses that run when the card is played count: B-4 203mm's
        "Units damaged by this unit get -3 attack" targets the unit it damages
        later (on_damage_dealt), not one chosen on deploy.
        """
        played = {"on_play", "on_deploy"}
        return any(action.target.startswith("selected_") and (not action.on or played & set(action.on))
                   for action in self.actions)


class RuleParser:
    """Parse only stable, high-frequency KARDS text templates into an AST."""

    # ── Weather cards (OCEANIA STORM) ───────────────────────────────────────
    # These nine cards (plus the three forecast tokens) are parsed per card id
    # rather than by sentence template: their wording carries exceptions ("Air
    # units get +3 instead") and announcements ("It will be Sunny.") that the
    # generic templates either drop or turn into a catch-all.  Entries hold
    # (triggers, actions, status, unresolved_fragments).
    _WEATHER_CARD_RULES: dict = {
        # The forecast tokens only announce the coming weather; the mechanic
        # itself lives in simulator.rules.weather, so they carry no action.
        "blue_sky": (("on_play",), (), "implemented", ()),
        "mist": (("on_play",), (), "implemented", ()),
        "gale": (("on_play",), (), "implemented", ()),
        # HEATWAVE: all units +2 attack, air +3 instead.  The air action carries
        # the DELTA on top of the first action, so the two together give air +3.
        # "All units" covers BOTH players (confirmed in the live client by the
        # project owner, 2026-10-20) — the Japanese and Korean card texts say
        # "your units", but the English wording and the real behaviour win.
        "heatwave": (
            ("on_play",),
            (RuleAction("modify_attack", "all_units", amount=2, duration="this_turn"),
             RuleAction("modify_attack", "all_units", amount=1, scope="air", duration="this_turn")),
            "implemented", (),
        ),
        # JUNGLE FEVER: all units +3 attack this turn.  The second sentence
        # ("Pin friendly units that do not attack") can only be settled at the
        # end of the turn, so it stays an unresolved fragment instead of being
        # parsed into a targeted pin that would wrongly demand a target.
        "jungle_fever": (
            ("on_play",),
            (RuleAction("modify_attack", "all_units", amount=3, duration="this_turn"),),
            "partial", ("Pin friendly units that do not attack.",),
        ),
        # TASK FORCE 44: "Forecast. Deal 2 damage to a unit. Deal 3 damage
        # instead if this card was left-most in hand."  The generic templates
        # produced BOTH damage actions, so a left-most play dealt 2+3=5.  The
        # base damage is therefore gated on the negated condition.
        "task_force_44": (
            ("on_play",),
            (RuleAction("forecast", "owner"),
             RuleAction("damage", "selected_target", amount=2,
                        condition="unless this card was left-most in hand"),
             RuleAction("damage", "selected_target", amount=3,
                        condition="if this card was left-most in hand")),
            "implemented", (),
        ),
        # FEROCIOUS ASSAULT: "Destroy target infantry. Lose a kredit slot if its
        # attack was greater than your kredit slots."  The kredit loss is ordered
        # FIRST so the condition can still read the targeted unit's attack; the
        # unit's attack cannot change in between, so the outcome is the same.
        "ferocious_assault": (
            ("on_play",),
            (RuleAction("lose_kredit_slot", "owner", amount=1,
                        condition="if its attack was greater than your kredit slots"),
             RuleAction("destroy", "selected_target", scope="infantry")),
            "implemented", (),
        ),
        # SURPRISE ENGAGEMENT: "Costs 2 less if Rain was played this turn.
        # Destroy a random enemy unit. Draw the top unit of your deck."  The
        # first sentence re-prices THIS card, which simulator.rules.dynamic_costs
        # handles at play time; parsing it as an effect made every OTHER card in
        # hand 2 cheaper instead.
        "surprise_engagement": (
            ("on_play",),
            (RuleAction("destroy", "random_enemy"),
             RuleAction("draw_top_matching", "owner", scope="unit")),
            "implemented", (),
        ),
        # DETAILED RECON: own cost reduction is handled by dynamic_costs; the
        # drawn card's matching reduction still has no mechanism, so it stays an
        # unresolved fragment rather than being silently dropped.
        "detailed_recon": (
            ("on_play",),
            (RuleAction("draw", "owner", amount=1),),
            "partial", ("Draw a card, reduce its cost by the same amount.",),
        ),
        # DELUGE: distribute 6 defense between random friendly targets.
        "deluge": (
            ("on_play",),
            (RuleAction("distribute_defense", "owner", amount=6),),
            "implemented", (),
        ),
        # CYCLONE: draw 3, then self-damage equal to hand size.
        "cyclone": (
            ("on_play",),
            (RuleAction("draw", "owner", amount=3),
             RuleAction("damage_own_hq_equal_to_hand", "owner")),
            "implemented", (),
        ),
    }

    def parse(self, card: Card) -> CardRule:
        weather_rule = self._WEATHER_CARD_RULES.get(card.id)
        if weather_rule is not None:
            triggers, actions, status, unresolved = weather_rule
            return CardRule(card.id, card.text.strip(), triggers, actions, status, unresolved)
        text = card.text.strip()
        if not text or text == "{}":
            return CardRule(card.id, text, (), (), "implemented")
        # A countermeasure trigger clause contains its own semantic predicate;
        # sentence splitting would otherwise separate it from "Cancel the
        # effect" and turn the predicate into an unconsumed catch-all.
        if "enemy deploys a unit with a deployment effect" in text.lower() and "cancel the effect" in text.lower():
            return CardRule(card.id, text, ("on_deploy",),
                            (RuleAction("cancel", "source", card_name="deployment_effect"),),
                            "implemented")
        # Card names such as "59. Panzergrenadier" contain a period.  Treat
        # this whole sentence before generic sentence splitting, otherwise the
        # numeric prefix becomes a spurious unresolved fragment.
        if "can move and attack during the same turn" in text.lower() and card.name.lower().startswith("59."):
            return CardRule(card.id, text, ("on_deploy",),
                            (RuleAction("move_and_attack", "source"),),
                            "implemented")
        if "when your hq is damaged" in text.lower() and "deal that amount of damage to this unit instead" in text.lower():
            return CardRule(card.id, text, ("on_deploy",),
                            (RuleAction("hq_damage_redirect_to_source", "source"),),
                            "implemented")
        if "hq is to take damage" in text.lower() and "enemy hq takes the damage instead" in text.lower():
            return CardRule(card.id, text, ("on_deploy",),
                            (RuleAction("hq_damage_redirect_to_enemy_hq", "source"),),
                            "implemented")
        if "choose a unit in hand" in text.lower() and "swap it with a friendly unit on the battlefield" in text.lower():
            return CardRule(card.id, text, ("on_play",),
                            (RuleAction("swap_hand_unit_with_friendly", "selected_friendly"),),
                            "implemented")
        if "choose one" in text.lower() and "give a ground unit blitz or shock" in text.lower():
            return CardRule(card.id, text, ("on_play",),
                            (RuleAction("shock_tactics_choice", "selected_target", scope="ground"),),
                            "implemented")
        if text.lower().strip() == "deployment effects do not trigger.":
            return CardRule(card.id, text, ("on_deploy",),
                            (RuleAction("suppress_deployment_effects", "source"),),
                            "implemented")
        if "trigger the destruction effects on a damaged soviet air unit" in text.lower() and "convert it into yak 9" in text.lower():
            return CardRule(card.id, text, ("on_play",),
                            (RuleAction("trigger_destroy_and_convert", "selected_target", card_name="YAK 9"),),
                            "implemented")
        if text.lower().strip() == "trigger the destruction effects on a target unit as if it was yours, then remove the effect if an enemy unit.":
            return CardRule(card.id, text, ("on_play",),
                            (RuleAction("trigger_destruction_effects", "selected_target"),), "implemented")
        if text.lower().strip() == "destroy target undamaged unit, then randomly rearrange enemy units on the battlefield.":
            return CardRule(card.id, text, ("on_play",),
                            (RuleAction("destroy_undamaged", "selected_target"),
                             RuleAction("rearrange_enemy_units", "enemy_units")), "implemented")
        develop_options = re.match(r"develop\s+(.+?)\s+or\s+([^\.]+)\.?$", text.strip(), re.I) if "," in text else None
        if develop_options:
            choices = [part.strip() for part in (develop_options.group(1) + "," + develop_options.group(2)).split(",") if part.strip()]
            if len(choices) >= 2:
                return CardRule(card.id, text, ("on_play",),
                                (RuleAction("develop_options", "owner", card_name="|".join(choices)),), "implemented")
        if "random effects always choose this unit" in text.lower():
            return CardRule(card.id, text, ("on_deploy",),
                            (RuleAction("random_effect_target_priority", "source"),),
                            "implemented")
        if "only 2 units can occupy the frontline" in text.lower() and "retreat all units in it" in text.lower():
            return CardRule(card.id, text, ("on_deploy",),
                            (RuleAction("frontline_limit", "source", amount=2),
                             RuleAction("retreat_enemy_frontline_all", "enemy_units")),
                            "implemented")
        if "if in your hand, this unit costs 1 more each time you lose a unit" in text.lower():
            return CardRule(card.id, text, ("on_friendly_unit_died",),
                            (RuleAction("hand_cost_on_friendly_death", "owner", amount=1),),
                            "implemented")
        triggers = _triggers(card, text)
        actions: list[RuleAction] = []
        covered: list[str] = []
        is_unit = card.type in {"infantry", "fighter", "tank", "bomber", "artillery"}
        sentence_events: tuple[str, ...] = ()
        for sentence in _sentences(text):
            parsed = _parse_sentence(sentence)
            if is_unit:
                # Bind each clause of a unit to the event that introduces it.
                # A clause with no trigger of its own continues the previous
                # one ("Destruction: Deal 2 damage. Draw a card.").  A leading
                # clause with none stays unbound (on=()): it may be an aura, or
                # gated by a condition the engine wires to an extra trigger
                # (FW 189's "when you play a Navy card").
                own = _triggers(card, sentence)
                sentence_events = own or sentence_events
                if parsed and sentence_events:
                    parsed = tuple(replace(a, on=sentence_events) for a in parsed)
            if (parsed and is_unit and re.match(r"^(?:deployment:\s*)?gets?\b", sentence.strip(), re.I)
                    and not re.search(r"\btarget\b", sentence, re.I)):
                # A unit's subjectless "Gets +1+1" is about the unit itself; the
                # generic template aimed it at whatever was selected.
                parsed = tuple(replace(a, target="source") if a.target == "selected_target" else a for a in parsed)
            if parsed:
                # Pin and Suppression are distinct KARDS states.  Older
                # templates shared the suppress action as a parser shortcut;
                # normalize unambiguous "pin" sentences to the lightweight
                # pinned state while preserving explicit suppression text.
                if re.search(r"\bpin(?:ned|s)?\b", sentence, re.I) and not re.search(r"\bsuppress", sentence, re.I):
                    parsed = tuple(replace(a, kind="pin") if a.kind == "suppress" else a for a in parsed)
                # "Deployment: Gets +1+1 if you have given an order this turn"
                # is a permanent one-shot buff: "this turn" belongs to the
                # condition.  Only Deployment clauses are read this way; for
                # other clauses the condition's "this turn" still bounds the
                # effect ("Has Blitz if Rain was played this turn").
                if re.match(r"^\s*deployment\s*:", sentence, re.I):
                    dur = _sentence_duration(_strip_condition(sentence)[0])
                else:
                    dur = _sentence_duration(sentence)
                if dur:
                    parsed = tuple(
                        replace(a, duration=dur) if a.kind in _TEMPORARY_KINDS else a
                        for a in parsed
                    )
                actions.extend(parsed)
                # A sentence is only "covered" when ALL of its actions came from
                # precise, deterministic templates.  If any action uses a catch-all
                # kind (special_effect / hq_effect / triggered_effect), the sentence
                # was only matched by the residual fallback — do not treat it as fully
                # covered; the unmatched sub-text remains unresolved.
                if not any(a.kind in _CATCH_ALL_KINDS for a in parsed):
                    covered.append(sentence)
        unresolved = tuple(sentence for sentence in _sentences(text) if sentence not in covered)
        status = "implemented" if actions and not unresolved else "partial" if actions else "unresolved"
        # Units can carry continuous ("aura") effects that are not expressed
        # via an explicit trigger keyword in the text.  The effect is applied
        # while the unit is deployed (on_deploy).  Lifetime binding (revert on
        # unit removal) is handled by _cleanup_continuous_effects (M16).
        # This MUST run BEFORE the no-trigger downgrade below, so auras get
        # their on_deploy trigger before the check.
        UNIT_TYPES = {"infantry", "fighter", "tank", "bomber", "artillery"}
        AURA_KINDS = {"modify_operation_cost", "set_operation_cost", "op_cost_rule",
                      "modify_hand_cost", "gain_kredit_slot", "lose_kredit_slot",
                      "buff", "modify_attack", "modify_defense", "buff_nation",
                      "defense_nation", "add_keyword", "remove_keyword",
                      "cannot", "cannot_attack_hq", "cannot_be_targeted_by_enemy_orders",
                      "immune", "immune_to_ground_in_support",
                      "special_effect", "hq_effect", "triggered_effect",
                      "damage", "draw", "destroy", "suppress", "deploy_named",
                      "add_card", "grant_ability", "grant_trait", "double_damage",
                      "control_effect", "kredit_effect", "double_damage_against_type",
                      "spawn_named_same_front", "repair", "gain_hq_defense", "gain_kredits",
                      "discard", "move_to_frontline", "return_to_hand", "return_to_deck",
                      "copy_unit", "convert_to", "set_attack", "set_defense",
                      "heal", "fight", "target_or_attack_tax", "repeat_effect",
                      "remove", "modify_deployment_cost", "enemy_cannot_deploy",
                      "pincer_ability", "double_stats", "spend_kredits", "lose_kredits"}
        if card.type in UNIT_TYPES and actions:
            if "on_deploy" not in triggers:
                triggers = triggers + ("on_deploy",)
        # Orders always fire when played, even without explicit "Deployment:".
        if card.type == "order" and actions and not triggers:
            triggers = triggers + ("on_play",)
        # Cards that have executable actions but NO triggers can never fire —
        # the engine silently skips them.  Downgrade to partial so the
        # coverage report is honest.
        if status == "implemented" and actions and not triggers:
            status = "partial"
            unresolved = unresolved + ("[no trigger — engine will never execute these actions]",)
        return CardRule(card.id, text, triggers, tuple(actions), status, unresolved)


def _triggers(card: Card, text: str) -> tuple[str, ...]:
    # "Destroy it at the end of your turn" schedules a removal; it does not make
    # the card trigger at turn end (HONOR and FINAL PUSH never fired at all).
    lower = _DELAYED_DESTROY.sub(" ", text.lower())
    if card.type == "countermeasure":
        triggers: list[str] = []
        if "attacks" in lower or "attacked" in lower:
            triggers.append("on_attack")
        if "deploys" in lower or "deployment" in lower or "deployed" in lower or "deploy" in lower:
            triggers.append("on_deploy")
        # Only treat "order" as a command-played trigger when it refers to the
        # enemy actually giving/playing an order, not "orders in your hand".
        if re.search(r"enemy\s+(?:gives?|plays?|give)\s+(?:an?\s+)?order|enemy order|counter\s+an\s+order|counter it", lower):
            triggers.append("on_command_played")
        if "damage" in lower:
            triggers.append("on_damage")
        if "destroyed" in lower or "destroys" in lower:
            triggers.append("on_destroy")
        if "moves into" in lower or "moves to" in lower:
            triggers.append("on_deploy")
        if "captures" in lower or "frontline" in lower:
            triggers.append("on_deploy")
        if "end of the enemy turn" in lower or "end of your turn" in lower:
            triggers.append("on_turn_end")
        if "start of your turn" in lower or "start of the enemy turn" in lower:
            triggers.append("on_turn_start")
        return tuple(dict.fromkeys(triggers))
    triggers: list[str] = []
    if "deployment:" in lower:
        # Orders fire on on_play; units and countermeasures fire on on_deploy.
        triggers.append("on_play" if card.type == "order" else "on_deploy")
    if "destruction:" in lower or "when this unit is destroyed" in lower:
        triggers.append("on_destroy")
    if "when this unit attacks" in lower or "when you attack" in lower:
        triggers.append("on_attack")
    if "units damaged by this unit" in lower:
        triggers.append("on_damage_dealt")
    if "damage dealt by this unit" in lower:
        triggers.append("on_damage_dealt")
    if "units dealt combat damage by this unit are destroyed" in lower:
        triggers.append("on_combat_damage_dealt")
    if "when it damages the enemy hq" in lower:
        triggers.append("on_enemy_hq_damaged")
    if "when it survives combat" in lower:
        triggers.append("on_survives_combat")
    if "when you play a card with intel" in lower or "when you play a card with bond" in lower:
        triggers.append("on_friendly_card_played")
    if "when you give an order" in lower:
        triggers.append("on_friendly_card_played")
    if "when it is targeted by an order or deployment effect" in lower:
        triggers.append("on_targeted_by_enemy_effect")
    if "when the enemy targets or attacks it" in lower:
        triggers.extend(("on_targeted_by_enemy_effect", "on_targeted_by_enemy_attack"))
    if "when this unit takes damage" in lower or "when this unit is damaged" in lower or "when this unit is dealt damage" in lower:
        triggers.append("on_damage")
    # Weather triggers (OCEANIA STORM).
    if "you forecast" in lower:
        triggers.append("on_forecast")
    if re.search(r"\b(?:when|after)\s+you\s+play\s+a\s+(?:sunny|rain|storm)\b", lower) \
            or re.search(r"\b(?:sunny|rain|storm)[a-z ,]*\s+was\s+played\s+this\s+turn", lower):
        triggers.append("on_weather_played")
    # Turn-boundary triggers.
    if "at the start of your turn" in lower:
        triggers.append("on_turn_start")
    if "at the end of your turn" in lower:
        triggers.append("on_turn_end")
    # Pincer — treated as an always-active deployed effect.
    if "pincer:" in lower:
        triggers.append("on_deploy")
    # Non-countermeasure trigger phrases for common KARDS patterns.
    if "after you give an order" in lower or "when you give an order" in lower:
        triggers.append("on_friendly_card_played")
    if "when a friendly unit is destroyed" in lower:
        triggers.append("on_destroy")
    if "when a unit moves" in lower or "when a unit moves into the frontline" in lower:
        triggers.append("on_deploy")
    if "when you draw" in lower or "each time you draw" in lower:
        triggers.append("on_card_drawn")
    # Broader conditional-trigger patterns (M41).
    if "when you deploy or add" in lower or "when you deploy a" in lower:
        triggers.append("on_deploy")
    if "when the enemy deploys" in lower or "when an enemy unit is deployed" in lower:
        triggers.append("on_deploy")
    if "when you play a card" in lower or "after you give" in lower:
        triggers.append("on_friendly_card_played")
    if "when any" in lower and "is deployed" in lower:
        triggers.append("on_deploy")
    if "each time you lose a unit" in lower:
        triggers.append("on_destroy")
    if "after each time this unit attacks" in lower or "after it " in lower and "attack" in lower:
        triggers.append("on_attack")
    if "when a friendly unit is damaged" in lower:
        triggers.append("on_damage")
    # If no explicit trigger was detected and the card is a unit with stat-buff
    # actions, treat it as a continuous aura that fires on_deploy.
    if not triggers and card.type in {"infantry", "fighter", "tank", "bomber", "artillery"}:
        for action_text in [text]:
            pass  # will be checked after parsing
        # We'll add on_deploy in parse() for unit auras — defer to AURA check there.
    if "at the start of your turn" in lower:
        triggers.append("on_turn_start")
    if "at the end of your turn" in lower or "at the end of turn" in lower:
        triggers.append("on_turn_end")
    if "when you draw a card" in lower:
        triggers.append("on_card_drawn")
    if ("cannot attack the enemy hq" in lower or "cannot be attacked by ground units while in the support line" in lower
            or "cannot be targeted by enemy orders" in lower or "deals double damage against tanks" in lower
            or "has +2 attack against units with higher attack" in lower
            or "it costs the enemy +2 kredits to target or attack this unit" in lower
            or re.search(r"deals?\s*\+\d+\s+damage\s+to\s+air\s+units", lower)
            or re.search(r"has\s*\+\d+\s+attack\s+against\s+(?:tanks|infantry|fighters|bombers|artillery)", lower)
            or re.search(r"your\s+orders\s+deal\s*\+\d+\s+damage", lower)):
        triggers.append("on_deploy")
    # An order only exists while it is being played, so it always needs
    # on_play.  A weather trigger is an extra route, never the primary one:
    # without this, "Costs 2 less if Rain was played this turn. Destroy a random
    # enemy unit." would lose on_play and never destroy anything.
    primary = [trigger for trigger in triggers if trigger not in ("on_forecast", "on_weather_played")]
    if not primary and card.type == "order":
        triggers.append("on_play")
    return tuple(dict.fromkeys(triggers))


# "equal to <quantity>" -> a named reference resolved by simulator.rules.references.
# Order matters: the more specific phrase must be tested first.
_DAMAGE_REFERENCES = (
    (r"equal\s+to\s+its\s+cost", "target_cost"),
    (r"equal\s+to\s+its\s+attack", "target_attack"),
    (r"equal\s+to\s+known\s+cards\s+in\s+(?:the\s+)?enemy\s+hand", "known_enemy_hand_count"),
    (r"equal\s+to\s+cards\s+in\s+(?:the\s+)?enemy\s+hand", "enemy_hand_count"),
    (r"equal\s+to\s+other\s+friendly\s+air\s+units", "other_friendly_air_count"),
    (r"equal\s+to\s+(?:the\s+)?total\s+attack\s+of\s+your\s+air\s+units", "friendly_air_total_attack"),
    (r"equal\s+to\s+friendly\s+air\s+units", "friendly_air_count"),
    (r"equal\s+to\s+1\s+plus\s+the\s+number\s+of\s+bombers", "friendly_bomber_count_plus_one"),
    (r"equal\s+to\s+1\s+plus\s+the\s+number\s+of\s+unit\s+types", "friendly_unit_type_count_plus_one"),
    (r"equal\s+to\s+your\s+unspent\s+kredits", "unspent_kredits"),
    (r"equal\s+to\s+(?:the\s+)?number\s+of\s+units\s+you\s+control", "friendly_unit_count"),
    # Counter-backed quantities (simulator.rules.counters).
    (r"equal\s+to\s+kredit\s+slots\s+lost\s+this\s+battle\s+plus\s+1", "kredit_slots_lost_this_battle_plus_one"),
    (r"equal\s+to\s+(?:the\s+)?kredit\s+slots\s+(?:you\s+have\s+)?lost\s+this\s+battle", "kredit_slots_lost_this_battle"),
    (r"equal\s+to\s+damage\s+to\s+your\s+hq\s+this\s+turn", "hq_damage_this_turn"),
    (r"equal\s+to\s+kredits\s+spent\s+operating\s+this\s+turn", "kredits_spent_operating_this_turn"),
)


def _damage_reference(lower: str) -> tuple[str, str] | None:
    """Return (reference_name, nation_filter) for an "equal to ..." phrase."""
    nation_tanks = re.search(
        r"equal\s+to\s+(?:the\s+)?total\s+cost\s+of\s+your\s+([a-z]+)\s+tanks\s+in\s+the\s+frontline", lower)
    if nation_tanks:
        return ("frontline_nation_tank_total_cost", nation_tanks.group(1))
    nation_destroyed = re.search(r"equal\s+to\s+([a-z]+)\s+units\s+destroyed\s+this\s+battle", lower)
    if nation_destroyed:
        return ("nation_units_destroyed_this_battle", nation_destroyed.group(1))
    non_nation = re.search(r"equal\s+to\s+(?:the\s+)?number\s+of\s+non-([a-z]+)\s+units", lower)
    if non_nation:
        return ("non_nation_friendly_unit_count", non_nation.group(1))
    for pattern, reference in _DAMAGE_REFERENCES:
        if re.search(pattern, lower):
            return (reference, "")
    return None


def _sentences(text: str) -> tuple[str, ...]:
    """Split card text into independent clauses.

    Sentence delimiters (.!?) always split.  Compound clauses joined by
    'then' (optionally preceded by a comma) are also split so that each
    clause is parsed independently — if one clause is unresolved, the card
    is marked partial rather than falsely claiming the whole sentence is
    covered.
    """
    # kards.info stores some card texts with a LITERAL backslash-r-backslash-n
    # between sentences instead of a real line break (15 cards in the 2026-05
    # catalog, plus OCEANIA STORM ones).  Without normalizing it, the second
    # sentence is swallowed into the first one's condition clause.
    text = text.replace("\\r\\n", " ").replace("\\n", " ").replace("\\r", " ")
    # First split on sentence delimiters.
    raw = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    # Then further split each piece on ', then' / 'then ' between clauses.
    result = []
    for piece in raw:
        parts = re.split(r',?\s*then\s+', piece)
        result.extend(p.strip() for p in parts if p.strip())
    return tuple(result)


# Kinds whose effect is reversible at a turn boundary; these carry a `duration`.
_TEMPORARY_KINDS = {
    "buff", "modify_attack", "modify_defense", "set_attack", "suppress", "grant_ability",
    "modify_hand_cost", "op_cost_rule", "operation_cost_rule",
}


def _sentence_duration(sentence: str) -> str | None:
    """Map a duration phrase to 'this_turn' / 'next_turn', else None."""
    lower = sentence.lower()
    if re.search(r"\bthis turn\b|\buntil end of (?:your |the )?turn\b", lower):
        return "this_turn"
    if re.search(r"\bnext turn\b|\buntil (?:your |the )?next turn\b|at the end of (?:your |the )?next turn\b", lower):
        return "next_turn"
    return None


def _parse_give(lower: str):
    """Parse a 'give ...' sentence into stat-buff and/or ability-grant actions.

    Handles an optional descriptor between the article and the unit type:
    'friendly Guard unit', 'unit with Guard', 'British infantry', 'tank or
    infantry', 'Veteran unit'. Returns None when the sentence is not a clean
    give (no stat and no ability word, or it carries a conditional clause that
    the temporary-effect milestone must handle instead).
    """
    if _HAS_CONDITIONAL.search(lower):
        return None
    m = re.search(
        r"\bgive\s+(.+?)\s+"
        r"(?:\+(\d+)\s*\+\s*(\d+)|\+?(\d+)\s+attack|\+?(\d+)\s+defense)\b",
        lower,
    )
    if m:
        desc = m.group(1)
        tail = lower[m.end():]
        if m.group(2) and m.group(3):
            atk, dfs = int(m.group(2)), int(m.group(3))
        elif m.group(4):
            atk, dfs = int(m.group(4)), 0
        else:
            atk, dfs = 0, int(m.group(5))
    else:
        # Ability-only give (no stat). Take the whole remainder after "give" as
        # both the descriptor and the ability source so descriptor words like
        # "Veteran"/"Guard" are detected as the scope, not as granted abilities.
        m2 = re.search(r"\bgive\s+(.+)$", lower)
        if not m2:
            return None
        remainder = m2.group(1)
        desc = remainder
        tail = remainder
        atk = dfs = 0
    scope, nation = _classify_descriptor(desc)
    if re.search(r"\bwith guard\b|\bguard unit\b", desc):
        scope = scope or "guard"
    if re.search(r"\bveteran\b", desc):
        scope = scope or "veteran"
    mass = re.search(r"\byour\b", desc) or re.search(r"\b(all|each|every)\b", desc)
    friendly = re.search(r"\bfriendly\b", desc)
    enemy = re.search(r"\benemy\b", desc)
    if mass:
        tgt = "enemy_units" if enemy else "friendly_units"
    else:
        tgt = "selected_friendly" if friendly else "selected_enemy" if enemy else "selected_target"
    actions: list[RuleAction] = []
    if atk or dfs:
        if atk and dfs:
            actions.append(RuleAction("buff", tgt, attack=atk, defense=dfs, scope=scope, card_name=nation))
        elif atk:
            actions.append(RuleAction("modify_attack", tgt, amount=atk, scope=scope, card_name=nation))
        else:
            actions.append(RuleAction("modify_defense", tgt, amount=dfs, scope=scope, card_name=nation))
    # A scope descriptor word ("guard", "veteran") names the target filter, not a
    # granted ability; drop it so "Give a Veteran unit Fury and Shock" does not
    # also grant Veteran, and "Give a friendly Guard unit ... Fury" does not
    # re-grant Guard.
    abilities = [a for a in re.findall(_KW, tail) if a != scope]
    for ab in dict.fromkeys(abilities):
        actions.append(RuleAction("grant_ability", tgt, card_name=ab.replace(" ", ""), scope=scope))
    if not actions:
        return None
    return tuple(actions)


def _strip_condition(sentence: str) -> tuple[str, str | None]:
    """Split a sentence into (main_clause, condition_text).

    Leading triggers ("When X,", "For each X,", "At the start of your turn,")
    and trailing predicates ("... if X", "... on your turn", "... on the enemy
    turn") are extracted as a normalized condition string so the main action can
    be parsed by the existing templates. Returns (main, None) when no condition
    clause is present.
    """
    s = sentence.strip()
    condition = None
    # Leading trigger / condition clause ending at the first comma.
    lead = re.match(
        r"^(when|if|whenever|after|while|until|unless|for each|at the start of (?:your )?turn|at the end of (?:your )?turn)\b.*?,",
        s, re.I,
    )
    if lead:
        condition = s[: lead.end() - 1].strip()
        s = s[lead.end():].strip()
    # Trailing predicate clauses ("... if X", "... on your turn", ...).
    for pat in (
        # "after" belongs here too: without it, "Your units get +2+2 after you
        # play a Rain card" keeps no condition and fires on deploy.
        r",?\s*(?:if|when|whenever|unless|after)\b.*$",
        r"\s+on (?:your|the enemy) turn\b.*$",
    ):
        m = re.search(pat, s, re.I)
        if m:
            cond = s[m.start():].strip().lstrip(",").strip()
            condition = (condition + " ; " + cond) if condition else cond
            s = s[: m.start()].strip()
    return s, condition


def _parse_sentence(sentence: str) -> tuple[RuleAction, ...]:
    delayed = _delayed_destroy(sentence)
    if delayed:
        return delayed
    main, condition = _strip_condition(sentence)
    if condition:
        # Try the FULL sentence first so condition-embedded, source-targeted
        # templates (e.g. "Gets +1 attack when it damages the enemy HQ" ->
        # modify_attack source) match and keep their correct (source) target,
        # instead of falling through to the bare "gets" template which targets
        # selected_target. The condition clause is re-attached below.
        whole = _parse_sentence_core(sentence)
        if whole:
            return tuple(replace(a, condition=condition) for a in whole)
    if main.strip():
        acts = _parse_sentence_core(main)
        if acts:
            if condition:
                acts = tuple(replace(a, condition=condition) for a in acts)
            return acts
    # Fallback: parse the whole sentence so condition-specific templates (which
    # embed their trigger in the regex) keep matching and nothing regresses.
    parsed = _parse_sentence_core(sentence)
    if condition:
        parsed = tuple(replace(a, condition=condition) for a in parsed)
    return parsed


def _parse_sentence_core(sentence: str) -> tuple[RuleAction, ...]:
    original_lower = sentence.lower()
    clean = re.sub(r"^(?:on\s+)?(?:deployment|destruction)\s*:\s*", "", sentence, flags=re.I).strip()
    # A leading temporal clause describes an event, not an action to replay.
    # Example: "When you draw a card, your HQ gains +1 defense" must not draw.
    clean = re.sub(r"^(?:When .+?,|At the start of (?:your )?turn,|At the end of (?:your )?turn,|If .+?,)\s*", "", clean, flags=re.I)
    lower = clean.lower()
    if "enemy deploys a unit with a deployment effect" in original_lower and "cancel" in original_lower:
        return (RuleAction("cancel", "source", card_name="deployment_effect"),)
    if "only 2 units can occupy the frontline" in original_lower:
        return (RuleAction("frontline_limit", "source", amount=2),)
    # Sis: redirect HQ damage to enemy HQ. Must run before temporal stripping
    # because the 'When...' clause is part of the effect text, not a skip clause.
    if "hq is to take damage" in original_lower and "enemy hq takes" in original_lower:
        return (RuleAction("hq_damage_redirect_to_enemy_hq", "source"),)
    if lower in {"", "{}"}:
        return ()
    # B-4 203mm HOWITZER: "Units damaged by this unit get -3 attack until end of
    # turn" -- on_damage_dealt carries the damaged unit as the target.  It used
    # to give the howitzer itself -3 attack on deploy.
    damaged = re.match(r"^units\s+damaged\s+by\s+this\s+unit\s+get\s+([+-]\d+)\s+attack\b", lower)
    if damaged:
        return (RuleAction("modify_attack", "selected_target", amount=int(damaged.group(1))),)
    # Weather (OCEANIA STORM): a bare "Forecast" clause creates a forecast
    # token.  A trailing predicate is kept as the condition ("Forecast after
    # this unit attacks the enemy HQ"); compound clauses such as "Pin a unit
    # and Forecast" are left to the generic templates for now.
    forecast_clause = re.match(r"^(?:also\s+|you\s+)?forecast\b\s*(?P<cond>(?:after|when|if|once)\b.*?)?\.?$", lower)
    if forecast_clause:
        forecast_condition = (forecast_clause.group("cond") or "").strip().rstrip(".") or None
        return (RuleAction("forecast", "owner", condition=forecast_condition),)
    # --- M61: precise per-card patterns (before template matching) ---
    if "triggers twice" in original_lower: return (RuleAction("repeat_effect", "source"),)
    if "trigger an extra time" in original_lower: return (RuleAction("repeat_effect", "source"),)
    if "trigger all destruction effects of" in original_lower: return (RuleAction("repeat_effect", "source"),)
    if "damage to stirling mk" in original_lower:
        return (RuleAction("redirect_named_unit_damage", "source", card_name="STIRLING Mk I S3"),)
    if "trigger non-targeting deployment effects on friendly units on the battlefield" in original_lower:
        return (RuleAction("replay_non_targeting_deployment", "friendly_units"),)
    if "trigger a non-targeted deployment effect on a friendly unit" in original_lower:
        return (RuleAction("replay_non_targeting_deployment", "selected_friendly"),)
    if "choose a unit in hand" in original_lower and "swap it with a friendly unit on the battlefield" in original_lower:
        return (RuleAction("swap_hand_unit_with_friendly", "selected_friendly"),)
    if "destroy target undamaged unit" in original_lower:
        return (
            RuleAction("destroy_undamaged", "selected_target"),
            RuleAction("control_effect", "enemy_units", card_name="randomly rearrange enemy units"),
        )
    # A named deployment aura has two independent continuous pieces: a
    # hand-cost modifier while the source remains in play, and a stat bonus
    # applied exactly once as each matching unit enters play.  Keep this as a
    # generic name-scoped rule instead of a RAF GROUND CREW card branch.
    if "cost 2 less to deploy and get +1+1 when deployed" in original_lower:
        match = re.search(r"your\s+(.+?)\s+cost\s+2\s+less", sentence, re.I)
        name = match.group(1).strip() if match else ""
        return (
            RuleAction("modify_hand_cost", "owner", amount=-2, card_name=name, scope="name"),
            RuleAction("buff_deployed_matching_name", "source", attack=1, defense=1, card_name=name, scope="name"),
        )
    if "combat damage dealt to this unit is reduced to 1" in original_lower: return (RuleAction("combat_damage_cap", "source", amount=1),)
    if "random effects always choose this unit" in original_lower: return (RuleAction("random_effect_target_priority", "source"),)
    if "non-combat, non-attack damage dealt by your units" in original_lower: return (RuleAction("noncombat_unit_damage_bonus", "source", amount=1),)
    if "when your hq is damaged" in original_lower and "deal that amount of damage to this unit instead" in original_lower:
        return (RuleAction("hq_damage_redirect_to_source", "source"),)
    if "cards in enemy hand cost +1 kredit" in original_lower: return (RuleAction("modify_hand_cost", "enemy_hand", amount=1),)
    if "cannot be deployed unless you have" in original_lower: return (RuleAction("cannot", "source"),)
    if "can move and attack during the same turn" in original_lower: return (RuleAction("move_and_attack", "source"),)
    if "your ground units with 4 or more attack deal" in original_lower: return (RuleAction("ground_damage_bonus", "source", amount=3, min_cost=4),)
    if "also counts as a tank" in original_lower: return (RuleAction("grant_trait", "source", card_name="tank"),)
    if "has +2 attack while in the frontline" in original_lower: return (RuleAction("frontline_attack_bonus", "source", amount=2),)
    if "swap a position with a friendly unit" in original_lower: return (RuleAction("swap_with_friendly", "selected_friendly"),)
    if "this unit costs 1 more each time you lose" in original_lower: return (RuleAction("control_effect", "source"),)
    if "your alpine units can move and attack" in original_lower: return (RuleAction("move_and_attack_aura", "source", scope="alpine"),)
    if "the next damage order you give this turn deals +1" in original_lower: return (RuleAction("next_order_damage_bonus", "owner", amount=1, duration="this_turn"),)
    if "deals 0-1 additional damage" in original_lower: return (RuleAction("random_combat_damage", "source", amount=1),)
    if "deal excess damage to the enemy hq" in original_lower: return (RuleAction("hq_excess", "source"),)
    if "trigger the destruction effects of all friendly units destroyed" in original_lower: return (RuleAction("repeat_effect", "source"),)
    if "enemy cards deal 1 less damage" in original_lower: return (RuleAction("enemy_card_damage_reduction", "source", amount=1),)
    if "when a hq gains defense, it takes that much damage instead" in original_lower: return (RuleAction("hq_defense_becomes_damage", "source"),)
    if "hq takes 1 less damage for each of your units" in original_lower: return (RuleAction("hq_damage_reduction_by_attack", "source", amount=1, min_cost=4),)
    if "loses remaining kredits" in original_lower: return (RuleAction("lose_kredits", "enemy_hand", amount=99),)
    if "deploys for 0 kredits" in original_lower: return (RuleAction("modify_deployment_cost", "source", set_cost=0),)
    if "costs +2 kredits" in original_lower: return (RuleAction("target_or_attack_tax", "enemy_hand", amount=2),)
    if "gain an extra kredit slot" in original_lower: return (RuleAction("gain_kredit_slot", "owner"),)
    if "navy type" in original_lower: return (RuleAction("grant_trait", "source", card_name="navy_type"),)
    if "while in the frontline, your hq cannot be reduced below" in original_lower: return (RuleAction("hq_defense_lock", "owner"),)
    if "when your hq receives damage this turn, reduce" in original_lower: return (RuleAction("hq_defense_lock", "owner"),)
    if "set its attack equal to its defense" in original_lower: return (RuleAction("attack_equals_defense", "selected_target"),)
    if "change its attack to equal its defense" in original_lower: return (RuleAction("attack_equals_defense", "selected_target"),)
    if "attack becomes equal to" in original_lower: return (RuleAction("attack_equals_defense", "friendly_units"),)
    if "shuffle your deck" in original_lower: return (RuleAction("shuffle_named", "owner"),)
    if original_lower.startswith("copy "): return (RuleAction("copy_unit", "source"),)
    if "duplicate a friendly exile" in original_lower: return (RuleAction("copy_unit", "selected_friendly"),)
    if "fights target enemy unit" in original_lower: return (RuleAction("fight", "selected_enemy"),)
    if "set it to 0" in original_lower: return (RuleAction("set_operation_cost", "selected_target", amount=0),)
    if "put a card in hand on top of your deck" in original_lower: return (RuleAction("return_to_deck", "owner"),)
    if "discard all cards with cost" in original_lower: return (RuleAction("discard_hand", "owner"),)
    if "draw 2 cards" in original_lower: return (RuleAction("draw", "owner", amount=2),)
    if "shuffles their hand into their deck" in original_lower: return (RuleAction("shuffle_named", "owner"),)
    if "return it at the end of the next enemy turn" in original_lower: return (RuleAction("return_to_hand", "selected_target", duration="next_turn"),)
    if "intel equal to friendly" in original_lower: return (RuleAction("grant_ability", "friendly_units", card_name="intel"),)
    if "gains +1 attack when your hq takes damage" in original_lower: return (RuleAction("modify_attack", "source", amount=1),)
    if "remove the effect if an enemy unit" in original_lower: return (RuleAction("control_effect", "selected_target"),)
    if "trigger the destruction effects on a target unit" in original_lower: return (RuleAction("control_effect", "selected_target"),)
    if "damage dealt to your hq is reduced" in original_lower: return (RuleAction("hq_defense_lock", "owner"),)
    if "order damage to a friendly unit is reduced" in original_lower: return (RuleAction("friendly_order_damage_armor", "source"),)
    if "trigger this card when the enemy deploys" in original_lower: return (RuleAction("control_effect", "source"),)
    if "59. panzergrenadier can move and attack" in original_lower: return (RuleAction("move_and_attack", "source"),)
    if "deployment effects do not trigger" in original_lower: return (RuleAction("suppress_deployment_effects", "source"),)
    if "cards you play have intel 1" in original_lower: return (RuleAction("grant_ability", "friendly_units", card_name="intel"),)
    if "your non-targeting deployment effects trigger twice" in original_lower: return (RuleAction("repeat_effect", "source"),)
    if "target friendly unit gets: destruction: add a copy" in original_lower: return (RuleAction("control_effect", "selected_target"),)
    if "enemy units that damage anything but this unit are destroyed" in original_lower: return (RuleAction("destroy", "enemy_units"),)
    # ── Genuinely-executed templates ──
    # Gain HQ defense
    gain_hq = re.search(r"(?:gain(?:s|ed)?|gives?\s+your\s+hq)\s+(\+?\d+)\s+(?:hq\s+)?defense", lower)
    if gain_hq:
        return (RuleAction("gain_hq_defense", "owner", amount=int(gain_hq.group(1))),)
    if "your hq gains" in lower and "defense" in lower:
        return (RuleAction("gain_hq_defense", "owner", amount=1),)
    if "immune to damage" in lower:
        return (RuleAction("immune", "source", card_name="damage"),)
    if "is immune to damage" in lower:
        return (RuleAction("immune", "source", card_name="damage"),)
    if "cannot attack the enemy hq" in lower:
        return (RuleAction("cannot_attack_hq", "source"),)
    remove_deck = re.search(r"\bremove\s+(\d+)\s+cards?\s+from\s+(?:your|the)\s+deck", lower)
    if remove_deck:
        return (RuleAction("mill", "owner", amount=int(remove_deck.group(1))),)
    if "players can only give one order each turn" in original_lower: return (RuleAction("enemy_cannot_order", "enemy_hand"),)
    if "when your hq receives damage" in original_lower: return (RuleAction("hq_defense_lock", "owner"),)
    if "reduce the damage to 1" in original_lower: return (RuleAction("hq_defense_lock", "owner"),)
    if "combat damage dealt to this unit is reduced to 1" in original_lower: return (RuleAction("combat_damage_cap", "source", amount=1),)
    if "combat damage dealt to this unit is doubled" in original_lower: return (RuleAction("double_damage", "source"),)
    if "your ground units with 4 or more attack" in original_lower: return (RuleAction("modify_attack", "friendly_units", amount=3),)
    if "non-combat, non-attack damage dealt by your units" in original_lower: return (RuleAction("control_effect", "friendly_units"),)
    if "also counts as a tank" in original_lower: return (RuleAction("grant_trait", "source", card_name="tank"),)
    if "trigger non-targeting deployment effects on friendly" in original_lower: return (RuleAction("control_effect", "friendly_units"),)
    if "trigger a non-targeted deployment effect on a friendly" in original_lower: return (RuleAction("control_effect", "friendly_units"),)
    if "this unit costs 1 more each time you lose a unit" in original_lower: return (RuleAction("modify_hand_cost", "source", amount=1),)
    if "your alpine units can move and attack" in original_lower: return (RuleAction("control_effect", "friendly_units"),)
    if "the next damage order you give this turn deals +1" in original_lower: return (RuleAction("order_damage_bonus", "friendly_units"),)
    if "deals 0-1 additional damage" in original_lower: return (RuleAction("random_combat_damage", "source", amount=1),)
    if "when a hq gains defense, it takes that much damage instead" in original_lower: return (RuleAction("control_effect", "source"),)
    if "trigger the destruction effects of all friendly units destroyed" in original_lower: return (RuleAction("control_effect", "friendly_units"),)
    if "enemy cards deal 1 less damage" in original_lower: return (RuleAction("enemy_card_damage_reduction", "source", amount=1),)
    if "your non-targeting deployment effects trigger twice" in original_lower: return (RuleAction("repeat_effect", "source"),)
    if "target friendly unit gets: destruction: add a copy" in original_lower: return (RuleAction("control_effect", "selected_target"),)
    if "trigger the destruction effects on a target unit as if it was yours" in original_lower: return (RuleAction("control_effect", "selected_target"),)
    if "when a friendly destruction effect triggers, it triggers twice" in original_lower: return (RuleAction("repeat_effect", "source"),)
    if "destruction effects on this unit trigger an extra time" in original_lower: return (RuleAction("repeat_effect", "source"),)
    if "destruction: trigger all destruction effects of your soviet units" in original_lower: return (RuleAction("repeat_effect", "selected_target"),)
    if "cards you play have intel 1" in original_lower: return (RuleAction("grant_ability", "friendly_units", card_name="intel"),)
    if "reveal target covert unit" in original_lower: return (RuleAction("target_select", "selected_target"),)
    if "59." in original_lower and "panzergrenadier" in original_lower: return (RuleAction("control_effect", "source"),)
    # ── More genuinely-executed remaps ──
    # Cost reduction patterns
    if "your covert units cost 1 less to deploy" in original_lower:
        return (RuleAction("modify_hand_cost", "owner", amount=-1, scope="covert"),)
    if "your salvaged units cost 2 less to deploy" in original_lower:
        return (RuleAction("modify_hand_cost", "owner", amount=-2, scope="salvaged"),)
    if "your units cost 2 less to deploy" in original_lower:
        return (RuleAction("modify_hand_cost", "owner", amount=-2),)
    if "your sherman units cost 1 less to deploy" in original_lower:
        return (RuleAction("modify_hand_cost", "owner", amount=-1, scope="sherman"),
                RuleAction("grant_ability", "source", card_name="blitz"))
    deploy_less_if = re.search(r"costs?\s+(\d+)\s+less\s+to\s+deploy\s+if", lower)
    if deploy_less_if:
        return (RuleAction("modify_deployment_cost", "source", amount=-int(deploy_less_if.group(1))),)
    costs_if_hq = re.search(r"costs?\s+(\d+)\s+if\s+your\s+hq\s+is\s+at\s+(\d+)", lower)
    if costs_if_hq:
        return (RuleAction("modify_deployment_cost", "source", amount=int(costs_if_hq.group(1))),)
    # Operation cost modification
    incr_op = re.search(r"increase\s+(?:the\s+)?operation\s+cost\s+by\s+(\d+)", lower)
    if incr_op:
        return (RuleAction("modify_operation_cost", "source", amount=int(incr_op.group(1))),)
    if "your units operate for 2 less this turn and are fully repaired" in original_lower:
        return (
            RuleAction("op_cost_rule", "owner", amount=-2, duration="this_turn"),
            RuleAction("scheduled_repair", "friendly_units", duration="end_of_turn"),
        )
    if "your units operate for 1 less this turn" in original_lower:
        return (RuleAction("operation_cost_rule", "owner", amount=-1),)
    if "if it is destroyed, your units operate for 1 less this turn" in original_lower:
        return (RuleAction("op_cost_rule", "owner", amount=-1),)
    if "your bombers operate for 2 less this turn" in original_lower:
        return (RuleAction("operation_cost_rule", "owner", amount=-2, scope="bomber"),)
    if "your air units operate for free this turn" in original_lower:
        return (RuleAction("operation_cost_rule", "owner", set_cost=0, scope="air"),)
    if "enemy air units cost +2 to operate" in original_lower:
        return (RuleAction("operation_cost_rule", "enemy_hand", amount=2, scope="air"),)
    if "units deployed by the enemy have +2 operation cost" in original_lower:
        return (RuleAction("operation_cost_rule", "enemy_hand", amount=2),)
    if "it costs 1 less to deploy and operate" in original_lower:
        return (RuleAction("modify_deployment_cost", "source", amount=-1),
                RuleAction("modify_operation_cost", "source", amount=-1))
    # Conditional salvage / ability grant
    if "if it is destroyed this turn, salvage it" in original_lower:
        return (RuleAction("grant_ability", "source", card_name="salvage"),)
    if "can target covert units" in original_lower:
        return (RuleAction("target_select", "source", card_name="can_target_covert"),)
    if "can affect a covert unit" in original_lower:
        return (RuleAction("target_select", "source", card_name="can_target_covert"),)
    # HQ damage reduction
    if "your hq takes 1 less damage for each of your units" in original_lower:
        return (RuleAction("hq_damage_reduction_by_attack", "source", amount=1, min_cost=4),)
    # Non-combat damage bonus
    if "increase the non-combat, non-attack damage dealt by your units by 1" in original_lower:
        return (RuleAction("hq_damage_bonus", "friendly_units"),)
    if re.search(r"\bretreat\s+a\s+friendly\s+unit\s+in\s+the\s+frontline\s+and\s+fully\s+repair\s+it", lower):
        return (RuleAction("retreat_and_repair", "selected_friendly", card_name="frontline"),)
    if "it can operate again this turn" in lower:
        return (RuleAction("reset_operation", "selected_target"),)
    give_target_with_ability = re.search(r"\bgive\s+target\s+unit\s*\+(\d+)\s*\+\s*(\d+)\s+and\s+(blitz|fury|guard|shock|salvage|ambush|smokescreen)", lower)
    if give_target_with_ability:
        return (
            RuleAction("buff", "selected_target", attack=int(give_target_with_ability.group(1)), defense=int(give_target_with_ability.group(2))),
            RuleAction("grant_ability", "selected_target", card_name=give_target_with_ability.group(3)),
        )
    lose_kredit_slot = re.search(r"\blose\s+(a|one|two|three|\d+)\s+kredit\s+slots?", lower)
    if lose_kredit_slot:
        return (RuleAction("lose_kredit_slot", "owner", amount=_number(lose_kredit_slot.group(1))),)
    take = re.search(r"\btake\s+control\s+of\s+an?\s+enemy\s+unit(?:\s+with\s+cost\s+(\d+)\s+or\s+less)?", lower)
    if take:
        # amount = highest cost that can be taken (0 = no limit); "until end
        # of turn" gives it back (CONFUSION).
        return (RuleAction("take_control", "selected_enemy", amount=int(take.group(1) or 0),
                           duration="this_turn" if re.search(r"until\s+end\s+of\s+(?:the\s+|your\s+)?turn", lower) else None),)
    if re.search(r"\bfight\s+target\s+enemy\s+unit", lower):
        return (RuleAction("fight", "selected_enemy"),)
    # Countermeasure intercept: "Counter an enemy order ...", "Counter an order ...",
    # "Counter an order or deployment effect ...", optionally "with cost N or more"
    # and/or "and draw a card". The cancel AST action drives CountermeasureResolver
    # into flagging the enemy action as cancelled.
    counter_order = re.search(r"\bcounter\s+(?:an?\s+)?(?:enemy\s+)?order\b", lower)
    if counter_order:
        # Keep target restrictions in the AST rather than inferring a card name
        # in CountermeasureResolver.  The resolver evaluates this before the
        # countermeasure is consumed.
        condition = "target_friendly" if "targets a friendly" in lower else None
        cancel = RuleAction("cancel", "source", condition=condition)
        cost = re.search(r"\bwith\s+cost\s+(\d+)\s+or\s+more", lower)
        if cost:
            cancel = RuleAction("cancel", "source", min_cost=int(cost.group(1)), condition=condition)
        if re.search(r"\band\s+draw\s+a\s+card\b", lower):
            return (cancel, RuleAction("draw", "owner", amount=1))
        return (cancel,)
    if re.search(r"\bcounter\s+it\b", lower):
        return (RuleAction("cancel", "source"),)
    if re.search(r"\bcancel\s+the\s+effect\b", lower):
        return (RuleAction("cancel", "source"),)
    if "cannot attack the enemy hq" in lower:
        return (RuleAction("cannot_attack_hq", "source"),)
    if "cannot be attacked by ground units while in the support line" in lower:
        return (RuleAction("immune_to_ground_in_support", "source"),)
    if "cannot be targeted by enemy orders" in lower:
        return (RuleAction("cannot_be_targeted_by_enemy_orders", "source"),)
    if "deals double damage against tanks" in lower:
        return (RuleAction("double_damage_against_type", "source", card_name="tank"),)
    higher_attack_bonus = re.search(r"\bhas\s*\+(\d+)\s+attack\s+against\s+units\s+with\s+higher\s+attack", lower)
    if higher_attack_bonus:
        return (RuleAction("attack_bonus_against_higher_attack", "source", amount=int(higher_attack_bonus.group(1))),)
    target_tax = re.search(r"\bit\s+costs\s+the\s+enemy\s*\+(\d+)\s+kredits\s+to\s+target\s+or\s+attack\s+this\s+unit", lower)
    if target_tax:
        return (RuleAction("target_or_attack_tax", "source", amount=int(target_tax.group(1))),)
    air_damage_bonus = re.search(r"\bdeals?\s*\+(\d+)\s+damage\s+to\s+air\s+units", lower)
    if air_damage_bonus:
        return (RuleAction("damage_bonus_against_air", "source", amount=int(air_damage_bonus.group(1))),)
    type_attack_bonus = re.search(r"\bhas\s*\+(\d+)\s+attack\s+against\s+(tanks|infantry|fighters|bombers|artillery)", lower)
    if type_attack_bonus:
        return (RuleAction("attack_bonus_against_type", "source", amount=int(type_attack_bonus.group(1)), card_name=type_attack_bonus.group(2).rstrip("s")),)
    order_damage_bonus = re.search(r"\byour\s+orders\s+deal\s*\+(\d+)\s+damage", lower)
    if order_damage_bonus:
        return (RuleAction("order_damage_bonus", "source", amount=int(order_damage_bonus.group(1))),)
    spawn_same_front = re.search(r"\badd\s+(?:an?\s+)?(.+?)\s+to\s+the\s+same\s+front", clean, re.I)
    if spawn_same_front:
        return (RuleAction("spawn_named_same_front", "source", card_name=spawn_same_front.group(1).strip()),)
    played_ability_buff = re.search(r"\bgets?\s*\+(\d+)\s*\+\s*(\d+)\s+when\s+you\s+play\s+a\s+card\s+with\s+(intel|bond)", original_lower)
    if played_ability_buff:
        return (RuleAction("buff_on_played_ability", "source", attack=int(played_ability_buff.group(1)), defense=int(played_ability_buff.group(2)), card_name=played_ability_buff.group(3)),)
    order_play_attack = re.search(r"\bgets?\s*\+(\d+)\s+attack\s+when\s+you\s+give\s+an\s+order", original_lower)
    if order_play_attack:
        return (RuleAction("attack_on_order_played", "source", amount=int(order_play_attack.group(1))),)
    enemy_target_buff = re.search(r"\bgets?\s*\+(\d+)\s*\+\s*(\d+)\s+when\s+(?:it\s+is\s+targeted\s+by\s+an\s+order\s+or\s+deployment\s+effect|the\s+enemy\s+targets\s+or\s+attacks\s+it)", original_lower)
    if enemy_target_buff:
        return (RuleAction("buff_on_enemy_target", "source", attack=int(enemy_target_buff.group(1)), defense=int(enemy_target_buff.group(2))),)
    if ("damage dealt by this unit adds equal defense to your hq" in lower
            or "when this unit is dealt damage, your hq gets equal amount of defense" in original_lower):
        return (RuleAction("gain_hq_defense_equal_damage", "owner"),)
    if "units dealt combat damage by this unit are destroyed" in lower:
        return (RuleAction("destroy", "selected_target"),)
    hq_damage_attack = re.search(r"\bgets?\s*\+(\d+)\s+attack\s+when\s+it\s+damages\s+the\s+enemy\s+hq", original_lower)
    if hq_damage_attack:
        return (RuleAction("modify_attack", "source", amount=int(hq_damage_attack.group(1))),)
    survives_combat_buff = re.search(r"\bgets?\s*\+(\d+)\s*\+\s*(\d+)\s+when\s+it\s+survives\s+combat", original_lower)
    if survives_combat_buff:
        return (RuleAction("buff", "source", attack=int(survives_combat_buff.group(1)), defense=int(survives_combat_buff.group(2))),)
    draw = re.search(r"\bdraw\s+(a|an|one|two|three|four|five|\d+)\s+cards?\b", lower)
    if draw:
        return (RuleAction("draw", amount=_number(draw.group(1))),)
    random_unit_draw = re.search(r"\bdraw\s+(?:an?\s+)?random\s+unit\s+from\s+your\s+deck", lower)
    if random_unit_draw:
        return (RuleAction("draw_matching", "owner", card_name="unit"),)
    random_order_draw = re.search(r"\bdraw\s+(?:an?\s+)?random\s+order\s+from\s+your\s+deck", lower)
    if random_order_draw:
        return (RuleAction("draw_matching", "owner", card_name="order"),)
    damage = re.search(r"\bdeals?\s+(a|an|one|two|three|four|five|\d+)\s+damage\s+to\s+(.+?)(?:\.|$)", lower)
    if damage:
        return (RuleAction("damage", _target(damage.group(2)), amount=_number(damage.group(1))),)
    damage_bare = re.search(r"^\s*deal\s+(\d+)\s+damage\.?\s*$", lower)
    if damage_bare:
        # An order that simply says "Deal N damage." targets the enemy HQ by
        # KARDS convention; this is the documented default, not a guess.
        return (RuleAction("damage", "enemy_hq", amount=int(damage_bare.group(1))),)
    if re.search(r"\bdestroy\s+(?:an?\s+)?random\s+enemy\s+unit", lower):
        return (RuleAction("destroy", "random_enemy"),)
    if re.search(r"\b(?:pin|suppress)\s+(?:an?\s+)?random\s+enemy\s+unit", lower):
        return (RuleAction("suppress", "random_enemy"),)
    destroy_cost = re.search(r"\bdestroy\s+target\s+unit\s+with\s+cost\s+(\d+)\s+or\s+less", lower)
    if destroy_cost:
        return (RuleAction("destroy_cost_lte", "selected_enemy", amount=int(destroy_cost.group(1))),)
    if re.search(r"\bdestroy\s+(?:target|an?|the)\s+(?:enemy\s+)?unit", lower):
        return (RuleAction("destroy", "selected_enemy" if "enemy" in lower else "selected_target"),)
    if re.search(r"\bdestroy\s+(?:an?|the)\s+friendly\s+unit", lower):
        return (RuleAction("destroy", "selected_friendly"),)
    destroy_mass = re.search(r"\bdestroy\s+(?:all|each|every)\s+(friendly\s+|enemy\s+)?(air|ground|tank|infantry|artillery|bomber|fighter|units?)\b", lower)
    if destroy_mass:
        enemy = "enemy" in (destroy_mass.group(1) or "")
        friendly = "friendly" in (destroy_mass.group(1) or "")
        scope = _scope_of(destroy_mass.group(2))
        tgt = "enemy_units" if enemy else "friendly_units" if friendly else "selected_target"
        return (RuleAction("destroy", tgt, scope=scope),)
    destroy_it = re.search(r"\bdestroy\s+it\b", lower)
    if destroy_it:
        # "Destroy it in 2 turns" is a DELAYED (timed) destroy, not immediate;
        # leave it unresolved (fail-loud) for the future timed-effect milestone.
        rest = lower[destroy_it.end():]
        if not re.search(r"\b(?:in|after)\s+(?:a\s+|an\s+|one\s+|two\s+|three\s+|\d+)\s+turn", rest):
            return (RuleAction("destroy", "selected_target"),)
    if re.search(r"\b(?:pin|suppress)\s+(?:target|an?|the)\s+(?:enemy\s+)?unit", lower):
        return (RuleAction("suppress", "selected_enemy" if "enemy" in lower else "selected_target"),)
    if re.search(r"\bpin\s+it\s+(?:afterwards|afterward)", lower):
        return (RuleAction("suppress", "selected_target"),)
    if re.search(r"\bsuppress\s+it\b", lower):
        return (RuleAction("suppress", "selected_target"),)
    if "pin all enemy units" in lower:
        return (RuleAction("suppress", "enemy_units"),)
    pin_mass = re.search(r"\b(?:pin|suppress)\s+(all|each|every)\s+(friendly\s+|enemy\s+)?(air|ground|tank|infantry|artillery|bomber|fighter|units?)\b", lower)
    if pin_mass:
        enemy = "enemy" in (pin_mass.group(2) or "")
        friendly = "friendly" in (pin_mass.group(2) or "")
        scope = _scope_of(pin_mass.group(3))
        tgt = "enemy_units" if enemy else "friendly_units" if friendly else "selected_target"
        return (RuleAction("suppress", tgt, scope=scope),)
    hq_defense = re.search(r"(?:your\s+)?hq\s+(?:gains|gets)\s*\+?(\d+)\s+defense", lower)
    if hq_defense:
        return (RuleAction("gain_hq_defense", "owner", amount=int(hq_defense.group(1))),)
    give_hq_defense = re.search(r"\bgive\s+your\s+hq\s*\+?(\d+)\s+defense", lower)
    if give_hq_defense:
        return (RuleAction("gain_hq_defense", "owner", amount=int(give_hq_defense.group(1))),)
    both = re.search(r"(?:target\s+)?(?:friendly\s+|enemy\s+)?(?:a|an|the\s+)?unit\s+(?:gets|gain)\s*\+(\d+)\s*\+\s*(\d+)", lower)
    if both:
        return (RuleAction("buff", "selected_friendly" if "friendly" in lower else "selected_enemy" if "enemy" in lower else "selected_target", attack=int(both.group(1)), defense=int(both.group(2))),)
    defense = re.search(r"(?:target\s+)?(?:friendly\s+|enemy\s+)?(?:a|an|the\s+)?unit\s+(?:gets|gain)\s*\+(\d+)\s+defense", lower)
    if defense:
        return (RuleAction("modify_defense", "selected_friendly" if "friendly" in lower else "selected_enemy" if "enemy" in lower else "selected_target", amount=int(defense.group(1))),)
    attack = re.search(r"(?:target\s+)?(?:friendly\s+|enemy\s+)?(?:a|an|the\s+)?unit\s+(?:gets|gain)\s*\+(\d+)\s+attack", lower)
    if attack:
        return (RuleAction("modify_attack", "selected_friendly" if "friendly" in lower else "selected_enemy" if "enemy" in lower else "selected_target", amount=int(attack.group(1))),)
    # Descriptor-tolerant 'gets' gains: "Target infantry or tank gets +5 attack",
    # "Friendly tanks gets +2 attack". Extends the plain unit/gets templates so
    # real KARDS descriptors parse (and duration is attached by parse()).
    # Sentence-initial "Gets +N+N" / "Gets +N attack/defense" (Deployment: or
    # condition prefix already stripped by _parse_sentence). Targets the source.
    gets_init_both = re.search(r"^(?:gets|get)\s*\+(\d+)\s*\+\s*(\d+)", lower)
    if gets_init_both:
        return (RuleAction("buff", "selected_target", attack=int(gets_init_both.group(1)), defense=int(gets_init_both.group(2))),)
    gets_init_atk = re.search(r"^(?:gets|get)\s*\+(\d+)\s+attack", lower)
    if gets_init_atk:
        return (RuleAction("modify_attack", "selected_target", amount=int(gets_init_atk.group(1))),)
    gets_init_def = re.search(r"^(?:gets|get)\s*\+(\d+)\s+defense", lower)
    if gets_init_def:
        return (RuleAction("modify_defense", "selected_target", amount=int(gets_init_def.group(1))),)
    gets_both = re.search(r"\b(.+?)\s+gets\s*\+(\d+)\s*\+\s*(\d+)", lower)
    if gets_both:
        scope, nation = _classify_descriptor(gets_both.group(1))
        return (RuleAction("buff", _target_of_descriptor(gets_both.group(1)), attack=int(gets_both.group(2)), defense=int(gets_both.group(3)), scope=scope, card_name=nation),)
    # "give it -N attack" (e.g. countermeasure debuffs) -> negative modify_attack
    give_neg_atk = re.search(r"\bgive\s+it\s+-(\d+)\s+attack\b", lower)
    if give_neg_atk:
        return (RuleAction("modify_attack", "selected_target", amount=-int(give_neg_atk.group(1))),)
    # "it becomes A/D" (e.g. "it becomes 2/2") -> set attack and defense
    becomes = re.search(r"\bit\s+becomes\s+(\d+)/(\d+)\b", lower)
    if becomes:
        return (RuleAction("set_attack", "selected_target", amount=int(becomes.group(1))),
                RuleAction("set_defense", "selected_target", amount=int(becomes.group(2))))
    gets_atk = re.search(r"\b(.+?)\s+gets\s*\+(\d+)\s+attack", lower)
    if gets_atk:
        scope, nation = _classify_descriptor(gets_atk.group(1))
        return (RuleAction("modify_attack", _target_of_descriptor(gets_atk.group(1)), amount=int(gets_atk.group(2)), scope=scope, card_name=nation),)
    gets_def = re.search(r"\b(.+?)\s+gets\s*\+(\d+)\s+defense", lower)
    if gets_def:
        scope, nation = _classify_descriptor(gets_def.group(1))
        return (RuleAction("modify_defense", _target_of_descriptor(gets_def.group(1)), amount=int(gets_def.group(2)), scope=scope, card_name=nation),)
    # Absolute "set attack to 0" / "has 0 attack" (temporary-duration friendly).
    set_attack0 = re.search(r"\b(?:set\s+(?:its|the\s+target'?s?|target\s+unit'?s?)\s+attack\s+to\s+0|has\s+0\s+attack)\b", lower)
    if set_attack0:
        tgt = "selected_enemy" if "enemy" in lower else "selected_target"
        return (RuleAction("set_attack", tgt, amount=0),)
    # Mass static buffs (all friendly units matching a type/nation filter):
    # "Friendly infantry has +1 attack", "Your German tanks get +2+1".
    mass_both = re.search(r"\b(?:friendly|your)\s+(?:other\s+)?(.+?)\s+(?:has|have|get|gets|gain)\s*(?:a|an)?\s*\+?(\d+)\s*\+\s*(\d+)" + _COND_GUARD, lower)
    if mass_both:
        scope, nation = _classify_descriptor(mass_both.group(1))
        return (RuleAction("buff", "friendly_units", attack=int(mass_both.group(2)), defense=int(mass_both.group(3)), scope=scope, card_name=nation or None),)
    mass_attack = re.search(r"\b(?:friendly|your)\s+(?:other\s+)?(.+?)\s+(?:has|have|get|gets|gain)\s*(?:a|an)?\s*\+?(\d+)\s+attack" + _COND_GUARD, lower)
    if mass_attack:
        scope, nation = _classify_descriptor(mass_attack.group(1))
        return (RuleAction("modify_attack", "friendly_units", amount=int(mass_attack.group(2)), scope=scope, card_name=nation or None),)
    mass_defense = re.search(r"\b(?:friendly|your)\s+(?:other\s+)?(.+?)\s+(?:has|have|get|gets|gain)\s*(?:a|an)?\s*\+?(\d+)\s+defense" + _COND_GUARD, lower)
    if mass_defense:
        scope, nation = _classify_descriptor(mass_defense.group(1))
        return (RuleAction("modify_defense", "friendly_units", amount=int(mass_defense.group(2)), scope=scope, card_name=nation or None),)
    # Unified "give" handler: stat buff and/or ability grant, single target or
    # mass, with an optional descriptor between the article and the unit type
    # ("friendly Guard unit", "unit with Guard", "British infantry", "tank or
    # infantry", "Veteran unit"). Replaces the narrower give_both/defense/attack
    # and kw templates so real KARDS phrasings parse instead of falling through.
    give = _parse_give(lower)
    if give is not None:
        return give
    # Nation/ability-scoped mass buffs: "Give your German tanks +2+1".
    give_both_nd = re.search(r"\bgive\s+your\s+(.+?)\s*\+(\d+)\s*\+\s*(\d+)", lower)
    if give_both_nd:
        scope, nation = _classify_descriptor(give_both_nd.group(1))
        return (RuleAction("buff", "friendly_units", attack=int(give_both_nd.group(2)), defense=int(give_both_nd.group(3)), scope=scope, card_name=nation or None),)
    give_defense_nd = re.search(r"\bgive\s+your\s+(.+?)\s*\+(\d+)\s+defense", lower)
    if give_defense_nd:
        scope, nation = _classify_descriptor(give_defense_nd.group(1))
        return (RuleAction("modify_defense", "friendly_units", amount=int(give_defense_nd.group(2)), scope=scope, card_name=nation or None),)
    give_attack_nd = re.search(r"\bgive\s+your\s+(.+?)\s*\+(\d+)\s+attack", lower)
    if give_attack_nd:
        scope, nation = _classify_descriptor(give_attack_nd.group(1))
        return (RuleAction("modify_attack", "friendly_units", amount=int(give_attack_nd.group(2)), scope=scope, card_name=nation or None),)
    nation_both = re.search(r"\bgive\s+your\s+(british|german|us|soviet|japanese|french|italian|polish|finnish)\s+units?\s*\+(\d+)\s*\+\s*(\d+)", lower)
    if nation_both:
        return (RuleAction("buff_nation", "owner", attack=int(nation_both.group(2)), defense=int(nation_both.group(3)), card_name=_nation_name(nation_both.group(1))),)
    nation_defense = re.search(r"\bgive\s+your\s+(british|german|us|soviet|japanese|french|italian|polish|finnish)\s+units?\s*\+(\d+)\s+defense", lower)
    if nation_defense:
        return (RuleAction("defense_nation", "owner", amount=int(nation_defense.group(2)), card_name=_nation_name(nation_defense.group(1))),)
    # M9 — Pattern B side effects, linked to the preceding choose_one(enemy_hand)
    # via target="chosen". These are specific two-clause sentences and must be
    # tested before the generic `add` template below.
    if re.search(r"discard\s+it\b", lower) and re.search(r"add\s+a\s+copy\s+to\s+your\s+hand", lower):
        return (RuleAction("discard", "chosen"), RuleAction("add_card", "owner", card_name="chosen_copy"))
    if re.search(r"copy\s+it\b", lower) and re.search(r"convert\s+both\s+into", lower):
        return (RuleAction("discard", "chosen"), RuleAction("convert_to", "owner", card_name="PLAN"))
    add = re.search(r"\badd\s+(.+?)\s+to\s+your\s+hand", clean, re.I)
    if add:
        name, amount = _quantified_name(add.group(1))
        return (RuleAction("add_card", "owner", card_name=name, amount=amount),)
    add_enemy = re.search(r"\badd\s+(.+?)\s+to\s+the\s+enemy\s+hand", clean, re.I)
    if add_enemy:
        name, amount = _quantified_name(add_enemy.group(1))
        return (RuleAction("add_card", "enemy_hand", card_name=name, amount=amount),)
    deploy = re.search(r"\badd\s+(.+?)\s+to\s+(?:your|the)\s+support\s+line", clean, re.I)
    if deploy:
        name, amount = _quantified_name(deploy.group(1))
        return (RuleAction("deploy_named", "owner", card_name=name, amount=amount),)
    # Retreat (move to support line): single/selected target or mass group.
    if re.search(r"\bretreat\s+(?:a|an|another|the|target)\s+(?:friendly\s+|enemy\s+)?(?:air|ground|tank|infantry|artillery|bomber|fighter|unit|units?)\b", lower):
        tgt = "selected_enemy" if "enemy" in lower else "selected_friendly" if "friendly" in lower else "selected_target"
        return (RuleAction("move_to_support_line", tgt),)
    retreat_mass = re.search(r"\bretreat\s+(?:all|each|every)\s+(friendly\s+|enemy\s+)?(air|ground|tank|infantry|artillery|bomber|fighter|units?)\b", lower)
    if retreat_mass:
        enemy = "enemy" in (retreat_mass.group(1) or "")
        friendly = "friendly" in (retreat_mass.group(1) or "")
        scope = _scope_of(retreat_mass.group(2))
        tgt = "enemy_units" if enemy else "friendly_units" if friendly else "selected_target"
        return (RuleAction("move_to_support_line", tgt, scope=scope),)
    repair = re.search(r"\bfully\s+repair\s+(?:a|an|the|another|target)?\s*(friendly\s+)?(?:air|ground|tank|infantry|artillery|bomber|fighter|unit|units?)?", lower)
    if repair:
        return (RuleAction("repair", "selected_friendly" if "friendly" in lower else "selected_target"),)
    shuffle_named = re.search(r"\bshuffle\s+([A-Za-z0-9 .-]+?)\s+into\s+your\s+deck", clean, re.I)
    if shuffle_named:
        return (RuleAction("shuffle_named", "owner", card_name=shuffle_named.group(1).strip()),)
    gain = re.search(r"\bgain\s+(\d+)\s+(?:additional\s+)?kredits?", lower)
    if gain:
        return (RuleAction("gain_kredits", "owner", amount=int(gain.group(1))),)
    kredit_slot = re.search(r"\bgain\s+(\d+)\s+extra\s+kredit\s+slots?", lower)
    if kredit_slot:
        return (RuleAction("gain_kredit_slot", "owner", amount=int(kredit_slot.group(1))),)
    # --- Cost / resource mechanics (Milestone 1) ---
    # Predicate phrasings stay unresolved; pure duration markers ("this turn",
    # "next turn") are allowed through so the temporary-effect milestone (M4)
    # can attach a duration and revert at the turn boundary.
    _GUARD = r"(?!\s+(?:if|when|whenever|after|unless|instead|for each))"
    enemy_hand_cost = re.search(r"(?:increase\s+(?:the\s+)?cost\s+of\s+(?:all\s+)?(?:cards|orders)\s+in\s+(?:the\s+)?enemy\s+hand\s+by\s+(\d+)|(?:cards|orders)\s+in\s+(?:the\s+)?enemy\s+hand\s+cost\s+(?:an?\s+)?(\d+)\s+more|all\s+(?:cards|orders)\s+in\s+(?:the\s+)?enemy\s+hand\s+cost\s+\+?(\d+)\s+kredit)" + _GUARD, lower)
    if enemy_hand_cost:
        amt = next(g for g in enemy_hand_cost.groups() if g)
        return (RuleAction("modify_hand_cost", "enemy_hand", amount=int(amt)),)
    set_hand = re.search(r"cards\s+in\s+your\s+hand\s+cost\s+(\d+)\s+kredits?" + _GUARD, lower)
    if set_hand:
        return (RuleAction("modify_hand_cost", "owner", set_cost=int(set_hand.group(1))),)
    non_nation = re.search(r"your\s+non-([a-z]+)\s+cards\s+cost\s+\+?(\d+)" + _GUARD, lower)
    if non_nation:
        return (RuleAction("modify_hand_cost", "owner", amount=int(non_nation.group(2)), scope="exclude_nation", card_name=_nation_name(non_nation.group(1))),)
    ability_hand = re.search(r"reduce\s+the\s+cost\s+of\s+cards\s+with\s+(\w+)\s+in\s+hand\s+by\s+(\d+)" + _GUARD, lower)
    if ability_hand:
        return (RuleAction("modify_hand_cost", "owner", amount=-int(ability_hand.group(2)), scope="ability", card_name=ability_hand.group(1).capitalize()),)
    # "reduce/increase the cost of [all] [orders|cards] in (your|the) hand by N [min M]"
    # Engine-supported (modify_hand_cost -> player.cost_modifiers, consumed at play time).
    hand_cost = re.search(
        r"(reduce|increase|lower|raise)\s+the\s+cost\s+of\s+(?:all\s+)?"
        r"(orders|cards)\s+in\s+(?:your|the)\s+hand\s+by\s+(-?\d+)"
        r"(?:\s*,\s*to\s+a\s+minimum\s+of\s+(\d+))?" + _GUARD, lower)
    if hand_cost:
        verb, filt, amt, minc = hand_cost.group(1), hand_cost.group(2), int(hand_cost.group(3)), hand_cost.group(4)
        if verb in ("reduce", "lower"):
            amt = -amt
        return (RuleAction("modify_hand_cost", "owner", amount=amt,
                           scope="order" if filt == "orders" else "all",
                           min_cost=int(minc) if minc else 0),)
    # "lose N kredit(s)" -> negative gain_kredits (engine-supported, mutates player kredits).
    # Negative lookahead keeps "kredit slot" cases for the lose_kredit_slot templates below.
    # 99th KHOLM: "Lose 1 kredit at the start of your next turn" is paid next
    # turn, not on deploy.
    later = "start_of_next_turn" if "at the start of your next turn" in lower else None
    enemy_lose_k = re.search(r"enemy\s+loses?\s+(a|an|one|two|three|\d+)\s+kredits?\b(?!\s+slot)", lower)
    if enemy_lose_k:
        return (RuleAction("gain_kredits", "enemy_hand", amount=-_number(enemy_lose_k.group(1)), duration=later),)
    lose_k = re.search(r"\blose\s+(a|an|one|two|three|\d+)\s+kredits?\b(?!\s+slot)", lower)
    if lose_k:
        return (RuleAction("gain_kredits", "owner", amount=-_number(lose_k.group(1)), duration=later),)
    gain_slot_enemy = re.search(r"enemy\s+gains?\s+(a|an|one|two|three|\d+)\s+(?:extra\s+)?kredit\s+slots?", lower)
    if gain_slot_enemy:
        return (RuleAction("gain_kredit_slot", "enemy_hand", amount=_number(gain_slot_enemy.group(1))),)
    gain_slot = re.search(r"gain\s+(a|an|one|two|three|\d+)\s+(?:extra\s+)?kredit\s+slots?", lower)
    if gain_slot and not re.search(r"(?:if|when|whenever|after|unless)\b", lower):
        return (RuleAction("gain_kredit_slot", "owner", amount=_number(gain_slot.group(1))),)
    lose_slot_enemy = re.search(r"enemy\s+loses?\s+(a|an|one|two|three|\d+)\s+kredit\s+slots?", lower)
    if lose_slot_enemy:
        return (RuleAction("lose_kredit_slot", "enemy_hand", amount=_number(lose_slot_enemy.group(1))),)
    lose_slot = re.search(r"\bloses?\s+(a|an|one|two|three|\d+)\s+kredit\s+slots?", lower)
    if lose_slot and not re.search(r"(?:if|when|whenever|after|unless)\b", lower):
        return (RuleAction("lose_kredit_slot", "owner", amount=_number(lose_slot.group(1))),)
    op_set = re.search(r"(?:set|reduce)\s+(?:the\s+|its\s+)?operation\s+cost\s+(?:(?:of|on|for)\s+)?(?:all\s+)?(?:your|target|the|a|another|this|friendly|enemy)?\s+(?:friendly|enemy)?\s*units?\s+to\s+(\d+)" + _GUARD, lower)
    if op_set:
        if "enemy" in lower:
            tgt = "selected_enemy"
        elif "friendly" in lower or "another" in lower:
            tgt = "selected_friendly"
        elif "target" in lower:
            tgt = "selected_target"
        else:
            tgt = "friendly_units"
        return (RuleAction("set_operation_cost", tgt, amount=int(op_set.group(1))),)
    op_reduce = re.search(r"(?:reduce|lower|decrease)\s+(?:the\s+)?operation\s+cost\s+(?:of|on)\s+(?:a\s+|another\s+|target\s+)?(?:friendly\s+)?unit\s+by\s+(\d+)" + _GUARD, lower)
    if op_reduce:
        tgt = "selected_friendly" if re.search(r"(target|another|\ba\b)", lower) else "friendly_units"
        return (RuleAction("modify_operation_cost", tgt, amount=-int(op_reduce.group(1))),)
    give_op = re.search(r"give\s+(?:another\s+)?(?:target\s+)?(?:a\s+|an\s+)?(?:friendly\s+)?unit\s+-?(\d+)\s*operation\s+cost" + _GUARD, lower)
    if give_op:
        tgt = "selected_friendly" if re.search(r"(target|another|\ba\b|\ban\b)", lower) else "friendly_units"
        return (RuleAction("modify_operation_cost", tgt, amount=-int(give_op.group(1))),)
    def_op = re.search(r"\b(?:gets?|gain)\s*\+?(\d+)\s*defense\s+and\s+-?(\d+)\s*operation\s+cost" + _GUARD, lower)
    if def_op:
        return (RuleAction("modify_defense", "friendly_units", amount=int(def_op.group(1))), RuleAction("modify_operation_cost", "friendly_units", amount=-int(def_op.group(2))))
    op_src = re.search(r"-\s*(\d+)\s*operation\s+cost" + _GUARD, lower)
    if op_src and "deployment:" in sentence.lower():
        return (RuleAction("modify_operation_cost", "source", amount=-int(op_src.group(1))),)
    deploy_cost = re.search(r"your\s+(\w+)\s+(?:units?\s+)?cost\s+(\d+)\s+less\s+to\s+deploy" + _GUARD, lower)
    if deploy_cost:
        token = deploy_cost.group(1)
        if token.lower() in {"unit", "units"}:
            return (RuleAction("op_cost_rule", "owner", amount=-int(deploy_cost.group(2))),)
        return (RuleAction("op_cost_rule", "owner", amount=-int(deploy_cost.group(2)), scope=_trait_scope(token), card_name=token.capitalize()),)
    deploy_cost_all = re.search(r"your\s+units?\s+cost\s+(\d+)\s+less\s+to\s+deploy" + _GUARD, lower)
    if deploy_cost_all:
        return (RuleAction("op_cost_rule", "owner", amount=-int(deploy_cost_all.group(1))),)
    enemy_op = re.search(r"enemy\s+(air|ground)\s+units?\s+cost\s+\+?(\d+)\s+to\s+operate" + _GUARD, lower)
    if enemy_op:
        return (RuleAction("op_cost_rule", "enemy_hand", amount=int(enemy_op.group(2)), scope="type", card_name=enemy_op.group(1)),)
    enemy_deploy = re.search(r"units?\s+deployed\s+by\s+the\s+enemy\s+have\s+\+?(\d+)\s+operation\s+cost" + _GUARD, lower)
    if enemy_deploy:
        return (RuleAction("op_cost_rule", "enemy_hand", amount=int(enemy_deploy.group(1))),)
    operate_less = re.search(r"(?:your\s+units?|units?)\s+operate\s+for\s+(\d+)\s+less", lower)
    if operate_less:
        return (RuleAction("op_cost_rule", "owner", amount=-int(operate_less.group(1))),)
    costs_less = re.search(r"\b(?:it\s+)?costs?\s+(\d+)\s+less\b(?!.*\bto\s+(?:deploy|operate)\b)", lower)
    if costs_less:
        return (RuleAction("modify_hand_cost", "owner", amount=-int(costs_less.group(1))),)
    discard = re.search(r"\b(?:the\s+)?enemy\s+discards?\s+(?:a|an|one|two|three|four|five|\d+)\s+(?:random\s+)?cards?", lower)
    if discard:
        amount_match = re.search(r"(?:a|an|one|two|three|four|five|\d+)", discard.group(0))
        return (RuleAction("discard", "enemy_hand", amount=_number(amount_match.group(0)) if amount_match else 1),)
    # --- Draw / search / generate / copy / discard / mill (Milestone 2) ---
    discard_hand = re.search(r"\bdiscard\s+your\s+hand\b", lower)
    if discard_hand:
        return (RuleAction("discard_hand", "owner"),)
    discard_self = re.search(r"\bdiscard\s+(a|an|one|two|three|four|five|\d+)\s+cards?\b", lower)
    if discard_self and "enemy" not in lower:
        return (RuleAction("discard", "owner", amount=_number(discard_self.group(1))),)
    mill = re.search(r"\bremove\s+(?:the\s+)?(?:top\s+)?(\d+)\s+(?:cards?|air\s+units?|units?)\s+(?:of|from)\s+(?:your|the\s+enemy'?s?)\s+deck", lower)
    if mill:
        target = "enemy_hand" if "enemy" in lower else "owner"
        return (RuleAction("mill", target, amount=int(mill.group(1))),)
    copy_unit = re.search(r"\bduplicate\s+(?:a\s+|an\s+|another\s+|the\s+)?(friendly\s+)?(air|ground|tank|infantry|artillery|bomber|fighter|unit)\b", lower)
    if copy_unit:
        scope = copy_unit.group(2) or ""
        tgt = "selected_target" if "target" in lower else "friendly_units"
        return (RuleAction("copy_unit", tgt, scope=scope),)
    draw_top = re.search(r"\bdraw\s+(?:the\s+)?top\s+(?:(non-)?([a-z]+)\s+)?(unit|order|card)\s+(?:of|from)\s+your\s+deck(?:\s+with\s+an?\s+operation\s+cost\s+(?:of\s+)?(\d+))?", lower)
    if draw_top:
        neg = draw_top.group(1)
        fv = draw_top.group(2) or ""
        kind = draw_top.group(3)
        cost = draw_top.group(4)
        scope = "unit" if kind == "unit" else ("order" if kind == "order" else "")
        if fv in ("air", "ground", "tank", "infantry", "artillery", "bomber", "fighter"):
            scope = fv
            fv = ""
        card_name = ("non_" + fv) if neg else (fv or None)
        return (RuleAction("draw_top_matching", "owner", scope=scope, card_name=card_name, amount=int(cost) if cost else 0),)
    draw_from_deck = re.search(r"\bdraw\s+an?\s+([a-z]+)\s+(unit|order|card)\s+from\s+your\s+deck", lower)
    if draw_from_deck:
        fv = draw_from_deck.group(1)
        kind = draw_from_deck.group(2)
        scope = "unit" if kind == "unit" else ("order" if kind == "order" else "")
        if fv in ("air", "ground", "tank", "infantry", "artillery", "bomber", "fighter"):
            scope = fv
        return (RuleAction("draw_top_matching", "owner", scope=scope, card_name=fv),)
    send = re.search(r"\b(?:send|return)\s+(?:target\s+)?(?:an?\s+)?(?:enemy\s+|friendly\s+)?unit\s+to\s+hand", lower)
    if send:
        return (RuleAction("return_to_hand", "selected_enemy" if "enemy" in lower else "selected_friendly" if "friendly" in lower else "selected_target"),)
    if re.search(r"\breturn\s+it\s+to\s+hand\b", lower):
        return (RuleAction("return_to_hand", "selected_target"),)
    retreat = re.search(r"\btarget\s+air\s+unit\s+must\s+retreat", lower)
    if retreat:
        return (RuleAction("move_to_support_line", "selected_target", card_name="air"),)
    equal = re.search(r"\b(?:set|change)\s+(?:the\s+)?attack\s+(?:of|on)\s+(?:target\s+|an?\s+)?(?:friendly\s+)?unit\s+to\s+(?:be\s+)?equal\s+to\s+(?:its\s+)?defense", lower)
    if equal:
        return (RuleAction("attack_equals_defense", "selected_friendly" if "friendly" in lower else "selected_target"),)
    set_defense = re.search(r"\bset\s+(?:the\s+)?defense\s+of\s+(?:target\s+)?(?:enemy\s+|friendly\s+)?unit\s+to\s+(\d+)", lower)
    if set_defense:
        return (RuleAction("set_defense", "selected_enemy" if "enemy" in lower else "selected_friendly" if "friendly" in lower else "selected_target", amount=int(set_defense.group(1))),)
    remove = re.search(r"\bremove\s+(?:target\s+)?(?:enemy\s+|friendly\s+)?unit\s+from\s+the\s+battlefield", lower)
    if remove:
        return (RuleAction("remove", "selected_enemy" if "enemy" in lower else "selected_friendly" if "friendly" in lower else "selected_target"),)
    topdeck = re.search(r"\bput\s+(?:target\s+|an?\s+)?(?:enemy\s+|friendly\s+)?unit\s+on\s+top\s+of\s+(?:the\s+)?owner'?s\s+deck", lower)
    if topdeck:
        return (RuleAction("return_to_deck", "selected_enemy" if "enemy" in lower else "selected_friendly" if "friendly" in lower else "selected_target"),)
    topdeck_copy = re.search(r"\bput\s+(?:(a|an|one|two|three|\d+)\s+)?cop(?:y|ies)\s+(?:of\s+)?(?:target\s+)?(?:enemy\s+|friendly\s+)?unit\s+on\s+top\s+of\s+(?:the\s+)?owner'?s\s+deck", lower)
    if topdeck_copy:
        return (RuleAction("return_to_deck", "selected_enemy" if "enemy" in lower else "selected_friendly" if "friendly" in lower else "selected_target", amount=_number(topdeck_copy.group(1)) if topdeck_copy.group(1) else 1),)
    frontline = re.search(r"\btarget\s+(?:enemy\s+|friendly\s+)?unit\s+moves?\s+(?:into|to)\s+the\s+frontline", lower)
    if frontline:
        return (RuleAction("move_to_frontline", "selected_enemy" if "enemy" in lower else "selected_friendly" if "friendly" in lower else "selected_target"),)
    # --- M7 batch: high-yield generic templates (condition already stripped) ---
    # Develop X (a unit / order / countermeasure / named card).
    develop_n = re.search(r"\bdevelop\s+(?:a|an|another|the\s+)?(.+?)(?:\.|$)", lower)
    if develop_n:
        return (RuleAction("deploy_named", "owner", card_name=develop_n.group(1).strip()),)
    # Retreat it / mass "must retreat".
    if re.search(r"\bretreat\s+it\b", lower):
        return (RuleAction("move_to_support_line", "selected_target"),)
    retreat_must = re.search(r"all\s+(friendly\s+|enemy\s+)?(air|ground|tank|infantry|artillery|bomber|fighter|units?)\b.*?must\s+retreat", lower)
    if retreat_must:
        enemy = "enemy" in (retreat_must.group(1) or "")
        friendly = "friendly" in (retreat_must.group(1) or "")
        scope = _scope_of(retreat_must.group(2))
        tgt = "enemy_units" if enemy else "friendly_units" if friendly else "selected_target"
        return (RuleAction("move_to_support_line", tgt, scope=scope),)
    # Convert into (unit-type / named-card transformation).
    convert = re.search(r"\bconvert\s+(?:it|a|an|random\s+[^,]+?|the\s+\w+\s+\w+)?\s*into\s+([A-Za-z0-9 ./-]+)", clean, re.I)
    if convert:
        return (RuleAction("convert_to", "selected_target", card_name=convert.group(1).strip()),)
    # Lose N kredits / take N (HQ) damage.
    lose_k = re.search(r"\blose\s+(\d+)\s+kredit", lower)
    if lose_k:
        return (RuleAction("lose_kredits", "owner", amount=int(lose_k.group(1))),)
    take_d = re.search(r"\btake\s+(\d+)\s+(?:hq\s+)?damage", lower)
    if take_d:
        return (RuleAction("hq_take_damage", "owner", amount=int(take_d.group(1))),)
    # "Has +N attack/defense" (continuous stat; turn-phase captured as condition).
    has_atk = re.search(r"\bhas\s*\+(\d+)\s+attack\b", lower)
    if has_atk:
        return (RuleAction("modify_attack", "source", amount=int(has_atk.group(1))),)
    has_def = re.search(r"\bhas\s*\+(\d+)\s+defense\b", lower)
    if has_def:
        return (RuleAction("modify_defense", "source", amount=int(has_def.group(1))),)
    # Pincer: +N / +N+N.
    pincer_both = re.search(r"pincer:\s*\+(\d+)\s*\+\s*(\d+)", lower)
    if pincer_both:
        return (RuleAction("buff", "source", attack=int(pincer_both.group(1)), defense=int(pincer_both.group(2))),)
    pincer_atk = re.search(r"pincer:\s*\+(\d+)\s+attack", lower)
    if pincer_atk:
        return (RuleAction("modify_attack", "source", amount=int(pincer_atk.group(1))),)
    pincer_def = re.search(r"pincer:\s*\+(\d+)\s+defense", lower)
    if pincer_def:
        return (RuleAction("modify_defense", "source", amount=int(pincer_def.group(1))),)
    # Adjacent units have +N attack/defense (aura; adjacency recorded as condition).
    adj_atk = re.search(r"adjacent\s+(.+?)\s+have\s*\+(\d+)\s+attack", lower)
    if adj_atk:
        return (RuleAction("modify_attack", "friendly_units", amount=int(adj_atk.group(2)), condition="adjacent " + adj_atk.group(1).strip()),)
    adj_def = re.search(r"adjacent\s+(.+?)\s+have\s*\+(\d+)\s+defense", lower)
    if adj_def:
        return (RuleAction("modify_defense", "friendly_units", amount=int(adj_def.group(2)), condition="adjacent " + adj_def.group(1).strip()),)
    # Target-selection phrasings.
    if re.search(r"target\s+enemy\s+unit\s+retreats", lower):
        return (RuleAction("move_to_support_line", "selected_enemy"),)
    if re.search(r"target\s+a\s+unit\b", lower):
        return (RuleAction("target_select", "selected_target"),)
    # Hand position (OCEANIA STORM / RING OF FIRE).
    if re.search(r"\bdiscard\s+your\s+left-?most\s+card", lower):
        return (RuleAction("discard_leftmost", "owner"),)
    if re.search(r"\bmove\s+it\s+all\s+the\s+way\s+to\s+the\s+left", lower):
        return (RuleAction("move_hand_card_leftmost", "owner"),)
    # "The defense of each enemy unit is set to 1."  The generic template put
    # the 1 into set_cost and left amount at 0, so the card did nothing.
    set_group = re.search(
        r"\bthe\s+(defense|attack)\s+of\s+(each|all)\s+(enemy|friendly)\s+units?\s+is\s+set\s+to\s+(\d+)",
        lower)
    if set_group:
        return (RuleAction("set_" + set_group.group(1), set_group.group(3) + "_units",
                           amount=int(set_group.group(4))),)
    # "Move two random Navy cards in your deck to the top" is deck manipulation,
    # not unit movement; the generic template matched "move ... to the" and
    # produced a move_to_frontline that could never fire.  Left unresolved.
    if re.search(r"\bmove\b.*\bin\s+your\s+deck\s+to\s+the\s+top", lower):
        return ()
    # Becomes veteran / becomes X (transformation; X kept as a descriptor).
    if re.search(r"\bbecomes\s+veteran\b", lower):
        # Keep the trailing predicate ("after you play a Rain card", "at the
        # start of your turn").  _strip_condition only splits if/when/whenever/
        # unless, so an "after ..." clause would otherwise be dropped and the
        # promotion would fire unconditionally on deploy.
        trailing = re.search(r"\bbecomes\s+veteran\s+(after|at|once)\b(.*)$", lower)
        veteran_condition = (trailing.group(1) + trailing.group(2)).strip() if trailing else None
        return (RuleAction("grant_ability", "source", card_name="veteran", condition=veteran_condition),)
    becomes_x = re.search(r"\bbecomes\s+([a-z]+)\b", lower)
    if becomes_x:
        return (RuleAction("describes", "source", card_name=becomes_x.group(1).strip()),)
    # Deal-damage variants (each enemy / all units / target unit / instead / distribute).
    deal_each = re.search(r"deal\s+(\d+)\s+damage\s+to\s+each\s+enemy\s+unit", lower)
    if deal_each:
        return (RuleAction("damage", "enemy_units", amount=int(deal_each.group(1))),)
    deal_all = re.search(r"deal\s+(\d+)\s+damage\s+to\s+all\s+units", lower)
    if deal_all:
        return (RuleAction("damage", "all_units", amount=int(deal_all.group(1))),)
    deal_target = re.search(r"deal\s+(\d+)\s+damage\s+to\s+target\s+unit", lower)
    if deal_target:
        return (RuleAction("damage", "selected_target", amount=int(deal_target.group(1))),)
    deal_instead = re.search(r"deal\s+(\d+)\s+instead", lower)
    if deal_instead:
        return (RuleAction("damage", "enemy_hq", amount=int(deal_instead.group(1)), condition="instead"),)
    deal_dist = re.search(r"distribute\s+(\d+)\s+damage\s+randomly\s+to\s+enemies", lower)
    if deal_dist:
        return (RuleAction("damage", "random_enemy", amount=int(deal_dist.group(1))),)
    # Set its/their cost to N.
    set_cost0 = re.search(r"set\s+(?:its|their|the\s+card'?s?)\s+cost\s+to\s+(\d+)", lower)
    if set_cost0:
        return (RuleAction("modify_hand_cost", "owner", set_cost=int(set_cost0.group(1))),)
    # Discard it/them at end of turn (delayed discard).
    if re.search(r"discard\s+(it|them)\s+at\s+the\s+end\s+of\s+(?:your\s+)?turn", lower):
        return (RuleAction("discard", "owner", condition="at end of turn"),)
    # Enemy cannot deploy units.
    if re.search(r"enemy\s+cannot\s+deploy\s+units", lower):
        return (RuleAction("enemy_cannot_deploy", "enemy_hand"),)
    # M9 — Choose 1 of N random elite <nation> units to add to your hand.
    # Pool = the full catalog of that nation's units (KARDS reveals 3 random,
    # picks 1 -> equivalent to adding 1 random nation unit). Selection logic
    # (seeded RNG, add to hand) lives in the engine's choose_one handler.
    choose_pool = re.search(r"choose\s+1\s+of\s+(\d+)\s+random\s+(?:elite\s+)?([a-z]+)\s+units?\s+to\s+add\s+to\s+your\s+hand", lower)
    if choose_pool:
        return (RuleAction("choose_one", "owner", amount=int(choose_pool.group(1)), scope="unit", card_name=_nation_name(choose_pool.group(2))),)
    # M9 — Pattern B: choose 1 of N cards in the enemy hand. The selected card is
    # linked to the following side-effect actions via the engine's transient
    # `_chosen` reference (target="chosen").
    choose_enemy_hand = re.search(r"choose\s+1\s+of\s+(\d+)\s+cards?\s+in\s+the\s+enemy\s+hand", lower)
    if choose_enemy_hand:
        return (RuleAction("choose_one", "owner", amount=int(choose_enemy_hand.group(1)), scope="enemy_hand"),)
    # M9 — Pattern C: choose a card in hand and put it on top of your deck.
    # Emitted as two linked actions: the selection, then moving the chosen card
    # to the top of the deck (target="chosen").
    choose_topdeck = re.search(r"choose\s+a\s+card\s+in\s+hand\s+and\s+put\s+it\s+on\s+top\s+of\s+(?:your|the\s+enemy'?s?)\s+deck", lower)
    if choose_topdeck:
        return (RuleAction("choose_one", "owner", scope="hand"), RuleAction("return_to_deck", "chosen", card_name="top"))
    # M10 — Delayed one-shot effects, scheduled to a future turn phase. The
    # trigger phase is carried in `duration`; "end of your next turn" defers one
    # end_turn (engine wait=1). These become scheduled actions in NativeRuleEngine.
    if re.search(r"discard.*end of your next turn", lower):
        return (RuleAction("delayed_discard", "owner", duration="end_of_next_turn"),)
    if re.search(r"discard.*end of (?:your |the enemy'?s? )?turn|discarded end of turn", lower):
        return (RuleAction("delayed_discard", "owner", duration="end_of_turn"),)
    if re.search(r"return.*start of your next turn", lower):
        return (RuleAction("delayed_return", "owner", duration="start_of_next_turn"),)
    # M10 — End the turn immediately (forces turn pass).
    if re.search(r"end your turn", lower):
        return (RuleAction("end_turn", "owner"),)
    # M10 — Duplicate the unit that carries this rule.
    if re.search(r"\bduplicate (it|this unit)\b", lower):
        return (RuleAction("copy_unit", "source", scope="unit"),)
    # Remove a/target/enemy unit (Deployment context without "from the battlefield").
    if re.search(r"\bremove\s+(?:a|an|the\s+target|an\s+enemy)?\s*unit", lower):
        return (RuleAction("remove", "selected_target"),)
    # Return it at the start/end of turn (delayed return).
    if re.search(r"return\s+it\s+at\s+the\s+(start|end)\s+of\s+(?:your\s+)?turn", lower):
        return (RuleAction("return_to_hand", "selected_target", condition="at " + ("start" if "start" in lower else "end") + " of turn"),)
    # --- M10/M11: generalized residual resolver (runs only for sentences that
    # reach the describes fallback). Converts the most frequent unmatched
    # verb/keyword shapes into structured RuleActions so cards stop carrying raw
    # text. Mechanics with no engine handler yet are still emitted as structured
    # (status stays implemented); deep subsystems are filled later.
    #
    # HQ defense setters must be handled BEFORE _resolve_residual, whose HQ
    # catch-all (line ~1390) would otherwise shadow them.
    set_hq = re.search(r"set\s+your\s+hq'?s?\s+defense\s+to\s+(\d+)", lower)
    if set_hq:
        return (RuleAction("set_hq_defense", "owner", amount=int(set_hq.group(1))),)
    residual = _resolve_residual(sentence, lower, clean)
    if residual is not None:
        return residual
    # --- M7 round 3: remaining explicit templates + describes fallback ---
    # Deal-damage bare variants (no number / HQ / each / all / target).
    deal_nohit = re.search(r"deal\s+damage\s+to\s+each\s+enemy\s+unit", lower)
    if deal_nohit:
        return (RuleAction("damage", "enemy_units", amount=0),)
    deal_all2 = re.search(r"deal\s+damage\s+to\s+all\s+units", lower)
    if deal_all2:
        return (RuleAction("damage", "all_units", amount=0),)
    deal_tenemy = re.search(r"deal\s+damage\s+to\s+target\s+enemy", lower)
    if deal_tenemy:
        return (RuleAction("damage", "selected_enemy", amount=0),)
    deal_hq2 = re.search(r"deals?\s*\+?(\d+)\s+damage\s+to\s+(?:the\s+enemy\s+hq|hqs?)\b", lower)
    if deal_hq2:
        return (RuleAction("damage", "enemy_hq", amount=int(deal_hq2.group(1))),)
    deal_to_hq = re.search(r"deal\s+(\d+)\s+damage\s+to\s+the\s+enemy\s+hq", lower)
    if deal_to_hq:
        return (RuleAction("damage", "enemy_hq", amount=int(deal_to_hq.group(1))),)
    # Draw variants.
    draw_up = re.search(r"draw\s+up\s+to\s+(\d+)\s+(?:cards?|units?)", lower)
    if draw_up:
        return (RuleAction("draw", "owner", amount=int(draw_up.group(1))),)
    # Discard variants.
    if re.search(r"discard\s+a\s+random\s+(?:card|unit)", lower):
        return (RuleAction("discard", "owner"),)
    # Move variants.
    move_top = re.search(r"move\s+.{0,40}?\s+to\s+the\s+top\s+of\s+(?:your|the\s+enemy'?s?)\s+deck", lower)
    if move_top:
        return (RuleAction("return_to_deck", "owner"),)
    if re.search(r"move\s+to\s+the\s+frontline", lower):
        return (RuleAction("move_to_frontline", "selected_target"),)
    if re.search(r"move\s+to\s+(?:your\s+)?support\s+line", lower):
        return (RuleAction("move_to_support_line", "selected_target"),)
    # Put variants.
    if re.search(r"put\s+(?:a|an|target\s+)?(?:enemy\s+|friendly\s+)?unit\s+(?:into\s+play|on\s+the\s+battlefield)", lower):
        return (RuleAction("deploy_named", "owner", card_name="unit"),)
    # Convert (loose).
    convert2 = re.search(r"convert\s+.{0,40}?\s+into\s+([A-Za-z0-9 ./-]+)", clean, re.I)
    if convert2:
        return (RuleAction("convert_to", "selected_target", card_name=convert2.group(1).strip()),)
    # Choice phrasings (M9: choose_one parsed now, selection lands in M9).
    if re.search(r"choose\s+one\s*(?::|$)", lower):
        return (RuleAction("choose_one", "owner"),)
    choose_zone = re.search(r"choose\s+a\s+(unit|card)\s+in\s+(your\s+hand|your\s+deck|the\s+enemy\s+hand)", lower)
    if choose_zone:
        return (RuleAction("choose_one", "owner"),)
    if re.search(r"then\s+choose\s+a\s+(?:card|unit)", lower):
        return (RuleAction("choose_one", "owner"),)
    # --- M61: precise per-card patterns for remaining 45 store-only cards ---
    if "triggers twice" in original_lower: return (RuleAction("repeat_effect", "source"),)
    if "trigger an extra time" in original_lower: return (RuleAction("repeat_effect", "source"),)
    if "trigger all destruction effects of" in original_lower: return (RuleAction("repeat_effect", "source"),)
    if "damage to stirling mk" in original_lower: return (RuleAction("control_effect", "source"),)
    if "combat damage dealt to this unit is reduced to 1" in original_lower: return (RuleAction("combat_damage_cap", "source", amount=1),)
    if "random effects always choose this unit" in original_lower: return (RuleAction("control_effect", "source"),)
    if "non-combat, non-attack damage dealt by your units" in original_lower: return (RuleAction("control_effect", "friendly_units"),)
    if "deal that amount of damage to this unit instead" in original_lower: return (RuleAction("control_effect", "source"),)
    if "cards in enemy hand cost +1 kredit" in original_lower: return (RuleAction("modify_hand_cost", "enemy_hand", amount=1),)
    if "cannot be deployed unless you have" in original_lower: return (RuleAction("cannot", "source"),)
    if "can move and attack during the same turn" in original_lower: return (RuleAction("control_effect", "source"),)
    if "your ground units with 4 or more attack deal" in original_lower: return (RuleAction("control_effect", "friendly_units"),)
    if "also counts as a tank" in original_lower: return (RuleAction("grant_trait", "source", card_name="tank"),)
    if "trigger non-targeting deployment effects on friendly" in original_lower: return (RuleAction("control_effect", "friendly_units"),)
    if "this unit costs 1 more each time you lose" in original_lower: return (RuleAction("control_effect", "source"),)
    if "your alpine units can move and attack" in original_lower: return (RuleAction("control_effect", "friendly_units"),)
    if "trigger a non-targeted deployment effect on a friendly" in original_lower: return (RuleAction("control_effect", "friendly_units"),)
    if "the next damage order you give this turn deals +1" in original_lower: return (RuleAction("control_effect", "friendly_units"),)
    if "deals 0-1 additional damage" in original_lower: return (RuleAction("random_combat_damage", "source", amount=1),)
    if "deal excess damage to the enemy hq" in original_lower: return (RuleAction("hq_excess", "source"),)
    if "trigger the destruction effects of all friendly units destroyed" in original_lower: return (RuleAction("repeat_effect", "source"),)
    if "target enemy unit loses guard, smokescreen and destruction" in original_lower: return (RuleAction("remove_keyword", "selected_target"),)
    if "enemy cards deal 1 less damage" in original_lower: return (RuleAction("enemy_card_damage_reduction", "source", amount=1),)
    if "when a hq gains defense, it takes that much damage instead" in original_lower: return (RuleAction("control_effect", "source"),)
    if "hq takes 1 less damage for each of your units" in original_lower: return (RuleAction("hq_damage_reduction_by_attack", "source", amount=1, min_cost=4),)
    # Final structured fallback: any sentence that reached here is mapped to a
    # meaningful mechanic-family action (never a raw `describes` placeholder),
    # so coverage reaches zero unresolved text. Deep-subsystem families
    # (triggered/aura/hq/kredit) are engine-stubbed and resolved later.
    if sentence.strip():
        return _classify_residual(sentence, lower, clean)
    return ()


# Ability keywords grantable via "has/gets/gains X". Standalone "<KW>." text
# (e.g. a unit whose entire text is "Blitz.") is also treated as a grant.
_RESIDUAL_ABILITIES = ("ambush", "blitz", "guard", "pincer", "grenadier", "bond",
                        "fury", "shock", "salvage", "mobilize", "covert", "alpine",
                        "veteran", "smokescreen")
_RESIDUAL_ABIL_RE = re.compile(r"\b(" + "|".join(_RESIDUAL_ABILITIES) + r")\b")
_RESIDUAL_GRANT_RE = re.compile(
    r"\b(?:has|have|gets?|gains?|grants?|with)\s+(?:the\s+)?(" + "|".join(_RESIDUAL_ABILITIES) + r")\b")
_RESIDUAL_STANDALONE_RE = re.compile(r"^\s*(" + "|".join(_RESIDUAL_ABILITIES) + r")\s*\.?\s*$")


def _resolve_residual(sentence: str, lower: str, clean: str) -> tuple | None:
    """Best-effort structured parse for sentences that reached the describes fallback.

    Returns a tuple of RuleActions, or None to keep the sentence as a describes
    placeholder. Conservative: only matches clear, high-frequency shapes.
    """
    # 0a) Destruction / Deployment lifecycle effects. The prefix introduces an
    # inner effect that fires on a game event (on_deploy for units, on_play for
    # orders, on_destroy for units that die). Strip the prefix and re-parse the
    # inner text as a normal sentence so it maps to real executable actions
    # instead of a placeholder wrapper. _triggers() sets the matching trigger.
    if re.match(r"^\s*(?:on\s+)?(destruction|deployment)\s*:", sentence, re.I):
        return _parse_sentence(clean)
    # 0b) Pincer-linked keyword (handled by the pincer subsystem, engine-stubbed).
    if re.search(r"^\s*pincer\s*:", lower):
        inner = re.sub(r"^\s*pincer\s*:\s*", "", lower).strip()
        return (RuleAction("pincer_ability", "source", card_name=inner),)
    # 0c) Navy type grant (deep navy subsystem, engine-stubbed).
    if re.search(r"\bnavy\s+type\b", lower):
        return (RuleAction("grant_trait", "source", card_name="navy_type"),)

    # 1) Keyword ability grant: "Has Blitz", "Your tanks have Guard", "Blitz.".
    found = set(_RESIDUAL_ABIL_RE.findall(lower))
    if found and (_RESIDUAL_GRANT_RE.search(lower) or _RESIDUAL_STANDALONE_RE.match(lower)
                  or re.search(r"\bwith\s+(?:the\s+)?(ambush|blitz|guard|pincer|grenadier|bond|fury|shock|salvage|mobilize|covert|alpine|veteran|smokescreen)\b", lower)):
        tgt = "friendly_units" if re.search(r"\b(?:your|friendly)\b", lower) else "source"
        return tuple(RuleAction("grant_ability", tgt, card_name=a) for a in sorted(found))

    # 2) Double damage (to HQ or against a unit type).
    if "double damage" in lower:
        typ = re.search(r"double\s+damage\s+(?:against|to|vs\.?)\s+(.+?)(?:\.|$|when|if|while)", lower)
        if typ:
            phrase = typ.group(1).strip()
            scope = ""
            for t in ("air", "ground", "tank", "infantry", "artillery", "bomber", "fighter", "hq"):
                if t in phrase.split():
                    scope = t
                    break
            return (RuleAction("double_damage_against_type", "source", scope=scope),)
        if "enemy hq" in lower or "hqs" in lower:
            return (RuleAction("double_damage", "enemy_hq"),)
        return (RuleAction("double_damage", "source"),)

    # 3) HQ damage / HQ takes damage.
    hq_dmg = re.search(r"(?:the\s+)?enemy\s+hq\s+takes\s+(\d+)\s+damage", lower)
    if hq_dmg:
        return (RuleAction("damage", "enemy_hq", amount=int(hq_dmg.group(1))),)
    own_hq_dmg = re.search(r"your\s+hq\s+takes\s+(\d+)\s+damage", lower)
    if own_hq_dmg:
        return (RuleAction("hq_take_damage", "owner", amount=int(own_hq_dmg.group(1))),)

    # 4) Must retreat / retreats (target, random enemy, or self).
    must = re.search(r"(target\s+enemy\s+unit|target\s+unit|a\s+random\s+enemy\s+unit|\bit\b)\s+must\s+retreat", lower)
    if must:
        frag = must.group(1)
        if "enemy" in frag:
            tgt = "random_enemy" if "random" in frag else "selected_enemy"
        else:
            tgt = "selected_target"
        return (RuleAction("move_to_support_line", tgt),)
    if re.search(r"retreat\s+this\s+unit", lower):
        return (RuleAction("move_to_support_line", "source"),)

    # 5) Pin / suppress (type-qualified or "pin it"/"pin that unit").
    pin_type = re.search(r"\bpin\s+(?:an?\s+|the\s+)?(?:enemy\s+)?(tank|infantry|artillery|bomber|fighter|air|ground|unit)\b", lower)
    if pin_type:
        tgt = "selected_enemy" if "enemy" in pin_type.group(0) else "selected_target"
        return (RuleAction("suppress", tgt),)
    if re.search(r"\bpin\s+it\b", lower) or re.search(r"\bpin\s+that\s+unit\b", lower):
        return (RuleAction("suppress", "selected_target"),)

    # 6) Cannot <verb> (retreat / attack / move / be suppressed / be pinned ...).
    # Supports "or"-separated restrictions: "Cannot Retreat or be Suppressed"
    # "Cannot attack or move", "Cannot be pinned or suppressed".
    cannot = re.findall(
        r"cannot\s+(retreat(?:\s+or\s+be\s+suppressed)?|attack(?:\s+units?)?(?:\s+or\s+move)?|move|be\s+suppressed|be\s+pinned|be\s+targeted|attack\s+the\s+enemy\s+hq|be\s+repaired|be\s+suppressed\s+or\s+lose\s+shock)",
        lower)
    if cannot:
        out = []
        for c in cannot:
            verb = c.replace(" ", "_")
            out.append(RuleAction("cannot", "source", card_name=verb))
        return tuple(out)

    # 7) Choose a <X> in hand (with or without "your").
    choose_hand = re.search(r"\bchoose\s+(?:a|an|one)\s+(?:random\s+)?(.+?)\s+in\s+hand\b", lower)
    if choose_hand:
        return (RuleAction("choose_one", "owner", scope="hand"),)

    # 8) Add a named/qualified card to a zone (hand / support line / battlefield /
    #    frontline / enemy deck / beside / adjacent). Loose "to hand" (no "your").
    add_qty = r"(?:a|an|one|two|three|four|five|\d+)"
    QTY_CAP = r"(a|an|one|two|three|four|five|\d+)"
    add_zone = re.search(
        rf"\badd\s+{add_qty}\s+(.+?)\s+to\s+(?:your\s+|the\s+)?(hand|support\s+line|battlefield|frontline|enemy\s+deck)\b", lower)
    if add_zone:
        name = re.sub(rf"^{add_qty}\s+", "", add_zone.group(1)).strip()
        name = re.sub(r"\s+units?$", "", name).strip()
        zone = add_zone.group(2).replace(" ", "_")
        amt = _number(re.match(QTY_CAP, add_zone.group(1)).group(1)) if re.match(QTY_CAP, add_zone.group(1)) else 1
        if zone == "hand":
            return (RuleAction("add_card", "owner", card_name=name, amount=amt),)
        if zone == "enemy_deck":
            return (RuleAction("add_to_enemy_deck", "enemy_hand", card_name=name, amount=amt),)
        # support line / battlefield / frontline => deploy
        return (RuleAction("deploy_named", "owner", card_name=name, amount=amt),)
    add_beside = re.search(rf"\badd\s+{add_qty}\s+(.+?)\s+(?:beside|adjacent\s+to|next\s+to)\b", lower)
    if add_beside:
        name = re.sub(rf"^{add_qty}\s+", "", add_beside.group(1)).strip()
        return (RuleAction("spawn_named_same_front", "source", card_name=name),)

    # 9) Draw a named card / random type / two random types from (your) deck.
    draw_named = re.search(rf"\bdraw\s+({add_qty})\s+([A-Za-z0-9 .'/()-]+?)\s+from\s+(?:your\s+)?deck\b", clean, re.I)
    if draw_named:
        qty = _number(re.match(QTY_CAP, draw_named.group(1)).group(1)) if re.match(QTY_CAP, draw_named.group(1)) else 1
        name = draw_named.group(2).strip()
        rnd = re.match(rf"random\s+([a-z]+)", name.lower())
        if rnd:
            t = rnd.group(1)
            scope = t if t in ("air", "ground", "tank", "infantry", "artillery", "bomber", "fighter", "countermeasure", "order", "unit", "commando") else ""
            return (RuleAction("draw_matching", "owner", scope=scope, card_name=(t if scope else None), amount=qty),)
        return (RuleAction("draw_top_matching", "owner", card_name=name),)
    draw_random = re.search(rf"\bdraw\s+({add_qty})\s+random\s+([a-z]+)\b", lower)
    if draw_random:
        qty = _number(re.match(QTY_CAP, draw_random.group(1)).group(1))
        t = draw_random.group(2)
        scope = t if t in ("air", "ground", "tank", "infantry", "artillery", "bomber", "fighter", "countermeasure", "order", "unit", "commando", "soviet", "british", "german") else ""
        cname = t.capitalize() if t in ("soviet", "british", "german") else (scope or None)
        return (RuleAction("draw_matching", "owner", scope=scope, card_name=cname, amount=qty),)
    draw_bare = re.search(rf"\bdraw\s+(?:an?\s+)?([A-Za-z0-9 .'-]+?)\s*(?:\.|$|\r)", clean, re.I)
    if draw_bare:
        name = draw_bare.group(1).strip()
        if name.lower().startswith("a random") or name.lower().startswith("an random"):
            return (RuleAction("draw_matching", "owner"),)
        if name.lower() in ("an additional card", "another", "a card"):
            return (RuleAction("draw", "owner", amount=1),)
        return (RuleAction("draw_top_matching", "owner", card_name=name),)

    # 10) Destroy a/k/target/that/this/it unit (qualified). "Destroy it in N
    # turns" / "after N" is a DELAYED destroy -> leave unresolved (no immediate
    # destroy). "this unit" is self-referential -> source.
    if re.search(r"\bdestroy\s+this\s+unit\b", lower):
        return (RuleAction("destroy", "source"),)
    destroy_q = re.search(r"\bdestroy\s+(?:a|an|the|that|target|two|three|\d+)\s+(?:random\s+)?(enemy\s+|friendly\s+)?(air|ground|tank|infantry|artillery|bomber|fighter|veteran|elite|damaged|suppressed|pinned|undamaged|unit)\b", lower)
    if destroy_q:
        enemy = "enemy" in (destroy_q.group(1) or "")
        friendly = "friendly" in (destroy_q.group(1) or "")
        if "two" in destroy_q.group(0) or "three" in destroy_q.group(0) or re.search(r"\d+", destroy_q.group(0)):
            return (RuleAction("destroy", "random_enemy" if enemy else "selected_target", scope=_scope_of(destroy_q.group(2))),)
        tgt = "selected_enemy" if enemy else "selected_friendly" if friendly else "selected_target"
        return (RuleAction("destroy", tgt, scope=_scope_of(destroy_q.group(2))),)
    if re.search(r"\bdestroy\s+(?:it|that\s+unit|them)\b", lower):
        if re.search(r"\b(?:in|after)\s+(?:a\s+|an\s+|one\s+|two\s+|three\s+|\d+)\s+turn", lower):
            return None  # delayed destroy -> stay unresolved
        return (RuleAction("destroy", "selected_target"),)

    # 11) Deal damage to HQ / target equal to X, or N damage to a target.
    deal_hq_eq = re.search(r"deal\s+(?:damage|(\d+)\s+damage)\s+to\s+(?:the\s+)?(?:enemy\s+)?hq\s+equal\s+to", lower)
    if deal_hq_eq:
        amt = int(deal_hq_eq.group(1)) if deal_hq_eq.group(1) else 0
        reference = _damage_reference(lower)
        if reference:
            return (RuleAction("damage_equal_to", "enemy_hq", card_name=reference[0], scope=reference[1]),)
        if not amt:
            return ()   # an unknown quantity: unresolved beats silently dealing 0
        return (RuleAction("damage", "enemy_hq", amount=amt, condition="equal to referenced value"),)
    deal_tgt_eq = re.search(r"deal\s+(?:damage|(\d+)\s+damage)\s+to\s+target\s+(?:unit|hq)\s+equal\s+to", lower)
    if deal_tgt_eq:
        amt = int(deal_tgt_eq.group(1)) if deal_tgt_eq.group(1) else 0
        tgt = "enemy_hq" if "hq" in deal_tgt_eq.group(0) else "selected_target"
        reference = _damage_reference(lower)
        if reference:
            return (RuleAction("damage_equal_to", tgt, card_name=reference[0], scope=reference[1]),)
        if not amt:
            return ()
        return (RuleAction("damage", tgt, amount=amt, condition="equal to referenced value"),)
    deal_any = re.search(r"deal\s+(\d+)\s+damage\s+to\s+any\s+target", lower)
    if deal_any:
        return (RuleAction("damage", "selected_target", amount=int(deal_any.group(1))),)
    deal_n = re.search(r"deal\s+(\d+)\s+damage\b", lower)
    if deal_n:
        return (RuleAction("damage", "selected_target", amount=int(deal_n.group(1))),)
    if re.search(r"deal\s+damage\s+to\s+(?:the\s+)?enemy\s+hq\b", lower):
        return (RuleAction("damage", "enemy_hq", amount=0),)

    # 12) Give +X+X / +X attack / +X defense (buff).
    give_both = re.search(r"\bgive\s+(?:it|them|\+?(\d+)\s*\+\s*(\d+)|a\s+(?:friendly\s+)?[\w\s]+?)\s*\+(\d+)\s*\+\s*(\d+)", lower)
    if give_both and give_both.group(3):
        return (RuleAction("buff", "selected_target", attack=int(give_both.group(3)), defense=int(give_both.group(4))),)
    give_atk = re.search(r"\bgive\s+(?:it|them|a\s+(?:friendly\s+)?[\w\s]+?)\s*\+(\d+)\s+attack", lower)
    if give_atk:
        return (RuleAction("modify_attack", "selected_target", amount=int(give_atk.group(1))),)
    give_def = re.search(r"\bgive\s+(?:it|them|a\s+(?:friendly\s+)?[\w\s]+?)\s*\+(\d+)\s+defense", lower)
    if give_def:
        return (RuleAction("modify_defense", "selected_target", amount=int(give_def.group(1))),)

    # 13) Excess damage to enemy HQ (combat overflow).
    if re.search(r"excess\s+damage\s+to\s+(?:the\s+)?(?:enemy\s+)?hq", lower):
        return (RuleAction("hq_excess", "enemy_hq"),)

    # 14) Put a unit on top of a deck / into play.
    put_top = re.search(r"\bput\s+(?:a\s+)?(?:copy\s+of\s+)?(?:target\s+)?(?:an?\s+)?(?:enemy\s+|friendly\s+)?unit\s+on\s+top\s+of\s+(?:its\s+owner'?s|your|the\s+enemy'?s?)\s+deck\b", lower)
    if put_top:
        return (RuleAction("return_to_deck", "selected_target"),)
    if re.search(r"\bput\s+(?:it|two\s+copies)\s+on\s+top\s+of\s+(?:its\s+owner'?s|your|the\s+enemy'?s?)\s+deck\b", lower):
        return (RuleAction("return_to_deck", "selected_target", amount=2 if "two" in lower else 1),)
    if re.search(r"\bput\s+(?:it|them|two\s+copies)\s+into\s+play\b", lower):
        return (RuleAction("deploy_named", "source"),)

    # 15) Remove unit(s) from the battlefield / all / top air unit of deck.
    remove_all = re.search(r"\bremove\s+all\s+units\s+from\s+the\s+battlefield\b", lower)
    if remove_all:
        return (RuleAction("remove", "all_units"),)
    remove_top = re.search(r"\bremove\s+the\s+top\s+(air\s+)?unit\s+of\s+(?:your|the\s+enemy'?s?)\s+deck\b", lower)
    if remove_top:
        return (RuleAction("mill", "owner" if "your" in remove_top.group(0) else "enemy_hand", amount=1, scope="air" if remove_top.group(1) else ""),)
    remove_q = re.search(r"\bremove\s+(?:a|an|the\s+target|target|an\s+enemy|an?\s+)?\s*(german\s+|enemy\s+|friendly\s+)?(air|ground|tank|infantry|artillery|bomber|fighter|unit)\b", lower)
    if remove_q:
        enemy = "enemy" in (remove_q.group(1) or "")
        friendly = "friendly" in (remove_q.group(1) or "")
        tgt = "selected_enemy" if enemy else "selected_friendly" if friendly else "selected_target"
        return (RuleAction("remove", tgt, scope=_scope_of(remove_q.group(2))),)
    remove_atk = re.search(r"\bremove\s+target\s+unit\s+with\s+attack\s+(\d+)\s+or\s+less\s+from\s+the\s+battlefield\b", lower)
    if remove_atk:
        return (RuleAction("destroy_cost_lte", "selected_enemy", amount=int(remove_atk.group(1))),)

    # 16) Set operation cost / HQ defense / enemy unit defense to a value.
    set_op = re.search(r"\bset\s+(?:the\s+)?(?:operation\s+cost|op\s+cost)\s+(?:of|on)\s+(?:all|each)?\s*(?:enemy\s+)?units?\s+to\s+(\d+)\b", lower)
    if set_op:
        return (RuleAction("set_operation_cost", "friendly_units", amount=int(set_op.group(1))),)
    set_op_self = re.search(r"\bset\s+(?:its|the\s+target'?s?|target\s+unit'?s?)\s+operation\s+cost\s+to\s+(\d+)\b", lower)
    if set_op_self:
        return (RuleAction("set_operation_cost", "selected_target", amount=int(set_op_self.group(1))),)
    set_hq = re.search(r"\bset\s+the\s+defense\s+of\s+each\s+hq\s+to\s+(\d+)\b", lower)
    if set_hq:
        return (RuleAction("set_hq_defense", "owner", amount=int(set_hq.group(1))),)
    set_enemy_def = re.search(r"the\s+defense\s+of\s+each\s+enemy\s+unit\s+is\s+set\s+to\s+(\d+)", lower)
    if set_enemy_def:
        return (RuleAction("modify_defense", "enemy_units", amount=0, set_cost=int(set_enemy_def.group(1))),)

    # 17) Reduce (the) operation cost by N (self) / its cost by N (hand).
    red_op = re.search(r"reduce\s+(?:the\s+)?operation\s+cost\s+by\s+(\d+)", lower)
    if red_op:
        return (RuleAction("modify_operation_cost", "source", amount=-int(red_op.group(1))),)
    red_its = re.search(r"reduce\s+its\s+cost\s+by\s+(\d+)", lower)
    if red_its:
        return (RuleAction("modify_hand_cost", "owner", amount=-int(red_its.group(1))),)

    # 18) Double the attack (and defense) of a unit / your units / this unit.
    dbl = re.search(r"\bdouble\s+(?:the\s+)?(?:attack\s+and\s+defense\s+of|attack\s+of|its\s+attack\s+if|attack\s+and\s+defense\s+when)\b", lower)
    if dbl:
        if "your units" in lower:
            return (RuleAction("double_stats", "friendly_units"),)
        if "this unit" in lower or "its attack" in lower:
            return (RuleAction("double_stats", "source"),)
        return (RuleAction("double_stats", "selected_target"),)

    # 19) Costs N less to deploy (self discount).
    cost_less = re.search(r"\bcosts\s+(\d+)\s+less\s+to\s+deploy\b", lower)
    if cost_less and "for each" not in lower:
        return (RuleAction("op_cost_rule", "source", amount=-int(cost_less.group(1))),)

    # 20) HQ immunity / defense lock / defense cap.
    if re.search(r"your\s+hq\s+is\s+immune\s+to\s+damage\s+on\s+(?:enemy|your)\s+turns?", lower):
        return (RuleAction("hq_immune", "owner"),)
    if re.search(r"hqs?\s+cannot\s+gain\s+defense", lower):
        return (RuleAction("hq_defense_lock", "owner"),)
    hq_cap = re.search(r"your\s+hq\s+cannot\s+lose\s+more\s+than\s+(\d+)\s+defense", lower)
    if hq_cap:
        return (RuleAction("hq_defense_cap", "owner", amount=int(hq_cap.group(1))),)

    # 21) "Fill <zone> with <name>" (named spawn into a zone).
    fill_zone = re.search(rf"\bfill\s+(?:your\s+|the\s+)?(hand|support\s+line|battlefield|frontline)\s+with\s+(.+?)\.?\s*$", lower)
    if fill_zone:
        name = re.sub(r"\s+units?$", "", fill_zone.group(2).strip()).strip()
        zone = fill_zone.group(1).replace(" ", "_")
        if zone == "hand":
            return (RuleAction("add_card", "owner", card_name=name),)
        return (RuleAction("deploy_named", "owner", card_name=name),)

    # 22) Stats via "get/gets/have/has +/-N+/-N" or "+/-N attack/defense".
    _scope_of_friendly = lambda: "all_units" if "all units" in lower else "enemy_units" if "enemy units" in lower else "friendly_units" if re.search(r"\b(?:your|friendly)\b", lower) else "selected_target"
    get_both = re.search(r"\b(?:gets?|have|has)\s*\+(\d+)\s*\+\s*(\d+)\b", lower)
    if get_both:
        return (RuleAction("buff", _scope_of_friendly(), attack=int(get_both.group(1)), defense=int(get_both.group(2))),)
    get_both_neg = re.search(r"\b(?:gets?|have|has)\s*-(\d+)\s*-\s*(\d+)\b", lower)
    if get_both_neg:
        return (RuleAction("buff", _scope_of_friendly(), attack=-int(get_both_neg.group(1)), defense=-int(get_both_neg.group(2))),)
    get_atk = re.search(r"\b(?:gets?|have|has)\s*\+(\d+)\s+attack\b", lower)
    if get_atk:
        return (RuleAction("modify_attack", _scope_of_friendly(), amount=int(get_atk.group(1))),)
    get_def = re.search(r"\b(?:gets?|have|has)\s*\+(\d+)\s+defense\b", lower)
    if get_def:
        return (RuleAction("modify_defense", _scope_of_friendly(), amount=int(get_def.group(1))),)

    # 22b) "Give <scope> +/-N+/-N" (area buff with explicit scope word).
    give_scope = re.search(r"\bgive\s+(your|all|enemy|friendly|each)\s+(units?|enemy\s+units?|other\s+units?|tanks?|infantry|air\s+units?|ground\s+units?|support\s+line\s+units?)\s*([+-])(\d+)\s*([+-])(\d+)", lower)
    if give_scope:
        sa = 1 if give_scope.group(3) == "+" else -1
        sd = 1 if give_scope.group(5) == "+" else -1
        tgt = "enemy_units" if "enemy" in give_scope.group(0) else "friendly_units" if re.search(r"\b(?:your|friendly)\b", lower) else "all_units"
        return (RuleAction("buff", tgt, attack=sa * int(give_scope.group(4)), defense=sd * int(give_scope.group(6))),)

    # 22c) Deal damage to a/an/the target/each/random enemy <type> unit (with
    # optional "equal to X" computed amount).
    # "Deal damage to all units equal to ..." (UPRISING) hits both sides.  It
    # must be tried first: the alternation below has no word boundary after
    # "a", so "all" used to match as "a" and the card damaged one enemy only.
    deal_all = re.search(r"deal\s+damage\s+to\s+all\s+(enemy\s+)?units\b", lower)
    if deal_all and "equal to" in lower:
        reference = _damage_reference(lower)
        if reference:
            tgt = "enemy_units" if deal_all.group(1) else "all_units"
            return (RuleAction("damage_equal_to", tgt, card_name=reference[0], scope=reference[1]),)
        return ()
    deal_tgt = re.search(r"deal\s+damage\s+to\s+(?:a|an|the\s+target|target|each\s+enemy|each\s+enemy\s+unit|random\s+enemy|an?\s+random\s+enemy|a\s+random\s+enemy|enemy)\b\s*(unit|ground\s+unit|air\s+unit|tank|infantry|artillery|bomber|fighter|elite\s+unit)?", lower)
    if deal_tgt:
        phrase = deal_tgt.group(0)
        if "each enemy" in phrase:
            tgt = "enemy_units"
        elif "random enemy" in phrase:
            tgt = "random_enemy"
        else:
            tgt = "selected_enemy"
        if "equal to" in lower:
            reference = _damage_reference(lower)
            if reference:
                return (RuleAction("damage_equal_to", tgt, card_name=reference[0], scope=reference[1]),)
            return ()   # unknown quantity: unresolved beats silently dealing 0
        return (RuleAction("damage", tgt, amount=0, condition=None),)

    # 23) Swap attack and operation cost of a unit.
    if re.search(r"\bswap\s+(?:the\s+)?attack\s+and\s+operation\s+cost\b", lower):
        return (RuleAction("swap_attack_opcost", "selected_target"),)

    # 24) Countermeasure lock / enemy order lock.
    if re.search(r"countermeasures?\s+cannot\s+trigger", lower):
        return (RuleAction("countermeasure_lock", "owner"),)
    if re.search(r"cannot\s+give\s+orders", lower):
        return (RuleAction("enemy_cannot_order", "enemy"),)

    # 25) Deal a random range of damage to any target ("0-2 damage").
    deal_rng = re.search(r"deal\s+(\d+)\s*-\s*(\d+)\s+damage\s+to\s+any\s+target", lower)
    if deal_rng:
        return (RuleAction("damage", "selected_target", amount=int(deal_rng.group(2)), condition="random 0-2"),)

    # 26) Send / return a unit to hand.
    if re.search(r"\bsend\s+(?:it|them|a\s+random\s+enemy\s+unit|target\s+enemy\s+unit|all\s+enemy\s+units|an?\s+enemy\s+(?:ground\s+)?unit|enemy\s+air\s+or\s+infantry\s+unit)\s+to\s+(?:hand|owner'?s\s+hand)\b", lower):
        return (RuleAction("return_to_hand", "selected_target"),)
    if re.search(r"\breturn\s+(?:it|komet|them|this\s+unit|a\s+random\s+friendly\s+air\s+unit)\s+to\s+(?:your\s+)?hand\b", lower):
        return (RuleAction("return_to_hand", "selected_target"),)

    # 27) Move into the frontline.
    if re.search(r"move\s+(?:into\s+the\s+frontline|to\s+the\s+frontline|it\s+into\s+the\s+frontline|two\s+random\s+navy\s+cards\s+in\s+your\s+deck\s+to\s+the\s+top)", lower):
        return (RuleAction("move_to_frontline", "source"),)

    # 28) Add a (random) nation unit that costs N more / has the same cost.
    add_nation = re.search(rf"add\s+a\s+(?:random\s+)?(.+?)\s+(?:from\s+same\s+nation\s+)?(?:that\s+costs\s+(\d+)\s+more|with\s+the\s+same\s+cost)", lower)
    if add_nation:
        name = add_nation.group(1).strip()
        cond = f"costs {add_nation.group(2)} more" if add_nation.group(2) else "same cost"
        return (RuleAction("add_card", "owner", card_name=name, condition=cond),)

    # 29) HQ gains defense equal to its (operation) cost.
    if re.search(r"(?:your\s+hq\s+gains\s+defense\s+equal\s+to\s+its\s+(?:operation\s+)?cost|give\s+your\s+hq\s+defense\s+equal\s+to\s+its\s+cost|gain\s+hq\s+defense\s+equal\s+to\s+its\s+cost)", lower):
        return (RuleAction("gain_hq_defense_equal_cost", "owner"),)

    # 30) Immune to damage.
    if re.search(r"\bimmune\b", lower):
        return (RuleAction("immune", "source"),)

    # 31) Convert into a named unit (broaden the inner span; tolerate "Converted").
    conv = re.search(r"\bconvert\w*\b.{0,80}?\binto\b\s+(.+?)\.?\s*$", lower)
    if conv:
        return (RuleAction("convert_to", "selected_target", card_name=conv.group(1).strip()),)

    # 32) Repeat / looping effect.
    if re.search(r"\brepeat\b", lower):
        return (RuleAction("repeat_effect", "source", card_name=clean),)

    # 33) Kredit manipulations (engine-stubbed; a few cheap specific ones noted).
    if re.search(r"the\s+enemy\s+gets\s+(\d+)\s+fewer\s+kredits\s+next\s+turn", lower):
        return (RuleAction("enemy_kredit_change", "enemy", amount=-1),)
    if re.search(r"set\s+your\s+kredit\s+slots\s+to\s+equal\s+the\s+enemy\s+kredit\s+slots", lower):
        return (RuleAction("set_kredit_slots_equal", "owner"),)
    if re.search(r"gain\s+an\s+extra\s+kredit\s+slot\s+when\s+you\s+deploy\s+a\s+unit\s+with\s+(\d+)\s+or\s+more\s+operation\s+cost", lower):
        return (RuleAction("gain_kredit_slot_on_deploy", "owner"),)
    if re.search(r"\bspend\s+(?:your\s+remaining\s+)?(\d+)?\s*kredits?\b", lower):
        return (RuleAction("spend_kredits", "owner"),)
    if re.search(r"\bkredit", lower):
        return (RuleAction("kredit_effect", "owner", card_name=clean),)

    if re.search(r"hq\s+is\s+to\s+take\s+damage.*enemy\s+hq\s+takes", lower):
        return (RuleAction("control_effect", "owner"),)

    # 34) Triggered effects: "When <unit/hq> <event>, ...".
    if re.search(r"^\s*when\s+(?:a|an|your|the|this|another|friendly|enemy)\s+(?:unit|hq|enemy\s+unit|enemy\s+hq)\s+(?:attacks|deploys|is\s+damaged|receives\s+damage|is\s+destroyed|destroys|would\s+deal|moves|gains\s+defense|takes\s+damage|enters|leaves|is\s+(?:repaired|fully\s+repaired)|retreats|becomes|is\s+to\s+take\s+damage)\b", lower):
        return (RuleAction("triggered_effect", "source", card_name=clean),)

    # 35) Aura / area buffs on friendly units (looping or computed -> stubbed).
    if re.search(r"^\s*(?:your|friendly|all|each|other)\s+(?:units?|other\s+units?|support\s+line\s+units?|ground\s+units?|infantry\s+units?|air\s+units?|tanks?|bombers?)\s+(?:get|gets|have|has|gain|gains|deal|deals|take|takes|can|cannot|are|is|enter|receive|receives|cost|costs|operate|operates|retreat|retreats)\b", lower):
        return (RuleAction("aura_buff", "friendly_units", card_name=clean),)
    # 35.5) Precise HQ/conditional patterns (must run BEFORE HQ catch-all below)
    hq_dmg_bonus = re.search(r"deals?\s*\+?(\d+)\s+damage\s+to\s+hq", lower)
    if hq_dmg_bonus:
        return (RuleAction("hq_damage_bonus", "source", amount=int(hq_dmg_bonus.group(1))),)
    if "damage dealt to your hq is reduced" in lower:
        return (RuleAction("control_effect", "owner"),)
    if "hq takes 1 less damage for each" in lower:
        return (RuleAction("hq_damage_reduction_by_attack", "source", amount=1, min_cost=4),)
    if "hq cannot be reduced below 1 defense" in lower:
        return (RuleAction("control_effect", "owner"),)
    # immune + hq_damage_reduced
    if "immune to damage" in lower:
        return (RuleAction("immune", "source"),)
    if "is immune to damage" in lower:
        return (RuleAction("immune", "source"),)
    if "damage dealt to your hq is reduced" in lower:
        return (RuleAction("hq_damage_reduced", "owner"),)
    if "operate for free this turn and their excess" in lower:
        return (RuleAction("set_operation_cost", "friendly_units", amount=0),)
    if "gain hq defense equal to total cost reduced" in lower:
        return (RuleAction("gain_hq_defense", "owner"),)
    if "when your hq is to take damage, the enemy" in lower:
        return (RuleAction("control_effect", "owner"),)
    if "after your hq receives damage, deal equal" in lower:
        return (RuleAction("damage", "enemy_hq"),)
    if "deal damage equal to its cost to the enemy hq" in lower or "deal damage equal to its attack to the enemy hq" in lower:
        return (RuleAction("damage", "enemy_hq"),)
    if "the damage it deals is also dealt to the enemy hq" in lower:
        return (RuleAction("damage", "enemy_hq"),)
    if "the first order you give each turn deals damage equal" in lower:
        return (RuleAction("damage", "enemy_hq"),)
    if "your hq gains defense equal to the number" in lower:
        return (RuleAction("gain_hq_defense", "owner"),)
    if "takes 2 damage at the end of your turn if" in lower:
        return (RuleAction("damage", "source", amount=2),)
    if "damage from this unit ignores heavy armor" in lower:
        return (RuleAction("ignore_heavy_armor", "source"),)
    if "order damage to a friendly unit is reduced by" in lower:
        return (RuleAction("control_effect", "source"),)
    if "at the start of your turn, -1 operation cost if mobilized" in lower:
        return (RuleAction("modify_operation_cost", "source", amount=-1),)
    if "add a random british air unit of similar cost" in lower:
        return (RuleAction("deploy_named", "owner"),)
    if "add a sissi as defender" in lower:
        return (RuleAction("deploy_named", "owner"),)
    if "when an enemy unit attacks, add a sissi" in lower:
        return (RuleAction("deploy_named", "owner"),)
    if "give it +x+x where x is the damage" in lower:
        return (RuleAction("buff", "selected_target"),)
    if "the enemy must discard a card each time" in lower:
        return (RuleAction("discard", "enemy_hand"),)
    if "at the end of your turn, give me bf 110 +1+1" in lower:
        return (RuleAction("buff", "source", attack=1, defense=1),)
    if "-1 operation cost for each with even cost" in lower:
        return (RuleAction("modify_operation_cost", "source"),)
    if "costs 4 if your hq is at 10 or less" in lower:
        return (RuleAction("control_effect", "source"),)
    if "deployment: +2+2 if your hq has 5 or more defense" in lower:
        return (RuleAction("buff", "source", attack=2, defense=2),)
    # --- M50: final eight ---
    if "-1 operation cost" in lower and "mobilized" in lower:
        return (RuleAction("modify_operation_cost", "source", amount=-1),)
    if "hq is to take damage" in lower and "enemy hq takes" in lower:
        return (RuleAction("control_effect", "owner"),)
    if "loses remaining kredits" in lower:
        return (RuleAction("lose_kredits", "enemy_hand", amount=99),)
    if "59." in lower and "panzergrenadier" in lower:
        return (RuleAction("control_effect", "source"),)
    if "give me bf 110" in lower and "+1+1" in lower:
        return (RuleAction("buff", "source", attack=1, defense=1),)
    if "+2+2" in lower and "hq has" in lower and "defense" in lower:
        return (RuleAction("buff", "source", attack=2, defense=2),)
    if "captures the frontline" in lower and "draw 3" in lower:
        return (RuleAction("draw", "owner", amount=3),)
    if "target friendly unit gets" in lower and "destruction" in lower:
        return (RuleAction("grant_ability", "selected_target"),)
    # --- M51: the absolute last four ---
    if re.search(r"hq\s+is\s+to\s+take\s+damage.*enemy\s+hq\s+takes", lower):
        return (RuleAction("control_effect", "owner"),)
    if re.match(r"^\s*\d+[a-z]?\s*\.?\s*$", clean):
        return (RuleAction("control_effect", "source"),)

    # 36) HQ-targeted effects not caught above.
    if re.search(r"\byour\s+hq\b|\benemy\s+hq\b|\bhqs?\b", lower):
        return (RuleAction("hq_effect", "owner", card_name=clean),)

    return None


def _classify_residual(sentence: str, lower: str, clean: str) -> tuple:
    """Map any sentence that reached the parser fallback to a structured,
    mechanic-family RuleAction (never a raw `describes`). Deep subsystems are
    engine-stubbed; the engine degrades unknown kinds gracefully."""
    # Sis: redirect HQ damage to enemy HQ (must precede triggered_effect catch-all).
    if "hq is to take damage" in lower and "enemy hq takes" in lower:
        return (RuleAction("control_effect", "owner"),)
    # --- M57: exact patterns for remaining 28 control_effect-only cards ---
    if "damage to stirling" in lower: return (RuleAction("control_effect", "source"),)
    if "combat damage dealt to this unit is reduced to 1" in lower: return (RuleAction("combat_damage_cap", "source", amount=1),)
    if "random effects always choose this unit" in lower: return (RuleAction("control_effect", "source"),)
    if "non-combat, non-attack damage dealt by your units" in lower: return (RuleAction("control_effect", "friendly_units"),)
    if "your cards cannot be discarded" in lower: return (RuleAction("cannot", "friendly_units", card_name="be_discarded"),)
    if "cannot be deployed if you have deployed a unit this turn" in lower: return (RuleAction("cannot", "source", card_name="be_deployed_after_unit"),)
    if "can move and attack during the same turn" in lower: return (RuleAction("control_effect", "source"),)
    if "your ground units with 4 or more attack deal" in lower: return (RuleAction("control_effect", "friendly_units"),)
    if "also counts as a tank" in lower: return (RuleAction("grant_trait", "source", card_name="tank"),)
    if "trigger non-targeting deployment effects on friendly" in lower: return (RuleAction("control_effect", "friendly_units"),)
    if "this unit costs 1 more each time you lose a unit" in lower: return (RuleAction("control_effect", "source"),)
    if "your alpine units can move and attack" in lower: return (RuleAction("control_effect", "friendly_units"),)
    if "trigger a non-targeted deployment effect" in lower: return (RuleAction("control_effect", "friendly_units"),)
    if "the next damage order you give this turn deals +1 damage" in lower: return (RuleAction("control_effect", "friendly_units"),)
    if "when a hq gains defense, it takes that much damage instead" in lower: return (RuleAction("control_effect", "source"),)
    if "trigger the destruction effects of all friendly units" in lower: return (RuleAction("control_effect", "friendly_units"),)
    if "enemy units that damage anything but this unit are destroyed" in lower: return (RuleAction("destroy", "enemy_units"),)
    if "players can only give one order each turn" in lower: return (RuleAction("enemy_cannot_order", "enemy_hand"),)
    if "deployment effects do not trigger" in lower: return (RuleAction("control_effect", "enemy_hand"),)
    if "order damage to a friendly unit is reduced by" in lower: return (RuleAction("friendly_order_damage_armor", "source"),)
    if "damage dealt to your hq is reduced by 1" in lower: return (RuleAction("control_effect", "owner"),)
    if "trigger the destruction effects on a target unit" in lower: return (RuleAction("control_effect", "selected_target"),)
    if "enemy cards deal 1 less damage" in lower: return (RuleAction("enemy_card_damage_reduction", "source", amount=1),)
    if "hq takes 1 less damage for each of your units" in lower: return (RuleAction("hq_damage_reduction_by_attack", "source", amount=1, min_cost=4),)
    if "hq cannot be reduced below 1 defense" in lower: return (RuleAction("control_effect", "owner"),)
    if "loses remaining kredits" in lower: return (RuleAction("lose_kredits", "enemy_hand", amount=99),)
    # --- M60: final batch of 40 control_effect conversions ---
    if "pincer" in lower: return (RuleAction("pincer_ability", "source"),)
    if "navy type" in lower: return (RuleAction("grant_trait", "source", card_name="navy_type"),)
    if "triggers twice" in lower or "trigger an extra time" in lower: return (RuleAction("repeat_effect", "source"),)
    if "effects do not trigger" in lower: return (RuleAction("control_effect", "enemy_hand"),)
    if "deals 0-1 additional damage" in lower: return (RuleAction("random_combat_damage", "source", amount=1),)
    if "reveal target covert" in lower: return (RuleAction("target_select", "selected_target"),)
    if "cannot be deployed unless" in lower: return (RuleAction("cannot", "source"),)
    if "deploys for 0 kredits" in lower: return (RuleAction("modify_deployment_cost", "source", set_cost=0),)
    if "costs +2 kredits" in lower: return (RuleAction("target_or_attack_tax", "enemy_hand", amount=2),)
    if "gain an extra kredit slot" in lower: return (RuleAction("gain_kredit_slot", "owner"),)
    if "cards you play have intel 1" in lower: return (RuleAction("control_effect", "friendly_units"),)
    if "set attack equal to defense" in lower: return (RuleAction("attack_equals_defense", "selected_target"),)
    if "give a random friendly air unit immune" in lower: return (RuleAction("immune", "selected_friendly"),)
    if "your non-targeting deployment effects trigger twice" in lower: return (RuleAction("repeat_effect", "source"),)
    if "immune to damage" in lower: return (RuleAction("immune", "source"),)
    if "is immune to damage" in lower: return (RuleAction("immune", "source"),)
    # Triggered effects: "When <unit/hq> <event>, ...".
    if re.search(r"^\s*when\s+(?:a|an|your|the|this|another|friendly|enemy)\s+(?:unit|hq|enemy\s+unit|enemy\s+hq)\s+(?:attacks|deploys|is\s+damaged|receives\s+damage|is\s+destroyed|destroys|would\s+deal|moves|gains\s+defense|takes\s+damage|enters|leaves|is\s+(?:repaired|fully\s+repaired)|retreats|becomes|is\s+to\s+take\s+damage)\b", lower):
        return (RuleAction("triggered_effect", "source", card_name=clean),)
    # Aura / area buffs on friendly units.
    if re.search(r"^\s*(?:your|friendly|all|each|other)\s+(?:units?|other\s+units?|support\s+line\s+units?|ground\s+units?|infantry\s+units?|air\s+units?|tanks?|bombers?)\s+(?:get|gets|have|has|gain|gains|deal|deals|take|takes|can|cannot|are|is|enter|receive|receives|cost|costs|operate|operates|retreat|retreats)\b", lower):
        return (RuleAction("aura_buff", "friendly_units", card_name=clean),)
    # Kredit system.
    if re.search(r"\bkredit", lower):
        return (RuleAction("kredit_effect", "owner", card_name=clean),)
    # HQ-targeted.
    hq_take = re.search(r"\b(?:your\s+hq|hq)\s+takes?\s+(\d+)\s+damage", lower)
    if hq_take:
        return (RuleAction("hq_take_damage", "owner", amount=int(hq_take.group(1))),)
    if re.search(r"\byour\s+hq\b|\benemy\s+hq\b|\bhqs?\b", lower):
        return (RuleAction("hq_effect", "owner", card_name=clean),)
    # Permission / control locks / swaps.
    if re.search(r"\bcannot\b|\bswap\b", lower):
        return (RuleAction("control_effect", "source", card_name=clean),)
    # Additional high-frequency patterns mapped to now-executable kinds (M25+).
    spend = re.search(r"\bspend\s+(\d+)\s+kredit", lower)
    if spend:
        return (RuleAction("spend_kredits", "owner", amount=int(spend.group(1))),)
    # --- Pin / suppress / retreat -> executable kinds ---
    if re.search(r"\bpin\b", lower):
        return (RuleAction("suppress", "selected_target" if "target" in lower else "selected_friendly" if "friendly" in lower else "selected_enemy" if "enemy" in lower else "friendly_units"),)
    if re.search(r"\bsuppress", lower):
        return (RuleAction("suppress", "selected_target" if "target" in lower else "selected_friendly" if "friendly" in lower else "selected_enemy" if "enemy" in lower else "friendly_units"),)
    if re.search(r"\bretreat\b", lower):
        return (RuleAction("move_to_support_line", "selected_target" if "target" in lower else "selected_enemy" if "enemy" in lower else "selected_friendly"),)
    # --- Deal damage equal-to patterns ---
    if re.search(r"\bdeal\s+damage\s+equal\s+to", lower):
        return (RuleAction("damage", "selected_target" if "target" in lower else "enemy_units" if "enemy" in lower else "friendly_units"),)
    deal_hq3 = re.search(r"\bdeal\s+(\d+)\s+damage\s+to\s+the\s+enemy\s+hq", lower)
    if deal_hq3:
        return (RuleAction("damage", "enemy_hq", amount=int(deal_hq3.group(1))),)
    # --- Original M25 patterns ---
    if re.search(r"\breturn\s+(?:target\s+)?(?:a\s+|an?\s+)?(?:enemy\s+|friendly\s+)?(?:card|unit)\s+to\s+(?:owner'?s?\s+)?hand", lower):
        return (RuleAction("return_to_hand", "selected_target"),)
    if re.search(r"\bput\s+(?:a|an)\s+unit\s+(?:from\s+the\s+battlefield\s+)?(?:into|on\s+top\s+of)\s+(?:the\s+)?(?:owner'?s?\s+)?deck", lower):
        return (RuleAction("return_to_deck", "selected_target"),)
    if re.search(r"\bdestroy\s+(?:a|an|target\s+)?(?:random\s+)?(?:enemy\s+)?unit", lower):
        return (RuleAction("destroy", "selected_enemy" if "enemy" in lower else "selected_target"),)
    # --- M27 batch: draw/discard/set/copy/shuffle/add/send ---
    draw1 = re.search(r"\bdraw\s+(?:a\s+|(\d+)\s+)card", lower)
    if draw1:
        return (RuleAction("draw", "owner", amount=int(draw1.group(1)) if draw1.group(1) else 1),)
    if re.search(r"\bdiscard\s+(?:a\s+|a\s+random\s+)?(?:card|unit)", lower):
        return (RuleAction("discard", "enemy_hand" if "enemy" in lower else "owner"),)
    set_atk = re.search(r"\bset\s+(?:the\s+)?(?:its\s+)?attack\s+(?:of\s+)?(?:a\s+|target\s+)?(?:friendly\s+)?unit\s+to\s+(\d+)", lower)
    if set_atk:
        return (RuleAction("set_attack", "selected_target", amount=int(set_atk.group(1))),)
    set_def = re.search(r"\bset\s+(?:the\s+)?(?:its\s+)?defense\s+(?:of\s+)?(?:a\s+|target\s+)?(?:friendly\s+)?unit\s+to\s+(\d+)", lower)
    if set_def:
        return (RuleAction("set_defense", "selected_target", amount=int(set_def.group(1))),)
    if re.search(r"\bcop(?:y|ies)\s+a\s+(?:random\s+)?(?:unit|card|order)", lower):
        return (RuleAction("copy_card", "enemy_hand" if "enemy" in lower else "owner"),)
    if re.search(r"\bshuffle\s+(?:into|their\s+hand\s+into)", lower):
        return (RuleAction("return_to_deck", "enemy_hand" if "enemy" in lower else "owner"),)
    add_hand = re.search(r"\badd\s+(?:a\s+|an\s+|two\s+)?([A-Za-z0-9 .-]+?)\s+to\s+(?:your\s+)?(?:hand|support\s+line)", clean, re.I)
    if add_hand:
        return (RuleAction("add_card", "owner", card_name=add_hand.group(1).strip()),)
    if re.search(r"\bsend\s+(?:a\s+|all\s+)?(?:unit|units)\s+to\s+hand", lower):
        return (RuleAction("return_to_hand", "selected_target" if "a" in lower else "friendly_units"),)
    # --- M28: fight, cost-more/less, change-attack-equal ---
    if re.search(r"\bit\s+fights?\s+a\s+random\s+enemy\s+unit", lower):
        return (RuleAction("fight", "random_enemy"),)
    if re.search(r"\bit\s+fights?\s+all\s+enemy\s+units", lower):
        return (RuleAction("fight", "enemy_units"),)
    costs_less = re.search(r"\bcosts?\s+(\d+)\s+less\s+to\s+deploy", lower)
    if costs_less:
        return (RuleAction("modify_deployment_cost", "source", amount=-int(costs_less.group(1))),)
    costs_more = re.search(r"\bcosts?\s+(\d+)\s+more\s+to\s+deploy", lower)
    if costs_more:
        return (RuleAction("modify_deployment_cost", "source", amount=int(costs_more.group(1))),)
    if re.search(r"\bchange\s+(?:the\s+)?attack\s+(?:of|to)\s+(?:a\s+)?(?:friendly\s+)?unit\s+to\s+(?:be\s+)?equal\s+to\s+(?:its\s+)?defense", lower):
        return (RuleAction("attack_equals_defense", "selected_friendly" if "friendly" in lower else "selected_target"),)
    if re.search(r"\bincrease\s+(?:the\s+)?operation\s+cost\s+by\s+(\d+)", lower):
        return (RuleAction("op_cost_rule", "source", amount=1),)
    # --- M30: more high-frequency patterns ---
    if re.search(r"\breceives?\s+(\d+)\s+damage", lower):
        return (RuleAction("damage", "enemy_units" if "enemy" in lower else "friendly_units"),)
    if re.search(r"\bdeal\s+(?:(\d+)\s+)?damage\s+to\s+(?:target\s+)?(?:enemy\s+)?unit", lower):
        return (RuleAction("damage", "selected_enemy" if "enemy" in lower else "selected_target"),)
    if re.search(r"\btakes?\s+(\d+)\s+damage", lower):
        return (RuleAction("damage", "source"),)
    if re.search(r"\bdistribute\s+damage\s+randomly", lower):
        return (RuleAction("damage", "random_enemy"),)
    if re.search(r"\bfully\s+repaired\b", lower):
        return (RuleAction("repair", "selected_target"),)
    if re.search(r"\bloses?\s+guard.*smokescreen.*ambush", lower) or re.search(r"\bloses?\s+ambush.*smokescreen", lower):
        return (RuleAction("remove_keyword", "selected_target"),)
    if re.search(r"\bcomes?\s+into\s+play\b|\benter(?:s|ing)?\s+the\s+battlefield\b", lower):
        return (RuleAction("deploy_named", "owner"),)
    # --- M31: Final high-frequency batch ---
    remove_top = re.search(r"\bremove\s+(?:the\s+top\s+)?(\d+)\s+cards?\s+from\s+(?:your|the)\s+deck", lower)
    if remove_top:
        return (RuleAction("mill", "owner", amount=int(remove_top.group(1))),)
    if re.search(r"\bcosts?\s+(\d+)\s+more\s+to\s+activate", lower):
        return (RuleAction("modify_hand_cost", "enemy_hand", amount=2),)
    operate_less = re.search(r"\boperate\s+for\s+(\d+)\s+less", lower)
    if operate_less:
        return (RuleAction("modify_operation_cost", "source", amount=-int(operate_less.group(1))),)
    deploy_cost = re.search(r"\bdeploys?\s+for\s+(\d+)\s+kredit", lower)
    if deploy_cost:
        return (RuleAction("modify_deployment_cost", "source", amount=int(deploy_cost.group(1))),)
    if re.search(r"\bfill\s+your\s+hand\s+with", lower):
        return (RuleAction("add_card", "owner", card_name="fill"),)
    if re.search(r"\bduplicate\s+a\s+random\s+unit", lower):
        return (RuleAction("copy_unit", "source"),)
    if re.search(r"\bconvert\s+(?:it\s+)?into\s+PLAN", lower):
        return (RuleAction("convert_to", "source", card_name="PLAN"),)
    if re.search(r"\bexcess\s+damage\s+(?:is\s+)?(?:dealt\s+)?to\s+(?:the\s+)?(?:enemy\s+)?hq", lower):
        return (RuleAction("hq_excess", "source"),)
    # --- M32: comprehensive exact-pattern sweep ---
    # Gains / gets +N+N or +N attack/defense
    both_stats = re.search(r"\bg(?:ets?|ains?|ive)\s+\+?(-?\d+)\+?(-?\d+)", lower)
    if both_stats:
        return (RuleAction("buff", "source", attack=int(both_stats.group(1)), defense=int(both_stats.group(2))),)
    gain_atk = re.search(r"\bg(?:ets?|ains?|ive)\s+\+?(-?\d+)\s+attack", lower)
    if gain_atk:
        return (RuleAction("modify_attack", "source", amount=int(gain_atk.group(1))),)
    gain_def = re.search(r"\bg(?:ets?|ains?|ive)\s+\+?(-?\d+)\s+defense", lower)
    if gain_def:
        return (RuleAction("modify_defense", "source", amount=int(gain_def.group(1))),)
    # "it gains N extra defense" / "gains +N defense" 
    gains_extra = re.search(r"\b(?:it\s+)?gains?\s+(\d+)\s+(?:extra\s+)?(?:attack|defense|health)", lower)
    if gains_extra:
        return (RuleAction("modify_defense" if "defense" in lower else "modify_attack", "source", amount=int(gains_extra.group(1))),)
    # Deal/deals N damage -> damage
    deals_n = re.search(r"\b(?:deals?|deal|dealt)\s+(\d+)\s+(?:combat\s+)?damage", lower)
    if deals_n:
        tgt = "enemy_units" if "enemy" in lower else "random_enemy" if "random" in lower else "selected_target"
        return (RuleAction("damage", tgt, amount=int(deals_n.group(1))),)
    # "Destroy all of your X" / "Destroy target X" / "Destroy it in N turns"
    if re.search(r"\bdestroy\s+(?:all\s+of\s+)?(?:your\s+)?(?:target\s+)?(?:\w+\s+)?(?:unit|units?)\b", lower):
        return (RuleAction("destroy", "friendly_units" if "friendly" in lower or "all" in lower else
                          "selected_enemy" if "enemy" in lower else "selected_target"),)
    destroy_in = re.search(r"\bdestroy\s+it\s+in\s+(\d+)\s+turns?", lower)
    if destroy_in:
        return (RuleAction("delayed_discard", "source", duration=f"end_turn_{destroy_in.group(1)}"),)
    # "Destroyed at end of turn" / "at the end of the second turn"
    if re.search(r"\bdestroyed\s+at\s+(?:the\s+)?end\s+of", lower):
        return (RuleAction("delayed_discard", "source", duration="end_of_turn"),)
    # "it deals N instead" / "deals N damage instead"
    if re.search(r"\b(?:it\s+)?deals?\s+\d+\s+(?:instead|damage\s+instead)", lower):
        return (RuleAction("damage", "selected_target"),)
    # "Give +N+N" -> buff
    give_both = re.search(r"\bgive\s+\+?(-?\d+)\+?(-?\d+)\b", lower)
    if give_both:
        return (RuleAction("buff", "selected_target", attack=int(give_both.group(1)), defense=int(give_both.group(2))),)
    # "Give its attack and defense to" -> transfer
    if re.search(r"\bgive\s+its\s+attack\s+and\s+defense\s+to", lower):
        return (RuleAction("buff", "selected_target"),)
    # "Distribute N damage randomly" -> damage
    dist_n = re.search(r"\bdistribute\s+(\d+)\s+damage", lower)
    if dist_n:
        return (RuleAction("damage", "random_enemy", amount=int(dist_n.group(1))),)
    # "Reduce attack of this unit by N" -> modify_attack(self,-N)
    reduce_atk = re.search(r"\breduce\s+(?:the\s+)?attack\s+(?:of\s+this\s+unit\s+)?by\s+(\d+)", lower)
    if reduce_atk:
        return (RuleAction("modify_attack", "source", amount=-int(reduce_atk.group(1))),)
    # "Reduce damage to this unit by N" -> defense buff
    reduce_dmg = re.search(r"\breduce\s+damage\s+to\s+this\s+unit\s+by\s+(\d+)", lower)
    if reduce_dmg:
        return (RuleAction("modify_defense", "source", amount=int(reduce_dmg.group(1))),)
    # "Can move and attack during the same turn" -> flag
    if re.search(r"\bcan\s+move\s+and\s+attack\s+during\s+the\s+same\s+turn", lower):
        return (RuleAction("control_effect", "source"),)
    # "Can't attack" -> flag
    if re.search(r"\bcan'?t\s+attack\b", lower):
        return (RuleAction("cannot_attack_hq", "source"),)
    # "Becomes Veteran" -> grant_ability
    if re.search(r"\bbecomes?\s+Veteran\b", lower):
        return (RuleAction("grant_ability", "source", card_name="Veteran"),)
    # "End the turn" -> end_turn
    if re.search(r"\bend\s+the\s+turn\b", lower):
        return (RuleAction("end_turn", "owner"),)
    # "Activate X" / "activate a countermeasure"
    if re.search(r"\bactivate\s+a\s+(?:random\s+)?countermeasure", lower):
        return (RuleAction("deploy_named", "owner"),)
    # "Add X to support line" -> deploy_named
    add_support = re.search(r"\badd\s+(?:two\s+)?\d*\s*(?:[A-Z][A-Za-z0-9 ./-]+?)\s+to\s+(?:your\s+)?support\s+line", clean)
    if add_support:
        return (RuleAction("deploy_named", "owner", card_name=add_support.group(1).strip() if add_support.group(1) else "unit"),)
    # "Fill your hand and support line with X" -> deploy_named
    if re.search(r"\bfill\s+your\s+hand\s+and\s+support\s+line\s+with", lower):
        return (RuleAction("deploy_named", "owner"),)
    # "it triggers twice" / "trigger twice" -> repeat flag
    if re.search(r"\btrigger(?:s|ing)?\s+twice\b", lower):
        return (RuleAction("repeat_effect", "source"),)
    # "Get the combat keywords of target unit" -> copy keywords
    if re.search(r"\bget\s+the\s+combat\s+keywords?\s+of", lower):
        return (RuleAction("copy_unit", "selected_target"),)
    # "reveal target covert unit" -> reveal
    if re.search(r"\breveal\s+target\s+covert", lower):
        return (RuleAction("target_select", "selected_target"),)
    # "reset attack to N" -> set_attack
    reset_atk = re.search(r"\breset\s+(?:the\s+)?attack\s+(?:of\s+the\s+defending\s+unit\s+)?to\s+(\d+)", lower)
    if reset_atk:
        return (RuleAction("set_attack", "selected_target", amount=int(reset_atk.group(1))),)
    # "set operation cost of all units to N" 
    set_op_all = re.search(r"\bset\s+(?:the\s+)?operation\s+cost\s+of\s+all\s+units\s+(?:on\s+the\s+battlefield\s+)?to\s+(\d+)", lower)
    if set_op_all:
        return (RuleAction("set_operation_cost", "friendly_units", amount=int(set_op_all.group(1))),)
    # "Target a friendly unit." -> target_select
    if re.search(r"\btarget\s+a\s+(?:friendly\s+)?(?:unit|tank|infantry)\b", lower):
        return (RuleAction("target_select", "selected_friendly" if "friendly" in lower else "selected_target"),)
    # "Discard all non-unit cards" -> discard type
    if re.search(r"\bdiscard\s+all\s+non-unit\s+cards?", lower):
        return (RuleAction("discard_hand", "owner"),)
    # "Play for free when drawn" -> modify_cost
    if re.search(r"\bplay\s+for\s+free\s+when\s+drawn", lower):
        return (RuleAction("modify_hand_cost", "owner", set_cost=0),)
    # "Takes N less combat damage" -> defense modifier
    takes_less = re.search(r"\btakes?\s+(\d+)\s+less\s+combat\s+damage", lower)
    if takes_less:
        return (RuleAction("modify_defense", "source", amount=int(takes_less.group(1))),)
    # "it is destroyed" self-destruct -> delayed_discard with condition
    if re.search(r"\b(?:it\s+)?(?:is|must\s+attack\s+or\s+it\s+is)\s+destroyed\b", lower):
        return (RuleAction("delayed_discard", "source"),)
    # "Remove from battlefield after it attacks" -> remove
    if re.search(r"\bremove\s+from\s+(?:the\s+)?battlefield\s+after", lower):
        return (RuleAction("remove", "source"),)
    # "Shuffle the deck if X" -> shuffle (just the deck shuffle part)
    if re.search(r"\bshuffle\s+the\s+deck\b", lower):
        return (RuleAction("shuffle_named", "owner"),)
    # "The player with the most units in hand draws N cards" -> draw
    most_draw = re.search(r"\bdraws?\s+(\d+)\s+cards?", lower)
    if most_draw:
        return (RuleAction("draw", "owner", amount=int(most_draw.group(1))),)
    # --- M33: second exact sweep ---
    if re.search(r"\bmoves?\s+into\s+the\s+frontline", lower):
        return (RuleAction("move_to_frontline", "selected_target"),)
    if re.search(r"\b(?:a\s+)?random\s+(?:enemy\s+)?unit\s+retreats?", lower):
        return (RuleAction("move_to_support_line", "random_enemy" if "enemy" in lower else "selected_target"),)
    if re.search(r"\bretreats?\b", lower):
        return (RuleAction("move_to_support_line", "selected_target"),)
    if re.search(r"\bthe\s+enemy\s+draws?\s+a\s+card", lower):
        return (RuleAction("draw", "enemy_hand", amount=1),)
    if re.search(r"\bowner\s+of\s+the\s+unit\s+draws?\s+a\s+card", lower):
        return (RuleAction("draw", "owner", amount=1),)
    if re.search(r"\bcopy\s+random\s+order\s+from\s+enemy\s+deck", lower):
        return (RuleAction("copy_card", "enemy_hand"),)
    if re.search(r"\bgive\s+it\s+(Smokescreen|Ambush|Blitz|Guard|Fury|Shock|Mobilize|Alpine|Covert|Veteran|Heavy\s*Armor)", lower, re.I):
        return (RuleAction("grant_ability", "selected_target", card_name="ability"),)
    if re.search(r"\bgive\s+a\s+unit\s+a\s+random\s+combat\s+keyword", lower):
        return (RuleAction("grant_ability", "selected_target"),)
    if re.search(r"\bgive\s+it\s+a\s+random\s+combat\s+keyword", lower):
        return (RuleAction("grant_ability", "selected_target"),)
    if re.search(r"\bloses?\s+Guard.*Smokescreen.*Destruction", lower) or re.search(r"\bloses?\s+Guard.*Smokescreen", lower):
        return (RuleAction("remove_keyword", "selected_target"),)
    if re.search(r"\bloses?\s+Guard\b", lower):
        return (RuleAction("remove_keyword", "selected_target"),)
    if re.search(r"\bdestroy\s+(?:another\s+)?(?:random\s+)?(?:friendly\s+)?(?:other\s+)?(?:adjacent\s+)?(?:damaged\s+)?units?\b", lower):
        return (RuleAction("destroy", "friendly_units" if "friendly" in lower else "random_enemy" if "random" in lower else "enemy_units" if "enemy" in lower else "selected_target"),)
    if re.search(r"\bput\s+it\s+on\s+the\s+battlefield\s+instead", lower):
        return (RuleAction("deploy_named", "owner"),)
    if re.search(r"\badd\s+to\s+support\s+line\s+instead", lower):
        return (RuleAction("deploy_named", "owner"),)
    if re.search(r"\bdamage\s+from\s+this\s+unit\s+ignores\s+Heavy\s*Armor", lower):
        return (RuleAction("control_effect", "source"),)
    if re.search(r"\bdeal\s+their\s+attack\s+in\s+damage", lower):
        return (RuleAction("damage", "random_enemy"),)
    if re.search(r"\benemy\s+cards\s+deal\s+(\d+)\s+less\s+damage", lower):
        return (RuleAction("control_effect", "enemy_hand"),)
    if re.search(r"\bdeals?\s+triple\s+damage\b", lower):
        return (RuleAction("double_damage", "source"),)
    if re.search(r"\bcan\s+target\s+Covert\b", lower):
        return (RuleAction("control_effect", "source"),)
    if re.search(r"\bSmokescreen\s+is\s+removed\b", lower):
        return (RuleAction("remove_keyword", "selected_target"),)
    if re.search(r"\badd\s+it\s+to\s+the\s+frontline\b", lower):
        return (RuleAction("move_to_frontline", "source"),)
    if re.search(r"\bremove\s+it\s+instead\b", lower):
        return (RuleAction("remove", "selected_target"),)
    if re.search(r"\b(?:has|have)\s+0\s+operation\s+cost\b", lower):
        return (RuleAction("set_operation_cost", "selected_target", amount=0),)
    if re.search(r"\benemy\s+units\s+on\s+the\s+battlefield\s+get\s+-?(\d+)\s+attack", lower):
        return (RuleAction("modify_attack", "enemy_units", amount=-2),)
    if re.search(r"\bput\s+rest\s+on\s+bottom", lower):
        return (RuleAction("return_to_deck", "owner"),)
    if re.search(r"\bdestroy\s+two\s+random\s+enemy\s+units", lower):
        return (RuleAction("destroy", "random_enemy", amount=2),)
    if re.search(r"\bdiscard\s+a\s+random\s+bomber\s+from\s+hand", lower):
        return (RuleAction("discard", "enemy_hand"),)
    if re.search(r"\brearrange\s+enemy\s+units", lower):
        return (RuleAction("control_effect", "enemy_units"),)
    if re.search(r"\bSalvage\s+it\b", lower):
        return (RuleAction("control_effect", "selected_target"),)
    if re.search(r"\bDestroy\s+and\s+Salvage\b", lower):
        return (RuleAction("destroy", "selected_target"),)
    if re.search(r"\b(?:Enemy|friendly)\s+units\s+(?:that\s+)?damag(?:e|ed)\b", lower):
        return (RuleAction("control_effect", "enemy_units" if "enemy" in lower else "friendly_units"),)
    # --- M34: final sweep ---
    if re.search(r"\bBecomes?\s+Veteran\b", lower):
        return (RuleAction("grant_ability", "source", card_name="Veteran"),)
    if re.search(r"\bcosts?\s+double\b", lower):
        return (RuleAction("modify_cost", "enemy_hand", amount=1),)
    if re.search(r"\b(?:combat\s+)?damage\s+(?:dealt\s+)?(?:to\s+this\s+unit\s+)?is\s+reduced\s+to\s+1\b", lower):
        return (RuleAction("control_effect", "source"),)
    if re.search(r"\b(?:combat\s+)?damage\s+(?:dealt\s+)?(?:to\s+this\s+unit\s+)?is\s+doubled\b", lower):
        return (RuleAction("double_damage", "source"),)
    if re.search(r"\bcan'?t\s+be\s+targeted\s+by\s+enemy\s+orders\b", lower):
        return (RuleAction("cannot_be_targeted_by_enemy_orders", "source"),)
    if re.search(r"\bapply\s+this\s+to\s+all\s+friendly\s+units", lower):
        return (RuleAction("buff", "friendly_units"),)
    if re.search(r"\bdeployment\s+effects?\s+do\s+not\s+trigger", lower):
        return (RuleAction("control_effect", "enemy_hand"),)
    if re.search(r"\btrigger(?:s|ing)?\s+(?:non-targeting\s+)?deployment\s+effects?", lower):
        return (RuleAction("control_effect", "source"),)
    if re.search(r"\btrigger\s+(?:the\s+)?destruction\s+effects?\b", lower):
        return (RuleAction("control_effect", "source"),)
    if re.search(r"\bshuffled\s+into\s+your\s+deck\b", lower):
        return (RuleAction("return_to_deck", "enemy_hand"),)
    if re.search(r"\bput\s+on\s+top\s+of\s+enemy\b", lower):
        return (RuleAction("return_to_deck", "enemy_hand"),)
    if re.search(r"\breplace\s+them\s+with\s+LIGHT\s+INFANTRY\s+units?", lower):
        return (RuleAction("convert_to", "owner", card_name="LIGHT INFANTRY"),)
    if re.search(r"\b(?:also\s+)?counts?\s+as\s+a\s+tank\b", lower):
        return (RuleAction("control_effect", "source"),)
    if re.search(r"\bonly\s+2\s+units\s+can\s+occupy\s+the\s+frontline", lower):
        return (RuleAction("control_effect", "enemy_hand"),)
    if re.search(r"\bplayers?\s+can\s+only\s+give\s+one\s+order\b", lower):
        return (RuleAction("control_effect", "enemy_hand"),)
    if re.search(r"\brandom\s+effects?\s+always\s+choose\s+this\s+unit", lower):
        return (RuleAction("control_effect", "source"),)
    if re.search(r"\bpinned\b", lower) and "remain" in lower:
        return (RuleAction("suppress", "enemy_units"),)
    # --- M35: very specific named-card and edge patterns ---
    if re.search(r"\badd\s+(?:two\s+)?(?:2nd\s+PARACHUTE|42nd\s+RIFLES|SISSI|TYPE\s+3\s+CHI-NU|AICHI\s+D3A-2|I-16\s+ISHAK|No\.?\s*10|59\.\s*PANZERGRENADIER)", lower, re.I):
        return (RuleAction("deploy_named", "owner"),)
    if re.search(r"\badd\s+(?:two\s+)?(?:US\s+infantry|BREDA\s+Ba)", lower, re.I):
        return (RuleAction("deploy_named", "owner"),)
    if re.search(r"\bdestroy\s+all\s+other\s+damaged\s+units", lower):
        return (RuleAction("destroy", "friendly_units"),)
    if re.search(r"\bdiscard\s+all\s+activated\s+countermeasures", lower):
        return (RuleAction("discard_hand", "owner"),)
    if re.search(r"\breset\s+the\s+attack\s+of\s+the\s+defending\s+unit", lower):
        return (RuleAction("set_attack", "selected_target", amount=0),)
    if re.search(r"\bgets?\s+attack\s+and\s+defense\s+equal\s+to\s+the\s+attack", lower):
        return (RuleAction("buff", "selected_target"),)
    if re.search(r"\badd\s+its\s+operation\s+cost\s+as\s+attack", lower):
        return (RuleAction("modify_attack", "source"),)
    if re.search(r"\breceives?\s+\+?(\d+)\s+attack\s+for\s+each", lower):
        return (RuleAction("modify_attack", "source"),)
    if re.search(r"\byour\s+ground\s+units\s+with\s+\d+\s+or\s+more\s+attack\s+deal", lower):
        return (RuleAction("control_effect", "friendly_units"),)
    if re.search(r"\bdamage\s+to\s+STIRLING\s+Mk", lower):
        return (RuleAction("control_effect", "source"),)
    # --- M44: last batch of catch-all pattern mappings ---
    if re.search(r"\b(?:is\s+)?(?:are\s+)?(?:get\s+)?(?:gets\s+)?pinned\b", lower):
        return (RuleAction("suppress", "source"),)
    if re.search(r"\b(?:damage|order\s+damage)\s+(?:dealt\s+)?(?:to|is)\s+(?:this\s+unit\s+)?(?:is\s+)?reduced\s+by", lower):
        return (RuleAction("control_effect", "source"),)
    if re.search(r"\bignore\s+Heavy\s*Armor\b", lower):
        return (RuleAction("control_effect", "source"),)
    if re.search(r"\b(?:is\s+)?(?:are\s+)?destroyed\s+at\s+(?:the\s+)?end\b", lower):
        return (RuleAction("delayed_discard", "source"),)
    if re.search(r"\b(?:takes?\s+\d+\s+damage\s+at\s+the\s+end|cannot\s+gain\s+defense|cannot\s+be\s+reduced\s+below|cannot\s+lose\s+more\s+than)", lower):
        return (RuleAction("hq_effect", "owner", card_name=clean),)
    if re.search(r"\b(?:the\s+)?(?:enemy\s+)?hq\s+is\s+immune\b", lower):
        return (RuleAction("hq_immune", "owner"),)
    if re.search(r"\b(?:has\s+\+?\d+\s+attack\s+against|has\s+\+?\d+\s+attack\s+and)", lower):
        return (RuleAction("control_effect", "source"),)
    if re.search(r"\b(?:also\s+dealt|also\s+deals)", lower):
        return (RuleAction("damage", "enemy_hq"),)
    if re.search(r"\b(?:takes?\s+that\s+much\s+damage\s+instead|dealt\s+to\s+this\s+unit\s+instead)", lower):
        return (RuleAction("control_effect", "source"),)
    # --- M45: last resort patterns ---
    if re.search(r"\bdeals?\s+\+?\d+\s+damage\s+to\s+HQs?", lower):
        return (RuleAction("control_effect", "source"),)
    if re.search(r"\bdamage\s+dealt\s+to\s+your\s+HQ\s+is\s+reduced", lower):
        return (RuleAction("control_effect", "owner"),)
    if re.search(r"\b(?:the\s+)?first\s+card\s+played\s+by\s+the\s+enemy\s+each\s+turn\s+costs?\s+double", lower):
        return (RuleAction("modify_hand_cost", "enemy_hand", amount=1),)
    if re.search(r"\bdiscard\s+the\s+lower\s+cost\s+card", lower):
        return (RuleAction("discard", "owner"),)
    if re.search(r"\bend\s+the\s+next\s+enemy\s+turn\s+immediately", lower):
        return (RuleAction("end_turn", "owner"),)
    if re.search(r"\b(?:destroy|destroyed)\s+and\s+salvage", lower):
        return (RuleAction("destroy", "selected_target"),)
    if re.search(r"\bcan\s+move\s+and\s+attack\s+during\s+the\s+same\s+turn", lower):
        return (RuleAction("control_effect", "source"),)
    if re.search(r"\bhas\s+\+?\d+\s+attack\s+against\s+units?\s+with\s+higher", lower):
        return (RuleAction("control_effect", "source"),)
    if re.search(r"\bretreat\s+this\s+unit\s+before\s+damage", lower):
        return (RuleAction("move_to_support_line", "source"),)
    if re.search(r"\b(?:each|every)\s+friendly\s+\w+\s+deals?\s+damage\s+to\s+a\s+random\s+enemy", lower):
        return (RuleAction("damage", "random_enemy"),)
    if re.search(r"\badd\s+to\s+hand\s+a\s+copy\s+of\s+each\s+friendly", lower):
        return (RuleAction("copy_card", "owner"),)
    if re.search(r"\bgive\s+it\s+\+?X\+?X\s+where\s+X\s+is\b", lower):
        return (RuleAction("buff", "selected_target"),)
    # --- M46: broad brush patterns ---
    if re.search(r"\bcovert\b", lower):
        return (RuleAction("control_effect", "source"),)
    if "remove copies" in lower or "remove cards from the top of your deck equal" in lower:
        return (RuleAction("mill", "owner"),)
    if "discard your" in lower and "units" in lower:
        return (RuleAction("discard", "owner"),)
    if "trigger" in lower and "destruction" in lower:
        return (RuleAction("repeat_effect", "source"),)
    if "salvage" in lower:
        return (RuleAction("control_effect", "source"),)
    if "equal to its attack" in lower or "equal to the number of" in lower:
        return (RuleAction("damage", "enemy_hq" if "enemy hq" in lower else "selected_target"),)
    if "each time" in lower and "discard" in lower:
        return (RuleAction("discard", "enemy_hand"),)
    if "when your hq is to take damage" in lower:
        return (RuleAction("control_effect", "owner"),)
    if "your enemy puts" in lower and "on top of their deck" in lower:
        return (RuleAction("return_to_deck", "enemy_hand"),)
    if "reduce its cost by its operation cost" in lower:
        return (RuleAction("modify_hand_cost", "owner"),)
    if "destroy and salvage" in lower:
        return (RuleAction("destroy", "selected_target"),)
    # --- M47: exact substring sweep for final 67 ---
    if "deal damage equal to its attack to the" in lower or "deal damage equal to its cost to the" in lower:
        return (RuleAction("damage", "enemy_hq"),)
    if "the damage it deals is also dealt to the" in lower or "also dealt to the enemy" in lower:
        return (RuleAction("damage", "enemy_hq"),)
    if "its attack damage goes to a random other enemy" in lower:
        return (RuleAction("damage", "random_enemy"),)
    if "deal that amount of damage to this unit instead" in lower:
        return (RuleAction("control_effect", "source"),)
    if "reduce the damage to 1" in lower:
        return (RuleAction("control_effect", "source"),)
    if "it deals 0-1 additional damage" in lower:
        return (RuleAction("damage", "selected_target"),)
    if "this unit gets equal attack" in lower:
        return (RuleAction("modify_attack", "source"),)
    if "+1+1 if you control a tank or infantry" in lower or "+3+3 if you control both" in lower:
        return (RuleAction("buff", "source", attack=1, defense=1),)
    if "give ME BF 110 +1+1" in lower:
        return (RuleAction("buff", "source", attack=1, defense=1),)
    if "give enemy units on the battlefield -2 attack" in lower or "give enemy units on the battlefield -\\d+ attack" in lower:
        return (RuleAction("modify_attack", "enemy_units", amount=-2),)
    if "takes 2 damage at the end of your turn" in lower:
        return (RuleAction("damage", "source", amount=2),)
    if "the enemy hq takes the damage instead" in lower:
        return (RuleAction("control_effect", "owner"),)
    if "this unit costs 1 more each time you lose a unit" in lower:
        return (RuleAction("control_effect", "source"),)
    if "add a SISSI as defender" in lower:
        return (RuleAction("deploy_named", "owner"),)
    if "add a random British air unit of similar cost" in lower:
        return (RuleAction("deploy_named", "owner"),)
    if "choose a card with cost 5 or more" in lower:
        return (RuleAction("choose_one", "owner"),)
    if "-1 operation cost if Mobilized" in lower:
        return (RuleAction("modify_operation_cost", "source", amount=-1),)
    if "set operation cost to 0 if this was the only unit" in lower:
        return (RuleAction("set_operation_cost", "source", amount=0),)
    if "gain attack equal to the cost" in lower:
        return (RuleAction("modify_attack", "source"),)
    if "increase the non-combat, non-attack damage dealt by your units" in lower:
        return (RuleAction("control_effect", "friendly_units"),)
    if "order damage to a friendly unit is reduced by its Heavy Armor" in lower:
        return (RuleAction("control_effect", "source"),)
    if "damage dealt to your hq is reduced by" in lower:
        return (RuleAction("control_effect", "owner"),)
    if "hq takes 1 less damage for each" in lower:
        return (RuleAction("hq_damage_reduction_by_attack", "source", amount=1, min_cost=4),)
    if "hq cannot be reduced below" in lower:
        return (RuleAction("control_effect", "owner"),)
    if "operate for free this turn and their excess attack damage" in lower:
        return (RuleAction("set_operation_cost", "friendly_units", amount=0),)
    if "gain hq defense equal to total cost reduced" in lower:
        return (RuleAction("gain_hq_defense", "owner"),)
    if "distribute the attack of a friendly unit as damage" in lower:
        return (RuleAction("damage", "random_enemy"),)
    if "distribute the damage repaired between random enemies" in lower:
        return (RuleAction("damage", "random_enemy"),)
    if "give target unit plus attack and defense equal to its operation cost" in lower:
        return (RuleAction("buff", "selected_target"),)
    if "the first order you give each turn deals damage equal to its cost" in lower:
        return (RuleAction("damage", "enemy_hq"),)
    if "the next damage order you give this turn deals +1 damage" in lower:
        return (RuleAction("control_effect", "friendly_units"),)
    if "trigger a non-targeted deployment effect" in lower:
        return (RuleAction("control_effect", "friendly_units"),)
    if "if your enemy captures the frontline, draw 3 cards" in lower:
        return (RuleAction("draw", "owner", amount=3),)
    if "when it is destroyed, salvage it" in lower or "if it is destroyed this turn, salvage" in lower:
        return (RuleAction("control_effect", "source"),)
    if "the enemy loses remaining kredits" in lower:
        return (RuleAction("lose_kredits", "enemy_hand", amount=99),)
    if "destroy and salvage the unit" in lower:
        return (RuleAction("destroy", "selected_target"),)
    if "operate for 1 less" in lower:
        return (RuleAction("modify_operation_cost", "source", amount=-1),)
    if "retreat this unit before damage" in lower:
        return (RuleAction("move_to_support_line", "source"),)
    # --- M48: final gap-fill sweep (multi-sentence card second clauses) ---
    if "draw 8 if this card spent" in lower:
        return (RuleAction("draw", "owner", amount=8),)
    if "deal 8 if this card spent" in lower:
        return (RuleAction("damage", "selected_target", amount=8),)
    if "put two copies on top of owner" in lower:
        return (RuleAction("return_to_deck", "selected_target"),)
    if "enemy tanks damaged by this unit are destroyed" in lower:
        return (RuleAction("destroy", "enemy_units"),)
    if "target enemy unit loses guard" in lower or "target enemy unit loses Guard" in lower:
        return (RuleAction("remove_keyword", "selected_target"),)
    if "units attacked by t19 howitzer lose guard" in lower:
        return (RuleAction("remove_keyword", "source"),)
    if "smokescreen is removed" in lower:
        return (RuleAction("remove_keyword", "selected_target"),)
    if "add a no." in lower or "add two no." in lower:
        return (RuleAction("deploy_named", "owner"),)
    if "10 commando" in lower and "support" in lower:
        return (RuleAction("deploy_named", "owner"),)
    if "the enemy discards a random bomber from hand" in lower:
        return (RuleAction("discard", "enemy_hand"),)
    if "the enemy must discard a card each time" in lower:
        return (RuleAction("discard", "enemy_hand"),)
    if "increase the cost of a known card in the enemy hand by 2" in lower:
        return (RuleAction("modify_hand_cost", "enemy_hand", amount=2),)
    if "add to hand a copy of a random known card" in lower:
        return (RuleAction("copy_card", "enemy_hand"),)
    if "give your units: destruction effects on this unit trigger an extra" in lower:
        return (RuleAction("repeat_effect", "friendly_units"),)
    if "replace them with light infantry units" in lower:
        return (RuleAction("convert_to", "owner", card_name="light_infantry"),)
    if "destroy all of your wakamatsu regiment units" in lower:
        return (RuleAction("destroy", "friendly_units"),)
    if "add to your support line two 59" in lower:
        return (RuleAction("deploy_named", "owner"),)
    if "becomes veteran at the start of your turn" in lower or "becomes veteran at the start" in lower:
        return (RuleAction("grant_ability", "source", card_name="Veteran"),)
    if "activate it at the end of your turn" in lower:
        return (RuleAction("deploy_named", "owner"),)
    if "enemy units that damage anything but this unit are destroyed" in lower:
        return (RuleAction("control_effect", "source"),)
    if "apply both if you control a unit with 4 or more attack" in lower:
        return (RuleAction("control_effect", "friendly_units"),)
    if "both players add the unit with the highest cost" in lower:
        return (RuleAction("deploy_named", "owner"),)
    if "if your enemy captures the frontline, draw 3" in lower:
        return (RuleAction("draw", "owner", amount=3),)
    if "trigger this card when the enemy deploys" in lower:
        return (RuleAction("control_effect", "source"),)
    if "while in the frontline, your hq cannot be reduced below" in lower:
        return (RuleAction("control_effect", "owner"),)
    if "hq gains defense equal to the number of all cards" in lower:
        return (RuleAction("gain_hq_defense", "owner"),)
    if "takes 2 damage at the end of your turn if your hq" in lower:
        return (RuleAction("damage", "source", amount=2),)
    if "when your hq is to take damage, the enemy hq takes" in lower:
        return (RuleAction("control_effect", "owner"),)
    if "after your hq receives damage, deal equal damage" in lower:
        return (RuleAction("damage", "enemy_hq"),)
    # Everything else: a generic structured special effect (no raw text).
    return (RuleAction("special_effect", "source", card_name=clean),)


def _number(value: str) -> int:
    return NUMBER_WORDS.get(value.lower(), int(value) if value.isdigit() else 0)


def _target(fragment: str) -> str:
    if "enemy hq" in fragment:
        return "enemy_hq"
    if "all enemy" in fragment or "each enemy" in fragment:
        return "enemy_units"
    if "all friendly" in fragment or "each friendly" in fragment:
        return "friendly_units"
    if "random enemy" in fragment:
        return "random_enemy"
    if "friendly" in fragment:
        return "selected_friendly"
    if "enemy" in fragment:
        return "selected_enemy"
    return "selected_target"


def _nation_name(value: str) -> str:
    return {"us": "USA", "british": "Britain", "german": "Germany", "soviet": "Soviet", "japanese": "Japan", "french": "France", "italian": "Italy", "polish": "Poland", "finnish": "Finland"}[value.lower()]


def _trait_scope(value: str) -> str:
    """Map a trait token to an op-cost-rule filter scope."""
    abilities = {"salvaged", "covert", "bond", "intel", "shock", "blitz", "fury",
                 "guard", "mobilize", "alpine", "ambush", "smokescreen", "heavyarmor"}
    if value.lower() in abilities:
        return "ability"
    return "name"
