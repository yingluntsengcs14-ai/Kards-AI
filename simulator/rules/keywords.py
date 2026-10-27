"""Generic keyword predicates; no card-specific keyword logic lives in cards."""
import re

class KeywordEngine:
    @staticmethod
    def matches_type(card, unit, requested_type: str) -> bool:
        """Return whether a unit satisfies a card-type rule at runtime.

        Some effects grant an additional type (for example, "also counts as a
        tank").  Rules must consult this predicate rather than only immutable
        catalog data.
        """
        kind = requested_type.lower()
        if card.type == kind:
            return True
        return kind == "tank" and bool(unit and unit.status.get("also_tank"))

    @staticmethod
    def has(card, keyword, unit=None):
        """Check catalog and runtime-granted abilities with removal overrides."""
        if unit is not None and unit.status.get("suppressed"):
            return False
        name = keyword.lower()
        removed = {str(value).lower() for value in (unit.status.get("removed_abilities", ()) if unit else ())}
        added = {str(value).lower() for value in (unit.status.get("added_abilities", ()) if unit else ())}
        static = {value.lower().split(":", 1)[0] for value in card.abilities}
        # "Has Blitz if it has 3 or more attack" (M2A4): held while attack >= N.
        conditional = unit.status.get("keyword_if_attack", {}) if unit is not None else {}
        if name in conditional and unit.attack >= int(conditional[name]):
            added = added | {name}
        return name not in removed and (name in static or any(value.startswith(name) and value[len(name):].isdigit() for value in static) or name in added)

    @staticmethod
    def numeric(card, keyword: str, default: int = 0) -> int:
        prefix = keyword.lower()
        for value in card.abilities:
            value = value.lower().split(":", 1)[0]
            if value.startswith(prefix) and value[len(prefix):].isdigit():
                return int(value[len(prefix):])
        return default

    @staticmethod
    def can_attack_on_deploy(card, unit=None):
        return KeywordEngine.has(card, "blitz", unit)

    @staticmethod
    def can_ignore_guard(card):
        return card.type in {"bomber", "artillery"}

    @staticmethod
    def attack_limit(card, unit=None):
        return 2 if KeywordEngine.has(card, "fury", unit) else 1

    @staticmethod
    def heavy_armor(card, unit=None):
        if unit is not None and unit.status.get("suppressed"):
            return 0
        removed = {str(value).lower() for value in (unit.status.get("removed_abilities", ()) if unit else ())}
        if "heavyarmor" in removed or "heavy armor" in removed:
            return 0
        added = tuple(str(value) for value in (unit.status.get("added_abilities", ()) if unit else ()))
        for value in (*card.abilities, *added):
            match = re.search(r"heavy[ _-]?armor\D*(\d+)", value, re.I)
            if match: return int(match.group(1))
        return 0

    @staticmethod
    def can_attack_from_position(card, position):
        return position == "frontline" or card.type in {"artillery", "bomber", "fighter"}

    @staticmethod
    def is_pinned(unit) -> bool:
        return bool(unit.status.get("pinned"))

    @staticmethod
    def is_guarded(state, target, attacker_card, cards):
        if KeywordEngine.can_ignore_guard(attacker_card): return False
        # Each player has an independent line. The shared battlefield list is
        # used for stable serialization only, so filter it before adjacency.
        line = [
            unit_id for unit_id in state.battlefield.get(target.position, [])
            if any(unit.instance_id == unit_id and unit.owner_id == target.owner_id for unit in state.players[target.owner_id].units)
        ]
        try: index = line.index(target.instance_id)
        except ValueError: return False
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(line):
                unit_id=line[neighbor]
                for player in state.players.values():
                    for unit in player.units:
                        if unit.instance_id == unit_id and unit.owner_id == target.owner_id and KeywordEngine.has(cards.get(unit.card_id), "guard", unit):
                            return True
        return False

    @staticmethod
    def is_hq_guarded(state, player_id, attacker_card, cards) -> bool:
        """Return whether the support-line unit adjacent to HQ has Guard.

        The state stores only units in support-line order; HQ is the fixed
        endpoint immediately before index zero.
        """
        if KeywordEngine.can_ignore_guard(attacker_card):
            return False
        support = [
            unit_id for unit_id in state.battlefield.get("support_line", [])
            if any(unit.instance_id == unit_id and unit.owner_id == player_id for unit in state.players[player_id].units)
        ]
        if not support:
            return False
        adjacent_id = support[0]
        for unit in state.players[player_id].units:
            if unit.instance_id == adjacent_id:
                return KeywordEngine.has(cards.get(unit.card_id), "guard", unit)
        return False
