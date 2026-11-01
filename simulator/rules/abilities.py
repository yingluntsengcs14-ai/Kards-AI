"""Runtime ability predicates independent of individual card implementations."""

from __future__ import annotations


class AbilityEngine:
    """Combines static catalog abilities with per-unit consumed/added abilities."""

    @staticmethod
    def has(card, unit, ability: str) -> bool:
        if unit.status.get("suppressed"):
            return False
        name = ability.lower()
        removed = {item.lower() for item in unit.status.get("removed_abilities", ())}
        added = {item.lower() for item in unit.status.get("added_abilities", ())}
        added |= {item.lower() for item in unit.status.get("aura_abilities", ())}   # 909th ROSLAVL
        static = {item.lower().split(":", 1)[0] for item in card.abilities}
        return name not in removed and (name in static or name in added)

    @classmethod
    def has_smokescreen(cls, card, unit) -> bool:
        return cls.has(card, unit, "smokescreen")

    @classmethod
    def has_ambush(cls, card, unit) -> bool:
        return cls.has(card, unit, "ambush")

    @staticmethod
    def consume(unit, ability: str) -> None:
        removed = unit.status.setdefault("removed_abilities", [])
        if ability not in removed:
            removed.append(ability)
