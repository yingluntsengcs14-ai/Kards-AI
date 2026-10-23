"""Reviewed per-card AST overrides stored separately from kards.info data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from simulator.rules.parser import CardRule, RuleAction


class CardRuleStore:
    """Loads only reviewed rule overrides; malformed entries fail closed."""

    def __init__(self, rules: Mapping[str, CardRule] | None = None) -> None:
        self._rules = dict(rules or {})

    @classmethod
    def from_file(cls, path: str | Path) -> "CardRuleStore":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, Mapping) or not isinstance(document.get("rules"), list):
            raise ValueError("Rule store must contain a rules array")
        rules: dict[str, CardRule] = {}
        for entry in document["rules"]:
            if not isinstance(entry, Mapping):
                raise ValueError("Rule store entries must be objects")
            rule = _rule_from_document(entry)
            if rule.card_id in rules:
                raise ValueError("Rule store card IDs must be unique")
            rules[rule.card_id] = rule
        return cls(rules)

    def get(self, card_id: str) -> CardRule | None:
        return self._rules.get(card_id)


def _rule_from_document(entry: Mapping[str, Any]) -> CardRule:
    card_id = entry.get("card_id")
    source_text = entry.get("source_text")
    triggers = entry.get("triggers", [entry.get("trigger")])
    actions = entry.get("actions", [])
    status = entry.get("status", "implemented")
    if not isinstance(card_id, str) or not isinstance(source_text, str) or not isinstance(triggers, list) or not all(isinstance(item, str) for item in triggers):
        raise ValueError("Rule store entry has invalid identity fields")
    if status not in {"implemented", "partial", "unresolved"} or not isinstance(actions, list):
        raise ValueError("Rule store entry has invalid status or actions")
    parsed_actions: list[RuleAction] = []
    for action in actions:
        if not isinstance(action, Mapping) or not isinstance(action.get("kind"), str):
            raise ValueError("Rule action must contain kind")
        parsed_actions.append(RuleAction(
            kind=action["kind"], target=str(action.get("target", "owner")), amount=int(action.get("amount", 0)),
            attack=int(action.get("attack", 0)), defense=int(action.get("defense", 0)), card_name=action.get("card_name"),
            scope=str(action.get("scope", "")), min_cost=int(action.get("min_cost", 0)), set_cost=action.get("set_cost"),
            duration=action.get("duration"), condition=action.get("condition"),
            on=tuple(action.get("on", ()) or ()),
        ))
    return CardRule(card_id, source_text, tuple(item for item in triggers if item), tuple(parsed_actions), status, tuple(entry.get("unresolved_fragments", ())))
