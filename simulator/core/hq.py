"""Centralized headquarters damage, healing, defence, and victory resolution."""

from __future__ import annotations

from simulator.core.state import GameState, GameStatus


class HQResolver:
    """The only game-rules component allowed to mutate HQ health."""

    @staticmethod
    def damage(
        state: GameState,
        target_player_id: str,
        amount: int,
        source_player_id: str | None = None,
        apply_replacements: bool = True,
    ) -> int:
        hq = state.players[target_player_id].hq
        if apply_replacements and amount > 0:
            redirectors = [
                unit for unit in state.players[target_player_id].units
                if unit.status.get("hq_damage_redirect_to_enemy_hq")
            ]
            if redirectors:
                enemy_id = next(player_id for player_id in state.players if player_id != target_player_id)
                applied = HQResolver.damage(state, enemy_id, amount, redirectors[0].owner_id, apply_replacements=False)
                state.event_log.append({
                    "event": "hq_damage_redirected_to_enemy_hq",
                    "player_id": target_player_id,
                    "enemy_player_id": enemy_id,
                    "amount": applied,
                    "source_unit_id": redirectors[0].instance_id,
                })
                return applied
        # Check friendly units for HQ defense cap/lock/immune flags.
        for unit in state.players[target_player_id].units:
            if unit.status.get("hq_defense_lock"):
                state.event_log.append({"event": "hq_damage_blocked", "player_id": target_player_id, "reason": "hq_defense_lock"})
                return 0
            if unit.status.get("hq_defense_cap") and hq.current_health - max(0, int(amount) - hq.defense_modifier) < (unit.status.get("hq_defense_cap") or 1):
                amount = max(0, hq.current_health - (unit.status.get("hq_defense_cap") or 1) + hq.defense_modifier)
        applied = max(0, int(amount) - hq.defense_modifier)
        redirectors = [
            unit for unit in state.players[target_player_id].units
            if unit.status.get("hq_damage_redirect_to_source")
        ]
        if applied > 0 and redirectors:
            # This is a replacement effect, not an additional damage trigger:
            # the first in-play redirector takes the post-HQ-defense amount and
            # the HQ remains unchanged.  Card copies are deterministic by board
            # order, matching the engine's stable trigger ordering.
            from simulator.effects import basic_effects
            redirector = redirectors[0]
            basic_effects.damage(state, redirector, applied, source_player_id)
            state.event_log.append({
                "event": "hq_damage_redirected_to_unit",
                "player_id": target_player_id,
                "unit_id": redirector.instance_id,
                "amount": applied,
                "source_player_id": source_player_id,
            })
            return applied
        hq.current_health -= applied
        from simulator.rules import counters
        counters.note_hq_damage(state, target_player_id, applied)
        state.event_log.append({"event": "hq_damaged", "player_id": target_player_id, "amount": applied, "source_player_id": source_player_id})
        HQResolver.check_victory(state, source_player_id)
        return applied

    @staticmethod
    def heal(state: GameState, target_player_id: str, amount: int) -> int:
        hq = state.players[target_player_id].hq
        before = hq.current_health
        hq.current_health = min(hq.max_health, hq.current_health + max(0, int(amount)))
        applied = hq.current_health - before
        state.event_log.append({"event": "hq_healed", "player_id": target_player_id, "amount": applied})
        return applied

    @staticmethod
    def modify_defense(state: GameState, target_player_id: str, amount: int) -> int:
        # Some continuous effects replace a HQ defense gain with equal damage.
        # This belongs at the HQ mutation boundary so every source of HQ
        # defense (orders, triggers, and native rules) observes the same rule.
        replacements = [
            unit.owner_id
            for player in state.players.values()
            for unit in player.units
            if unit.status.get("hq_defense_becomes_damage")
        ]
        if amount > 0 and replacements:
            total = 0
            for source_player_id in replacements:
                total += HQResolver.damage(state, target_player_id, amount, source_player_id)
            state.event_log.append({
                "event": "hq_defense_replaced_with_damage",
                "player_id": target_player_id,
                "amount": amount,
                "source_count": len(replacements),
            })
            return total
        # An HQ's defense IS its health (confirmed in game, 2026-10-24).  This
        # used to raise ``defense_modifier``, which damage() subtracts from every
        # hit -- "+3 defense" made the HQ shrug off 3 damage per hit forever.
        # "+N defense" is not a repair, so it may lift the HQ above its
        # starting value; the ceiling moves with it.
        hq = state.players[target_player_id].hq
        hq.current_health = max(0, hq.current_health + int(amount))
        hq.max_health = max(hq.max_health, hq.current_health)
        state.event_log.append({"event": "hq_defense_modified", "player_id": target_player_id,
                                "amount": int(amount), "hq_health": hq.current_health})
        if hq.current_health <= 0:
            HQResolver.check_victory(state)
        return hq.current_health

    @staticmethod
    def set_defense(state: GameState, target_player_id: str, value: int) -> int:
        """"Set your HQ's defense to N": the HQ's health becomes N."""
        hq = state.players[target_player_id].hq
        hq.current_health = max(0, int(value))
        hq.max_health = max(hq.max_health, hq.current_health)
        state.event_log.append({"event": "hq_defense_set", "player_id": target_player_id, "amount": int(value)})
        if hq.current_health <= 0:
            HQResolver.check_victory(state)
        return hq.current_health

    @staticmethod
    def check_victory(state: GameState, source_player_id: str | None = None) -> GameStatus:
        if state.game_status != GameStatus.IN_PROGRESS:
            return state.game_status
        defeated = [player_id for player_id, player in state.players.items() if player.hq.current_health <= 0]
        if not defeated:
            return state.game_status
        if len(defeated) == len(state.players):
            state.game_status = GameStatus.DRAW
            winner = None
        else:
            winner = source_player_id or next(player_id for player_id in state.players if player_id not in defeated)
            state.game_status = GameStatus.PLAYER_ONE_WON if winner == "p1" else GameStatus.PLAYER_TWO_WON
        state.event_log.append({"event": "victory", "winner_id": winner, "defeated_player_ids": defeated})
        state.event_log.append({"event": "game_over", "status": state.game_status.value})
        return state.game_status
