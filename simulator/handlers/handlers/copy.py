from simulator.handlers.base_handler import BaseHandler, HandlerContext


class CopyHandler(BaseHandler):
    def execute(self, state, cards, context):
        card_id = str(context.parameters.get("card_id", context.source_card_id or ""))
        destination = str(context.parameters.get("destination", "hand"))
        if card_id not in cards:
            state.event_log.append({"event": "copy_failed", "card_id": card_id})
            return
        player = state.players[context.player_id]
        if destination == "deck":
            player.deck.append(card_id)
        else:
            from simulator.rules.native import _hand_or_graveyard
            _hand_or_graveyard(state, context.player_id, card_id)
