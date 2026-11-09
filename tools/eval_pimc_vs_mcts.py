"""PIMC vs 完美資訊 MCTS 勝率實驗（期末報告的實驗章節用）。

背景：``ai/mcts.py`` 的 ``MCTS``／``MCTSAgent`` 直接在真實 ``GameState`` 上
搜尋，等於看得到對手手牌跟兩邊牌庫順序——是一個「作弊」的完美資訊搜尋。
``ai/pimc.py`` 的 ``PIMCAgent`` 則是先用 ``determinize()`` 依照「這個玩家
實際看得到的資訊」抽樣出好幾個合法世界，在每個世界上各跑一次一般 MCTS，
再把各世界的 visit distribution 平均起來投票選動作——是隱藏資訊正確的搜尋。

這個工具讓兩種 agent 真的對打，量化「隱藏資訊正確」這件事要付出多少勝率代價
（PIMC 沒看到對手手牌，直覺上應該比完美資訊 MCTS 弱；這個數字就是報告裡
「PIMC vs 完美資訊 MCTS」那張表/那段文字的來源）。

用法：
    python tools/eval_pimc_vs_mcts.py --games 20 --worlds 4 --simulations 16
    python tools/eval_pimc_vs_mcts.py --games 40 --model runtime/kardsnet.pt --meta-decks

沒有 --model 時用隨機初始化的 KARDSNet（兩個 agent 共用同一份權重，公平比較
演算法本身的差異，不是模型強弱）；這個雲端環境目前沒有訓練過的 checkpoint
（訓練在使用者桌機跑），所以預設就是跑這個「演算法本身的勝率差距」版本。等
桌機那邊訓練出 checkpoint 後，可以用 --model 指到那個 .pt 檔重跑一次，兩個
數字都放進報告：「隨機網路下的勝率差距」＋「訓練後網路下的勝率差距」。

為了消除先後手優勢，``--games`` 會平分成兩半：前一半 PIMC 當 p1、MCTS 當
p2；後一半互換。回報的勝率是依「PIMC agent」／「MCTS agent」本身統計的，不
是依 p1/p2。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow `python3 tools/eval_pimc_vs_mcts.py ...` directly (python only puts
# this script's own directory on sys.path, not the repo root, so `import ai`
# would otherwise fail unless invoked as `python3 -m tools.eval_pimc_vs_mcts`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.agents import MCTSAgent
from ai.network import KARDSNet
from ai.observation import ObservationEncoder
from ai.pimc import PIMCAgent
from ai.replay_buffer import ReplayBuffer
from ai.selfplay import SelfPlayRunner
from simulator.cards.deck_code import load_meta_decks
from simulator.cards.loader import CardDatabase

ROOT = Path(__file__).parent.parent
DEFAULT_CATALOG = ROOT / "data/source/kards_info_cards.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=20, help="總局數（平分兩半，輪流當 p1/p2）")
    parser.add_argument("--worlds", type=int, default=4, help="PIMC 每次決策抽樣的世界數")
    parser.add_argument("--simulations", type=int, default=16, help="每個 MCTS 搜尋（含 PIMC 底下每個世界）跑幾次模擬")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=None, help="訓練過的 checkpoint 路徑；不給就用隨機初始化權重（兩個 agent 共用一份）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nation", default="France", help="沒有 --meta-decks 時用的隨機牌組國家")
    parser.add_argument("--meta-decks", action="store_true", help="改用官方 17 套大賽牌組隨機配對，比隨機牌組更貼近真實對局")
    parser.add_argument("--max-actions", type=int, default=300)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--output", default=None, help="把結果另外存成 JSON（預設只印在終端機）")
    args = parser.parse_args()

    cards = CardDatabase.from_file(Path(args.catalog))
    encoder = ObservationEncoder(cards)
    model = KARDSNet.load_checkpoint(args.model, device=args.device) if args.model else KARDSNet()
    model = model.to(args.device).eval()
    meta_decks = load_meta_decks(cards) if args.meta_decks else None

    def make_pimc(seed: int) -> PIMCAgent:
        return PIMCAgent(model, encoder, worlds=args.worlds, simulations=args.simulations, seed=seed)

    def make_mcts(seed: int) -> MCTSAgent:
        return MCTSAgent(model, encoder, simulations=args.simulations, seed=seed)

    half = args.games // 2
    remainder = args.games - half
    buffer = ReplayBuffer(seed=args.seed)
    runner = SelfPlayRunner(cards, encoder, buffer, max_actions=args.max_actions, seed=args.seed)

    pimc_wins = mcts_wins = draws = 0
    started = time.monotonic()

    if half:
        pimc_agent, mcts_agent = make_pimc(args.seed + 1), make_mcts(args.seed + 2)
        report = runner.run(half, pimc_agent, mcts_agent, nation=args.nation, meta_decks=meta_decks)
        pimc_wins += report.p1_wins
        mcts_wins += report.p2_wins
        draws += report.draws

    if remainder:
        mcts_agent, pimc_agent = make_mcts(args.seed + 3), make_pimc(args.seed + 4)
        report = runner.run(remainder, mcts_agent, pimc_agent, nation=args.nation, meta_decks=meta_decks)
        mcts_wins += report.p1_wins
        pimc_wins += report.p2_wins
        draws += report.draws

    elapsed = time.monotonic() - started
    total = pimc_wins + mcts_wins + draws
    result = {
        "games": total,
        "pimc_wins": pimc_wins,
        "mcts_wins": mcts_wins,
        "draws": draws,
        "pimc_win_rate": pimc_wins / total if total else 0.0,
        "mcts_win_rate": mcts_wins / total if total else 0.0,
        "worlds": args.worlds,
        "simulations": args.simulations,
        "model": args.model or "random_init",
        "meta_decks": args.meta_decks,
        "elapsed_seconds": elapsed,
        "seconds_per_game": elapsed / total if total else 0.0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
