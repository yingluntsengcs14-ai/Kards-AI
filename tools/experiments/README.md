# 實驗腳本（2026-11-25～26，人工評分與搜尋）

雲端 CPU 上跑的實驗，結果記在 claude 專案文件 `claude/實驗結果_1124-1126.md`。

| 腳本 | 用途 |
|---|---|
| `heur_eval.py rule\|random` | 貪婪人工評分 AI，配對 meta 評估 |
| `gen_features.py N SEED OUT` | epsilon-greedy 自我對局，記錄回合結束特徵與勝負 |
| `fit_weights.py DATA...` | 邏輯迴歸擬合權重，輸出 `DATA.weights.json` |
| `evolve.py START OUT GENS` | 演化調參（打 rule ＋ 打現任最佳） |
| `eval_w.py` / `eval_big.py` | 權重版本比較（272／544 局，換種子） |
| `hmcts_eval.py SIMS OUT` | 不用網路、只用人工評分的 MCTS 打 rule |
| `matchup.py` / `plot_matchup.py` | 17×17 牌組對戰矩陣與熱力圖（`matchup.py` 需在 v1 commit `c33aaff` 的工作樹下跑 v1 模型） |
