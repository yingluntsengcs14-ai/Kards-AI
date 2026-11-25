import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams["font.family"] = "Noto Sans CJK TC"
data = json.load(open(sys.argv[1])); cells = data["cells"]
names = data["decks"]
overall = {d: sum(v[0] for v in cells[d].values()) / sum(v[1] for v in cells[d].values()) for d in names}
order = sorted(names, key=lambda d: -overall[d]); n = len(order)
M = np.full((n, n), np.nan); G = np.zeros((n, n), int)
for i, a in enumerate(order):
    for j, b in enumerate(order):
        if a != b: w, g = cells[a][b]; M[i, j] = w / g; G[i, j] = g
cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"]); cmap.set_bad("#d9d8d4")
fig, ax = plt.subplots(figsize=(13, 11.5), dpi=150); fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
for i in range(n):
    for j in range(n):
        if i == j: ax.text(j, i, "—", ha="center", va="center", color="#7a7974", fontsize=9)
        else: ax.text(j, i, f"{M[i,j]*100:.0f}", ha="center", va="center", fontsize=8.5, color="#1f1e1c")
labels = [f"{d}  {overall[d]*100:.0f}%" for d in order]
ax.set_xticks(range(n)); ax.set_xticklabels(order, rotation=60, ha="right", fontsize=10, color="#52514e")
ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=10, color="#1f1e1c")
ax.set_xticks(np.arange(-.5, n), minor=True); ax.set_yticks(np.arange(-.5, n), minor=True)
ax.grid(which="minor", color="#fcfcfb", linewidth=2); ax.tick_params(which="both", length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xlabel("對手牌組", fontsize=11, color="#52514e"); ax.set_ylabel("使用的牌組（右側為對全部牌組的平均勝率）", fontsize=11, color="#52514e")
games = int(G[G > 0].min())
ax.set_title(f"{sys.argv[3]}：17 套 meta 牌組對戰勝率（列牌組打欄牌組，%）", fontsize=14, color="#1f1e1c", loc="left", pad=14)
fig.text(0.01, 0.01, f"同一個模型操作雙方，每格 {games} 局（先後手各半）。每格 95% 誤差約 ±{196*np.sqrt(.25/games):.0f} 個百分點，只看大差距。", fontsize=9.5, color="#52514e")
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("勝率", color="#52514e"); cb.outline.set_visible(False)
cb.set_ticks([0, .25, .5, .75, 1]); cb.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(sys.argv[2], facecolor=fig.get_facecolor())
print({d: round(overall[d], 3) for d in order}, "errors", data["errors"])
