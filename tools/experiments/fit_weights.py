import sys, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from ai.heuristic import FEATURES, HeuristicWeights
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
rows = [r for r in rows if r["y"] in (0, 1)]
X = np.array([r["x"] for r in rows]); y = np.array([r["y"] for r in rows])
print(len(rows), "rows")
m = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False)
print("cv acc", cross_val_score(m, X, y, cv=5).mean().round(3))
m.fit(X, y)
w0 = HeuristicWeights()
base = np.array([getattr(w0, k) for k in FEATURES]) * 2 / w0.scale  # hand weights in logit units
print("hand-set acc", ((X @ base > 0) == y).mean().round(3))
fitted = m.coef_[0] * w0.scale / 2
for k, a, b in zip(FEATURES, [getattr(w0, k) for k in FEATURES], fitted): print(f"{k:10s} hand {a:6.2f}  fitted {b:6.2f}")
json.dump(dict(zip(FEATURES, fitted.round(3).tolist())), open(sys.argv[1] + ".weights.json", "w"))
