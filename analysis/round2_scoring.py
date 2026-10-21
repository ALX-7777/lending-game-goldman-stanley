"""Round 2: score the new applicants (NewApplications_Lender2_Round2.csv) with our PD model, and check whether adding the
100,000 round-1 applicants (whose defaults are now observed in the diagnostic) to the training set improves the model.
Training sets compared by 5-fold out-of-sample AUC / log-loss on the round-1 applicants (a held-out population for the
past-loans-only model, in-fold for the augmented one -- so the augmented model is evaluated by cross-validation).
Writes output/round2/new_applications_scored.csv (id, segment, p2) with the chosen model, and output/round2/scoring_check.txt.
Run from the project folder:  python analysis/round2_scoring.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from lending_game.features import income_segment, make_features  # noqa: E402
from lending_game.models import fit_full  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = HERE / "output" / "round2"
OUT.mkdir(parents=True, exist_ok=True)
strategy = json.load(open(HERE / "output" / "round1" / "strategy_round1.json", encoding="utf-8"))
params, use_sex = strategy["model_params"], strategy.get("use_sex", False)

past = pd.read_csv(HERE / "PastLoans.csv")
r1 = pd.read_csv(HERE / "NewApplications_Lender2_Round1.csv").merge(
    pd.read_csv(HERE / "Round_1_report_new" / "Round1_Diagnostic_23.csv")[["id", "default"]], on="id")
r2 = pd.read_csv(HERE / "Round_1_report_new" / "NewApplications_Lender2_Round2.csv")
cols = ["sex", "employment", "married", "income", "signal2", "default"]
past_s, r1_s = past[cols], r1[cols]
print(f"past loans {len(past):,} (default {past.default.mean():.4f}) | round-1 applicants {len(r1):,} (default {r1.default.mean():.4f}) | round-2 applicants {len(r2):,}")
print("signal2 == 0 share: past %.3f, round 1 %.3f, round 2 %.3f" % ((past.signal2 == 0).mean(), (r1.signal2 == 0).mean(), (r2.signal2 == 0).mean()))

# 1. past-loans model evaluated on the round-1 applicants (a genuine out-of-sample population)
m_past = fit_full(make_features(past_s, ["signal2"], use_sex), past_s["default"].to_numpy(), params)
p_r1 = m_past.predict_proba(make_features(r1_s, ["signal2"], use_sex))[:, 1]
lines = [f"past-loans model on the 100,000 round-1 applicants: AUC {roc_auc_score(r1_s.default, p_r1):.4f}, log-loss {log_loss(r1_s.default, p_r1):.4f}, mean PD {p_r1.mean():.4f} vs realized {r1_s.default.mean():.4f}"]

# 2. augmented model (past + round 1), 5-fold CV where the round-1 rows are the test folds
both = pd.concat([past_s.assign(src="past"), r1_s.assign(src="r1")], ignore_index=True)
X_all = make_features(both, ["signal2"], use_sex)
y_all = both["default"].to_numpy()
is_r1 = (both.src == "r1").to_numpy()
pred = np.full(len(both), np.nan)
skf = StratifiedKFold(5, shuffle=True, random_state=0)
r1_idx = np.where(is_r1)[0]
for tr, te in skf.split(r1_idx, y_all[r1_idx]):
    test_rows = r1_idx[te]
    train_rows = np.setdiff1d(np.arange(len(both)), test_rows)  # all past loans + the other round-1 folds
    m = fit_full(X_all.iloc[train_rows], y_all[train_rows], params)
    pred[test_rows] = m.predict_proba(X_all.iloc[test_rows])[:, 1]
pa = pred[is_r1]
lines.append(f"augmented model (past + round 1, 5-fold on the round-1 rows): AUC {roc_auc_score(y_all[is_r1], pa):.4f}, log-loss {log_loss(y_all[is_r1], pa):.4f}")
gain = roc_auc_score(y_all[is_r1], pa) - roc_auc_score(r1_s.default, p_r1)
lines.append(f"AUC gain from augmentation: {gain:+.4f}")
print("\n".join(lines))

# 3. final model: train on both populations, score round 2
m_final = fit_full(X_all, y_all, params)
p2 = m_final.predict_proba(make_features(r2, ["signal2"], use_sex))[:, 1]
seg = income_segment(r2["income"])
scored = pd.DataFrame({"id": r2["id"].astype(int), "segment": seg, "p2": np.round(p2, 6)})
scored.to_csv(OUT / "new_applications_scored.csv", index=False)
summary = scored.groupby("segment")["p2"].agg(["size", "mean"]).rename(columns={"size": "applicants", "mean": "mean PD"})
summary["mean PD"] = summary["mean PD"].round(4)
lines.append("\nround-2 applicants scored with the augmented model (past + round 1):\n" + summary.to_string())
lines.append(f"overall mean PD round 2: {p2.mean():.4f}; PD bands: " + ", ".join(f"{lo*100:.0f}-{hi*100:.0f}%: {int(((p2 > lo) & (p2 <= hi)).sum()):,}" for lo, hi in [(0, .05), (.05, .10), (.10, .15), (.15, .20), (.20, .25), (.25, .30), (.30, 1)]))
print("\n".join(lines[-2:]))
(OUT / "scoring_check.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nwritten {OUT / 'new_applications_scored.csv'} and scoring_check.txt")
