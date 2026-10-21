"""Round 1 pipeline for the Lending Game (we are Lender 2: we observe signal2 only).

Steps
  1. Tune and fit gradient-boosted PD models; out-of-sample PDs on the past loans for
     us (X+s2), each competitor (X+s1, X+s3) and the full-information model (X+s1+s2+s3).
  2. Price the past loans the way each competitor archetype would (see competitors.py).
  3. Choose our pricing rule (margins by segment, rejection thresholds, safety-net rate)
     by maximizing expected profit averaged over competitor scenarios (strategy.py).
  4. Report diagnostics (profit, share, winner's curse) for our rule and naive baselines.
  5. Apply the rule to the new applications and write the submission CSV.

Usage, from the project folder:   python run_round1.py            (about 5 minutes)
                                  python run_round1.py --no-tune  (skip the grid search)
Outputs go to output/round1/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lending_game.competitors import SCENARIOS, competitor_rates, price
from lending_game.features import income_segment, make_features
from lending_game.models import BASE_PARAMS, calibration_table, fit_full, oos_predict, scores, tune
from lending_game.simulator import LOAN, US, draw_types, simulate, FastEvaluator
from lending_game.strategy import Rule, default_params, optimize

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 200)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_keys(seg, p2, cuts, fine):
    """Margin groups: income segment, optionally crossed with a risk tercile within the segment."""
    if not fine:
        return seg.copy()
    keys = np.empty(len(seg), dtype=object)
    for s, (q1, q2) in cuts.items():
        mask = seg == s
        bucket = np.where(p2[mask] <= q1, "lo", np.where(p2[mask] <= q2, "mid", "hi"))
        keys[mask] = np.char.add(s + "|", bucket.astype(str))
    return keys.astype(str)


def evaluate_strategy(r2, comp, p2, pF, y, seeds):
    """Average over type draws, per scenario: our outcome and the competitors' profits."""
    rows = []
    for name, (r1, r3) in comp.items():
        acc = []
        for seed in seeds:
            types = draw_types(len(y), seed)
            res, choice = simulate(np.column_stack([r1, r2, r3]), types, y, pF, seed)
            won = choice == US
            acc.append({"scenario": name, "share": res[US]["share"], "loans": res[US]["loans"],
                        "profit_exp_M": res[US]["profit_expected"] / 1e6,
                        "profit_real_M": res[US]["profit_realized"] / 1e6,
                        "our_p2_on_won": p2[won].mean() if won.any() else np.nan,
                        "realized_default": res[US]["default_realized"],
                        "mean_rate": res[US]["mean_rate"],
                        "lender1_profit_M": res[0]["profit_realized"] / 1e6,
                        "lender3_profit_M": res[2]["profit_realized"] / 1e6})
        rows.append(pd.DataFrame(acc).drop(columns="scenario").mean().rename(name))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--past", default=HERE / "PastLoans.csv")
    ap.add_argument("--new", default=HERE / "NewApplications_Lender2_Round1.csv")
    ap.add_argument("--out", default=HERE / "output" / "round1")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--no-sex", action="store_true", help="exclude sex from the PD models")
    ap.add_argument("--seeds", type=int, default=3, help="random applicant-type draws per scenario")
    ap.add_argument("--recompute", action="store_true", help="ignore cached out-of-sample PDs")
    ap.add_argument("--net", type=float, default=None,
                    help="fix the safety-net rate (e.g. 0.899) instead of searching; see analysis/net_rate_scenarios.py")
    ap.add_argument("--ramp-from", type=float, default=None,
                    help="make the net a ramp: this rate at the threshold, rising linearly to --net at PD 25%% "
                         "(see analysis/band_payoff_matrix.py); default: flat net")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    use_sex = not args.no_sex
    seeds = list(range(args.seeds))
    report = []  # lines of the diagnostics file

    def section(title, body=""):
        print(f"\n=== {title} ===\n{body}", flush=True)
        report.append(f"\n## {title}\n\n```\n{body}\n```\n" if body else f"\n## {title}\n")

    past = pd.read_csv(args.past)
    new = pd.read_csv(args.new)
    y = past["default"].to_numpy()
    n = len(past)

    # ------------------------------------------------------------------ 1. PD models
    X_ours = make_features(past, ["signal2"], use_sex)
    if args.no_tune:
        params = dict(BASE_PARAMS)
    else:
        log("tuning the boosted-tree PD model (3-fold CV log-loss)...")
        grid = [dict(learning_rate=0.05, max_leaf_nodes=lv, max_iter=it) for lv in (7, 15, 31) for it in (200, 400, 800)]
        best, table = tune(X_ours, y, grid)
        params = {**BASE_PARAMS, **best}
        section("Hyper-parameter search (our model, X + signal2)", table.to_string(index=False))

    cache = out / f"oos_probas_sex{int(use_sex)}.npz"
    if cache.exists() and not args.recompute and json.loads(str(np.load(cache, allow_pickle=True)["params"])) == params:
        z = np.load(cache, allow_pickle=True)
        pX, p1, p2, p3, pF = (z[k] for k in ["pX", "p1", "p2", "p3", "pF"])
        log(f"loaded cached out-of-sample PDs from {cache.name}")
    else:
        log("computing 5-fold out-of-sample PDs for X, X+s1, X+s2, X+s3, X+s1+s2+s3...")
        pX = oos_predict(make_features(past, [], use_sex), y, params)
        p1 = oos_predict(make_features(past, ["signal1"], use_sex), y, params)
        p2 = oos_predict(X_ours, y, params)
        p3 = oos_predict(make_features(past, ["signal3"], use_sex), y, params)
        pF = oos_predict(make_features(past, ["signal1", "signal2", "signal3"], use_sex), y, params)
        np.savez(cache, pX=pX, p1=p1, p2=p2, p3=p3, pF=pF, params=json.dumps(params))

    body = "\n".join(f"{name:28s} AUC={s['auc']:.4f}  logloss={s['logloss']:.4f}  brier={s['brier']:.4f}"
                     for name, s in [("X only", scores(y, pX)), ("X + signal2 (us)", scores(y, p2)),
                                     ("X + signal1 (lender 1)", scores(y, p1)), ("X + signal3 (lender 3)", scores(y, p3)),
                                     ("X + all signals (full info)", scores(y, pF))])
    if use_sex:
        p2_nosex = oos_predict(make_features(past, ["signal2"], False), y, params)
        s = scores(y, p2_nosex)
        body += f"\n{'X + signal2 WITHOUT sex':28s} AUC={s['auc']:.4f}  logloss={s['logloss']:.4f}  brier={s['brier']:.4f}"
    section("Model quality (out of sample)", body)
    section("Calibration of our model (deciles)", calibration_table(y, p2).round(4).to_string())
    hidden = pF - p2
    seg = income_segment(past["income"])
    section("Hidden information U = p_full - p_ours",
            f"std(U) = {hidden.std():.4f}\nby income segment:\n"
            + pd.Series(hidden).groupby(seg).std().round(4).to_string())

    # ------------------------------------------------------------------ 2. competitors
    comp = competitor_rates(p1, p3, seg)
    ev = FastEvaluator(comp, n, seeds, pF)

    # ------------------------------------------------------------------ 3. our rule
    cuts = {s: tuple(np.quantile(p2[seg == s], [1 / 3, 2 / 3])) for s in np.unique(seg)}
    candidates = {}
    for fine in (False, True):
        label = "segment x risk-tercile margins" if fine else "segment margins"
        log(f"optimizing our pricing rule ({label})...")
        rule = Rule(p2, seg, make_keys(seg, p2, cuts, fine))
        prm, best = optimize(rule, ev, default_params(rule, tail_rate=args.net or 0.989, ramp_from=args.ramp_from),
                             fixed_tail="search" if args.net is None else args.net)
        avg, worst, per = ev.objective(rule.rates(prm))
        candidates[fine] = (prm, avg, worst, rule)
        log(f"  -> mean expected profit {avg / 1e6:.3f} M EUR, worst scenario {worst / 1e6:.3f} M EUR")
    fine = candidates[True][1] > candidates[False][1] * 1.02  # finer rule must earn its complexity
    params_rule, avg, worst, rule = candidates[fine]
    r2_opt = rule.rates(params_rule)
    section("Chosen pricing rule" + (" (segment x risk tercile)" if fine else " (segment margins)"),
            json.dumps({**params_rule, "mean_expected_profit_M": round(avg / 1e6, 3),
                        "worst_scenario_profit_M": round(worst / 1e6, 3)}, indent=2))

    # ------------------------------------------------------------------ 4. diagnostics
    strategies = {
        "naive: break-even + 3%, lend to everyone": price(p2, margin=0.03),
        "naive: break-even + 3%, no offer if PD > 25%": price(p2, margin=0.03, reject_above=0.25),
        "naive + safety net 0.989 instead of no offer": price(p2, margin=0.03, reject_above=0.25, net=0.989),
        "OPTIMIZED rule": r2_opt,
    }
    summary = []
    for name, r2 in strategies.items():
        tab = evaluate_strategy(r2, comp, p2, pF, y, seeds)
        section(f"Strategy: {name}", tab.round(3).to_string())
        summary.append(pd.Series({"mean profit (M EUR)": tab["profit_real_M"].mean(),
                                  "worst scenario (M EUR)": tab["profit_real_M"].min(),
                                  "mean share": tab["share"].mean()}, name=name))
    section("Summary across scenarios (realized profit on past loans)", pd.DataFrame(summary).round(3).to_string())

    # per-segment view of the optimized rule in the base scenario
    r1, r3 = comp["base / base"]
    types = draw_types(n, 0)
    res, choice = simulate(np.column_stack([r1, r2_opt, r3]), types, y, pF, 0)
    won = choice == US
    df = pd.DataFrame({"segment": seg, "won": won, "rate": np.where(np.isfinite(r2_opt), r2_opt, np.nan),
                       "offered": np.isfinite(r2_opt), "p2": p2, "y": y,
                       "profit": np.where(won, np.where(y == 1, -LOAN, LOAN * np.where(np.isfinite(r2_opt), r2_opt, 0)), 0.0)})
    g = df.groupby("segment")
    per_seg = pd.DataFrame({"applicants": g.size(), "offered": g["offered"].mean(), "share_won": g["won"].mean(),
                            "mean_rate_offered": g["rate"].mean(),
                            "our_p2_on_won": df[won].groupby("segment")["p2"].mean(),
                            "realized_default_on_won": df[won].groupby("segment")["y"].mean(),
                            "profit_M": g["profit"].sum() / 1e6})
    section("Optimized rule by income segment (scenario base / base)", per_seg.round(3).to_string())

    # ------------------------------------------------------------------ 5. new applications
    log("fitting the final PD model on all past loans and pricing the new applications...")
    model = fit_full(X_ours, y, params)
    p2_new = model.predict_proba(make_features(new, ["signal2"], use_sex))[:, 1]
    seg_new = income_segment(new["income"])
    rule_new = Rule(p2_new, seg_new, make_keys(seg_new, p2_new, cuts, fine))
    r_new = rule_new.rates(params_rule)
    rate_col = np.where(np.isfinite(r_new), np.round(r_new, 4), np.nan)

    sub = pd.DataFrame({"id": new["id"].astype(int), "rate": rate_col})
    sub.to_csv(out / "rates_round1.csv", index=False, na_rep="")
    pd.DataFrame({"id": new["id"], "segment": seg_new, "p2": np.round(p2_new, 5), "rate": rate_col}) \
        .to_csv(out / "new_applications_scored.csv", index=False, na_rep="")
    with open(out / "strategy_round1.json", "w", encoding="utf-8") as f:
        json.dump({"model_params": params, "use_sex": use_sex, "fine_margins": bool(fine), "net_fixed": args.net,
                   "risk_tercile_cuts": {k: list(map(float, v)) for k, v in cuts.items()},
                   "rule": params_rule, "mean_expected_profit_M": avg / 1e6, "worst_scenario_profit_M": worst / 1e6}, f, indent=2)

    tail = params_rule["tail_rate"]
    ramp_from = params_rule.get("ramp_from")
    band_new = r_new >= (ramp_from if ramp_from is not None else tail) if tail else np.zeros(len(r_new), bool)
    band_desc = (f"on the ramp {ramp_from} -> {tail}" if ramp_from is not None else f"at safety-net rate {tail}") if tail else "in band"
    offers = pd.DataFrame({"segment": seg_new, "rate": rate_col, "p2": p2_new})
    body = (f"applicants: {len(sub)}   offers: {int(np.isfinite(r_new).sum())}   no offer: {int(np.isinf(r_new).sum())}"
            f"   {band_desc}: {int(band_new.sum())}\n"
            f"mean PD (our model): {p2_new.mean():.4f}\n\nby segment:\n"
            + offers.groupby("segment").agg(applicants=("rate", "size"), mean_p2=("p2", "mean"),
                                            offered=("rate", lambda s: s.notna().mean()),
                                            median_rate=("rate", "median"), min_rate=("rate", "min"),
                                            max_rate=("rate", "max")).round(4).to_string())
    section("Submission: output/round1/rates_round1.csv", body)

    (out / "diagnostics_round1.md").write_text("# Round 1 diagnostics\n" + "\n".join(report), encoding="utf-8")
    log(f"done. Files in {out}")


if __name__ == "__main__":
    main()
