"""Build the submission CSV from the ANALYSED rule, without re-running the optimizer.

Applies lending_game.strategy.Rule.rates to the per-applicant PD of the new applications
(output/round1/new_applications_scored.csv) with the margins and thresholds stored in
output/round1/strategy_round1.json, optionally overriding the net and turning it into a ramp.
This keeps the submitted rule identical to the one evaluated in analysis/band_payoff_matrix.py.

Usage, from the project folder:
    python make_submission.py --net 0.828 --ramp-from 0.70            # ramp 0.70 -> 0.828 over PD (tau, 0.25]
    python make_submission.py --net 0.828                             # flat net 0.828
    python make_submission.py --net 0.828 --ramp-from 0.60 --out some/other/path.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lending_game.strategy import Rule  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = HERE / "output" / "round1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", type=float, default=None, help="net (top of the ramp); default: the analysed rule's")
    ap.add_argument("--ramp-from", type=float, default=None, help="rate at the threshold; omit for a flat net")
    ap.add_argument("--ramp-top", type=float, default=0.25, help="PD at which the ramp reaches the net")
    ap.add_argument("--out", default=str(OUT / "rates_round1.csv"))
    ap.add_argument("--adjusted-x", type=float, default=None,
                    help="band = adjusted break-even(PD) + X (from output/round1/curse_floor.csv), phased in over "
                         "--phase-in points of PD at the threshold, capped at --cap; overrides --net/--ramp-from")
    ap.add_argument("--cap", type=float, default=0.989)
    ap.add_argument("--phase-delta", type=float, default=0.01,
                    help="centre of the margin transition, in PD above the segment threshold (sigmoid)")
    ap.add_argument("--phase-width", type=float, default=0.01, help="scale of the sigmoid transition, in PD")
    ap.add_argument("--floor-by-segment", action="store_true",
                    help="use the income-segment-specific adjusted break-even (output/round1/curse_floor_by_segment.csv) "
                         "instead of the pooled curve")
    ap.add_argument("--applications", default=str(HERE / "NewApplications_Lender2_Round1.csv"),
                    help="application file to price (round 2: Round_1_report_new/NewApplications_Lender2_Round2.csv)")
    ap.add_argument("--safe-extra", type=float, default=0.0, help="points added to every segment margin on the safe part (e.g. 0.02)")
    ap.add_argument("--augment-round1", action="store_true",
                    help="add the 100,000 round-1 applicants with their observed defaults (Round_1_report_new/Round1_Diagnostic_23.csv) to the training set")
    args = ap.parse_args()

    strategy = json.load(open(OUT / "strategy_round1.json", encoding="utf-8"))
    params = dict(strategy["rule"])
    if args.net is not None:
        params["tail_rate"] = args.net
    params["ramp_from"] = args.ramp_from
    params["ramp_top"] = args.ramp_top

    # full-precision PDs: refit the final model exactly as run_round1.py does (deterministic) and score the new file
    from lending_game.features import income_segment, make_features
    from lending_game.models import fit_full
    past = pd.read_csv(HERE / "PastLoans.csv")
    new = pd.read_csv(args.applications)
    use_sex = strategy.get("use_sex", False)
    train = past[["sex", "employment", "married", "income", "signal2", "default"]]
    if args.augment_round1:
        r1 = pd.read_csv(HERE / "NewApplications_Lender2_Round1.csv").merge(
            pd.read_csv(HERE / "Round_1_report_new" / "Round1_Diagnostic_23.csv")[["id", "default"]], on="id")
        train = pd.concat([train, r1[train.columns]], ignore_index=True)
    model = fit_full(make_features(train, ["signal2"], use_sex), train["default"].to_numpy(), strategy["model_params"])
    if args.safe_extra:
        params["margins"] = {s: m_ + args.safe_extra for s, m_ in params["margins"].items()}
    p2 = model.predict_proba(make_features(new, ["signal2"], use_sex))[:, 1]
    seg = income_segment(new["income"])
    scored = pd.DataFrame({"id": new["id"].astype(int)})
    if args.adjusted_x is not None:
        seg_arr = np.asarray(seg)
        if args.floor_by_segment:
            fs = pd.read_csv(OUT / "curse_floor_by_segment.csv")
            adj = np.full(len(p2), np.nan)
            for s in fs["segment"].unique():
                sub = fs[fs["segment"] == s]
                mk = seg_arr == s
                adj[mk] = np.interp(p2[mk], sub["pd"], sub["adjusted_breakeven"])
            assert not np.isnan(adj).any(), "a segment has no curve in curse_floor_by_segment.csv"
        else:
            floor = pd.read_csv(OUT / "curse_floor.csv")
            adj = np.interp(p2, floor["pd"], floor["adjusted_breakeven"])
        tau = np.array([params["tau"][s] for s in seg])
        m = np.array([params["margins"][s] for s in seg])
        safe = p2 / (1 - p2) + m
        band = np.minimum(adj + args.adjusted_x, args.cap)
        w = 1 / (1 + np.exp(-(p2 - tau - args.phase_delta) / args.phase_width))  # margin rises smoothly around tau + delta
        r = safe + w * (band - safe)
        params.update({"band": "adjusted break-even + X", "X": args.adjusted_x, "cap": args.cap,
                       "floor": "by income segment (curse_floor_by_segment.csv)" if args.floor_by_segment else "pooled (curse_floor.csv)",
                       "phase_delta": args.phase_delta, "phase_width": args.phase_width, "tail_rate": None, "ramp_from": None})
    else:
        r = Rule(p2, seg, seg).rates(params)
    rate = np.where(np.isfinite(r), np.round(r, 4), np.nan)

    sub = pd.DataFrame({"id": scored["id"].astype(int), "rate": rate})
    assert len(sub) == len(new) == 100_000 and (sub["id"].to_numpy() == new["id"].to_numpy()).all() and sub["id"].is_unique
    assert np.nanmax(rate) <= 1.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False, na_rep="")
    import hashlib
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    with open(out.with_name(out.stem + "_params.json"), "w", encoding="utf-8") as f:
        json.dump({"rule": params, "model_params": strategy["model_params"], "use_sex": use_sex,
                   "applications": str(args.applications), "training": "PastLoans.csv" + (" + round-1 applicants with observed defaults" if args.augment_round1 else ""),
                   "source": "model refitted (full-precision PDs); " + (
                       "curse_floor_by_segment.csv" if args.floor_by_segment else "curse_floor.csv") + " for the adjusted break-even",
                   "command": " ".join(sys.argv), "sha256": sha}, f, indent=2)

    tau = np.array([params["tau"][s] for s in seg])
    band = p2 > tau
    print(f"wrote {out}  ({len(sub)} rows, {int(np.isnan(rate).sum())} without offer, max rate {np.nanmax(rate):.4f})  sha256 {sha[:16]}...")
    print("rate buckets (all offers):\n" + pd.cut(pd.Series(rate), [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.9889, 1.0]).value_counts().sort_index().to_string())
    print(f"normal offers: {int((~band).sum())} at {np.nanmin(rate[~band]):.4f}-{np.nanmax(rate[~band]):.4f}; "
          f"band offers: {int(band.sum())} at {np.nanmin(rate[band]):.4f}-{np.nanmax(rate[band]):.4f}")
    if args.ramp_from is not None:
        print("band rates by bucket:\n" + pd.cut(pd.Series(rate[band]), [0, 0.65, 0.70, 0.75, 0.80, 1.0]).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
