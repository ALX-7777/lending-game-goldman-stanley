"""Band-price payoff matrix.

The only thing that matters about a competitor, for the risky band, is the price at which it serves the
band (if at all).  This script builds a grid of competitor 'worlds' (both competitors' band behaviour)
and a grid of our band-price schedules ('actions'), evaluates every (action, world) pair on the market
simulator (expected profit using the full-information PD, 3 applicant-type draws), and summarizes the
matrix with several decision criteria: worst case, uniform mean, prior-weighted mean, maximum regret,
and -- for a shortlist -- our rank among the three lenders (realized profits).
Confidence intervals and the across-world risk view are in band_payoff_ci.py (which imports this module).

Extra competitor behaviours can be dropped into analysis/ultra/archetypes/<slug>.py, each exposing
    NAME = "short label"            REGIME = "cheap" | "margin" | "net" | "reject" | "mixed" | "extra"
    def rates(p, ctx) -> np.ndarray  # p: the competitor's own PD estimate (n,), np.inf = no offer
    ALSO_VS_BASE = True              # optional: also pair it with a base competitor
ctx = dict(tau=our thresholds per applicant, m=our margins per applicant, seg=income segment labels,
           net=our net rate, n=number of applicants, be=lambda p: p/(1-p), below=lambda p: be(p)+m)

Run from the project folder after run_round1.py:  python analysis/band_payoff_matrix.py
Outputs: output/round1/band_matrix_{payoff,regret,summary,best_response,competitor_profit,rank}.csv
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lending_game.features import income_segment  # noqa: E402
from lending_game.simulator import LOAN, US, FastEvaluator, draw_types, simulate  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 40)
OUT = ROOT / "output" / "round1"
EXTRA_DIR = ROOT / "analysis" / "ultra" / "archetypes"

past = pd.read_csv(ROOT / "PastLoans.csv")
y = past["default"].to_numpy()
n = len(past)
strategy = json.load(open(OUT / "strategy_round1.json"))
z = np.load(OUT / f"oos_probas_sex{int(strategy['use_sex'])}.npz", allow_pickle=True)
pX, p1, p2, p3, pF = z["pX"], z["p1"], z["p2"], z["p3"], z["pF"]
seg = income_segment(past["income"])
employment = past["employment"].to_numpy()
sig_zero = {1: past["signal1"].to_numpy() == 0, 3: past["signal3"].to_numpy() == 0}
chosen = strategy["rule"]
NET = chosen["tail_rate"]
seeds = [0, 1, 2]
tau_arr = np.array([chosen["tau"][s] for s in seg])
m_arr = np.array([chosen["margins"][s] for s in seg])
PARANOID_TAU = {"0": 0.04, "1-5k": 0.08, "5-10k": 0.06, "10-20k": 0.06, "20-40k": 0.25, "40k+": 0.25}  # clone-aware re-optimization
paranoid_tau = np.array([PARANOID_TAU[s] for s in seg])


def be(p):
    return p / (1 - p)


def below(p):
    """How everybody prices the safe part: break-even + our segment margins (3-4%)."""
    return be(p) + m_arr


def ramp_frac(p, s=0.25, tau=None):
    t = tau_arr if tau is None else tau
    return np.clip((p - t) / np.maximum(s - t, 1e-9), 0, 1)


def logit_shift(p, shift):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + shift)))


# ------------------------------------------------------------------ competitor band behaviours
# Each returns a function of the competitor's OWN PD estimate p (and, when it needs it, the lender index k).
def w_margin(M):      # keeps lending to the band at model + M (curse-aware margin-adder)
    return lambda p, k=None: np.where(p <= tau_arr, below(p), np.minimum(be(p) + M, 1.0))


def w_reject(t):      # base-like: break-even + 3% up to PD t, refuses above
    return lambda p, k=None: np.where(p <= t, be(p) + 0.03, np.inf)


def w_net(X):         # clone-like: our thresholds, net X above
    return lambda p, k=None: np.where(p <= tau_arr, below(p), X)


def w_net_t(t, X):    # +3% up to its own PD t, net X above (a clone with a different threshold)
    return lambda p, k=None: np.where(p <= t, be(p) + 0.03, X)


def w_cap(c):         # +3% for everyone but refuses any rate above a cap c
    return lambda p, k=None: np.where(be(p) + 0.03 <= c, be(p) + 0.03, np.inf)


def w_everyone(m):    # lends to everyone at break-even + m
    return lambda p, k=None: np.minimum(be(p) + m, 1.0)


def w_slide(t):       # lecture formula p + 3%, refuses above t
    return lambda p, k=None: np.where(p <= t, p + 0.03, np.inf)


def w_pdmargin(k_):   # margin growing with the PD, everyone served
    return lambda p, k=None: np.minimum(be(p) + 0.03 + k_ * np.maximum(p - 0.05, 0), 1.0)


def w_ramp(L, H, tau=None):     # a ramp like ours
    return lambda p, k=None: np.where(p <= (tau_arr if tau is None else tau), below(p), L + (H - L) * ramp_frac(p, tau=tau))


def w_flat(r):        # the same rate for everyone, never refuses
    return lambda p, k=None: np.full(n, r)


def w_weak_model():   # base rule, but the PD ignores the competitor's own signal (X only)
    return lambda p, k=None: np.where(pX <= 0.25, be(pX) + 0.03, np.inf)


def w_zero_as_worst():  # base rule with a model that extrapolates signal == 0 as a bad value (+0.26 logit; -0.08 otherwise)
    def f(p, k=None):
        q = logit_shift(p, np.where(sig_zero[k], 0.26, -0.08))
        return np.where(q <= 0.25, be(q) + 0.03, np.inf)
    return f


def w_cherry():       # break-even + 1% if PD < 10%, refuses everyone else
    return lambda p, k=None: np.where(p < 0.10, be(p) + 0.01, np.inf)


def w_clone_pd(M):    # a clone whose net is min(0.828, model + M)
    return lambda p, k=None: np.where(p <= tau_arr, below(p), np.minimum(be(p) + M, NET))


def w_paranoid():     # the clone-aware re-optimized rule: thresholds 4-8% at low income, net 0.70
    return lambda p, k=None: np.where(p <= paranoid_tau, below(p), 0.70)


def w_refuse_group(group):  # refuses a visible group outright, base rule otherwise
    mask = employment == group
    return lambda p, k=None: np.where(mask, np.inf, np.where(p <= 0.25, be(p) + 0.03, np.inf))


WORLDS = {}  # name -> (regime, f_lender1, f_lender3)


def add_world(name, regime, f1, f3=None):
    WORLDS[name] = (regime, f1, f3 or f1)


for M in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]:
    add_world(f"both: model + {M:.2f} in band", "cheap" if M < 0.08 else "margin", w_margin(M))
for m in [0.01, 0.03, 0.05]:
    add_world(f"both: everyone at +{m:.2f}", "cheap", w_everyone(m))
for k_ in [0.5, 1.0, 2.0]:
    add_world(f"both: PD-growing margin k={k_}", "cheap" if k_ < 1 else "margin", w_pdmargin(k_))
for r in [0.08, 0.10, 0.12]:
    add_world(f"both: flat {r:.2f} for everyone", "cheap", w_flat(r))
add_world("both: base rule, model ignores own signal", "cheap", w_weak_model())
add_world("both: base rule, signal=0 treated as worst", "cheap", w_zero_as_worst())
add_world("both: lecture formula p+3%, reject >25%", "cheap", w_slide(0.25))
for M in [0.20, 0.30, 0.40]:
    add_world(f"both: clone, net = min(0.828, model + {M:.2f})", "margin", w_clone_pd(M))
for t in [0.12, 0.20, 0.25, 0.30]:
    add_world(f"both: reject above PD {t:.2f}", "reject", w_reject(t))
for c in [0.25, 0.35, 0.50]:
    add_world(f"both: max-rate cap {c:.2f}", "reject", w_cap(c))
add_world("both: cherry-picker (+1% if PD<10%, else refuse)", "reject", w_cherry())
add_world("both: refuse unemployed, base otherwise", "reject", w_refuse_group("unemployed"))
add_world("both: refuse students, base otherwise", "reject", w_refuse_group("student"))
for X in [0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.83, 0.84, 0.85, 0.90, 0.95, 0.99]:
    add_world(f"both: clone net {X:.2f}", "net", w_net(X))
for t in [0.15, 0.20]:
    for X in [0.80, 0.90]:
        add_world(f"both: +3% to PD {t:.2f}, net {X:.2f} above", "net", w_net_t(t, X))
for X in [0.90, 0.99]:
    add_world(f"both: +3% to 25%, net {X:.2f} above", "net", w_net_t(0.25, X))
add_world("both: paranoid clone (tau 4-8%, net 0.70)", "net", w_paranoid())
for L, H in [(0.60, 0.828), (0.70, 0.828), (0.70, 0.90), (0.75, 0.99), (0.80, 0.99), (0.50, 0.90)]:
    add_world(f"both: ramp {L:.2f}->{H:.3f}", "net", w_ramp(L, H))
# asymmetric pairs
add_world("model + 0.20 / reject >25%", "mixed", w_margin(0.20), w_reject(0.25))
add_world("model + 0.10 / everyone +3%", "mixed", w_margin(0.10), w_everyone(0.03))
add_world("clone net 0.90 / model + 0.10", "mixed", w_net(0.90), w_margin(0.10))
add_world("reject >20% / everyone +3%", "mixed", w_reject(0.20), w_everyone(0.03))
add_world("clone net 0.898 / max-rate cap 0.35", "mixed", w_net(0.898), w_cap(0.35))
add_world("model + 0.30 / clone net 0.95", "mixed", w_margin(0.30), w_net(0.95))
add_world("clone net 0.828 / flat 0.10", "mixed", w_net(0.828), w_flat(0.10))
add_world("clone min(0.828, model+0.30) / model ignores own signal", "mixed", w_clone_pd(0.30), w_weak_model())
for X in [0.99, 0.95, 0.898, 0.85, 0.828, 0.80, 0.75, 0.70, 0.65, 0.60]:
    add_world(f"clone net {X:.3f} / reject >25%", "mixed", w_net(X), w_reject(0.25))
for a, b in [(0.99, 0.75), (0.898, 0.60), (0.80, 0.90), (0.83, 0.85), (0.80, 0.85), (0.75, 0.99)]:
    add_world(f"two clones, nets {a:.3f} / {b:.3f}", "net", w_net(a), w_net(b))

# extra behaviours contributed as modules
ctx = dict(tau=tau_arr, m=m_arr, seg=seg, net=NET, n=n, be=be, below=below)
if EXTRA_DIR.exists():
    for path in sorted(EXTRA_DIR.glob("*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            f = (lambda mod: (lambda p, k=None: np.asarray(mod.rates(p, ctx), dtype=float)))(mod)
            add_world(f"both: {mod.NAME}", getattr(mod, "REGIME", "extra"), f)
            if getattr(mod, "ALSO_VS_BASE", True):
                add_world(f"{mod.NAME} / reject >25%", "mixed", f, w_reject(0.25))
        except Exception as e:  # a broken module must not kill the run
            print(f"[warning] could not load {path.name}: {e}")

cases = {name: (f1(p1, 1), f3(p3, 3)) for name, (regime, f1, f3) in WORLDS.items()}
regime_of = pd.Series({name: regime for name, (regime, _, _) in WORLDS.items()})

# ------------------------------------------------------------------ our band-price schedules
def ours(band):
    return np.where(p2 > tau_arr, band, below(p2))


ACTIONS = {}
for X in [0.70, 0.75, 0.80, 0.828, 0.85, 0.90, 0.95, 0.989]:
    ACTIONS[f"flat net {X:.3f}"] = ours(np.full(n, X))
for L in [0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80]:
    for H in [0.828, 0.85, 0.90, 0.989]:
        if L < H:
            ACTIONS[f"ramp {L:.2f}->{H:.3f}"] = ours(L + (H - L) * ramp_frac(p2))
for M in [0.05, 0.10, 0.15, 0.20, 0.30]:
    ACTIONS[f"model + {M:.2f}"] = ours(np.minimum(be(p2) + M, 1.0))
for M in [0.10, 0.15, 0.20]:
    for s in [0.15, 0.18]:
        ACTIONS[f"hybrid: model + {M:.2f} to PD {s:.2f}, then net 0.828"] = ours(np.where(p2 <= s, be(p2) + M, NET))
r70 = 0.70 + (NET - 0.70) * ramp_frac(p2)
ACTIONS["ramp 0.70->0.828, then 0.989 above PD 0.30"] = ours(np.where(p2 > 0.30, 0.989, r70))
ACTIONS["ramp 0.70->0.828, no offer above PD 0.35"] = ours(np.where(p2 > 0.35, np.inf, r70))
ACTIONS["ramp 0.70->0.90 at income 0, 0.70->0.828 elsewhere"] = ours(np.where(seg == "0", 0.70 + (0.90 - 0.70) * ramp_frac(p2), r70))
ACTIONS["no offer in band"] = ours(np.full(n, np.inf))
ACTIONS["model + 0.03 (naive)"] = ours(np.minimum(be(p2) + 0.03, 1.0))

REGIMES = ["cheap", "margin", "net", "reject", "mixed", "extra"]
PRIORS = {  # probability of each regime; uniform over the worlds inside a regime
    "skeptical":    dict(cheap=0.50, margin=0.10, net=0.10, reject=0.15, mixed=0.10, extra=0.05),
    "moderate":     dict(cheap=0.35, margin=0.15, net=0.15, reject=0.15, mixed=0.12, extra=0.08),
    "margin-heavy": dict(cheap=0.25, margin=0.35, net=0.10, reject=0.10, mixed=0.12, extra=0.08),
    "net-heavy":    dict(cheap=0.25, margin=0.10, net=0.35, reject=0.10, mixed=0.12, extra=0.08),
}
SHORTLIST = ["flat net 0.828", "flat net 0.700", "flat net 0.989", "ramp 0.70->0.828", "ramp 0.65->0.828", "ramp 0.60->0.828",
             "ramp 0.50->0.828", "ramp 0.70->0.850", "ramp 0.70->0.900", "ramp 0.60->0.900", "ramp 0.60->0.989",
             "ramp 0.70->0.828, then 0.989 above PD 0.30", "ramp 0.70->0.828, no offer above PD 0.35",
             "ramp 0.70->0.90 at income 0, 0.70->0.828 elsewhere", "model + 0.15", "model + 0.20",
             "hybrid: model + 0.15 to PD 0.15, then net 0.828", "no offer in band", "model + 0.03 (naive)"]
RANKLIST = ["flat net 0.828", "ramp 0.70->0.828", "ramp 0.60->0.828", "ramp 0.60->0.900", "model + 0.20",
            "hybrid: model + 0.15 to PD 0.15, then net 0.828"]


def world_weights(prior: dict) -> pd.Series:
    """Probability of each world under a prior over regimes (uniform inside a regime)."""
    present = [r for r in REGIMES if (regime_of == r).any()]
    tot = sum(prior[r] for r in present)
    w = pd.Series(0.0, index=regime_of.index)
    for r in present:
        members = regime_of.index[regime_of == r]
        w[members] = prior[r] / tot / len(members)
    return w


def safe_to_csv(df, path):
    """Excel sometimes holds one of the output files open; fall back to a sibling name instead of crashing."""
    try:
        df.to_csv(path)
    except PermissionError:
        alt = path.with_name(path.stem + "_new" + path.suffix)
        df.to_csv(alt)
        print(f"[warning] {path.name} is locked (open in Excel?); written to {alt.name} instead")


if __name__ == "__main__":
    ev = FastEvaluator(cases, n, seeds, pF)
    P = pd.DataFrame({a: {w: float(np.mean(v)) / 1e6 for w, v in ev.profits(r2).items()} for a, r2 in ACTIONS.items()})
    P = P.loc[list(WORLDS)]  # rows = worlds, columns = actions
    R = P.max(axis=1).to_frame().values - P  # regret

    present = [r for r in REGIMES if (regime_of == r).any()]
    S = pd.DataFrame(index=P.columns)
    S["worst world"] = P.min()
    S["mean (uniform)"] = P.mean()
    S["max regret"] = R.max()
    for r in present:
        S[f"mean | {r}"] = P.loc[regime_of == r].mean()
    for pname, prior in PRIORS.items():
        S[f"prior {pname}"] = (P.mul(world_weights(prior), axis=0)).sum()
    S["worlds with a loss"] = (P < 0).sum()

    best = pd.DataFrame({"regime": regime_of, "best action": P.idxmax(axis=1), "best payoff": P.max(axis=1),
                         "flat net 0.828": P["flat net 0.828"], "ramp 0.70->0.828": P["ramp 0.70->0.828"],
                         "ramp 0.60->0.828": P["ramp 0.60->0.828"], "model + 0.15": P["model + 0.15"]})

    # realized profits of all three lenders and our rank, for the rank shortlist (3 type draws)
    rank_rows, comp_rows = [], {}
    for name, (regime, f1, f3) in WORLDS.items():
        r1, r3 = cases[name]
        for a in RANKLIST:
            ranks, ours_p, l1, l3 = [], [], [], []
            for seed in seeds:
                res, _ = simulate(np.column_stack([r1, ACTIONS[a], r3]), draw_types(n, seed), y, pF, seed)
                profits = [res[k]["profit_realized"] for k in range(3)]
                ranks.append(1 + sum(pk > profits[US] for k, pk in enumerate(profits) if k != US))
                ours_p.append(profits[US] / 1e6); l1.append(profits[0] / 1e6); l3.append(profits[2] / 1e6)
            rank_rows.append({"world": name, "regime": regime, "action": a, "our realized (M)": np.mean(ours_p),
                              "lender 1 (M)": np.mean(l1), "lender 3 (M)": np.mean(l3), "mean rank": np.mean(ranks),
                              "P(rank 1)": np.mean([r == 1 for r in ranks])})
            if a == "flat net 0.828" and name.startswith("both"):
                comp_rows[name] = {"regime": regime, "lender 1 profit": np.mean(l1), "lender 3 profit": np.mean(l3),
                                   "our profit": np.mean(ours_p)}
    RK = pd.DataFrame(rank_rows)
    C = pd.DataFrame(comp_rows).T

    safe_to_csv(P, OUT / "band_matrix_payoff.csv")
    safe_to_csv(R, OUT / "band_matrix_regret.csv")
    safe_to_csv(S, OUT / "band_matrix_summary.csv")
    safe_to_csv(best, OUT / "band_matrix_best_response.csv")
    safe_to_csv(C, OUT / "band_matrix_competitor_profit.csv")
    safe_to_csv(RK.set_index(["world", "action"]), OUT / "band_matrix_rank.csv")

    print(f"{len(WORLDS)} worlds ({dict(regime_of.value_counts())}), {len(ACTIONS)} actions\n")
    print("=== A. Shortlisted actions: mean expected profit (M EUR) by competitor regime, worst world, max regret, priors ===")
    print(S.loc[SHORTLIST].round(2).to_string())
    print("\n=== B. Top 5 actions by criterion ===")
    for crit in ["worst world", "max regret", "mean (uniform)"] + [f"prior {p}" for p in PRIORS]:
        asc = crit == "max regret"
        top = S[crit].sort_values(ascending=asc).head(5)
        print(f"{crit:22s}: " + " | ".join(f"{a} ({v:.2f})" for a, v in top.items()))
    print("\n=== C. Best response per world (the stage-2 playbook) ===")
    print(best.round(2).to_string())
    print("\n=== D. Competitors' realized profit when we play the flat net (is their behaviour sustainable?) ===")
    print(C.round(2).to_string())
    print("\n=== E. Our rank among the three lenders (realized profits): mean rank and P(rank 1) by regime ===")
    print(RK.pivot_table(index="regime", columns="action", values="mean rank", aggfunc="mean")[RANKLIST].round(2).to_string())
    print()
    print(RK.pivot_table(index="regime", columns="action", values="P(rank 1)", aggfunc="mean")[RANKLIST].round(2).to_string())
    print("\noverall mean rank: " + ", ".join(f"{a}: {RK[RK.action == a]['mean rank'].mean():.2f}" for a in RANKLIST))
