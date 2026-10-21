"""Which 'lender of last resort' rate should we use?

Two questions:
  1. Break-even: how bad is the pool nobody else lends to, and below which rate do we lose money on it?
  2. Competition: if a competitor also puts a safety net (at 1.00, 0.99, 0.95, 0.90, ...), how does
     each choice of OUR net rate fare?  (lower net = wins against their net, but earns less per
     loan and starts 'winning' terrible borrowers from competitors who price very high but do lend)
Run from the project folder after run_round1.py:  python analysis/net_rate_scenarios.py"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lending_game.competitors import price  # noqa: E402
from lending_game.features import income_segment  # noqa: E402
from lending_game.simulator import LOAN, US, draw_types, simulate  # noqa: E402
from lending_game.strategy import Rule  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "output" / "round1"

past = pd.read_csv(HERE / "PastLoans.csv")
y = past["default"].to_numpy()
n = len(past)
strategy = json.load(open(OUT / "strategy_round1.json"))
z = np.load(OUT / f"oos_probas_sex{int(strategy['use_sex'])}.npz", allow_pickle=True)
p1, p2, p3, pF = z["p1"], z["p2"], z["p3"], z["pF"]
seg = income_segment(past["income"])
rule = Rule(p2, seg, seg)
chosen = strategy["rule"]
seeds = [0, 1, 2]

# ---------------------------------------------------------------- 1. break-even of the abandoned pool
print("=== 1. The pool both competitors reject: default rate and break-even net rate r* = d/(1-d) ===")
rows = []
for t in [0.15, 0.20, 0.25, 0.30, 0.35]:
    pool = (p1 > t) & (p3 > t)
    d = y[pool].mean()
    rows.append({"competitors reject if PD >": t, "pool size": f"{pool.mean():.2%}", "realized default d": round(d, 3),
                 "break-even net rate d/(1-d)": round(d / (1 - d), 3) if d < 0.5 else "none (d > 50%)",
                 "profit/loan at 0.989": round((1 - d) * 9_890 - d * 10_000), "at 0.90": round((1 - d) * 9_000 - d * 10_000),
                 "at 0.80": round((1 - d) * 8_000 - d * 10_000), "at 0.70": round((1 - d) * 7_000 - d * 10_000)})
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------- 2. competition on the net
base = dict(rule="breakeven", margin=0.03, reject_above=0.25)
configs = {"base, no net": (base, base)}
for c in [1.0, 0.99, 0.95, 0.90, 0.85, 0.80, 0.70]:
    configs[f"base, both net {c:.2f}"] = ({**base, "net": c}, {**base, "net": c})
configs["base, only L1 nets 0.90"] = ({**base, "net": 0.90}, base)
configs["cautious(>20%), net 0.99"] = ({**base, "margin": 0.05, "reject_above": 0.20, "net": 0.99},) * 2
configs["loose(>30%), net 0.99"] = ({**base, "margin": 0.02, "reject_above": 0.30, "net": 0.99},) * 2
configs["no_reject (+3%, lend to all)"] = ({**base, "reject_above": None},) * 2
configs["aggressive (+1%, lend to all)"] = ({**base, "margin": 0.01, "reject_above": None},) * 2

our_nets = [0.989, 0.97, 0.949, 0.929, 0.899, 0.849, 0.799, 0.749, 0.699, 0.599, 0.499, None]
total, net_part, net_loans = {}, {}, {}
for cname, (a1, a3) in configs.items():
    r1, r3 = price(p1, **a1), price(p3, **a3)
    for net in our_nets:
        prm = copy.deepcopy(chosen)
        prm["tail_rate"] = net
        r2 = rule.rates(prm)
        tot, npart, nl = [], [], []
        for seed in seeds:
            res, choice = simulate(np.column_stack([r1, r2, r3]), draw_types(n, seed), y, pF, seed)
            won = choice == US
            at_net = won & (r2 == net) if net is not None else np.zeros(n, bool)
            tot.append(res[US]["profit_expected"] / 1e6)
            npart.append((LOAN * ((1 - pF[at_net]) * r2[at_net] - pF[at_net])).sum() / 1e6)
            nl.append(at_net.sum())
        key = f"{net:.3f}" if net is not None else "no offer"
        total.setdefault(key, {})[cname] = np.mean(tot)
        net_part.setdefault(key, {})[cname] = np.mean(npart)
        net_loans.setdefault(key, {})[cname] = np.mean(nl)

T = pd.DataFrame(total).T
N = pd.DataFrame(net_part).T
L = pd.DataFrame(net_loans).T
print("\n=== 2a. TOTAL expected profit (M EUR), rows = OUR net rate, columns = what competitors do ===")
print(T.round(2).to_string())
print("\n=== 2b. Profit coming from the net loans only (M EUR) ===")
print(N.round(2).to_string())
print("\n=== 2c. Number of loans made at our net rate ===")
print(L.round(0).astype(int).to_string())
print("\n=== Summary over the competitor cases above (total profit, M EUR) ===")
print(pd.DataFrame({"mean": T.mean(1), "min": T.min(1), "mean if someone nets (0.70-1.00)": T[[c for c in T.columns if "net" in c and "no net" not in c]].mean(1),
                    "mean if nobody nets": T[["base, no net", "no_reject (+3%, lend to all)", "aggressive (+1%, lend to all)"]].mean(1)}).round(2).to_string())
