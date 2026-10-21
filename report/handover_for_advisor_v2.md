# Lending Game — second brief for the advisor: the final stage-1 rule

*Team "Lender 2", 14 October 2026. Follows the first brief (`handover_for_advisor.md`, which describes the game, the data and all earlier experiments) and your review (`strategy_methodology_submission_advice.md`). This document describes what we did with your advice, the rule we are now submitting, why, and what we would like you to challenge.*

---

## 0. What changed since your review

Done as you asked: the evaluator now integrates the three applicant types exactly (weight 1/3 each) and splits exact ties evenly (with an adverse-tie variant; the final schedules create no exact ties, so both agree); every candidate is evaluated after 4-decimal rounding; the submission CSV is built from full-precision PDs (model refitted deterministically), with a SHA-256 checksum, the exact parameters and the reproduction command stored beside it; a decision criterion was declared before the final runs; the thirteen factual corrections to the first brief were accepted (two were genuine errors: the Monte Carlo rank ranges).

Not done: the stage-1 price experiment (we have not yet asked the instructor what the stage-1 feedback contains), the outer-fold policy evaluation (judged second-order; a paired bootstrap is planned), the fintech name.

**The rule itself changed**, for a reason that came from the team, not from the simulations: our stated belief is that the two rival teams will be curse-aware but will *not* use a lender-of-last-resort rate — they will price the risky band on their model with a safety margin. Against such rivals, every schedule we had considered (flat net 0.828, ramps 0.45–0.65 or 0.65–0.828) finishes last 85–97% of the time, because the rivals take the band at 25–45% and make money with it. So we replaced the ramp by a schedule derived from a measured quantity: the break-even of the applicants we actually win.

## 1. The rule we are submitting

Segments by income: 0 / 1–5k / 5–10k / 10–20k / 20–40k / 40k+. Margins m(s) = 4.0 / 3.5 / 3.5 / 3.0 / 3.0 / 3.0 %. Thresholds τ(s) = 12 / 12 / 10 / 8 / 25 / 25 % PD.

For applicant i with our PD estimate p̂ (boosted trees on employment, married, income, log income, signal2, "signal = 0" flag; OOS AUC 0.735, calibrated within 0.4 pt per decile):

```
r_safe(p̂) = p̂/(1−p̂) + m(s)
r_band(p̂) = min( π̃(p̂)/(1−π̃(p̂)) + X , 0.989 ),      X = 0.05
r(p̂)      = r_safe + w(p̂) · (r_band − r_safe),      w(p̂) = 1 / (1 + exp(−(p̂ − τ(s) − 0.01)/0.01))
```

π̃(p̂) is the **selection-adjusted default rate**: the default rate of the applicants we win when every lender prices on its own model (Section 2). w is a logistic weight ≈ 0.05 two points below the threshold, 0.5 one point above, 0.95 three points above — the margin rises smoothly from ~2 points over the adjusted break-even (the competitive level on the safe part) to X = 5 points in the band. Rates rounded to 4 decimals; everyone gets an offer.

The rule at a glance (segment 5–10k, τ = 10%, m = 3.5%):

| PD | plain break-even | adjusted break-even | margin over adjusted | **rate** |
|---|---|---|---|---|
| 6% | 6.4% | 7.5% | 2.4 pt | **9.9%** |
| 8% | 8.7% | 10.6% | 1.8 | **12.4%** |
| 10% | 11.1% | 13.7% | 2.0 | **15.7%** |
| 12% | 13.6% | 16.4% | 3.9 | **20.3%** |
| 14% | 16.3% | 21.2% | 4.7 | **25.9%** |
| 16% | 19.0% | 27.1% | 4.9 | **32.0%** |
| 20% | 25.0% | 36.1% | 5.0 | **41.1%** |
| 25% | 33.3% | 64.7% | 5.0 | **69.7%** |
| ≥ 27.3% | ≥ 37.5% | > 100% | — | **98.9% (cap)** |

The file: 100,000 offers from 3.04% to 98.9%; 65,145 applicants below their segment threshold (3–19%), 45,582 priced on the adjusted break-even (12.8–98.9%), 1,946 of them at the cap. SHA-256 `5aba4d61…`. Command: `python make_submission.py --adjusted-x 0.05 --cap 0.989 --phase-delta 0.01 --phase-width 0.01`.

## 2. Where π̃ comes from (the break-even "given that we win")

On the past loans all three signals are visible, so the selection can be reproduced. For each past loan i: p̂_{k,i} = E[D | X, s_k] out-of-sample for k = 1, 2, 3; p_full,i = E[D | X, s₁, s₂, s₃]. Every lender quotes p̂_k/(1−p̂_k) + m(s) — the same rule on its own information. For each applicant type t (one third each; the preferred lender's rate counts 0.02 lower), we win iff our effective rate is the lowest; ω_i = the average over t of that indicator = the probability that applicant i takes our offer. Per one-point PD bin B:

```
π̃(B)      = Σ_{i∈B} ω_i p_full,i / Σ_{i∈B} ω_i            (expected; used in the rule, smoothed with a 3-bin rolling mean)
π̃_real(B) = Σ ω_i D_i / Σ ω_i ;   π̃_95(B) = π̃_real + 1.645 · sqrt(π̃_real(1−π̃_real)/Σ ω_i)
```

| Our PD | share we win | π̃ | π̃_95 | penalty π̃ − p̂ | plain BE | adjusted BE |
|---|---|---|---|---|---|---|
| 5% | 44% | 5.1% | 5.5% | +0.6 pt | 4.7% | 5.3% |
| 8.5% | 39% | 10.2% | 11.3% | +1.7 | 9.2% | 11.3% |
| 10.5% | 38% | 12.7% | 14.2% | +2.1 | 11.7% | 14.5% |
| 12.5% | 44% | 14.4% | 15.8% | +1.9 | 14.3% | 16.9% |
| 14.5% | 29% | 18.2% | 19.9% | +3.7 | 16.9% | 22.2% |
| 16.5% | 17% | 22.9% | 26.7% | +6.4 | 19.7% | 29.6% |
| 20.5% | 11% | 26.9% | 30.1% | +6.5 | 25.7% | 36.8% |
| 24.5% | 13% | 35.5% | 39.7% | +11.0 | 32.5% | 55.0% |
| 26.5% | 4% | 46.8% | 58.2% | +20.2 | 36.2% | 88.1% |
| ≥ 28% | ≤ 1% | > 50% | — | > 20 | ≥ 40% | none below the cap |

Properties we rely on: (i) the selection is invariant to any common increasing transformation of the rule (if all lenders apply the same increasing function to their own PD, the same applicants are won), so π̃ does not depend on the rivals' margin — only on their models being about as informative as ours; (ii) the 95% bound is within 1–4 points of the expected curve up to 25% PD; (iii) above ~27% PD the competitive pool defaults > 50%, so the rule stops competing and caps.

## 3. Why X = 0.05, and the transition

**Monte Carlo over rival strategies** (exact evaluator, 120 rival pairs per profile). Profiles and our prior: *like us, no net* — rivals pricing the band at their own adjusted break-even + a margin ~U(3, 25) points, never netting (40%); *smart* — curse-aware rivals using a net, a ramp, a large margin, a data-driven schedule, refusal or a compromise (20%); *clones* of our method with other parameters (10%); *best-responders* undercutting our schedule (10%); *mixed*, including naive behaviours (10%); *netters* (5%); *model-followers*, model + 2–6% for everyone (5%).

| Band schedule | prior-weighted mean | P(first) | P(last) | "like us" mean / P(first) / P(last) |
|---|---|---|---|---|
| adjusted BE, X = 0 | 11.9 M€ | 79% | 2% | 9.7 / 62% / 4% |
| X = 0.02 | 13.6 | 82% | 1% | 11.0 / 70% / 2% |
| **X = 0.05** | **15.7** | **75%** | **3%** | **12.4 / 55% / 6%** |
| X = 0.075 | 17.1 | 70% | 5% | 12.9 / 47% / 11% |
| X = 0.10 | 18.3 | 65% | 7% | 13.1 / 36% / 17% |
| X = 0.15 | 19.6 | 54% | 15% | 12.2 / 14% / 35% |
| ramp 0.45→0.65 | 22.2 | 42% | 41% | 8.2 / 2% / 91% |
| flat net 0.828 (first file) | 11.0 | 13% | 68% | 6.7 / 0% / 98% |

**Declared criterion**: maximize the prior-weighted mean subject to P(last) ≤ 3% → X = 0.05. We did not use your "within 3% of the best mean, then P(first)" rule because here it selects the ramp (its mean is 10% above everything else, made of jackpots against netters) despite a 41% probability of finishing last; since the grade counts the ranking, we think a last-place constraint is the right gate. We'd like your view on this.

**Transition** (one-point step vs logistic widths; 100 pairs per profile): step 16.3 M€ / first 82% / last 1%; logistic centre τ+1 pt, scale 1 pt (≈ 8–14% PD, submitted) 15.8 / 82% / 1%; centre τ+2, scale 1.5: 15.2; centre τ+3, scale 2.5: 14.4. The step produced a 5-point jump in the rate between 10% and 11% PD with no economic event behind it; the one-point logistic removes it for 0.5 M€ (inside the noise). Economic reading: the margin rises as rivals disengage; their thresholds in every profile lie between 8% and 20% PD.

**The cap** (1,946 applicants with adjusted break-even > 100%): where we win any capped loans their expected default is 33–46% (break-even 50–85% < 98.9%); prior-weighted +0.10 M€ vs 0 for refusing them. Kept.

## 4. Robustness of the final rule across profiles (mean M€ / P(first) / P(last))

| Schedule | like us | smart | clones | best-responders | mixed | netters | model-followers |
|---|---|---|---|---|---|---|---|
| **adjusted BE + 0.05 (submitted)** | 12.4 / 55% / 6% | 20.9 / 82% / 2% | 16.5 / 88% / 0% | 26.0 / 100% / 0% | 7.2 / 96% / 0% | 24.3 / 99% / 0% | 7.7 / 59% / 0% |
| ramp 0.45→0.65 | 8.2 / 2% / 91% | 35.7 / 63% / 7% | 35.2 / 90% / 0% | 42.8 / 55% / 10% | 9.1 / 74% / 3% | 55.4 / 100% / 0% | 6.3 / 34% / 38% |
| flat net 0.828 | 6.7 / 0% / 98% | 19.5 / 14% / 40% | 10.0 / 10% / 60% | 4.0 / 0% / 98% | 10.1 / 62% / 13% | 33.1 / 28% / 44% | 6.3 / 34% / 39% |

The submitted rule gives up the large upside of the ramp against netters and clones (24 vs 55 M€, 17 vs 35) in exchange for never finishing last in the worlds we consider likely.

## 5. Assumptions and weaknesses we see

1. **π̃ assumes rivals' models are as informative as ours** (our GBM on their signal). If their models are worse, the curse is smaller and we over-price the band; if better, the five-point margin is the cushion.
2. **The 'like us' profile is our construct** and carries 40% of the prior; its rivals' margins are uniform on 3–25 points — the choice of X depends directly on that distribution.
3. π̃ is estimated in one-point bins and smoothed; bins above 25% PD are thin (effective sample 3–30), which is why the 95% bound is noisy there. The rule only uses the expected curve, which is based on p_full and is smooth.
4. The safe part (margins 3–4%, thresholds) is unchanged from the first submission and was optimized on a different (hand-built) scenario set; we have not re-optimized it under the Monte Carlo profiles.
5. The game's no-outside-option rule is what makes the cap harmless; tie-breaking is unspecified (immaterial here: no exact ties).
6. Everything rests on expected profit computed with p_full as the truth; realized-vs-expected checks agree within ~0.5 M€ in aggregate.

## 6. Questions

1. Is "selection-adjusted break-even + a constant margin" the right object to price the band, and is X = 5 points the right order of magnitude? Would you make X depend on PD (e.g. proportional to the dispersion of rivals' prices) rather than constant?
2. Is our π̃ — selection under a symmetric rule — the right break-even to use when rivals may use *different* rules? Our invariance argument covers common monotone transformations; what does it miss (different thresholds, refusals, nets)?
3. The declared criterion (P(last) ≤ 3%) versus yours (within 3% of the best mean): which is more defensible given the grading text "both in absolute terms and in relative terms"?
4. The transition width (8–14% PD) and the cap: anything wrong with the reasoning or the measurement?
5. Should the safe-part margins/thresholds be re-optimized under the same Monte Carlo rivals, or left as they are?
6. Any failure mode of this rule we have not simulated? In particular, rivals whose models are much better or much worse than ours, or rivals pricing by employment rather than income.
7. For the report (70% of the grade): is "price the pool you win, not the population" plus the measured π̃ curve the right central exhibit?

Files: `report/stage1_final_rule_note.tex` (the note describing the rule, with the figures `output/round1/figures/figA_curse_and_rule.png` and `fig13_final_margin.png`), `analysis/curse_adjusted_breakeven.py` (π̃), `analysis/final_margin_choice.py` (X), `analysis/margin_transition.py`, `analysis/cap_region.py`, `make_submission.py`.
