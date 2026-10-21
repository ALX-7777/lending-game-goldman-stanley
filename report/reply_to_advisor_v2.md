# Lending Game — reply to the review of the second brief

*Team "Lender 2", 14 October 2026. Answers your review of `handover_for_advisor_v2.md`, point by point, with the checks we ran and the corrections made. One point (income) led to a full analysis whose result the team decided not to put in the stage-1 file; it is documented here and kept for stage 2. Files are listed at the end.*

## Summary

You were right on all eight points. Two changed numbers in the documents (the file counts, the logistic weights), three changed wording (invariance, "never last", p_full), one changed the simulation (500 draws instead of 120, with standard errors). The income point was the material one: the selected default rate does depend on income at a given PD, the dependence is real out of sample, and a file priced on income-specific curves earns 0.46 M€ more in our simulation with half the probability of finishing last. The team nevertheless decided to **keep the pooled curve for stage 1** (reasons in Section 7); the submitted file is unchanged (SHA-256 `5aba4d61…`). The margin X = 0.05 is now selected cleanly by the declared criterion (Section 9).

## 1. "The selection is invariant to any common increasing transformation" — false with the 2% preference

Agreed: with the preference, two points of rate are a different PD gap under a steep rule and under a flat rule, so the set of applicants we win is not literally the same. We recomputed π̃(PD) under five other common rules — the submitted rule itself applied by all three lenders, the lecture formula p̂ + 3%, twice the break-even + m(s), break-even + 10%, and break-even + m(s) without preference (`analysis/curse_sensitivity.py`). Over 5–25% PD the curve moves by at most 1.3–1.7 points of default depending on the rule, 0.3 on average; a constant shift of the rule leaves it exactly unchanged. The claim is now stated as approximate, with these numbers, and the note says what π̃ does depend on: the rivals' models being about as informative as ours.

## 2. The counts exceeded 100,000

Correct. The file has 54,418 applicants below their segment threshold and 45,582 above it (100,000). The figure 65,145 was wrong. Both documents now carry 54,418 / 45,582.

## 3. The logistic weight at τ + 3 points is 0.88, not 0.95

Correct: w = 1/(1 + exp(−(p̂ − τ − 0.01)/0.01)) gives 0.05 at τ − 2 pt, 0.27 at τ, 0.50 at τ + 1, 0.88 at τ + 3, 0.95 at τ + 4. The sentence now reads that way.

## 4. "Never finishing last" contradicted the table (6% / 2%)

Agreed; "never" is gone. The text now gives the estimated probabilities with their standard errors (Section 9).

## 5. X = 0.05 is a constrained choice

Agreed, and the note now says so explicitly: the unconstrained maximum of the prior-weighted mean lies at a larger X, and X = 0.05 is the largest margin that satisfies the declared limit P(last) ≤ 3%. With 500 draws this is no longer a borderline statement: 1.6% ± 0.3 at 0.05, 3.4% ± 0.5 at 0.075 (Section 9).

## 6. 120 draws are weak for probabilities of 1–3%

Agreed. The final pass uses 500 rival pairs per profile (3,500 markets), the same draws for every schedule, and reports standard errors; P(last) is now resolved to ±0.2–0.5 points. A paired comparison on the same draws separates two schedules to ±0.02 M€.

## 7. Condition π̃ on income

This was the material point. At the same PD the pool we win is not the same across segments (raw one-point bins, weights = exact-type win probabilities, target = full-information PD; realized defaults in brackets):

| our PD | income 0 | 1–5k | 5–10k | 10–20k | pooled |
|---|---|---|---|---|---|
| 10–11% | 10.6% [12.1] (206 expected wins) | 11.8% [12.7] (163) | 12.9% [13.2] (1,180) | 16.7% [9.3] (65) | 12.5% |
| 12–13% | 13.5% [13.5] (1,196) | 13.9% [13.9] (365) | 16.1% [16.6] (722) | 26.4% [24.3] (12) | 14.8% |
| 16–17% | 19.9% [21.4] (260) | 22.8% [25.4] (81) | 26.0% [25.8] (249) | — | 22.0% |

**Mechanism** (checked, not assumed): the curse at a given PD is governed by how unusual that PD is for the segment, i.e. by how often we win. At 12–13% PD the zero-income segment (average PD 20%) has rivals estimating these applicants at 17% on average; we win 69% of them, including every case where the rivals' signals are neutral — little selection. The 10–20k segment (average PD 6%) has rivals estimating them at 9%; we win 4%, only when both rivals' signals are bad as well. The hidden-information dispersion σ(p_full − p̂), largest at zero income, is not what drives the curse at a fixed PD.

**Model.** Not a free curve per segment: above 20% PD (5–10k) or 12% PD (10–20k) there are no wins to fit on, and a free quadratic extrapolated to absurd values. On the logit scale each segment's curve is the pooled curve shifted by a segment-specific amount linear in logit(PD):

```
logit π̃_s(p) = logit π̃_pooled(p) + α_s + β_s · (logit p − logit 0.12)
```

(α_s, β_s) fitted by weighted least squares on the probability scale (a logit-scale fit targets E[logit p_full] and under-states the mean by Jensen; we checked: 0.5–1 point low) over the PD range from three points below the segment threshold up to the last one-point bin with ≥ 20 expected wins: 9–28% at income 0 (5,399 expected wins), 9–19% at 1–5k (1,746), 7–20% at 5–10k (8,160), 5–12% at 10–20k (3,786). Beyond the range the shift is held constant; the curve is floored at p̂; 20–40k and 40k+ keep the pooled curve (no wins at their 25% threshold). Fitted: α = −0.12, −0.08, +0.09, +0.33 and β = +0.07, +0.34, +0.34, +0.43 for income 0, 1–5k, 5–10k, 10–20k (`analysis/curse_floor_by_segment.py`, output `curse_floor_by_segment.csv`, figure `figC_curse_by_segment.png`).

**Out of sample** (`analysis/curse_floor_by_segment_cv.py`): random halves, pooled curve and shifts re-estimated on one half, compared with the raw bins of the other half in the band region, then swapped.

| segment | bias of the pooled curve (pts) | bias of the segment curve (pts) | MAE pooled → segment |
|---|---|---|---|
| income 0 | +1.4 / +1.5 | −0.1 / +0.1 | 1.5 → 0.6–0.7 |
| 1–5k | +0.5 / +0.4 | −0.1 / +0.1 | 0.8–0.9 → 0.4–0.5 |
| 5–10k | −0.8 / −1.3 | +0.4 / −0.2 | 0.9–1.3 → 0.4–0.5 |
| 10–20k | −1.3 / −1.2 | −0.1 / +0.5 | 1.2–1.3 → 0.4–0.6 |

Shifts stable across halves (α within 0.01–0.09, β within 0.04–0.10); overall MAE 1.22 → 0.54 points.

**What it would change.** Adjusted break-even at 12 / 16 / 20 / 25% PD: income 0: 14.6 / 24.6 / 33.4 / 61.1%; 1–5k: 15.1 / 28.0 / 40.1 / 71.7%; 5–10k: 18.0 / 33.2 / 48.6 / 87.0%; 10–20k: 22.8 / 37.6 / 50.1 / 89.7%; pooled: 16.4 / 27.1 / 36.1 / 64.7%. The pooled rate (+5) falls below the segment break-even above ~15% PD at 5–10k and ~11% at 10–20k. An income-conditioned file changes 64,942 offers: band rates 1–3 points lower at income 0, 0.5 (10–12% PD) to 16 points (22–25% PD) higher at 5–10k, 1 to 13 points higher at 10–20k; 2,173 at the cap instead of 2,027. In the simulation (same draws) it adds +0.60 / +0.46 / +0.33 M€ (± 0.02) prior-weighted at X = 0.03 / 0.05 / 0.075; by profile at 0.05: −0.25 against "like us" rivals (modelled on the pooled curve: they take thinly profitable 5–10k loans), +0.78 smart, +1.18 clones, +1.86 best-responders, +0.09 mixed, +1.77 netters, +0.11 model-followers. P(last) 0.7% vs 1.6%, P(first) unchanged at 78%.

**Decision: the pooled curve stays in the stage-1 file.** The team preferred the single curve for round 1. The arguments on that side: the gain is 3% of the expected profit and comes from the profiles we weight less, with a small loss against the profile we consider most likely; the largest shifts (10–20k above 12% PD, ~2,300 offers) are extrapolated from bins of 12–65 expected wins; and one curve is easier to explain in the report and to revise once the rivals' actual prices are observed. The analysis, the curves and the two files (X = 0.05 and 0.075) are kept in `output/round1/alternatives/` as the first candidate for stage 2. We would like your view on whether this is the right call (Section 11).

## 8. Call p_full a diagnostic estimate

Done. The note defines p_full as "a full-information estimate from a model trained on all three signals (out of sample), a diagnostic estimate, not the truth", and keeps the realized-default checks beside it (the realized 95% upper bound is within 1–4 points of the expected curve up to 25% PD; realized-vs-expected profit agrees within ~0.5 M€ in aggregate; the realized defaults in the table of Section 7 follow the model values).

## 9. The margin X with 500 draws

Prior-weighted over the seven profiles (likeus 40 / smart 20 / clones 10 / best-responders 10 / mixed 10 / netters 5 / model-followers 5), 500 rival pairs per profile, same draws for every schedule:

| schedule | mean (s.e.) | P(first) | P(last) (s.e.) | 5th pct | "like us": mean / P(first) / P(last) |
|---|---|---|---|---|---|
| pooled, X = 0.03 | 14.2 (0.12) | 84% | 1.0% (0.3) | 7.3 | 11.6 / 72% / 2.6% |
| **pooled, X = 0.05 (submitted)** | **15.5 (0.13)** | **78%** | **1.6% (0.3)** | **7.9** | **12.4 / 62% / 3.6%** |
| pooled, X = 0.075 | 16.8 (0.15) | 70% | 3.4% (0.5) | 8.3 | 12.9 / 47% / 7.6% |
| segment, X = 0.02 | 14.2 (0.12) | 84% | 0.3% (0.2) | 7.3 | 11.2 / 69% / 0.8% |
| segment, X = 0.03 | 14.8 (0.12) | 83% | 0.3% (0.2) | 7.5 | 11.5 / 67% / 0.8% |
| segment, X = 0.05 | 15.9 (0.14) | 78% | 0.7% (0.2) | 7.9 | 12.1 / 58% / 1.8% |
| segment, X = 0.075 | 17.1 (0.15) | 70% | 2.7% (0.4) | 8.3 | 12.6 / 44% / 6.2% |
| segment, X = 0.10 | 18.1 (0.17) | 64% | 5.4% (0.6) | 8.6 | 12.8 / 34% / 12.6% |

On the pooled curve the declared criterion (max prior-weighted mean s.t. P(last) ≤ 3%) selects X = 0.05 without ambiguity: 0.075 fails at 3.4% ± 0.5, and the one-sided 95% bound at 0.05 is 2.1%. Expected rank over the prior: 1.24 at 0.05, 1.33 at 0.075, for +1.3 M€ (8%). On the segment curve the limit would be met at 0.075 at the point estimate (2.7% ± 0.4) but not with its margin (3.4%); had we adopted that curve, this would have been the question to settle.

## 10. Other things worth knowing

- The transition table (one-point step vs logistic widths) is unchanged (pooled curve, 100 pairs per profile).
- The cap check was also re-run on the segment curve (`cap_region.py 60 seg`): same conclusion (keep the cap; +0.09 M€ vs 0 vs +0.05).
- "Like us" rivals are modelled on the pooled curve + U(3, 25) points. If the real rivals condition on income, the segment file's +0.46 M€ edge shrinks toward zero; if they do not, our pooled file concedes the 5–10k and 10–20k loans above ~15% / ~11% PD at thin or negative margins to whoever prices lowest there.
- Errata to the v2 brief, for the record: "invariant" → approximately invariant (≤ 1.7 pt); 65,145 → 54,418; "0.95 three points above" → 0.88; "never finishing last" → 1.6% (500 draws); "120 pairs" → 500; "p_full = E[D | X, s1, s2, s3]" → a diagnostic full-information estimate.

## 11. Questions

1. Keeping the pooled curve for stage 1 leaves, by our own simulation, +0.46 ± 0.02 M€ and half the last-place risk on the table. Is the simplicity argument defensible, or would you adopt the income-conditioned curve now?
2. If adopted later: should the "like us" rivals be modelled with income-conditioned curves of their own (from their signals), and should the shift be held constant beyond the data range (our choice) or revert to the pooled curve there?
3. Is a last-place limit the right form of the criterion, or would you use an expected-rank objective? The two readings disagree at the margin only on the segment curve.
4. For the report (70% of the grade): "price the pool you win, not the population", with the measured curve and the income decomposition as the mechanism — right central exhibit?

## Files

`report/stage1_final_rule_note.tex` (rewritten; PDF `stage1_final_rule_note_v5.pdf`), `analysis/curse_sensitivity.py` (point 1), `analysis/curse_floor_by_segment.py` and `_cv.py` (point 7), `analysis/final_margin_choice_v2.py` (points 5–6, 9; `output/round1/final_margin_choice_v2.csv`, per-draw file `final_margin_choice_v2_draws.csv`, figure `fig13b_final_margin_v2.png`), `analysis/cap_region.py`, `make_submission.py` (option `--floor-by-segment`). Submitted file `output/round1/rates_round1.csv`, SHA-256 `5aba4d61…`, unchanged since 13 October; alternatives in `output/round1/alternatives/` (README inside).
