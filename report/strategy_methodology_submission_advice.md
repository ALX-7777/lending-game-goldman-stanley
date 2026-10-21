# Lending Game: strategy, methodology, and submission recommendations

*Prepared as a self-contained handoff for an AI or team member working on the HEC Paris Lending Game. Date: 14 October 2026.*

## 1. Objective and current position

The objective is to maximize both absolute profit and relative rank in the two lending rounds while producing the strongest possible three-page explanation for the instructor.

The verified assignment weights are 10% for Stage-1 performance, 20% for Stage-2 performance, and 70% for the written report. Performance is assessed using both profit level and rank. This makes a modest, controlled Stage-1 learning investment potentially valuable, provided its expected cost is explicitly bounded.

The project already contains unusually strong work:

- the economically correct break-even rate, \(p/(1-p)\);
- out-of-sample probability-of-default estimates for every lender's information set;
- an explicit treatment of the winner's curse caused by competitors' private signals;
- market simulations covering many possible competitor behaviours;
- income-dependent margins and lending thresholds;
- analysis of lender-of-last-resort or fallback pricing.

The main safe-region policy is convincing. The largest remaining uncertainty concerns pricing the risky "band." The current generated submission still uses a flat rate of 0.828 for that band. A lower ramp appears more robust, but the final endpoints must not be chosen until a material tie-breaking bias in the evaluator has been corrected.

## 2. Executive recommendation

The work should proceed in this order:

1. Correct tie handling and integrate borrower types analytically.
2. Freeze a short list of pricing policies and rerun them under identical assumptions.
3. Select a policy using a declared profit-versus-rank criterion.
4. If applicant-level Stage-1 feedback will be available, reserve a small stratified sample for randomized price exploration.
5. Freeze and validate the exact submission CSV and its reproducibility information.
6. Use the actual Stage-1 outcome, rather than further subjective rival scenarios, to construct the Stage-2 best response.
7. Build the final report around one thesis: competition makes acceptance informative about credit risk.

If a submission had to be made immediately, before the corrected rerun, the provisional recommendation would be:

- retain the current safe-region margins and thresholds;
- replace the flat 0.828 band price with a ramp from 0.45 at the applicable threshold to 0.65 at a predicted PD of 25%, constant at 0.65 above 25%;
- describe 0.45 and 0.65 as robust hedge parameters, not structural or statistically precise optima.

This immediate recommendation is provisional because current policy comparisons do not all use the same tie-breaking rule.

## 3. Critical issue: the evaluator currently favours us on exact ties

This is the highest-priority methodological correction.

In `lending_game/simulator.py`, the full simulator adds random perturbations to all three lenders. By contrast, `FastEvaluator` adds a small positive perturbation only to competitors and then declares that we win whenever our effective rate is strictly below the perturbed competitor minimum. Therefore, an exact unperturbed tie is effectively awarded to us.

Other analysis scripts sometimes use `argmin`, which awards certain ties to lender 1 instead. Consequently, different tables use different tie conventions.

This is material, not cosmetic:

- one diagnostic with a competitor using the same flat 0.828 band price produced roughly 12,700 exact ties;
- the favourable `FastEvaluator` convention overstated our expected profit by approximately EUR 27.6-27.8 million in that test;
- among the large payoff grid, the worst observed tie-related bias was approximately EUR 38.8 million;
- ramps also create exact ties at clipped endpoints, with identified distortions of roughly EUR 0.2-2.6 million.

### Required correction

For expected-profit evaluation, do not simulate the borrower type. For every applicant:

1. evaluate the choice under type 1, type 2, and type 3;
2. give each type weight \(1/3\);
3. identify all lenders attaining the minimum effective rate;
4. under fair random tie-breaking, allocate win probability \(1/n_{\min}\) to every tied lender;
5. calculate expected profit using that fractional win probability;
6. repeat the analysis under an adverse convention in which we lose every exact tie.

All policies must be evaluated after the same rounding used in the submitted CSV. Four-decimal rounding can create ties that do not exist in full-precision arrays.

The simulator should also reject invalid inputs:

- finite rates must lie in \([0,1]\);
- no `NaN` or negative infinity may enter the simulator;
- `+inf` should be the only internal representation of no offer;
- borrower types must be in the valid set;
- PD inputs must lie in \([0,1]\).

No final band-price decision should be made from the existing ranking tables before this correction.

## 4. Safe-region policy to retain

For an applicant with our estimated PD \(\hat p_i\), income segment \(s_i\), margin \(m(s_i)\), and threshold \(\tau(s_i)\), retain:

\[
r_i=\frac{\hat p_i}{1-\hat p_i}+m(s_i),
\qquad \hat p_i\leq \tau(s_i).
\]

Current margins:

| Income segment | Margin |
|---|---:|
| income = 0 | 0.040 |
| EUR 1-5k | 0.035 |
| EUR 5-10k | 0.035 |
| EUR 10-20k | 0.030 |
| EUR 20-40k | 0.030 |
| EUR 40k+ | 0.030 |

Current thresholds:

| Income segment | Threshold |
|---|---:|
| income = 0 | 0.12 |
| EUR 1-5k | 0.12 |
| EUR 5-10k | 0.10 |
| EUR 10-20k | 0.08 |
| EUR 20-40k | 0.25 |
| EUR 40k+ | 0.25 |

The economic rationale is that hidden-information dispersion is much larger among lower-income borrowers, so adverse selection begins at a lower own-model PD in those segments.

The remaining time should not primarily be spent chasing small AUC gains. Pricing and selection uncertainty are currently much more important than marginal classifier improvements.

## 5. Final policy shortlist and selection rule

After repairing the evaluator, freeze the following small list before looking at new results:

1. the submitted flat band rate of 0.828;
2. a ramp from 0.50 to 0.70;
3. a ramp from 0.45 to 0.65;
4. the existing compromise policy: a large markup on the least risky part of the band followed by a ramp;
5. a monotone or two-tier high-tail policy that charges a competitive price in the moderately risky band but moves toward approximately 0.95 for the highest-risk tail;
6. at most one pre-specified data-driven step schedule.

The provisional simple ramp is:

\[
b(\hat p_i,s_i)
=0.45+0.20\min\left\{
\frac{\hat p_i-\tau(s_i)}{0.25-\tau(s_i)},1
\right\},
\qquad \hat p_i>\tau(s_i).
\]

For segments with \(\tau(s)=0.25\), use 0.65 directly above the threshold.

Under the existing cross-profile results:

| Simple policy | Average expected P&L | P(first) | P(last) |
|---|---:|---:|---:|
| ramp 0.50 to 0.70 | EUR 31.93M | 63.0% | 13.7% |
| ramp 0.45 to 0.65 | EUR 31.88M | 70.9% | 9.8% |

This makes 0.45 to 0.65 the stronger provisional simple policy when absolute profit is almost unchanged and the grading rule also values rank. These numbers must nevertheless be recomputed after fair tie handling.

The reported probabilities are frequencies under subjective simulated rival-profile generators. They are robustness diagnostics, not literal forecasts of the two classmates' behaviour.

### Declared decision criterion

Do not choose the final policy by an informal average over continually changing scenario families. Use a predeclared rule:

1. eliminate any policy with negative expected profit in a core plausible world;
2. keep policies whose average expected profit is within approximately 3% of the best remaining policy;
3. among those policies, maximize the average probability of finishing first;
4. use probability of finishing last, fifth-percentile profit, and maximum regret as secondary downside criteria;
5. rerun the choice under adverse tie-breaking and modest changes to the rival-profile weights.

Present the final policy as a robust choice on an efficient frontier, not as a precisely identified optimum.

## 6. Stage-1 exploration for Stage 2

Before adding exploration, ask the instructor what the phrase "results along with the complete dataset" means. Specifically ask whether the returned information contains:

- applicant-level lender choice;
- borrower type;
- applicant-level defaults;
- each lender's quoted rate or no-offer decision.

If applicant-level choice and default are returned, reserve approximately 10% of band applicants for a pricing experiment. If only aggregate results are returned, omit the experiment.

### Conservative experimental design

Within every income-segment by PD-band cell, assign applicants reproducibly using a hash of applicant ID:

- 90%: exploit using the final chosen ramp;
- 3.33%: offer \(\hat p/(1-\hat p)+0.15\), capped at 1;
- 3.33%: offer approximately 0.80;
- 3.33%: offer 0.99.

Suggested coarse PD strata within the band:

- threshold to 15%;
- 15% to 22%;
- above 22%.

With approximately 45,582 band applicants, this gives roughly 1,500 observations in each alternative arm while retaining 90% of the band for the main policy.

Before adopting the design, simulate its Stage-1 opportunity cost and impose a maximum acceptable expected cost, for example EUR 0.5-1.0 million. If all rival rates will be disclosed after Stage 1, the exploratory fraction can be reduced because competitor pricing functions can be learned directly.

The purpose is to estimate, by risk cell and price:

\[
Q_c(r)=P(\text{we win}\mid c,r)
\]

and

\[
\pi_c(r)=P(D=1\mid \text{we win},c,r).
\]

The direct per-offer target is:

\[
E\left[
1\{\text{win}\}\,10{,}000\big((1-D)r-D\big)
\mid c,r
\right].
\]

Randomization within narrow cells is important. A deterministic ramp alone changes price together with PD and cannot causally distinguish a price effect from a risk-composition effect.

## 7. Stage-2 best-response methodology

The central operating principle should be:

> Stage 1 is prior-driven; Stage 2 is evidence-driven.

Immediately after Stage-1 results arrive:

1. calculate total profit, rank, number of loans won, mean rate, and realized default;
2. decompose those quantities into safe region, intermediate-risk band, and high-risk fallback region;
3. within income and PD cells, compare our PD, the all-signal diagnostic PD, and realized default among loans won;
4. estimate the observed win curve \(Q_c(r)\) and selected-risk curve \(\pi_c(r)\);
5. replace subjective weights on hand-built competitor archetypes with the observed rival pair;
6. optimize the Stage-2 price grid using:

\[
r_c^*=\arg\max_r
Q_c(r)\left[(1-\pi_c(r))r-\pi_c(r)\right];
\]

7. bootstrap applicants and perturb estimated rival pricing rules to avoid overfitting one market realization;
8. prefer a lower-confidence-bound maximizer when two candidate prices are close.

If competitors' rates are disclosed, fit each competitor's pricing and rejection rule as a function of its public variables and private signal. For a new applicant, integrate over the historical conditional distribution of the rivals' hidden signals given our observed variables and signal. Evaluate prices around:

- concentrations in the rivals' rate distributions;
- the rivals' rejection boundaries;
- rates two percentage points above and below important rival prices because of borrower preference.

If competitors' rates are not disclosed, infer the response from randomized price arms and any choice-implied price bounds.

Do not automatically reuse the Stage-1 ramp in Stage 2.

## 8. Methodological validation

### 8.1 Conditional calibration

Overall calibration deciles are insufficient because the central problem is selection among accepted borrowers. For every shortlisted policy and representative competitor world, report on loans won:

- number of loans;
- mean own-model PD;
- mean all-signal diagnostic PD;
- realized default rate;
- expected and realized profit.

In an existing base-world diagnostic, the won-loan pool had approximately:

- own-model PD: 5.37%;
- all-signal diagnostic PD: 6.71%;
- realized default: 6.69%.

This is a clean demonstration of the winner's curse: the own-information model understates accepted risk by approximately 1.3 percentage points, while the all-signal proxy closely matches aggregate realized default in that backtest.

Attach confidence intervals to narrow tail cells. Several high-risk cells have few expected wins, so point estimates there should not be presented as precise facts.

### 8.2 Policy-selection stability

Out-of-fold PDs prevent ordinary in-sample prediction leakage, but the policy families, endpoints, and competitor worlds were repeatedly designed and inspected on the same 100,000 historical borrowers.

Now freeze the candidate list. Then perform either:

- an outer-fold policy evaluation; or
- a deterministic development/validation split used only as a split-stability check;
- paired borrower bootstraps using identical resampled borrowers for every compared policy.

A split created now is not a pristine independent holdout because the full sample has already influenced human decisions. Do not call it one. The truly new test will be the realized Stage-1 market.

### 8.3 Separate sources of uncertainty

Report separately:

- default and borrower-type sampling risk;
- model-estimation risk;
- uncertainty about the competitors' strategy class;
- uncertainty about the instructor's implementation of unspecified rules.

The last two dominate small Monte Carlo standard errors and should not be hidden by large numbers of artificial simulations.

## 9. Rule clarifications worth obtaining

The high-rate strategy depends on several implementation details. Seek written clarification, without describing the team's intended prices, on:

1. how exact ties are broken;
2. whether every borrower with at least one finite offer must accept one;
3. whether default probability is independent of the interest rate charged;
4. whether lenders face any funding, volume, or capital constraint;
5. which applicant-level fields and rival decisions are disclosed after Stage 1.

Run sensitivity analyses even if the instructor does not answer.

## 10. Three-page report for the instructor

Suggested title:

> **Pricing the Winner's Curse: Selection-Aware Lending with Private Signals**

The single thesis should be:

> A calibrated population probability of default is not sufficient for competitive pricing. Because competitors observe private information, whether a borrower accepts our offer is itself informative. We therefore price to default risk conditional on winning, not merely to the applicant's population PD.

### Page 1: model and economic problem

Briefly describe the cross-fitted gradient-boosted model. Emphasize calibration, log loss, and the incremental information in the private signal. Do not devote space to a long hyperparameter table.

Use the ordinary loan-profit equation:

\[
E[\ell_i(r)\mid p_i]
=10{,}000\left[(1-p_i)r-p_i\right],
\qquad
r_i^{BE}=\frac{p_i}{1-p_i}.
\]

Let \(Z_i=(X_i,s_{2i})\), let \(A_i(r)\) be the event that the applicant chooses us, define

\[
Q_i(r)=P(A_i(r)=1\mid Z_i)
\]

and

\[
\pi_i(r)=P(D_i=1\mid A_i(r)=1,Z_i).
\]

Then:

\[
\Pi_i(r)
=10{,}000\,Q_i(r)
\left[(1-\pi_i(r))r-\pi_i(r)\right].
\]

Explain that a rival's refusal or high price contains adverse information because that rival observes a signal that we do not.

Do not state that \(\pi_i(r)>\hat p_i\) or that \(\pi_i(r)\) is increasing in \(r\) as a universal theorem. Those properties hold empirically under relevant monotone rival policies but may fail under nets, rejection discontinuities, or non-monotone policies.

### Page 2: Stage-1 policy and realized diagnosis

State the exact submitted policy, not a later proposed policy. Use one compact table with rows for safe, intermediate, and fallback regions and columns for:

- loans won;
- mean rate;
- own predicted PD;
- realized default;
- profit.

Start with actual Stage-1 results rather than simulated forecasts:

> In Stage 1 we earned EUR X, ranked Y of three, won N loans, and observed a Z% default rate versus a predicted W% on the loans won.

Use the decomposition:

\[
\Pi_{\mathrm{actual}}
=\sum_{i\in W}10{,}000\left[r_i-D_i(1+r_i)\right].
\]

Compare actual profit with the same calculation using our PD and then the all-signal diagnostic PD. This separates ordinary pricing margin, hidden-information selection, and residual outcome noise.

### Page 3: Stage-2 learning and change

Explain what was learned about each rival, how accepted risk differed from predicted risk, and exactly why each Stage-2 parameter changed.

Good causal wording:

> Because realized default among won loans exceeded our own-information PD by X points in this band, we lowered the lending threshold from A to B.

or:

> Rival prices concentrated near C. We therefore priced just below that level where the rate remained above the selection-adjusted break-even floor.

Close with:

> Competition transforms private information into adverse selection. Our Stage-2 improvement came primarily from conditioning risk on the event that our offer wins, not from adding complexity to the classifier.

### Figures

Use no more than two compact figures:

1. by own-PD band, show own predicted PD, realized default among won loans, and acceptance share;
2. show the Stage-1 and Stage-2 pricing schedules together with the estimated selection-adjusted break-even curve.

Omit the 92-world heatmap, extensive clone tables, large Monte Carlo mixture charts, and lengthy hyperparameter descriptions from the three-page report. They can remain in supporting material.

### Language for high fallback rates

Use:

> The fallback rate is not an ordinary retail quote or a PD estimate. It is a contingent lender-of-last-resort offer, exercised mainly when both rivals quote more or decline to lend. It is profitable only when the rate exceeds the break-even rate of the jointly rejected pool. This mechanism relies on the stylized game's lack of an outside option and is not proposed as a real-world consumer-lending practice.

Avoid terms such as "jackpot," "exploit," "net war," or celebratory language concerning 50-80% rates.

## 11. Corrections required in `handover_for_advisor.md`

Before sharing that handover externally, correct or qualify the following points:

1. The statement that the current strategy is both "about to be submitted" and "submitted" is inconsistent. State the actual status.
2. The current generated `output/round1/rates_round1.csv` is the flat-0.828 policy. The recommended ramp has not yet been materialized as a CSV.
3. Bare `python run_round1.py` does not reproduce the current file. The current flat submission was produced with the equivalent of `python run_round1.py --no-sex --net 0.828`.
4. The required fintech name is not present in the repository material reviewed; only "Lender 2" appears.
5. Two strategy-note PDFs exist. Identify explicitly which file is current.
6. The Monte Carlo rank statement is inaccurate. `model + 0.20` finishes first over approximately 60-83% of the four reported mixtures, not 75-83% in every mixture. The relevant ramp results span much more than 57-66%.
7. The top calibration decile differs by approximately 0.36 percentage points, so "within roughly 0.4 points" is safer than "within 0.3 points."
8. The hidden-information contribution is not larger than our signal contribution in every segment. Above EUR 40k, the respective standard deviations are approximately 0.274 and 0.307 percentage points.
9. The statement that every category becomes profitable near the 100% cap cannot hold for a selected default rate of 51%, whose break-even rate is approximately 104.1%.
10. "Raising the rate does not fix it" must be qualified as a result for the benchmark rival policy. Other sections correctly show profitable interior responses to high-margin competitors.
11. "Thresholds are robust across everything" is too strong. Say that they were stable across the tested scenario families.
12. Expected and realized results do not agree within EUR 0.5 million in every reported cell; describe their agreement as approximate or aggregate.
13. The statement "no round numbers" is false for all normal offers. Only the deliberately non-round fallback level avoids an obvious round-number tie.

## 12. Submission controls

The final CSV should be built from full-precision model predictions, not the rounded PD values in an intermediate diagnostic CSV.

Validate all of the following:

- exactly 100,000 rows;
- exactly the required columns, `id` and `rate`;
- unique IDs;
- IDs equal the application's complete ID set;
- finite offered rates in \([0,1]\);
- missing values used only if deliberately representing no offer;
- no `+inf` or `-inf` written to the CSV;
- no unintended changes after four-decimal rounding;
- expected counts in every pricing region;
- min, max, mean, quantiles, and counts at repeated rates;
- the exact strategy JSON saved beside the CSV;
- a recorded reproduction command;
- a SHA-256 checksum of the final file;
- a fun fintech name included in the submission email;
- the written report describes the policy actually submitted.

The current `output/round1/rates_round1.csv` should not be sent accidentally if the policy is changed.

## 13. Practical work plan

### First work block

- repair fair and adverse tie handling;
- integrate the three borrower types exactly;
- add simulator and CSV input assertions;
- evaluate after submission rounding.

### Second work block

- freeze and rerun the short candidate list;
- calculate expected P&L, realized backtest P&L, P(first), P(last), fifth-percentile profit, and regret;
- perform paired bootstrap comparisons;
- choose the final policy using the declared criterion.

### Third work block

- decide whether to include the 10% experiment based on the instructor's feedback fields and its simulated opportunity cost;
- generate the final CSV from full-precision predictions;
- run the complete submission checklist;
- save configuration, command, summary, and checksum;
- update the handover so it states the actual submitted policy.

### After Stage 1

- prioritize the realized market outcome over prior scenario simulations;
- infer or fit each rival's policy;
- estimate acceptance and selected risk by price;
- calculate a Stage-2 best response;
- build the three-page report around the observed winner's curse and the evidence-driven change.

## 14. Final judgment

The core economics and modeling are already strong enough for an excellent submission. The largest attainable improvement is not a more complex classifier or a still larger library of hypothetical rivals. It is to make the market evaluator correct and internally consistent, use a small controlled experiment to learn the actual market, and present the result through a disciplined selection-adjusted-profit framework.

The strongest explanatory message is:

> We did not merely predict who defaults. We modeled how competitors' private information changes the risk of the borrowers whose offers we win, then priced the selected pool rather than the population pool.
