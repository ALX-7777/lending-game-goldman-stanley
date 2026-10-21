# Lending Game — complete brief for an external advisor

*Prepared 14 October 2026 by team "Lender 2" (HEC Paris, MSc Data Science & AI for Business, course "Introduction to Finance for Data Scientists"). Stage-1 submission due 17 October 2026 at midnight; stage 2 and a 3-page report due 23 October.*

**Purpose of this document.** We have worked on this problem for two days with a full simulation pipeline. We want an independent, critical opinion on (a) the strategy we are about to submit, (b) the reasoning behind it, and (c) anything we have missed. Section 10 lists the precise questions. Everything else is context and evidence; all numbers quoted come from our own simulations on the data described in Section 2.

---

## 1. The game (exact rules)

Three teams of students are lenders competing for the **same 100,000 loan applicants**. Each applicant asks for a loan of **10,000 €**. Each lender k submits, for every applicant i, an interest rate r(k,i) in [0, 1] or no offer. Rates above 100% are not allowed.

**Choice rule.** Applicants come in three types, one third each, drawn uniformly at random and unobservable. A type-k applicant prefers lender k and is willing to pay 2% more for it: formally, the applicant chooses the lowest of r(1,i) − 0.02·[k=1], r(2,i) − 0.02·[k=2], r(3,i) − 0.02·[k=3]. No offer counts as +∞. **There is no outside option: an applicant who receives at least one offer borrows.** Tie-breaking is not specified.

**Payoff.** If the borrower repays, the lender earns r × 10,000. If the borrower defaults, the lender loses 10,000 (zero recovery). A team's score is its total profit over the loans it wins.

**Grading.** Performance = 10% (stage 1) + 20% (stage 2), judged "both in absolute terms (level of profits) and in relative terms (ranking relative to other groups)"; the written report = 70%. The report must explain the methodology, "the problem created by the fact that the other lenders have information that you do not have and how you tried to overcome this problem", and the stage-2 diagnosis.

**Stage 2.** After stage 1 the instructor simulates the market and sends each team "the results … along with the complete dataset" (we do not know whether rivals' per-applicant rates are disclosed). Teams then price a new batch of 100,000 applicants.

**Information structure.** All lenders see sex, employment (4 categories), married, income (top-coded at 1M). Each lender also sees **one private signal** in [0, 1] about each applicant: lender 1 sees signal1, we (lender 2) see signal2, lender 3 sees signal3. A public file of 100,000 **past loans** contains all three signals *and* the default outcome. The new applicants are drawn from the same population with the same default process.

## 2. The data and what it says

* Past loans: 100,000 rows, default rate 9.95%, no missing values. New applications match the past population on every moment we checked.
* **Income dominates default**: 20.2% default at zero income (17% of applicants), 15.1% at 1–5k, 12.5% at 5–10k, 5.8% at 10–20k, 2.2% at 20–40k, 0.8% at 40–80k, 0.1% above 80k. Employment: unemployed 15.6%, student 12.9%, retired 9.0%, employed 7.9%. Women 10.7% vs men 9.4%; single 10.8% vs married 9.4%.
* **A signal of exactly 0 means "no signal"**, not "worst score": 21% of each signal is exactly 0, and those applicants default at the population average (9.7%). Non-zero signal values are monotone in risk but weak: signal2 in (0, 0.2] → 14.9% default, (0.8, 1] → 7.9%.
* The three signals are **different information**: pairwise correlation 0.18, zeros independent across signals.
* **Models.** Gradient-boosted trees (sklearn HistGradientBoosting; learning rate 0.05, 7 leaves, 200 iterations, ≥200 obs per leaf, L2 = 1; chosen by 3-fold CV log-loss) on employment dummies, married, income, log income, the signal, and a "signal = 0" indicator. Out-of-sample (5-fold) AUC: X only 0.717; X + signal2 (us) 0.735; X + signal1 0.735; X + signal3 0.736; X + all three signals 0.760. Calibration of our model by decile is within ±0.3 points. Sex adds nothing (identical AUC) and was dropped.
* **Hidden information.** Let p̂₂ = E[D | X, s₂] (ours) and p_full = E[D | X, s₁, s₂, s₃]. The std of U = p_full − p̂₂ is **4.5 points of default probability** overall, and it depends strongly on income: 7.1 points at zero income, 5.0–5.2 at 1–10k, 2.5 at 10–20k, 0.8 at 20–40k, 0.3 above 40k. Our own signal moves the PD by 4.9 / 3.4 / 3.4 / 1.8 / 0.7 / 0.3 points in the same segments. So the rivals' information advantage over us is always larger than our signal's contribution (two signals against one), and it is concentrated on low-income applicants.

## 3. The economics we rely on

### 3.1 Break-even rate
Interest is received only if the borrower repays, so the expected profit of a loan at rate r to a borrower with default probability p is 10,000·[(1 − p)·r − p] and the break-even rate is **r\* = p/(1 − p)** (not r = p, which is what the lecture's NPV = R − PD formula gives when interest is paid upfront; a team using r = p + m underprices risky borrowers).

### 3.2 The winner's curse through the rivals' signals
Our estimate is calibrated on the population but the applicants who *accept* our offer are not a random sample: an applicant takes our offer when both rivals' effective rates are above ours, and rivals price higher exactly when their signals are bad. Writing Q_i(r) for the probability that applicant i accepts our offer at rate r and π_i(r) = E[D_i | accepts at r], the expected profit of the offer is

Π_i(r) = 10,000 · Q_i(r) · [(1 − π_i(r))·r − π_i(r)],  with π_i(r) > p̂₂,ᵢ and π_i(r) increasing in r.

This is the lecture's M − E[U | U > M − R̄] with the acceptance event generated by the rivals' information. Measured on the past loans with rivals pricing at break-even + 3% (rejecting above 25% PD) and us lending to everyone at the model's rate:

| Our predicted PD | Share we win | Predicted PD on won loans | Realized default on won loans | Rate needed to break even on them | Our rate | Profit |
|---|---|---|---|---|---|---|
| 0–5% | 37% | 2.5% | 2.6% | 2.7% | 5.6% | +3.40 M€ |
| 5–10% | 37% | 7.3% | 8.9% | 9.7% | 11.2% | +1.07 |
| 10–15% | 32% | 12.3% | 15.4% | 18.2% | 17.7% | −0.33 |
| 15–20% | 16% | 17.0% | 24.2% | 31.9% | 24.4% | −1.06 |
| 20–25% | 10% | 22.8% | 32.3% | 47.6% | 33.6% | −0.72 |
| 25–30% | 7% | 26.6% | 41.5% | 70.9% | 40.3% | −0.36 |
| 30–40% | 10% | 34.0% | 54.8% | 121% | 55.8% | −0.28 |

The curse grows with risk (because the rivals' hidden information does) while a fixed margin does not; above ~10% PD, lending at the model's rate loses money, and **raising the rate does not fix it**, because a higher rate selects even worse borrowers (π(r) rises with r). We call the applicants above the threshold where this happens **"the band"**.

### 3.3 The 2% preference
For an applicant whose best rival rate is c, we win our own fans if r < c + 0.02 and all three types if r < c − 0.02. Undercutting by 4 points triples volume; ignoring selection, that pays only when r > p/(1 − p) + 0.06, i.e. when rivals sit more than 4 points above break-even. So on the safe part of the market the natural target is one's fans at break-even plus a few points, and market-share grabbing is a loss (tested: pricing at break-even + 1% gives 55% share and −2 M€, and drags the rivals into losses too).

### 3.4 Lender of last resort ("the net")
Because applicants have no outside option, an offer at a very high rate is accepted only when every rival asks more or refuses. It costs nothing when rivals lend and earns 10,000·[(1 − d)·r − d] when both refuse, where d = E[D | both reject]. On the past loans, if rivals reject above 25% PD the abandoned pool is 0.8% of applicants with 43% default (break-even 74%); if they reject above 20%, 5.3% with 30% default; above 15%, 14.4% with 24% default. If rivals use *low* thresholds like ours and net the band too, the pool where all three net is **34% of applicants with 18.7% default** (break-even 0.23) and whoever is lowest within 2% takes it: 65 M€ (own fans only) to 190 M€ (all), against ~5 M€ for the entire normal game. This quirk of the rules dominates the stage-1 ranking if any team nets.

## 4. What we built

* `lending_game/`: feature engineering; PD models (out-of-sample predictions for us, for each rival's information set, and for the full-information model, which we use as "the truth" when computing expected profits); a market simulator implementing the choice rule literally (random types, 2% preference, random tie-break, realized or expected profit); competitor archetypes; our pricing rule and a coordinate-descent optimizer.
* Every result below is a simulation **on the 100,000 past loans**, where all three signals and the outcome are visible, so rivals can be given their own information set and the curse is reproduced rather than assumed. Expected profit uses the full-information PD (less noisy than realized outcomes; both agree within 0.5 M€).
* `run_round1.py` reproduces the submission; `make_submission.py` builds a submission from the analysed rule without re-optimizing; `analysis/` holds all experiments; `output/round1/figures/fig0…fig10` the figures.

## 5. The rule we submitted (stage 1 file as it stands today)

For applicant i with our PD estimate p̂ and income segment s:

* if p̂ ≤ τ(s): r = p̂/(1 − p̂) + m(s), with m = 4% (income 0), 3.5% (1–10k), 3% (above 10k) and τ = 12% (income 0 and 1–5k), 10% (5–10k), 8% (10–20k), 25% (above 20k);
* otherwise ("the band"): a flat lender-of-last-resort rate of **0.828**.

Parameters were chosen by maximizing expected profit averaged over 12 hand-built competitor scenarios (rivals at break-even + 1/2/3/4/5%, with or without rejection at 20/25/30% PD, the lecture-formula mistake, a selection-aware rival with 2–6% margins and a 0.99 net, and mixed pairs). The file: 100,000 offers, 54,418 "normal" offers between 3.0% and 17.6%, **45,582 offers at 82.8%** (92% of zero-income applicants, 74% of 1–5k, 69% of 5–10k, 18% of 10–20k, none above 20k).

Results against the 12 scenarios (realized profit): mean **6.4 M€**, worst scenario +1.2 M€ (both rivals lending to everyone at +1%), market share 22%, first in every scenario. For comparison: break-even + 3% for everyone 1.7 M€ (worst −2.2), the same with no offer above 25% PD 2.3 M€, the same with a 0.989 net 2.9 M€. Threshold sensitivity: raising τ to 25% everywhere cuts the mean to 3.1 M€ and makes the worst case −1.1 M€; the low thresholds are driven by the curse, not by the net (re-optimizing without the net gives the same thresholds). The net adds ~0.4 M€ in the base scenario and far more when rivals reject.

Why 0.828 and not the maximum: a first analysis with rivals netting at 1.00 / 0.99 / 0.95 / 0.90 / 0.85 / 0.80 / 0.70 showed that lowering our net from 0.989 to 0.90 costs 0.14 M€ if nobody else nets and gains 1.4 M€ on average if somebody does; a second analysis with "clones" (rivals using our low thresholds and a net) showed that against a clone netting at x we win everything below x − 0.02, only our fans within ±0.02, and nothing above x + 0.02; a prior-weighted decision table made 0.83 the best level under skeptical, moderate and paranoid priors, and 0.828 was chosen to beat a clone at 0.85 for all applicant types.

## 6. The band-price question: what we tried next

Everything after the submission concerns one thing: **what to charge the 46% of applicants in the band**. The safe part (break-even + 3–4%) is settled: the profit curve as a function of our safe margin is hump-shaped, its optimum moves with the rivals' margin (2.5% against rivals at +1%, 3% at +2%, 4% at +3%, 4.5% at +4%), and our 3–4% sits on the plateau of every world (cost of being one point off: 0.3–1 M€).

### 6.1 Payoff matrix, 92 competitor worlds × 60 band schedules
Worlds grouped by how the rivals treat the band: *cheap* (model + 0–5 points), *margin* (model + 8–40 points, or margins growing with risk), *reject* (refusal above 12–30% PD, rate caps, cherry-pickers, refusal of a visible group), *net* (clones netting at 0.60–0.99 with various thresholds, two-tier nets, rival ramps), *mixed* (asymmetric pairs). Schedules: flat nets, ramps L → H rising with PD from the threshold to 25% PD, "model + M", hybrids. Expected profit (M€), key rows:

| Schedule | cheap | margin | reject | net | mixed | max regret | finishes 1st (net worlds) |
|---|---|---|---|---|---|---|---|
| flat net 0.828 (submitted) | 7.4 | 5.2 | 50.9 | 41.8 | 12.2 | 144 | 33% |
| ramp 0.65 → 0.828 | 7.4 | 5.4 | 43.9 | 79.1 | 11.8 | 75 | ~75–80% (0.60: 81%, 0.70: 72%) |
| ramp 0.60 → 0.828 | 7.4 | 5.8 | 41.9 | 80.7 | 11.6 | 83 | 81% |
| model + 0.20 | 6.7 | 24.1 | 18.9 | 47.9 | 8.6 | 163 | 100% |

Findings: (i) in cheap and margin worlds every net-type schedule earns the same ~5–7 M€ — the band price only matters against netters and rejecters; (ii) against curse-aware margin-adders (model + 10–40 points) the net earns nothing in the band while the rivals earn 12–54 M€, and our best response is "their margin minus 5–10 points" (e.g. 25 M€ against +0.20), but that schedule forfeits most of the net and reject worlds; (iii) within-world sampling noise is ±0.7–3 M€ (95%), i.e. all differences are about *which world*, not statistics; (iv) an adversarial check refuted our claim that the ramp is "never worse than the flat net": it is worse by up to 20 M€ when both rivals net at ≥ 0.85 and by 7–15 M€ against rivals refusing on a visible attribute; it gains 42–117 M€ when a rival's band price lies in [0.72, 0.848). The flat net is better only if each rival has < ~8% probability of a band price below 0.848.

### 6.2 "Lend to everyone", "increasing margin", copycats
* Lending to the band at the model's rate (no threshold, no net): mean 2.0 M€, worst −1.7 M€; the band loses 2.4 M€ (table in 3.2). Adding +5 or +15 points for PD > 20%: still losing in aggressive worlds.
* "Model + a margin that grows with PD" instead of a threshold: optimized, the slope goes to the top of its grid (it tries to *become* a net) and still earns 5.8 vs 6.6 M€; given the threshold and the net, the optimal slope is exactly zero.
* A rival copying our rule with a lower safe margin: each point costs us ~0.4–0.5 M€, but at 3 points below us the copycat earns less than we do and at 4 points it loses money (thin margins get eaten). A rival copying our rule with a lower band ramp: 2 points below takes the contested band (two such copycats: we fall from 45 to 26 M€; 5 points: to 13 M€). No schedule survives an exact copy of itself that is strictly lower.

### 6.3 Profit per PD category as a function of our rate (the clearest picture)
Under rivals at model + 3% refusing above 25%, every category above 10% PD has negative expected profit for every rate up to ~90% and turns barely positive only at the cap (where the accepted are the applicants both rivals refused, defaulting 34–51%). Under rivals adding a margin growing with risk, or model + 20 points, the curves have an interior maximum at *their rate minus 2–5 points* (22% for 10–15% PD, 32% for 15–20%, 44% for 20–25%) and nothing works above 25% PD. The lesson: the right rate for a category is argmax_r N_c·Q_c(r)·[(1 − π_c(r))·r − π_c(r)], and it is either "just below the rivals" or the cap, never in between.

### 6.4 Monte Carlo over rival strategies
Instead of hand-picked worlds, each rival is drawn from a distribution: safe margin 1–5%, model noise (logit sd 0–0.3), lecture-formula mistake 10%, band regime from a mixture (cheap / margin-adder / rejecter / netter at obvious or uniform levels / ramper), two independent rivals, 800 draws per mixture. Results (expected profit, M€):

| Mixture | flat 0.828 | ramp 0.65→0.828 | ramp 0.60→0.828 | flat 0.70 | model + 0.20 | compromise* |
|---|---|---|---|---|---|---|
| baseline (45% cheap / 15 margin / 15 reject / 15 net / 10 ramp) | 7.5 | 8.2 | 8.3 | 8.2 | 7.4 | 8.3 |
| naive rivals (65/10/15/7/3) | 7.1 | 7.2 | 7.2 | 6.8 | 5.5 | 6.7 |
| margin-adders (30/40/10/12/8) | 6.6 | 7.1 | 7.3 | 7.2 | 10.7 | 10.2 |
| sophisticated (25/15/10/30/20) | 9.9 | 12.7 | 13.1 | 13.2 | 10.9 | 11.8 |

\*compromise = model + 0.20 for PD ≤ 18%, ramp above. Two corrections came out of this: (1) the expected profit is **7–13 M€**, not the 20–40 M€ suggested by the hand grid, which over-weighted symmetric extreme worlds; (2) on *rank*, "model + 0.20" finishes first 75–83% of the time in every mixture (it is never shut out of the band) while the ramps finish first 57–66%; a compromise (model + 0.25 up to 15% PD, ramp above) keeps the ramp's mean with P(last) halved.

### 6.5 A ramp derived from selection-adjusted break-evens
For each PD category c and rate r, averaging over random rivals gives Q_c(r), π_c(r), the **selection-adjusted break-even** π_c(r)/(1 − π_c(r)), and the profit-maximizing rate. Fitted schedules (band only; categories 10–12 / 12–15 / 15–18 / 18–22 / 22–26 / 26–30 / 30–40% PD):

| Fitted against | Selection-adjusted break-even at the chosen rate | Data-driven rate per category |
|---|---|---|
| mixed rivals | 0.19 / 0.24 / 0.30 / 0.35 / 0.44 / 0.57 / 0.75 | 0.38 / 0.44 / 0.62 / 0.62 / 0.74 / 0.76 / 0.94 |
| smart rivals | 0.17 / 0.20 / 0.24 / 0.29 / 0.36 / 0.45 / 0.57 | 0.62 / 0.62 / 0.64 / 0.66 / 0.68 / 0.68 / 0.74 |
| clones of us | 0.16 / 0.20 / 0.24 / 0.30 / 0.36 / 0.46 / 0.58 | 0.54 / 0.56 / 0.56 / 0.58 / 0.60 / 0.62 / 0.68 |
| netters | 0.16 / 0.19 / 0.23 / 0.28 / 0.34 / 0.42 / 0.53 | 0.62 / 0.64 / 0.64 / 0.64 / 0.66 / 0.68 / 0.78 |
| very smart best-responders to us | 0.16 / 0.18 / 0.22 / 0.27 / 0.33 / 0.42 / 0.53 | 0.50 / 0.52 / 0.52 / 0.52 / 0.54 / 0.54 / 0.56 |
| model-followers | — | 0.18 / 0.22 / 0.32 / 0.50 / 0.98 / 1.00 / 1.00 |

So the structure is rate(c) = selection-adjusted break-even(c) + a margin that is a bet on how low the rivals go; each round of "best response to their best response" lowers it, toward the break-even floor plus the 2% rent.

### 6.6 Smart-rival profiles and the cross-table (our current state of knowledge)
Profiles: *smart* (calibrated models, correct break-even, low thresholds; band = net / ramp / margin / data-driven / reject / compromise), *clones* (our method with thresholds × 0.6–2 and safe margins 2–5%; band = ramp / data-driven ± 0.1 / flat net 0.65–0.99), *model-followers* (model + 2–6% or growing margins, no net), *netters* (obvious numbers, break-even-derived 0.65–0.90, two-tier), *very smart* (rivals using our data-driven schedule shifted 0–10 points below, undercutting our safe margins by ~1 point, some with the compromise). Fresh draws, 200 per profile. Mean expected profit (M€) / P(we finish first) / P(we finish last):

| Our band schedule | mixed | smart | clones | model-followers | netters | very smart | average |
|---|---|---|---|---|---|---|---|
| flat net 0.828 (submitted) | 7.5 / 61% / 18% | 21 / 16% / 55% | 10 / 10% / 60% | 6.2 / 39% / 36% | 36 / 30% / 40% | 4.0 / 0% / 97% | 14 / 26% / 51% |
| ramp 0.65 → 0.828 | 8.5 / 67% / 14% | 34 / 32% / 32% | 22 / 28% / 32% | 6.2 / 39% / 36% | 63 / 66% / 4% | 4.4 / 0% / 95% | 23 / 39% / 35% |
| compromise (model + 0.25 to 15% PD, then ramp 0.65→0.828) | 9.1 / 68% / 9% | 31 / 35% / 18% | 22 / 34% / 18% | 6.3 / 38% / 34% | 50 / 68% / 2% | 26 / 8% / 14% | 24 / 42% / 16% |
| ramp 0.55 → 0.75 | 8.9 / 70% / 10% | 38 / 46% / 20% | 34 / 68% / 5% | 6.2 / 39% / 36% | 70 / 92% / 0% | 18 / 6% / 44% | 29 / 53% / 19% |
| **ramp 0.50 → 0.70** | 9.0 / 72% / 8% | 37 / 53% / 16% | 37 / 82% / 1% | 6.2 / 39% / 35% | 68 / 98% / 0% | 34 / 33% / 22% | **32 / 63% / 14%** |
| ramp 0.45 → 0.65 | 9.1 / 74% / 7% | 35 / 60% / 11% | 36 / 94% / 0% | 6.3 / 38% / 36% | 62 / 100% / 0% | 43 / 57% / 4% | 32 / 71% / 10% |
| data-driven fitted vs clones (≈0.54→0.68) | 8.6 / 70% / 8% | 37 / 60% / 12% | 39 / 95% / 0% | 6.2 / 38% / 40% | 66 / 96% / 0% | 39 / 44% / 18% | 33 / 67% / 13% |
| data-driven fitted vs very smart (≈0.50→0.56) | 8.3 / 71% / 9% | 33 / 62% / 10% | 35 / 99% / 0% | 6.1 / 36% / 48% | 59 / 100% / 0% | 54 / 75% / 2% | 33 / 74% / 12% |

Reading: the submitted flat 0.828 and the 0.65→0.828 ramp **collapse against rivals who copy us and go slightly lower** (4 M€, last 95–97%). Lowering the band schedule to ~0.50→0.70 costs nothing in the mixed world, little against pure netters (the loss against the few netting at 0.85–0.99 is repaid by the many at 0.55–0.85), and is never the worst row in any profile. Model-followers make the band irrelevant (all schedules tie at 6.2 M€).

## 7. Where we stand and what we lean towards

* **Submitted file today:** flat net 0.828 (Section 5). Our written stage-1 note describes it.
* **Current recommendation:** replace the flat net by a **ramp 0.50 → 0.70** (0.50 at the threshold, rising linearly to 0.70 at 25% PD, flat above), keeping thresholds and safe margins unchanged. Rationale: Section 6.6. Alternatives on the table: ramp 0.45 → 0.65 or the data-driven schedule (≈ 0.50–0.56) if we believe rivals will literally best-respond to us; ramp 0.65 → 0.828 if we believe most netters sit at 0.85–1.0.
* The thresholds (8–12% PD below 20k income, 25% above) and safe margins (3–4%) are robust across everything we tried.

## 8. Assumptions and known weaknesses

1. **Rival behaviour is our construct.** The hand grid, the archetypes and the Monte Carlo distributions are our guesses; the profile mixture weights are subjective. The two other teams are students with access to the same data and to LLM assistants; we believe they will build similar boosted-tree models (so no modeling edge either way) and we expect at least one of them to be curse-aware.
2. **Rivals are modeled as "our GBM on their signal"**, possibly with logit noise. We never modeled a rival with a materially different feature set, a different threshold *shape* (e.g. by employment), or deliberately randomized prices.
3. **Full-information PD as the truth.** Expected profits use p_full = E[D | X, s₁, s₂, s₃] estimated out-of-sample; it is well calibrated (band realized 16.5% vs 16.4% predicted) but any misspecification shared by all our models would bias everything in the same direction.
4. **Types uniform and independent of characteristics**, random tie-breaks, exact rules as written. Our evaluator breaks exact ties in our favour (immaterial for the ramps, which create no exact ties). The simulator does not enforce the 100% cap (our rates never exceed 0.83).
5. **No outside option.** The 60–190 M€ band jackpots exist only because an applicant always borrows from someone. If the instructor's simulation applied an unstated reservation rate, every net/ramp would earn nothing on the band (and lose nothing).
6. **Magnitudes vs. order.** The net-world levels (60–190 M€) assume rivals copy our low thresholds; with rival thresholds at 15–20% they shrink 3–10× (order unchanged). The Monte Carlo (7–13 M€ expected under mixed rivals) is our better estimate of what to expect.
7. **Rank vs. level.** The grade weighs both, with unknown weights. Net-type schedules finish last in margin-adder worlds; model + margin schedules finish last in net worlds; the lower ramps are the compromise.
8. **Visibility and stage 2.** After stage 1 everyone sees everyone's outcomes (possibly rates). Whatever we do in the band will be copied or undercut in stage 2; conversely, we will learn the rivals' regime and can best-respond per PD category (Section 6.3). We have not modeled the value of information, nor whether a stage-1 schedule that *reveals* the rivals' band prices (a ramp spreads our prices over 0.5–0.7, a flat net does not) is worth something.
9. **Optics.** 45% of our offers at 50–83% interest will be noticed by the instructor. It is within the rules and economically sound (lender of last resort pricing), but we would like a second opinion on how it reads in a graded report.

## 9. Figures available (PNG, in output/round1/figures)
fig0 payoff matrix heatmap; fig1 where each schedule wins (vs rival net level, vs rival band margin, vs rejecters); fig2 prior-weighted profit and max regret; fig3 across-world risk; fig4 P(first) by regime; fig5 copycat tests; fig6 margin curve; fig7 profit per PD category vs our rate under three rival behaviours; fig8 Monte Carlo distributions; fig9 selection-adjusted break-evens and data-driven ramp; fig10 cross-table of schedules × smart-rival profiles.

## 10. Questions for the advisor

1. **The band price.** Given Sections 6.5–6.6, is a ramp 0.50 → 0.70 the right stage-1 choice? Would you go lower (toward the selection-adjusted break-evens + a small margin), higher (0.65 → 0.828, betting on naive netters), or to a per-category data-driven schedule? Is there a better *shape* than a linear ramp in predicted PD (e.g. constant markup over the selection-adjusted break-even, or a schedule in the rivals' likely price rather than in PD)?
2. **Equilibrium reasoning.** The best-response iteration races the band price down toward the break-even floor + 2% rent. How far should a first-mover go in a one-shot stage 1 when rivals may or may not reason the same way? Is there a cleaner way to pick the margin over the floor than our mixture priors (e.g. a level-k or quantal-response argument)?
3. **Rank vs. level.** How would you weigh "finish first" against expected profit here, given the grading text? Is a hybrid (model + 0.25 on the least risky part of the band, lower ramp above) worth its complexity?
4. **Thresholds and safe margins.** Any reason to doubt the thresholds (8–12% PD at low income, 25% above 20k) or the 3–4% margins? Should the threshold also depend on employment or on whether our signal is missing (we found no gain from the latter)?
5. **Blind spots.** Which rival behaviours would hurt us that we have not simulated? Which simulator assumption (types, no outside option, tie-breaking, full-information PD, rivals = our model on their signal) worries you most, and how would you test it?
6. **Stage 2.** What should we extract first from the stage-1 results, and is it worth sacrificing some stage-1 profit for information (e.g. randomized price perturbations, or a ramp whose spread of prices reveals the rivals' band prices)?
7. **The report (70% of the grade).** Which parts of this analysis are most valuable to present, and what is the cleanest formal statement of the problem and of our solution (the selection-adjusted profit Π_i(r), the break-even floor, the lender-of-last-resort argument)?
8. **Anything we are fundamentally wrong about.**

---

### Appendix A — parameter values of the analysed rule
Segments by income: 0 / 1–5k / 5–10k / 10–20k / 20–40k / 40k+. Margins m: 0.040 / 0.035 / 0.035 / 0.030 / 0.030 / 0.030. Thresholds τ: 0.12 / 0.12 / 0.10 / 0.08 / 0.25 / 0.25. Band schedule: submitted = flat 0.828; recommended = ramp from 0.50 at τ to 0.70 at PD 0.25 (constant above). Rates rounded to 4 decimals; no round numbers.

### Appendix B — scenario-by-scenario results of the submitted rule (12 hand-built scenarios, realized profit, M€; lender 1 / lender 3 in brackets)
base/base 5.1 (3.2 / 3.2); aggressive/aggressive 1.2 (−4.0 / −4.0); cautious/cautious 24.8 (6.5 / 6.2); no-reject/no-reject 4.7 (2.3 / 2.5); lecture-formula/lecture-formula 4.4 (−1.1 / −0.9); lecture no-reject 4.0 (−2.2 / −1.9); smart/smart 6.4 (6.2 / 6.3); base-with-net/base-with-net 5.1 (3.2 / 3.2); base/aggressive 2.2 (−1.1 / −2.1); cautious/lecture 8.7 (0.3 / 2.7); smart/base 5.6 (3.3 / 5.5); loose/tight 5.0 (2.6 / 1.9). Realized default on loans we win: 6.8% vs 5.4% predicted (the residual curse).

### Appendix C — the two formulas we would put in the report
Selection-adjusted expected profit of an offer: Π_i(r) = 10,000 · Q_i(r) · [(1 − π_i(r))·r − π_i(r)], with Q and π estimated by simulating the rivals on the past loans where their signals are visible.
Lender-of-last-resort condition: an applicant refused by every rival is profitable at rate r iff r > d/(1 − d), d = E[D | refused by all]; with a rival net at x, we win all abandoned applicants iff r < x − 0.02, only our fans iff |r − x| ≤ 0.02.
