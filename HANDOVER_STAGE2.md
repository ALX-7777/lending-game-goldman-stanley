# Lending Game — hand-over for stage 2 (team "Goldman Stanley", type-2 lender)

*Written 21 October 2026 for the teammates who carry stage 2. Stage 2 is due **23 October 2026 at 24:00** by email to hombert@hec.fr: (1) a CSV with the 100,000 round-2 applications and our interest rates, (2) a 3-page text report. Everything described here is in the git repo; paths are relative to the project folder.*

---

## 0. The short version

1. **What we did in stage 1.** A gradient-boosted PD model on the public variables plus our signal (out-of-sample AUC 0.735), then a price per applicant built on two ideas: the break-even rate is `p/(1−p)`, not `p`; and the applicants who accept our offer are riskier than our estimate because the rivals see signals we do not (winner's curse). We measured that curse on the past loans and priced the risky half of the applicants on the *default rate of the pool we actually win*, plus a 5-point margin, capped at 98.9%. Chosen by Monte Carlo against seven profiles of rivals.
2. **What happened.** 51% market share, **28.68 M€, 6th of 12 teams**. The leaders made 54–73 M€ with 37–41% share. We priced for careful rivals competing on their models; the real market had massive *refusals* (one rival in each pair made no offer to 34–38% of applicants, both refused 14%), a type-1 lender quoting a flat 92% to everything risky, and cheap pricers on the safe part. The curse we priced for in the band never materialised (won loans defaulted about 1.5 points above our PD instead of 5–20), so our band rates were 20–60 points below what the applicants we won would have accepted. 80% of our profit came from the 45% of applicants above our thresholds; the safe part, where we competed hardest, earned 225 € per loan.
3. **What to do in round 2.** Keep the model and the safe part; price the band (PD above the segment threshold) at a **flat high rate, about 0.78** (alternative 0.85). Two ready files: `output/round2/rates_round2_band078.csv` (recommended) and `output/round2/rates_round2_band085.csv`. Section 6 explains the choice and its risks; Section 7 gives the checks before sending; Section 8 outlines the 3-page report.
4. **Rules.** The folder `Round_1_report/` contains another team's files sent to us by mistake. It is in `.gitignore`; delete it, never use it, never share it. The guidelines forbid using other groups' data.

---

## 1. The game in ten lines

- 100,000 applicants per round, each asks 10,000 €. Profit on a loan: `r × 10,000` if repaid, `−10,000` if default (recovery zero).
- For each application we face one random type-1 and one random type-3 lender (four teams of each type exist, including one ChatGPT lender per type). We are **type 2**: we see `signal2`, they see `signal1` or `signal3`. Public variables for everyone: sex, employment, married, income.
- Each applicant has a preferred lender (one third each) and takes the lowest of the three rates after subtracting 2 points from the preferred lender's rate. No offer = infinite rate. **There is no outside option**: an applicant with at least one offer always borrows.
- `PastLoans.csv` (100,000 loans) contains all three signals and the default outcome; the applicants of each round come from the same population.
- Grading: stage-1 profit 10% (level and rank), stage-2 profit 20%, report 70%. Bonus for the funniest name.
- Rules: no data from other groups, no price coordination. AI may help with code and feedback, not design the strategy or write the report.

---

## 2. What we built in stage 1

### 2.1 The default model (`run_round1.py`, package `lending_game/`)

- Features: employment (one-hot), married, income, log income, `signal2`, and a flag `signal2 == 0`. Sex was tested and dropped (no information). **A signal of exactly 0 (21% of applicants) means "no signal": those applicants default at the population average, not at the worst rate.** Several rival teams apparently got this wrong (Section 4.3).
- Model: `HistGradientBoostingClassifier` (learning rate 0.05, 7 leaves, 200 iterations, min 200 per leaf, L2 = 1), 5-fold out-of-sample probabilities, calibrated within half a point per decile.
- Discrimination out of sample: public variables alone AUC 0.717; public + one signal 0.735 (identical for signals 1, 2 and 3: nobody has a better signal); public + all three signals 0.760. On the round-1 population our model scored AUC 0.728 and stayed calibrated.
- The hidden information of the rivals, measured as the standard deviation of `p_full − p̂` (full-information PD minus ours): 4.5 points overall, 7 points at zero income, 0.3 points above 40k of income.

### 2.2 The economics (this is the heart of the stage-1 report)

- **Break-even.** A loan at rate `r` with default probability `p` has expected profit `(1−p) r − p` (times 10,000). It breaks even at `r* = p/(1−p)`, not at `r = p`: 11.1% for a 10% PD, 25% for 20%, 43% for 30%.
- **The 2% preference.** Whoever is lowest within two points of the preferred lender takes the applicant. Undercutting a rival by more than ~2.5 points on the safe part is self-defeating; staying within 2 points of the market keeps our third of the applicants.
- **Winner's curse.** We win an applicant when the two rivals ask more, which happens when *their* signals are bad. So the default rate of the applicants we win, `π̃(p̂) = E[default | we win, p̂]`, exceeds our calibrated `p̂`, more so where our PD is far from the segment's typical risk. We measured it on the past loans by letting all three lenders price on their own model and recording who we win: `π̃` = 12.7% at 10% PD, 18% at 15%, 27% at 20%, 39% at 25%, above 50% beyond 27% (file `output/round1/curse_floor.csv`, script `analysis/curse_adjusted_breakeven.py`). The **adjusted break-even** `π̃/(1−π̃)` is the rate at which winning still breaks even: 13.7% at 10% PD, 36% at 20%, no finite rate beyond 27%.
- **Lender of last resort.** Because there is no outside option, an offer at 98.9% is taken only when both rivals refuse or quote above it; it cannot lose a competitive loan and earns a lot when accepted.

### 2.3 The submitted rule (file `output/round1/rates_round1.csv`, SHA-256 `5aba4d61…`)

Income segments 0 / 1–5k / 5–10k / 10–20k / 20–40k / 40k+; margins `m(s)` = 4.0 / 3.5 / 3.5 / 3.0 / 3.0 / 3.0 points; thresholds `τ(s)` = 12 / 12 / 10 / 8 / 25 / 25% PD.

```
r_safe = p̂/(1−p̂) + m(s)
r_band = min( π̃(p̂)/(1−π̃(p̂)) + 0.05 , 0.989 )
r      = r_safe + w(p̂) · (r_band − r_safe),   w(p̂) = 1 / (1 + exp(−(p̂ − τ(s) − 0.01)/0.01))
```

Composition: 54,418 applicants below their threshold (rates 3.0–18.6%), 45,582 above (12.8–98.9%), 2,027 at the cap. Built by `python make_submission.py --adjusted-x 0.05 --cap 0.989 --phase-delta 0.01 --phase-width 0.01` (the model is refitted at full precision; the checksum and parameters are stored in `rates_round1_params.json`).

### 2.4 How the margin was chosen (`analysis/final_margin_choice_v2.py`, note `report/stage1_final_rule_note_v5.pdf`)

A market simulator (`lending_game/simulator.py`, exact applicant types, fair ties) and seven profiles of rivals (`analysis/rivals.py`: rivals pricing like us with a random margin, curse-aware "smart" rivals, clones, best-responders, a mixed bag, netters, model-followers). 500 random rival pairs per profile. The margin X = 0.05 maximised the prior-weighted expected profit subject to a probability of finishing last of at most 3%: 15.5 M€ prior-weighted, first in 78% of simulated markets, last in 1.6%. An income-conditioned version of the curve was analysed and validated (+0.46 M€ in simulation) but not adopted for stage 1; its files are in `output/round1/alternatives/`.

The stage-1 documents, in order: `report/stage1_strategy_note*.pdf` (early, flat-net version, stale), `report/stage1_final_rule_note_v5.pdf` (the rule actually submitted; v4 is stale), `report/handover_for_advisor*.md` and `report/reply_to_advisor_v2.md` (an external methodological review and our answers), `output/round1/diagnostics_round1.md` (model diagnostics).

---

## 3. Round-1 outcome

League table (market share, P&L): LookingForAnInternshipInFinance 37.5% / 73.3 M; HECredit 38.4% / 62.5 M; The Walking Debt 41% / 53.9 M; Ponzi & Co 55.6% / 35.0 M; Happiness in the Household 49.2% / 30.9 M; **Goldman Stanley 51% / 28.7 M**; Lehman Sisters 45.5% / 23.4 M; Only God Can Trust Us 40.6% / 10.7 M; the three ChatGPT lenders 48% / 4.4–5.4 M; Loan and Behold 26.3% / 3.1 M.

Profit per loan won separates two groups: the leaders made 1,300–1,950 € per loan on 37–41% share; we and the other high-share teams made 500–630 € per loan. That difference is price level, not model quality (all signals are equally informative). Our 28.7 M€ was above the top of our own simulated range (7–25 M€ across profiles): the rivals were softer than any profile we had imagined.

---

## 4. The diagnostic, in full (`analysis/round1_diagnostic.py`, `analysis/round1_expost_optimum.py`; tables and figures in `output/round1/diagnostic/`)

Our file is `Round_1_report_new/Round1_Diagnostic_23.csv` (verified: profits sum to 28,677,721 and all 100,000 `own` rates equal our submission). For every applicant it gives both rival offers (or refusal), the borrower type, and the default outcome **for everyone**, won or not.

### 4.1 Where the money came from

| part of our rule | offers | won | share | profit | per loan won | realized default of won loans |
|---|---|---|---|---|---|---|
| safe part (below threshold) | 54,418 | 25,292 | 46% | 5.7 M€ | 225 € | 1.0–1.4 pts above our PD |
| band (adjusted break-even + 5) | 43,555 | 25,112 | 58% | 21.1 M€ | 838 € | about 1.5 pts above our PD |
| cap at 98.9% | 2,027 | 627 | 31% | 1.9 M€ | 3,070 € | 34–38% |

By segment: the zero-income band alone made 11.3 M€ (66% share, 1,207 € per loan); the 5–10k band 7.9 M€; the 10–20k band only 0.1 M€ (70 € per loan: won loans defaulted 14.4% against a 10.5% PD, the one place where the curse did bite, as the income analysis had predicted). The safe part of the 10–20k segment, our largest group of offers, made 1.6 M€ at 159 € per loan.

### 4.2 The realized winner's curse

Default of the loans we won versus our PD: +0.3 pt at 1% PD, +1.1 at 3%, +1.2 at 5%, +1.4 at 7%, +0.8 at 9%, +0.4 at 11%, +1.0 at 13%, +1.5 at 15%, +1.6 at 17%, +1.5 at 19%, +1.8 at 23%, −1 at 27%. We had priced the band for +5 to +20 points. The curse was mild because the rivals mostly did not bid on those applicants; an absent rival selects nothing. On the safe part it was slightly *larger* than priced, because the rivals undercut us on good risks and left us the applicants whose other signals were bad (figure `figD_round1_diagnostic.png`, panel C).

### 4.3 What the rivals did

- **Refusals.** Type-1 rivals made no offer to 34% of applicants, type-3 to 38%, both to 14%. At PD below 2% the refusal rate is a flat 25% for each type, identical across income, employment, sex, marital status, our signal and borrower type; it then rises with PD to 67% (type 1) and 50% (type 3). The flat quarter is consistent with every rival refusing applicants whose own signal is exactly zero (21% of applicants have `signal_k = 0`, independently of everything we observe), or with one absent lender per type. We cannot tell which; the difference matters for round 2 (Section 6).
- **A 92% net.** One type-1 lender quotes exactly 0.9199 to roughly everything above 8% PD: 14% of all type-1 draws. One type-3 lender uses a handful of flat levels between 37% and 54% in the band.
- **Cheap pricers on the safe part.** The lowest quarter of rival offers sits 0–2 points above the plain break-even (the LLM lenders and at least one student team). The median rival offer at PD below 2% was 2.5% against our 3.8%; our safe rate was below the median rival's in only 6% of cases there and 23–47% elsewhere below 12% PD.
- **Rival rate distributions** by PD bin, with deciles, refusal rates, mark-ups and a rough split into the four lenders behind each type: `rival1_by_pd_bin.csv`, `rival3_by_pd_bin.csv`, and section D of `run_diagnostic_23.log`.

### 4.4 Where we left money (rivals fixed, realized defaults)

- **Headroom.** On the loans we won, the median gap to the best rival offer was 1–3 points in the safe bins and 17–60 points in the band. 15–33% of band applicants had no other offer at all; 6.5% of the safest ones too.
- **Ex-post optimum.** The realized profit of a single rate per PD bin rises almost monotonically up to 89% in every bin (figure `figE_expost.png`, panel A); 89% is the 92% net minus the 2% preference. Per-bin optima sum to 128 M€; "0.89 for everyone" would have made 126.5 M€ at 21% share; "safe part unchanged, band at a flat 0.89" 76 M€ at 35% share; pricing everyone at 98.9% (winning only the applicants nobody else wanted) 99 M€. Those are upper bounds, with yesterday's rivals, but they explain the league table.
- **Lost loans.** Of 48,969 offers not accepted, 2,982 were lost on the preference alone; the rest to a cheaper rival (half to each type). The rivals earned 165–220 € per loan on the safe loans they took from us and 400–830 € on band loans.
- **The pool nobody wanted** (both rivals refused) defaulted 2.0% at PD 1% (vs 0.9% for the contested ones), 13.9% vs 12.3% at 13%, 25.3% vs 21.4% at 23%: slightly worse than the contested pool, far below the 50% break-even of a 98.9% rate (`expost_uncontested_default.csv`).

---

## 5. What we learned (for the report's diagnosis section)

1. Our economics were right and too cautious. The break-even and the curse are correct concepts; the curse was priced for a market of model-competing rivals that did not exist in round 1.
2. The binding fact of this market is **who makes no offer**, not who prices best. With no outside option, an applicant refused by both rivals accepts any rate up to 100%, and a 92% net sets the ceiling for everybody else.
3. The safe part is a thin business: 54% of applicants, 20% of profit, 225 € per loan, genuine competition within 1–3 points, cheap LLM-like pricers in a quarter of draws. Raising our safe margins would have lost share for little gain in every simulation; it only pays if the refusers persist.
4. The model is fine and cannot be the lever: adding the 100,000 round-1 outcomes to the training set improves AUC by 0.0008 (`analysis/round2_scoring.py`, `output/round2/scoring_check.txt`).
5. Income matters at a given PD (the 10–20k band lost money; the zero-income band was our best business); the income-conditioned curve is validated and available if a curve-based band is ever used again.

---

## 6. Round 2: the recommendation and why

### 6.1 The decision problem

Everything we know is in the diagnostic; what we do not know is how the rivals move. Everyone received a diagnostic and the league table; the cheap, high-share teams will raise prices; the refusers may or may not start bidding; the netter may keep 0.92, raise it, or be undercut by a copycat at 0.89 or 0.84. The scenario engine `analysis/round2_scenarios.py` evaluates candidate schedules on the round-1 applicants with *realized defaults* under transformations of the actual rival offers:

- S0: rivals exactly as in round 1.
- S1: the flat quarter of refusals disappears (replaced by draws from the same lender type's offers in the same PD bin).
- S2: S1 and the 92% net lender moves down to 0.80.
- S3: S1 and every rival offer rises by 5 points.
- S4: S1 and a copycat prices every applicant above 10% PD at 0.84 in one draw out of four.
- S5: no refusal at all and every offer +5 points (the toughest plausible market).

Weights used for the prior-weighted mean: 0.15 / 0.30 / 0.15 / 0.15 / 0.15 / 0.10 (judgment, not data; change them in the script if you disagree).

### 6.2 Results (realized profit in M€; `output/round2/scenarios.csv`)

| candidate (safe part unchanged unless stated) | S0 | S1 | S2 | S3 | S4 | S5 | prior mean | worst |
|---|---|---|---|---|---|---|---|---|
| submitted stage-1 rule | 28.7 | 17.9 | 17.6 | 34.8 | 17.6 | 29.3 | 23.1 | 17.6 |
| band = adjusted break-even + 0.20 | 41.7 | 21.0 | 20.8 | 36.6 | 20.6 | 25.7 | 26.8 | 20.6 |
| band flat 0.60 | 51.8 | 22.3 | 22.3 | 33.7 | 22.3 | 17.8 | 28.0 | 17.8 |
| band flat 0.75 | 69.4 | 28.3 | 28.3 | 39.0 | 28.3 | 15.9 | 34.8 | 15.9 |
| **band flat 0.78 (recommended)** | 72.8 | 29.4 | 23.7 | 40.2 | 29.4 | 15.7 | 35.3 | 15.7 |
| band flat 0.82 | 77.6 | 31.1 | 19.2 | 41.8 | 27.0 | 15.4 | 35.7 | 15.4 |
| band flat 0.85 (alternative) | 81.2 | 32.5 | 13.7 | 43.0 | 24.4 | 15.3 | 35.6 | 13.7 |
| band flat 0.89 | 86.0 | 34.3 | 14.4 | 44.7 | 21.3 | 15.1 | 36.8 | 14.4 |
| 0.89 for everyone | 126.5 | 35.8 | 10.9 | 35.9 | 22.1 | 0.8 | 40.1 | 0.8 |
| safe +5, band 0.85 | 82.5 | 31.2 | 12.5 | 41.5 | 23.2 | 13.2 | 34.6 | 12.5 |
| safe +20, band 0.89 | 94.0 | 32.5 | 12.7 | 33.5 | 19.6 | 2.4 | 34.0 | 2.4 |

Reading: (i) every flat band between 0.75 and 0.89 beats the stage-1 rule by 12–14 M€ on the prior mean and is never catastrophic; (ii) the higher the band, the more it depends on the 92% net staying put (S2 and S4); 0.78 keeps most of the upside while surviving a net at 0.80 or a copycat at 0.84; (iii) raising the safe part loses money in every scenario except the literal repeat of round 1, because the refused safe applicants are the only reason it ever pays; (iv) "a high rate for everyone" is the ex-post champion and the worst thing to do if the refusals stop. The 0.989 cap above 30% PD adds 0.2 M€ and is optional.

### 6.3 The recommended file

`output/round2/rates_round2_band078.csv` (SHA-256 `4fee915a…`), built by

```
python make_submission.py --applications Round_1_report_new/NewApplications_Lender2_Round2.csv --net 0.78 --augment-round1 --out output/round2/rates_round2_band078.csv
```

Safe part unchanged (`p̂/(1−p̂) + m(s)` below the segment thresholds), a flat 0.78 above them, model trained on past loans plus the round-1 applicants with their observed defaults. The alternative `rates_round2_band085.csv` (SHA-256 `ac6fc6bd…`) is the same with 0.85. Expect, if the market looks like S1–S4, a share around 35% and 25–40 M€; if the rivals do not move at all, 70–80 M€. The rate jumps at the threshold (from ~15% to 78%); that is deliberate, the band price is now a take-it-or-leave-it offer for applicants the rivals avoid.

If you prefer a different level, `--net X` sets it; `--safe-extra 0.02` adds two points to every safe margin; `--adjusted-x X` rebuilds a curve-based band instead. Rerun `analysis/round2_scenarios.py` after any change to the candidate list.

---

## 7. Before sending the CSV

1. Open the params file written next to the CSV (`..._params.json`): the command, the application file (`NewApplications_Lender2_Round2.csv`), the training set, and the SHA-256 of the file you send.
2. Check: 100,000 rows, ids 100001 to 200000 in order and unique, columns `id,rate`, no missing rate (we always offer), no rate above 1.0, rates with four decimals.
3. Recompute the checksum of the file you attach (`sha256sum` or `Get-FileHash`) and compare with the params file. Keep both under version control.
4. Do not open the CSV in Excel and save it (Excel rewrites numbers).

---

## 8. The 3-page report (70% of the grade): a suggested outline

The guidelines ask for (a) the methodology for the default probability and the rate in stage 1, with the problem created by the rivals' private information and how we tried to overcome it, and (b) the diagnosis of why we made money and how the strategy changed for stage 2. Text, not bullets; include formulas.

1. **Default probability** (half a page): data, features, why sex was dropped and why `signal = 0` is "no signal", the model, out-of-sample AUC 0.735 and calibration; the information structure: public 0.717, one signal 0.735, three signals 0.760; the rivals' hidden information (4.5 points, 7 at zero income).
2. **From PD to price** (one page): break-even `r* = p/(1−p)` and why `r = p` loses money; the 2% preference and the "fans vs everyone" logic; the winner's curse with the formula `π̃(p̂) = Σ ω_i p_full,i / Σ ω_i` and the measured curve (Table 3 and Figure 2 of `stage1_final_rule_note_v5.pdf`); the rule (formula above), the segment parameters, the cap as a lender-of-last-resort offer; the Monte Carlo and the criterion that fixed X = 0.05. Figures available: `output/round1/figures/figA_curse_and_rule.png`, `fig13b_final_margin_v2.png`.
3. **Diagnosis** (one page): the league table and profit per loan; the three tables of Section 4 (money by part, realized curse, refusals and the 92% net); the ex-post optimum and the headroom (figures `figD_round1_diagnostic.png`, `figE_expost.png`); the sentence that summarises it: *the curse we priced for assumed rivals compete on their models; they mostly stayed away, and an absent rival creates no selection but leaves an applicant who will pay anything*.
4. **Stage 2** (half a page): the scenario table of Section 6, the chosen band level and the reasoning about the net and the refusals, what would make it fail, and what we kept (model, safe part, cap logic).

---

## 9. Map of the repository

```
PastLoans.csv, NewApplications_Lender2_Round1.csv      data of stage 1 (round-2 applications are in Round_1_report_new/)
Project instructions.txt, Slides-5.pdf, Slides8_Scoring.pdf   course material
run_round1.py                 stage-1 pipeline: features, model, out-of-sample PDs, rule optimisation -> output/round1/
make_submission.py            builds a submission CSV from the analysed rule (options: --net, --ramp-from, --adjusted-x, --cap,
                              --phase-delta/--phase-width, --floor-by-segment, --safe-extra, --applications, --augment-round1, --out)
lending_game/                 features.py (make_features, income_segment), models.py (fit_full, oos_predict), simulator.py (market
                              rules, exact types), competitors.py (early rival archetypes), strategy.py (Rule, optimizer)
analysis/                     one script per question; the ones that matter now:
  rivals.py                       the seven rival profiles used by every Monte Carlo
  curse_adjusted_breakeven.py     measures pi~ and writes curse_floor.csv (the pooled curve of the submitted rule)
  curse_floor_by_segment.py, _cv.py   income-conditioned curve and its two-fold validation (not adopted)
  final_margin_choice_v2.py       the 500-draw Monte Carlo that fixed X = 0.05 (add "plot" to redraw from saved draws)
  final_figures.py                figure A of the note; plot_rule.py, band_figures.py, cross_profiles.py: other figures
  round1_calibration.py           which rival profiles reproduce the league-table outcome
  round1_diagnostic.py            the post-mortem on our diagnostic file (sections A-H)
  round1_expost_optimum.py        ex-post optimum per PD bin, refusal anatomy, candidate schedules
  round2_scoring.py               scores the round-2 applicants, tests the augmented training set
  round2_scenarios.py             the scenario engine of Section 6
  make_fake_diagnostic.py         synthetic diagnostic used to test round1_diagnostic.py
  (older: band_payoff_matrix.py, band_payoff_ci.py, monte_carlo_*.py, breakeven_ramp.py, clone_scenarios.py, net_rate_scenarios.py,
   margin_transition.py, smooth_transition.py, cap_region.py, curse_sensitivity.py, ... = the stage-1 exploration, documented in report/)
output/round1/                stage-1 outputs: rates_round1.csv (+ params), strategy_round1.json, curse_floor*.csv,
                              new_applications_scored.csv, oos_probas_sex0.npz (out-of-sample PDs of the past loans), all simulation tables,
                              figures/, diagnostic/ (the post-mortem), alternatives/ (income-segment files, not submitted)
output/round2/                new_applications_scored.csv, scoring_check.txt, scenarios.csv, rates_round2_band078.csv, rates_round2_band085.csv
report/                       stage-1 notes (LaTeX source + PDFs; v5 is current), advisor briefs and replies, this file's companions
Round_1_report_new/           OUR round-1 results: Results_Round1.pdf, Round1_Diagnostic_23.csv, NewApplications_Lender2_Round2.csv
Round_1_report/               ANOTHER TEAM'S files received by mistake: ignored by git; delete, do not use
```

Environment: Python 3.13 with numpy, pandas, scikit-learn, scipy, matplotlib; LaTeX via TinyTeX (`latexmk -pdf`; the `enumitem` package is missing, do not use it). Scripts are run from the project folder (`python analysis/<script>.py`). The heavy ones (500-draw Monte Carlo) take about ten minutes; the diagnostic scripts a minute.

---

## Appendix: the formulas in one place

- Expected profit of a loan at rate `r` to an applicant with default probability `p`: `10,000 · [(1−p) r − p]`; break-even `r* = p/(1−p)`.
- Choice rule: lender `k` wins applicant `i` of type `t` if `r_k,i − 0.02·1[t = k]` is the smallest of the three effective rates; no offer counts as infinite.
- Win probability under symmetric pricing (used to measure the curse): `ω_i = (1/3) Σ_t 1{ r_2,i − 0.02·1[t=2] < min_{k≠2}( r_k,i − 0.02·1[t=k] ) }`.
- Selected default rate per PD bin B: `π̃(B) = Σ_{i∈B} ω_i p_full,i / Σ_{i∈B} ω_i`, with `p_full` a diagnostic estimate from a model trained on all three signals; realized version with `D_i` instead of `p_full,i`.
- Adjusted break-even: `π̃/(1−π̃)`; the stage-1 band: `min(π̃/(1−π̃) + 0.05, 0.989)`; the stage-2 band: a flat 0.78.
- Headroom on a won loan: `(min rival effective rate + 0.02·1[t=2]) − our rate`; ex-post profit at the maximum winning rate is the upper bound of what the loan could have earned against those two rivals.
