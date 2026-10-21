# Alternatives to the stage-1 submission (NOT submitted)

Built on 2026-10-14 with `make_submission.py --floor-by-segment`: the band priced on the income-segment adjusted
break-even curve (`output/round1/curse_floor_by_segment.csv`, `analysis/curse_floor_by_segment.py`) instead of the
pooled curve. Analysed in the note (Section 2.1) and in `report/reply_to_advisor_v2.md`; the team kept the pooled
curve for stage 1 (`output/round1/rates_round1.csv`, sha256 5aba4d61...). Candidate for stage 2.

- `rates_round1_income_segment_x050.csv` (+ `_params.json`): X = 0.05, sha256 35369fc7...
- `rates_round1_income_segment_x075.csv` (+ `_params.json`): X = 0.075, sha256 88ddd130...
- `cap_region_income_segment.log`: cap-region check with the segment curve (`python analysis/cap_region.py 60 seg`)

Rebuild (from the project folder):

    python make_submission.py --adjusted-x 0.05 --cap 0.989 --phase-delta 0.01 --phase-width 0.01 --floor-by-segment --out output/round1/alternatives/rates_round1_income_segment_x050.csv

Simulation comparing both curves on the same rival draws: `analysis/final_margin_choice_v2.py` (500 draws per profile;
`output/round1/final_margin_choice_v2.csv`, figure `figures/fig13b_final_margin_v2.png`).
