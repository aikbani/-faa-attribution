# FAA Attribution Forecast — polymarket-faa-attribution-02

- Repo: **aikbani/-faa-attribution** (public GitHub, created 2026-11-14T13:37:46Z — after the 02:00Z open, default branch main, https://github.com/aikbani/-faa-attribution; workdir `~/workspace/makefun/work/faa-attribution/` stands in for it until pushed)
- Competition: polymarket-faa-attribution-02 — "Forecast Polymarket's FAA outage attribution prices minute by minute, 4 hours ahead"
- Prize: $250, top 100 places, ESCROWED (submit signed). Deadline: 2026-11-14 21:00 WIB.
- Baseline to beat (practice): house baseline loss 0.363589 pinball points.

## Design (main.py, Python 3.13, stdlib only)

- Reads `$MAKEFUN_INPUT_DIR` else `/input/`, every `*.in` in name order; writes one `/output/<id>.out` per case (`$MAKEFUN_OUTPUT_DIR` else `/output/`).
- Per market, per horizon minute h = 1..240, five quantiles (0.05/0.25/0.5/0.75/0.95):
  `q_tau = clamp(x + z_tau * sigma_1min * sqrt(h), 0, 1)`, rounded to 6 decimals, enforced non-decreasing per row.
- `x` = last price at checkpoint (martingale median, no drift).
- `sigma_1min` = robust realized volatility: median(|1-min moves|) / 0.6745 over recent windows (last 120, 360, 720 minutes), median of the window estimates; floored at 0.0012, capped at 0.02. Falls back to 0.0025 (house baseline) when a market has < 2 prices.
- Fatter outer tails for jumps (real family: Iran 0.49 -> 0.275 in one minute): z = (-1.75, -0.67449, 0, 0.67449, 1.75) instead of the normal ±1.644854.
- Deterministic, no network, no randomness; per-market try/except falls back to the baseline-style forecast so a bad market can never invalidate a case.

## Local validation (2026-11-14 ~20:40 WIB)

- Run: `mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py` — all 6 practice cases in 0.62s total (limit 8s/case).
- Format: all 6 outputs valid — right market ids/counts (19/19/19/23/24/24), 240 rows/market, 5 non-decreasing numbers in [0,1] per row, sizes 200–258 KB (< 1 MiB cap).
- Determinism: 2 runs byte-identical (`diff -r out out2` clean).
- Practice scoring (local, vs real Polymarket CLOB answers fetched for every case/market, `score.py`, answers cached in `/tmp/faa-answers.json`):
  - **Mine: 0.326694** (median err 0.7675 pp, 90% coverage 0.9004)
  - **Starter baseline: 0.363589** (median err 0.7675 pp, 90% coverage 0.9496) — reproduces the rules' stated 0.363589 exactly, which validates the scoring pipeline.
  - **Mine beats baseline by 0.036896 (~10%)**. Coverage 0.9004 is almost exactly the honest 0.9 target (baseline is over-wide at 0.9496).
- Note: rules say the practice score is no guide to the final (different event, NVIDIA vs FAA), but the model beats the house baseline on practice and is well-calibrated.

## Repo / commit notes

- Files at root: `main.py`, `LICENSE` (MIT, copied from kit starter), `STATUS.md`.
- Entry: Python (main.py only, no main.ts).
