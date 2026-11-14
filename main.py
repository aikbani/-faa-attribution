#!/usr/bin/env python3
"""polymarket-faa-attribution-02 entry: adaptive robust-volatility quantile forecast.

For each market and each of the 240 minutes after a checkpoint, forecasts five
quantiles of the Yes price:

    q_tau(h) = clamp(x + z_tau * sigma_1min * sqrt(h), 0, 1)

where x is the market's last price at the checkpoint (prediction-market prices
are martingales, so the median stays at x), sigma_1min is a robust estimate of
the market's own 1-minute move size, and the outer z values are slightly
fatter than normal because these prices jump (the real family saw Iran
0.49 -> 0.275 in a single minute).

sigma_1min: median absolute 1-minute move / 0.6745, computed over recent
windows (last 120, 360 and 720 minutes), taking the median of the window
estimates; floored at 0.0012 and capped at 0.02 price units per sqrt-minute.
With fewer than 2 prices it falls back to the house baseline's 0.0025.

Python 3.13, standard library only. No network, no randomness, deterministic:
two runs write byte-identical output. Reads $MAKEFUN_INPUT_DIR else /input/,
writes $MAKEFUN_OUTPUT_DIR else /output/, one <id>.out per <id>.in found.
"""

import json
import math
import os
from statistics import median

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The quantile levels the contest scores, in the order each row lists them.
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
# z for each level: normal quantiles, except the outer pair widened to +-1.75
# so the 5%/95% band survives the jumps these markets actually make.
Z = (-1.75, -0.67449, 0.0, 0.67449, 1.75)

# Robust-volatility settings (price units per square-root minute).
SIGMA_FLOOR = 0.0012
SIGMA_CAP = 0.02
SIGMA_FALLBACK = 0.0025  # house baseline spread, used when a market has no moves
# Recent windows (in minutes of 1-minute moves) whose robust sigmas are combined.
WINDOWS = (120, 360, 720)
# 0.6745 = Phi^-1(0.75): median absolute deviation of a normal -> its sigma.
MAD_TO_SIGMA = 1.0 / 0.6745


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation).

    Only used when a case asks for quantile levels other than the standard
    five; the standard levels use the fixed Z table above (with the widened
    outer pair). Accurate to ~1e-9, far below the 1e-6 output rounding.
    """
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def z_for_levels(levels):
    """z values for a case's quantile levels: fixed table for the standard
    five (with fat outer tails), Acklam otherwise."""
    if tuple(levels) == QUANTILES:
        return Z
    return tuple(norm_ppf(t) for t in levels)


def clamp6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0


def robust_sigma(prices):
    """Per-minute move size from recent 1-minute moves, robust to jumps.

    Median absolute move / 0.6745 per window, median across windows.
    Returns SIGMA_FALLBACK when there are too few prices to measure.
    """
    n = len(prices)
    if n < 2:
        return SIGMA_FALLBACK
    moves = [abs(prices[i] - prices[i - 1]) for i in range(1, n)]
    estimates = []
    for w in WINDOWS:
        window = moves[-w:] if len(moves) > w else moves
        if window:
            estimates.append(median(window) * MAD_TO_SIGMA)
    # One more estimate from the whole history: it steadies markets whose
    # recent windows are too short or too quiet to measure on their own.
    estimates.append(median(moves) * MAD_TO_SIGMA)
    if not estimates:
        return SIGMA_FALLBACK
    sigma = median(estimates)
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))


def forecast_market(prices, horizon, z):
    """240 rows x 5 quantiles for one market, median = last price."""
    x = prices[-1]
    sigma = robust_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        row = [clamp6(x + zv * spread) for zv in z]
        # Enforce non-decreasing after rounding (rounding can tie, never invert,
        # but keep the guarantee explicit so a row is always valid).
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def baseline_market(prices, horizon, z):
    """House-baseline-style fallback: fixed 0.0025 spread, normal z."""
    x = prices[-1]
    z_base = tuple(norm_ppf(t) for t in QUANTILES) if tuple(z) != Z else \
        (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)
    rows = []
    for h in range(1, horizon + 1):
        spread = SIGMA_FALLBACK * math.sqrt(h)
        row = [clamp6(x + zv * spread) for zv in z_base]
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def process_case(case):
    """One case dict -> {"forecasts": [{"id", "q"}, ...]}; never raises per market."""
    horizon = int(case["horizon_minutes"])
    z = z_for_levels(case.get("quantiles", QUANTILES))
    forecasts = []
    for m in case["markets"]:
        prices = m["prices"]
        if not prices:
            raise ValueError("market %s has no prices" % m["id"])
        try:
            q = forecast_market(prices, horizon, z)
        except Exception:
            q = baseline_market(prices, horizon, z)
        forecasts.append({"id": m["id"], "q": q})
    return {"forecasts": forecasts}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in inputs:
        case_id = name[: -len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = process_case(case)
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
