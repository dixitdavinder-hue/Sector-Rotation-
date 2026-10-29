"""
Indian Sector Rotation Screener
================================
Ranks NSE sectoral indices against Nifty 50 and flags which sectors are
OUTPERFORMING now and which are SET UP TO OUTPERFORM next.

Method (all relative to Nifty 50, weekly data):
  1. RS line            = Sector / Nifty
  2. RS-Ratio  (trend)  = 100 + z-score of RS vs its moving average  (>100 = beating Nifty)
  3. RS-Momentum        = 100 + z-score of RS-Ratio's rate of change (>100 = gaining)
  4. RRG quadrant:
        Leading   (Ratio>100, Mom>100)  -> outperforming and still gaining
        Weakening (Ratio>100, Mom<100)  -> outperforming but losing steam
        Lagging   (Ratio<100, Mom<100)  -> underperforming
        Improving (Ratio<100, Mom>100)  -> underperforming but turning up  <- early rotation candidates
  5. Extra confirmation: relative returns (1M/3M/6M), Mansfield RS, sector above its 50-day SMA,
     and the direction of the RRG tail over the last few weeks.
  6. Leading stocks: inside sectors that are outperforming / turning up, every stock is scored on
     RS vs its sector and vs Nifty, 20/50/200 EMA trend, RSI(14), MACD(12,26,9), ADX(14),
     Supertrend(10,3), volume surge, distance from 52-week high, with a suggested stop loss.
  7. Swing (2-20 days) and Positional (1-6 months) scores: weekly trend, rising 200 EMA, base /
     consolidation breakouts, pullbacks to 20 EMA, stock Mansfield RS, fresh MACD crosses,
     with entry, stop loss and 2R / 3R targets for each.

Usage:
    python -m pip install yfinance pandas numpy matplotlib
    python sector_rotation_screener.py                 # default settings
    python sector_rotation_screener.py --top 8         # 8 stocks per sector
    python sector_rotation_screener.py --all-sectors   # scan stocks of every sector
    python sector_rotation_screener.py --no-stocks     # sectors only (faster)

Outputs (in ./sector_rotation_output/):
    sector_rotation_report.html   – open in a browser: RRG chart + sector table + leading stocks
    sector_rotation.csv           – sector table
    stock_screener.csv            – every scanned stock with all technical values
    rrg_chart.png                 – Relative Rotation Graph

This is a technical screening tool, not investment advice.
"""

import argparse
import base64
import io
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

BENCHMARK = ("NIFTY 50", "^NSEI")

# Yahoo Finance tickers for NSE sector indices. Any ticker that returns no data is skipped.
SECTORS = {
    "Bank":            "^NSEBANK",
    "IT":              "^CNXIT",
    "Auto":            "^CNXAUTO",
    "Pharma":          "^CNXPHARMA",
    "FMCG":            "^CNXFMCG",
    "Metal":           "^CNXMETAL",
    "Realty":          "^CNXREALTY",
    "Energy":          "^CNXENERGY",
    "Infra":           "^CNXINFRA",
    "PSU Bank":        "^CNXPSUBANK",
    "Media":           "^CNXMEDIA",
    "PSE":             "^CNXPSE",
    "MNC":             "^CNXMNC",
    "Consumption":     "^CNXCONSUM",
    "Commodities":     "^CNXCMDT",
    "Fin Services":    "NIFTY_FIN_SERVICE.NS",
    "Pvt Bank":        "NIFTY_PVT_BANK.NS",
    "Healthcare":      "NIFTY_HEALTHCARE.NS",
    "Oil & Gas":       "NIFTY_OIL_AND_GAS.NS",
    "Consumer Durables": "NIFTY_CONSR_DURBL.NS",
    "Midcap 100":      "NIFTY_MIDCAP_100.NS",
    "Smallcap 100":    "^CNXSC",
}

# Major stocks of each sector index (NSE symbols). Index constituents change over time —
# edit these lists as you like. Symbols with no data on Yahoo are skipped automatically.
SECTOR_STOCKS = {
    "Bank": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK",
             "BANKBARODA", "PNB", "CANBK", "FEDERALBNK", "AUBANK", "IDFCFIRSTB"],
    "Pvt Bank": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "INDUSINDBK",
                 "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "RBLBANK", "CUB"],
    "PSU Bank": ["SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "INDIANB",
                 "BANKINDIA", "MAHABANK", "IOB", "CENTRALBK"],
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "PERSISTENT",
           "COFORGE", "MPHASIS", "OFSS"],
    "Auto": ["M&M", "MARUTI", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "TVSMOTOR",
             "ASHOKLEY", "BOSCHLTD", "MOTHERSON", "BHARATFORG", "TIINDIA", "EXIDEIND"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN", "AUROPHARMA",
               "ZYDUSLIFE", "TORNTPHARM", "ALKEM", "GLENMARK", "BIOCON", "MANKIND"],
    "Healthcare": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
                   "MAXHEALTH", "FORTIS", "LUPIN", "ZYDUSLIFE", "TORNTPHARM"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR",
             "GODREJCP", "MARICO", "COLPAL", "VBL", "UBL", "EMAMILTD"],
    "Metal": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "JINDALSTEL", "SAIL",
              "NMDC", "NATIONALUM", "HINDZINC", "APLAPOLLO", "COALINDIA", "JSL"],
    "Realty": ["DLF", "GODREJPROP", "LODHA", "OBEROIRLTY", "PRESTIGE", "PHOENIXLTD",
               "BRIGADE", "SOBHA", "ANANTRAJ"],
    "Energy": ["RELIANCE", "NTPC", "POWERGRID", "ONGC", "COALINDIA", "TATAPOWER",
               "ADANIGREEN", "IOC", "BPCL", "GAIL", "ADANIPOWER", "JSWENERGY"],
    "Oil & Gas": ["RELIANCE", "ONGC", "IOC", "BPCL", "GAIL", "HINDPETRO", "OIL",
                  "PETRONET", "IGL", "MGL"],
    "Infra": ["RELIANCE", "LT", "BHARTIARTL", "ULTRACEMCO", "NTPC", "POWERGRID",
              "ADANIPORTS", "GRASIM", "DLF", "SIEMENS", "INDIGO", "AMBUJACEM"],
    "PSE": ["NTPC", "POWERGRID", "ONGC", "COALINDIA", "BEL", "HAL", "IOC", "BPCL",
            "GAIL", "RECLTD", "PFC", "BHEL"],
    "Media": ["ZEEL", "SUNTV", "PVRINOX", "NETWORK18", "NAZARA", "SAREGAMA",
              "TIPSMUSIC", "DISHTV"],
    "MNC": ["MARUTI", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "SIEMENS", "ABB",
            "CUMMINSIND", "BOSCHLTD", "COLPAL", "VBL", "ESCORTS", "GILLETTE"],
    "Consumption": ["HINDUNILVR", "ITC", "BHARTIARTL", "TITAN", "ASIANPAINT", "M&M",
                    "MARUTI", "TRENT", "DMART", "ETERNAL", "NESTLEIND", "INDHOTEL"],
    "Commodities": ["RELIANCE", "ULTRACEMCO", "TATASTEEL", "JSWSTEEL", "HINDALCO",
                    "ONGC", "GRASIM", "SHREECEM", "AMBUJACEM", "PIDILITIND", "UPL"],
    "Fin Services": ["HDFCBANK", "ICICIBANK", "BAJFINANCE", "BAJAJFINSV", "SBILIFE",
                     "HDFCLIFE", "SHRIRAMFIN", "CHOLAFIN", "KOTAKBANK", "AXISBANK",
                     "SBIN", "JIOFIN"],
    "Consumer Durables": ["TITAN", "HAVELLS", "DIXON", "VOLTAS", "BLUESTARCO",
                          "CROMPTON", "KAJARIACER", "AMBER", "KALYANKJIL", "WHIRLPOOL"],
}

# Sector signals whose stocks get scanned (unless --all-sectors is used)
SCAN_SIGNALS = ("OUTPERFORMING", "LIKELY", "Watch", "Re-accelerating")


# ----------------------------------------------------------------------------- data
def fetch_prices(period="2y"):
    import yfinance as yf

    tickers = [BENCHMARK[1]] + list(SECTORS.values())
    print(f"Downloading {len(tickers)} indices from Yahoo Finance ...")
    raw = yf.download(tickers, period=period, interval="1d",
                      auto_adjust=True, progress=False, group_by="column")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]

    name_of = {BENCHMARK[1]: BENCHMARK[0], **{v: k for k, v in SECTORS.items()}}
    close = close.rename(columns=name_of)
    close = close.dropna(axis=1, how="all").ffill()

    missing = [k for k in SECTORS if k not in close.columns]
    if missing:
        print(f"  Skipped (no data from Yahoo): {', '.join(missing)}")
    if BENCHMARK[0] not in close.columns:
        sys.exit("Could not download Nifty 50 (^NSEI). Check your internet connection.")
    return close


# ----------------------------------------------------------------------------- maths
def zscore(s, window):
    return (s - s.rolling(window).mean()) / s.rolling(window).std(ddof=0)


def rrg_series(sector, bench, window=10, mom_lag=1):
    """Weekly RS-Ratio and RS-Momentum (JdK-style approximation)."""
    rs = (100 * sector / bench).ewm(span=3, adjust=False).mean()
    rs_ratio = (100 + zscore(rs, window)).ewm(span=3, adjust=False).mean()
    roc = rs_ratio.diff(mom_lag)
    rs_mom = (100 + zscore(roc, window)).ewm(span=3, adjust=False).mean()
    return rs_ratio, rs_mom


def quadrant(ratio, mom):
    if ratio >= 100 and mom >= 100:
        return "Leading"
    if ratio >= 100:
        return "Weakening"
    if mom >= 100:
        return "Improving"
    return "Lagging"


def rel_return(sec, bench, days):
    if len(sec) <= days:
        return np.nan
    return ((sec.iloc[-1] / sec.iloc[-1 - days]) - (bench.iloc[-1] / bench.iloc[-1 - days])) * 100


def analyse(daily, window=10, tail=6):
    bench_d = daily[BENCHMARK[0]]
    weekly = daily.resample("W-FRI").last().dropna(how="all")
    bench_w = weekly[BENCHMARK[0]]

    rows, tails = [], {}
    for name in [c for c in daily.columns if c != BENCHMARK[0]]:
        sec_d, sec_w = daily[name].dropna(), weekly[name].dropna()
        b_w = bench_w.loc[sec_w.index]
        if len(sec_w) < window * 3:
            continue

        ratio, mom = rrg_series(sec_w, b_w, window)
        pts = pd.DataFrame({"ratio": ratio, "mom": mom}).dropna()
        if len(pts) < tail + 1:
            continue
        tails[name] = pts.tail(tail)

        r_now, m_now = pts["ratio"].iloc[-1], pts["mom"].iloc[-1]
        q_now = quadrant(r_now, m_now)
        q_prev = quadrant(pts["ratio"].iloc[-2], pts["mom"].iloc[-2])
        q_4w = quadrant(pts["ratio"].iloc[-5], pts["mom"].iloc[-5]) if len(pts) >= 5 else q_prev

        # direction of tail over last 3 weeks (degrees; 0-90 = heading towards Leading)
        dx = r_now - pts["ratio"].iloc[-4]
        dy = m_now - pts["mom"].iloc[-4]
        heading = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        speed = float(np.hypot(dx, dy))

        b_d = bench_d.loc[sec_d.index]
        rs_d = sec_d / b_d
        mansfield = (rs_d.iloc[-1] / rs_d.rolling(250, min_periods=150).mean().iloc[-1] - 1) * 100
        rs_vs_20 = (rs_d.iloc[-1] / rs_d.rolling(20).mean().iloc[-1] - 1) * 100
        above_50dma = sec_d.iloc[-1] > sec_d.rolling(50).mean().iloc[-1]

        r1m, r3m, r6m = (rel_return(sec_d, b_d, d) for d in (21, 63, 126))

        # ---------- signal logic
        heading_up_right = 0 <= heading <= 90
        if q_now == "Leading" and (heading <= 120 or heading >= 330):
            signal = "OUTPERFORMING - strong"
        elif q_now == "Leading":
            signal = "OUTPERFORMING - momentum cooling"
        elif q_now == "Improving" and heading_up_right and rs_vs_20 > 0:
            signal = "LIKELY TO OUTPERFORM - early"
        elif q_now == "Improving":
            signal = "Watch - turning up"
        elif q_now == "Weakening" and heading_up_right:
            signal = "Re-accelerating - watch"
        elif q_now == "Weakening":
            signal = "Losing leadership - book/trim"
        else:
            signal = "UNDERPERFORMING - avoid"

        # composite score 0-100 (relative strength + momentum + confirmation)
        score = (
            30 * np.clip((r_now - 97) / 6, 0, 1)
            + 30 * np.clip((m_now - 97) / 6, 0, 1)
            + 10 * (1 if heading_up_right else 0)
            + 10 * np.clip((r1m + 5) / 10, 0, 1)
            + 10 * np.clip((r3m + 10) / 20, 0, 1)
            + 10 * (1 if above_50dma else 0)
        )

        rows.append({
            "Sector": name,
            "Quadrant": q_now,
            "Prev wk": q_prev,
            "4 wks ago": q_4w,
            "Signal": signal,
            "Score": round(score, 1),
            "RS-Ratio": round(r_now, 2),
            "RS-Mom": round(m_now, 2),
            "Heading°": round(heading),
            "Tail speed": round(speed, 2),
            "Rel 1M %": round(r1m, 2),
            "Rel 3M %": round(r3m, 2),
            "Rel 6M %": round(r6m, 2),
            "Mansfield RS": round(mansfield, 2),
            ">50DMA": "Yes" if above_50dma else "No",
            "Fresh rotation": "Yes" if q_now != q_4w else "",
        })

    table = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    table.index += 1
    return table, tails, daily.index[-1]  # last actual trading day


# ----------------------------------------------------------------------------- stocks
def fetch_stocks(symbols, period="2y"):
    import yfinance as yf

    symbols = sorted(set(symbols))
    tickers = [s + ".NS" for s in symbols]
    print(f"Downloading {len(tickers)} stocks ...")
    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    out = {}
    for sym, t in zip(symbols, tickers):
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        df = df.dropna(subset=["Close"])
        if len(df) >= 210:
            out[sym] = df
    skipped = [s for s in symbols if s not in out]
    if skipped:
        print(f"  Skipped (no/short data): {', '.join(skipped)}")
    return out


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
    a = atr(h, l, c, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def supertrend(h, l, c, n=10, m=3):
    a = atr(h, l, c, n).values
    hl2 = ((h + l) / 2).values
    ub, lb, cv = hl2 + m * a, hl2 - m * a, c.values
    fu, fl = ub.copy(), lb.copy()
    direction = np.ones(len(cv))
    for i in range(1, len(cv)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or cv[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or cv[i - 1] < fl[i - 1]) else fl[i - 1]
        if cv[i] > fu[i - 1]:
            direction[i] = 1
        elif cv[i] < fl[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    line = np.where(direction == 1, fl, fu)
    return direction[-1], line[-1]


def pct(a, b):
    return (a / b - 1) * 100


def base_info(h, l, c, days):
    """Range of the last `days` sessions (excluding today) and whether today broke out of it."""
    prior_h = h.iloc[-days - 1:-1].max()
    prior_l = l.iloc[-days - 1:-1].min()
    depth = (prior_h - prior_l) / prior_h * 100
    return depth, prior_h, prior_l, c.iloc[-1] > prior_h


def swing_positional(df, c, h, l, v, sp, bp, close, e20, e50, e200, r, hist, atr_now,
                     adx_now, di_bull, st_dir, vol_ratio, from_high, rs_sec_1m, rs_nifty_3m,
                     sector_quadrant):
    """Swing (2-20 days) and positional (1-6 months) scores, setups and trade plans."""
    ema20_s, ema200_s = ema(c, 20), ema(c, 200)
    rsi_s = rsi(c)
    vol20 = v.iloc[-21:-1].mean()
    vol_today = v.iloc[-1] / vol20 if vol20 > 0 else np.nan
    up_day = c.iloc[-1] > c.iloc[-2]

    rs_nifty_1m = pct(close, c.iloc[-22]) - pct(bp.iloc[-1], bp.iloc[-22])
    rs_nifty_6m = pct(close, c.iloc[-127]) - pct(bp.iloc[-1], bp.iloc[-127]) if len(c) > 127 else np.nan
    rs_n = c / bp
    mansfield = (rs_n.iloc[-1] / rs_n.rolling(250, min_periods=150).mean().iloc[-1] - 1) * 100

    # weekly trend
    wk = c.resample("W-FRI").last().dropna()
    w10, w30 = wk.rolling(10).mean(), wk.rolling(30).mean()
    weekly_up = bool(wk.iloc[-1] > w30.iloc[-1] and w10.iloc[-1] > w30.iloc[-1]
                     and w30.iloc[-1] > w30.iloc[-5])
    ema200_rising = ema200_s.iloc[-1] > ema200_s.iloc[-21]

    # momentum timing
    macd_fresh = hist.iloc[-1] > 0 and (hist.iloc[-6:-1] <= 0).any()
    macd_up = hist.iloc[-1] > 0 and hist.iloc[-1] >= hist.iloc[-2]
    rsi_rising = r > rsi_s.iloc[-6]

    # bases
    s_depth, s_hi, s_lo, s_break = base_info(h, l, c, 15)     # ~3-week base for swing
    p_depth, p_hi, p_lo, p_break = base_info(h, l, c, 40)     # ~8-week base for positional
    vol_confirm = vol_today >= 1.5 and up_day
    swing_tight = s_depth <= 10
    pos_base = p_depth <= 18

    pullback = (close > e20 > e50 and l.iloc[-3:].min() <= ema20_s.iloc[-1] * 1.015
                and close > ema20_s.iloc[-1])
    extended = pct(close, e20) > 8 or r > 75

    # ---------------- swing
    if s_break and swing_tight and vol_confirm:
        s_setup = "Base breakout"
    elif pullback and 40 <= r <= 60:
        s_setup = "Pullback to 20 EMA"
    elif extended:
        s_setup = "Extended - wait"
    elif swing_tight and close > e50 and pct(close, s_hi) > -4:
        s_setup = "Tight base - watch"
    elif close < e50:
        s_setup = "No setup - below 50 EMA"
    else:
        s_setup = "No clear setup"

    s_score = (
        (15 if close > e20 > e50 else 8 if close > e50 else 0)
        + (15 if 50 <= r <= 65 and rsi_rising else 8 if 45 <= r <= 70 else 0)
        + (20 if s_setup in ("Base breakout", "Pullback to 20 EMA") else
           10 if s_setup == "Tight base - watch" else 0)
        + (10 if vol_confirm else 5 if vol_ratio > 1.2 else 0)
        + (10 if macd_fresh or macd_up else 0)
        + (10 if rs_sec_1m > 0 else 0)
        + (10 if rs_nifty_1m > 0 else 0)
        + (10 if st_dir == 1 else 0)
    )

    s_entry = close
    s_stop = max(l.iloc[-5:].min(), close - 2 * atr_now)
    if close - s_stop < atr_now:                      # too tight: give at least 1 ATR
        s_stop = close - atr_now
    s_risk = close - s_stop
    s_t1, s_t2 = close + 2 * s_risk, close + 3 * s_risk

    # ---------------- positional
    if p_break and pos_base and vol_ratio > 1.1:
        p_setup = "Base breakout"
    elif weekly_up and from_high > -5:
        p_setup = "Near 52W high"
    elif weekly_up and pos_base:
        p_setup = "Building base"
    elif weekly_up:
        p_setup = "Weekly uptrend"
    elif close > e200 and ema200_rising:
        p_setup = "Early trend"
    else:
        p_setup = "Avoid - no trend"

    p_score = (
        (15 if close > e200 and ema200_rising else 0)
        + (5 if e50 > e200 else 0)
        + (15 if weekly_up else 0)
        + (10 if from_high > -5 else 6 if from_high > -15 else 0)
        + (10 if mansfield > 0 else 0)
        + (5 if rs_nifty_3m > 0 else 0)
        + (5 if (rs_nifty_6m or -1) > 0 else 0)
        + (10 if sector_quadrant in ("Leading", "Improving") else 0)
        + (10 if adx_now > 25 and di_bull else 0)
        + (10 if p_setup == "Base breakout" else 5 if pos_base else 0)
        + (5 if vol_ratio > 1.2 else 0)
    )

    p_stop = max(l.iloc[-20:].min(), close * 0.88)
    if close - p_stop < close * 0.03:
        p_stop = close * 0.95
    p_risk = close - p_stop
    p_t1, p_t2 = close + 2 * p_risk, close + 3 * p_risk

    return {
        "Swing score": int(s_score),
        "Swing setup": s_setup,
        "Swing stop": round(s_stop, 2),
        "Swing T1": round(s_t1, 2),
        "Swing T2": round(s_t2, 2),
        "Swing risk %": round(s_risk / s_entry * 100, 2),
        "Positional score": int(p_score),
        "Positional setup": p_setup,
        "Positional stop": round(p_stop, 2),
        "Positional T1": round(p_t1, 2),
        "Positional T2": round(p_t2, 2),
        "Positional risk %": round(p_risk / close * 100, 2),
        "Weekly uptrend": "Yes" if weekly_up else "No",
        "200 EMA rising": "Yes" if ema200_rising else "No",
        "Mansfield RS": round(mansfield, 2),
        "RS vs Nifty 1M": round(rs_nifty_1m, 2),
        "RS vs Nifty 6M": round(rs_nifty_6m, 2) if not np.isnan(rs_nifty_6m) else None,
        "3W base depth %": round(s_depth, 1),
        "8W base depth %": round(p_depth, 1),
        "Vol today x": round(vol_today, 2),
        "MACD fresh cross": "Yes" if macd_fresh else "No",
    }


def analyse_stock(df, sec_px, bench_px, sector_quadrant=""):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    sp = sec_px.reindex(c.index).ffill()
    bp = bench_px.reindex(c.index).ffill()
    close = c.iloc[-1]

    e20, e50, e200 = ema(c, 20).iloc[-1], ema(c, 50).iloc[-1], ema(c, 200).iloc[-1]
    if close > e20 > e50 > e200:
        trend, t_pts = "Strong uptrend", 20
    elif close > e50 > e200:
        trend, t_pts = "Uptrend", 14
    elif close > e200:
        trend, t_pts = "Above 200 EMA", 7
    else:
        trend, t_pts = "Downtrend", 0

    r = rsi(c).iloc[-1]
    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)
    if hist.iloc[-1] > 0 and hist.iloc[-1] >= hist.iloc[-2]:
        macd_state, m_pts = "Bullish", 10
    elif hist.iloc[-1] > 0:
        macd_state, m_pts = "Bullish, fading", 5
    elif hist.iloc[-1] > hist.iloc[-2]:
        macd_state, m_pts = "Bearish, improving", 3
    else:
        macd_state, m_pts = "Bearish", 0

    adx_v, pdi, mdi = adx(h, l, c)
    adx_now = adx_v.iloc[-1]
    adx_bull = adx_now > 20 and pdi.iloc[-1] > mdi.iloc[-1]
    st_dir, st_line = supertrend(h, l, c)
    atr_now = atr(h, l, c).iloc[-1]

    vol_ratio = v.tail(5).mean() / v.tail(50).mean() if v.tail(50).mean() > 0 else np.nan
    high52 = h.tail(252).max()
    from_high = pct(close, high52)

    rs_sec_1m = pct(close, c.iloc[-22]) - pct(sp.iloc[-1], sp.iloc[-22])
    rs_sec_3m = pct(close, c.iloc[-64]) - pct(sp.iloc[-1], sp.iloc[-64])
    rs_nifty_3m = pct(close, c.iloc[-64]) - pct(bp.iloc[-1], bp.iloc[-64])
    rs_line = c / sp
    rs_rising = rs_line.iloc[-1] > rs_line.rolling(20).mean().iloc[-1]

    # setup tag
    if from_high > -3 and vol_ratio > 1.3 and close > e20:
        setup = "Breakout / near 52W high"
    elif r > 75:
        setup = "Overbought - wait for dip"
    elif t_pts >= 14 and 40 <= r <= 55 and abs(pct(close, e20)) < 3:
        setup = "Pullback to 20 EMA"
    elif t_pts == 20 and r >= 55 and m_pts >= 5:
        setup = "Momentum"
    elif trend == "Downtrend":
        setup = "Weak - avoid"
    else:
        setup = "Neutral"

    rsi_pts = 10 if 50 <= r <= 70 else 5 if (40 <= r < 50 or 70 < r <= 80) else 0
    score = (
        20 * np.clip((rs_sec_3m + 10) / 20, 0, 1)
        + 10 * np.clip((rs_nifty_3m + 10) / 20, 0, 1)
        + (10 if rs_rising else 0)
        + t_pts + rsi_pts + m_pts
        + (10 if adx_bull else 0)
        + (5 if st_dir == 1 else 0)
        + (5 if vol_ratio > 1.2 else 0)
    )

    stop = st_line if st_dir == 1 else close - 2 * atr_now

    swing = swing_positional(df, c, h, l, v, sp, bp, close, e20, e50, e200, r, hist, atr_now,
                             adx_now, pdi.iloc[-1] > mdi.iloc[-1], st_dir, vol_ratio, from_high,
                             rs_sec_1m, rs_nifty_3m, sector_quadrant)
    return {
        "Score": round(float(score), 1),
        "Setup": setup,
        "Close": round(close, 2),
        "1W %": round(pct(close, c.iloc[-6]), 2),
        "RS vs Sector 1M": round(rs_sec_1m, 2),
        "RS vs Sector 3M": round(rs_sec_3m, 2),
        "RS vs Nifty 3M": round(rs_nifty_3m, 2),
        "RS rising": "Yes" if rs_rising else "No",
        "Trend": trend,
        "RSI": round(r, 1),
        "MACD": macd_state,
        "ADX": round(adx_now, 1),
        "+DI>-DI": "Yes" if pdi.iloc[-1] > mdi.iloc[-1] else "No",
        "Supertrend": "Buy" if st_dir == 1 else "Sell",
        "Vol ratio": round(vol_ratio, 2),
        "From 52W high %": round(from_high, 2),
        "Stop loss": round(stop, 2),
        "Risk %": round((close - stop) / close * 100, 2),
        **swing,
    }


def analyse_stocks(daily, sector_table, stock_data, top_n=5, all_sectors=False):
    bench = daily[BENCHMARK[0]]
    target = sector_table if all_sectors else \
        sector_table[sector_table["Signal"].str.startswith(SCAN_SIGNALS)]
    rows = []
    for _, srow in target.iterrows():
        sec = srow["Sector"]
        if sec not in SECTOR_STOCKS or sec not in daily.columns:
            continue
        for sym in SECTOR_STOCKS[sec]:
            df = stock_data.get(sym)
            if df is None:
                continue
            try:
                m = analyse_stock(df, daily[sec], bench, srow["Quadrant"])
            except Exception as e:  # bad/short data for one stock shouldn't stop the run
                print(f"  {sym}: skipped ({e})")
                continue
            rows.append({"Sector": sec, "Sector signal": srow["Signal"], "Stock": sym, **m})
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    full = pd.DataFrame(rows)
    full["Sector rank"] = full.groupby("Sector")["Score"].rank(ascending=False, method="first").astype(int)
    order = {s: i for i, s in enumerate(sector_table["Sector"])}
    full = full.sort_values(["Sector", "Score"], key=lambda col: col.map(order) if col.name == "Sector" else -col)
    top = full[full["Sector rank"] <= top_n].reset_index(drop=True)
    return full.reset_index(drop=True), top


# ----------------------------------------------------------------------------- chart
QCOL = {"Leading": "#1a9850", "Weakening": "#d9a400", "Lagging": "#d73027", "Improving": "#2c7bb6"}


def rrg_chart(tails, table, as_of):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 9))
    allx = pd.concat([t["ratio"] for t in tails.values()])
    ally = pd.concat([t["mom"] for t in tails.values()])
    pad = 0.6
    xlo, xhi = min(allx.min(), 99) - pad, max(allx.max(), 101) + pad
    ylo, yhi = min(ally.min(), 99) - pad, max(ally.max(), 101) + pad

    ax.fill_between([100, xhi], 100, yhi, color=QCOL["Leading"], alpha=.08)
    ax.fill_between([100, xhi], ylo, 100, color=QCOL["Weakening"], alpha=.08)
    ax.fill_between([xlo, 100], ylo, 100, color=QCOL["Lagging"], alpha=.08)
    ax.fill_between([xlo, 100], 100, yhi, color=QCOL["Improving"], alpha=.08)
    ax.axhline(100, color="grey", lw=.8)
    ax.axvline(100, color="grey", lw=.8)
    for txt, x, y, ha, va in [("LEADING", xhi, yhi, "right", "top"), ("WEAKENING", xhi, ylo, "right", "bottom"),
                              ("LAGGING", xlo, ylo, "left", "bottom"), ("IMPROVING", xlo, yhi, "left", "top")]:
        q = txt.capitalize()
        ax.text(x, y, f" {txt} ", ha=ha, va=va, fontsize=13, weight="bold", color=QCOL[q], alpha=.6)

    qmap = table.set_index("Sector")["Quadrant"].to_dict()
    for name, t in tails.items():
        c = QCOL[qmap.get(name, "Lagging")]
        ax.plot(t["ratio"], t["mom"], "-", color=c, lw=1.4, alpha=.75)
        ax.scatter(t["ratio"].iloc[:-1], t["mom"].iloc[:-1], s=12, color=c, alpha=.5)
        ax.scatter(t["ratio"].iloc[-1], t["mom"].iloc[-1], s=70, color=c, edgecolor="black", zorder=5)
        ax.annotate(name, (t["ratio"].iloc[-1], t["mom"].iloc[-1]), xytext=(5, 4),
                    textcoords="offset points", fontsize=9, weight="bold")

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("RS-Ratio  (relative strength vs Nifty 50)  →")
    ax.set_ylabel("RS-Momentum  →")
    ax.set_title(f"Relative Rotation Graph – NSE sectors vs Nifty 50 (weekly, as of {as_of:%d %b %Y})",
                 fontsize=12, weight="bold")
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- app report
APP_NAME = "Sector Rotation"


def make_icons(out_dir):
    """App icon (four RRG quadrants with a rotation arrow) + web-app manifest."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc

    for size in (192, 512):
        fig = plt.figure(figsize=(1, 1), dpi=size)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0", color="#12161f"))
        g, s = 0.04, 0.30
        for (x, y), q in zip([(0.5 + g / 2, 0.5 + g / 2), (0.5 + g / 2, 0.5 - g / 2 - s),
                              (0.5 - g / 2 - s, 0.5 - g / 2 - s), (0.5 - g / 2 - s, 0.5 + g / 2)],
                             ["Leading", "Weakening", "Lagging", "Improving"]):
            ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0,rounding_size=0.07",
                                        color=QCOL[q]))
        ax.add_patch(Arc((0.5, 0.5), 0.5, 0.5, theta1=200, theta2=110, color="white", lw=size / 40))
        ax.add_patch(FancyArrowPatch((0.36, 0.745), (0.47, 0.75), color="white",
                                     arrowstyle=f"-|>,head_length={size/600:.2f},head_width={size/900:.2f}",
                                     lw=0))
        fig.savefig(os.path.join(out_dir, f"icon-{size}.png"), dpi=size)
        plt.close(fig)

    manifest = {
        "name": "Sector Rotation – NSE vs Nifty",
        "short_name": APP_NAME,
        "start_url": ".",
        "display": "standalone",
        "background_color": "#0f1115",
        "theme_color": "#12161f",
        "icons": [{"src": "icon-192.png", "sizes": "192x192", "type": "image/png"},
                  {"src": "icon-512.png", "sizes": "512x512", "type": "image/png",
                   "purpose": "any maskable"}],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


APP_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Sector Rotation</title>
<meta name="theme-color" content="#12161f">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Sector Rotation">
<link rel="manifest" href="manifest.json">
<link rel="icon" href="icon-192.png">
<link rel="apple-touch-icon" href="icon-192.png">
<style>
:root{
  --bg:#f3f4f7;--card:#ffffff;--card2:#f7f8fa;--text:#141821;--muted:#6b7280;--line:#e6e8ec;
  --brand:#12161f;--accent:#2c7bb6;
  --lead:#1a9850;--weak:#d9a400;--lag:#d73027;--imp:#2c7bb6;
  --pos:#15803d;--neg:#c0392b;--shadow:0 1px 2px rgba(16,24,40,.06),0 1px 3px rgba(16,24,40,.08);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0f1115;--card:#181b22;--card2:#1f232c;--text:#e8eaf0;--muted:#9aa1ad;--line:#2a2f3a;
  --brand:#0b0d12;--accent:#5aa9e6;--pos:#4ade80;--neg:#f87171;--shadow:none;
  --lead:#2fbf6e;--weak:#e8b923;--lag:#ef5a4c;--imp:#4a9be0;}}
:root[data-theme="dark"]{
  --bg:#0f1115;--card:#181b22;--card2:#1f232c;--text:#e8eaf0;--muted:#9aa1ad;--line:#2a2f3a;
  --brand:#0b0d12;--accent:#5aa9e6;--pos:#4ade80;--neg:#f87171;--shadow:none;
  --lead:#2fbf6e;--weak:#e8b923;--lag:#ef5a4c;--imp:#4a9be0;}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;background:var(--bg);color:var(--text);
  font:15px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
.appbar{position:sticky;top:0;z-index:10;background:var(--brand);color:#fff;
  padding:calc(12px + env(safe-area-inset-top)) 16px 12px}
.appbar .in{max-width:820px;margin:0 auto;display:flex;align-items:center;gap:12px}
.appbar img{width:34px;height:34px;border-radius:9px}
.appbar h1{font-size:18px;margin:0;font-weight:700;letter-spacing:.2px}
.appbar .d{font-size:12px;opacity:.7}
.appbar .sp{flex:1}
.iconbtn{background:rgba(255,255,255,.1);border:0;color:#fff;width:36px;height:36px;border-radius:10px;
  display:grid;place-items:center;cursor:pointer}
main{max-width:820px;margin:0 auto;padding:14px 16px calc(96px + env(safe-area-inset-bottom))}
.view{display:none}.view.on{display:block;animation:f .2s ease}
@keyframes f{from{opacity:0;transform:translateY(4px)}to{opacity:1}}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:22px 4px 10px;font-weight:700}
.card{background:var(--card);border-radius:16px;box-shadow:var(--shadow);border:1px solid var(--line);
  padding:14px;margin-bottom:10px}
.tap{cursor:pointer;transition:transform .1s}.tap:active{transform:scale(.985)}
.row{display:flex;align-items:center;gap:10px}.grow{flex:1;min-width:0}
.t{font-weight:700;font-size:16px}.s{color:var(--muted);font-size:12.5px}
.pill{font-size:11.5px;font-weight:700;padding:3px 9px;border-radius:999px;white-space:nowrap;color:#fff}
.q-Leading{background:var(--lead)}.q-Weakening{background:var(--weak);color:#1b1b1b}
.q-Lagging{background:var(--lag)}.q-Improving{background:var(--imp)}
.tag{font-size:11.5px;font-weight:600;padding:3px 8px;border-radius:8px;background:var(--card2);
  border:1px solid var(--line);white-space:nowrap;display:inline-block}
.tag.g{color:var(--pos);border-color:color-mix(in srgb,var(--pos) 40%,transparent)}
.tag.r{color:var(--neg);border-color:color-mix(in srgb,var(--neg) 40%,transparent)}
.tag.b{color:var(--accent);border-color:color-mix(in srgb,var(--accent) 40%,transparent)}
.tag.y{color:var(--weak);border-color:color-mix(in srgb,var(--weak) 45%,transparent)}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.bar{height:6px;background:var(--card2);border-radius:9px;overflow:hidden;margin-top:10px}
.bar i{display:block;height:100%;border-radius:9px;background:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px}
.grid.g3{grid-template-columns:repeat(3,1fr)}
.m{background:var(--card2);border-radius:10px;padding:8px 9px;min-width:0}
.m b{display:block;font-size:14.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.m span{font-size:11px;color:var(--muted);display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.tile{border-radius:14px;padding:12px 10px;color:#fff;cursor:pointer}
.tile b{font-size:24px;display:block;line-height:1.1}.tile span{font-size:12px;opacity:.95}
.tile.q-Weakening{color:#1b1b1b}
.mini{display:flex;align-items:center;gap:10px;padding:11px 2px;border-bottom:1px solid var(--line);cursor:pointer}
.mini:last-child{border-bottom:0}
.score{width:44px;height:44px;border-radius:50%;display:grid;place-items:center;font-weight:800;font-size:14px;
  flex:none;background:conic-gradient(var(--c) calc(var(--v)*1%),var(--card2) 0)}
.score>span{width:36px;height:36px;border-radius:50%;background:var(--card);display:grid;place-items:center}
.hero{background:linear-gradient(135deg,#12161f,#23324a);color:#fff;border:0}
.hero .s{color:#c7cede}
.filters{display:flex;gap:8px;overflow-x:auto;padding:2px 0 10px;scrollbar-width:none}
.filters::-webkit-scrollbar{display:none}
.chip{border:1px solid var(--line);background:var(--card);color:var(--text);padding:7px 12px;border-radius:999px;
  font-size:13px;font-weight:600;white-space:nowrap;cursor:pointer}
.chip.on{background:var(--text);color:var(--bg);border-color:var(--text)}
.search{display:flex;gap:8px;margin-bottom:10px}
.search input,.search select{flex:1;min-width:0;padding:11px 12px;border-radius:12px;border:1px solid var(--line);
  background:var(--card);color:var(--text);font-size:15px}
.search select{flex:0 0 40%}
.empty{text-align:center;color:var(--muted);padding:30px 10px}
.chartwrap{background:#fff;border-radius:12px;padding:4px;overflow:hidden}
.chartwrap img{width:100%;display:block}
.legend .row{padding:8px 0;border-bottom:1px solid var(--line);align-items:flex-start}
.legend .row:last-child{border:0}
.note{font-size:12px;color:var(--muted);margin:18px 4px;line-height:1.5}
nav{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);
  padding:6px 8px calc(6px + env(safe-area-inset-bottom));z-index:10}
nav .in{max-width:820px;margin:0 auto;display:grid;grid-template-columns:repeat(5,1fr)}
nav button{background:none;border:0;color:var(--muted);font-size:11.5px;font-weight:600;padding:6px 0;
  display:flex;flex-direction:column;align-items:center;gap:3px;cursor:pointer}
nav button svg{width:23px;height:23px}
nav button.on{color:var(--accent)}
.sl{font-size:12px;color:var(--muted);margin-top:10px}
.seg{display:grid;grid-template-columns:repeat(3,1fr);background:var(--card2);border:1px solid var(--line);
  border-radius:12px;padding:3px;margin-bottom:10px}
.seg button{border:0;background:none;color:var(--muted);font-weight:700;font-size:14px;padding:9px 0;border-radius:9px;cursor:pointer}
.seg button.on{background:var(--card);color:var(--text);box-shadow:var(--shadow)}
.hrow{display:flex;justify-content:space-between;align-items:baseline}
.gauge{margin:10px 0}.gauge .lab{display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px}
.gbar{position:relative;height:10px;background:var(--card2);border-radius:9px;overflow:visible}
.gbar i{display:block;height:100%;border-radius:9px}
.gbar b{position:absolute;left:50%;top:-3px;width:2px;height:16px;background:var(--muted)}
.on-pill{font-weight:800;padding:4px 10px;border-radius:999px;font-size:12px;white-space:nowrap}
.on-pill.g{background:color-mix(in srgb,var(--pos) 18%,transparent);color:var(--pos)}
.on-pill.r{background:color-mix(in srgb,var(--neg) 18%,transparent);color:var(--neg)}
.inp{display:flex;gap:8px}.inp label{flex:1;font-size:12px;color:var(--muted)}
.inp input,.exit textarea{width:100%;padding:9px 10px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--text);font-size:15px;margin-top:4px;font-family:inherit}
.exit textarea{min-height:64px;resize:vertical}
.dim{opacity:.55}
details.rules summary{cursor:pointer;font-weight:700}
details.rules li{margin:4px 0;font-size:13.5px}
.hrow a{color:var(--accent);text-transform:none;letter-spacing:0;font-size:13px;cursor:pointer}
@media (max-width:480px){.grid{gap:6px}.m{padding:7px 6px}.m b{font-size:13.5px}.m span{font-size:10.5px}
  .tile b{font-size:20px}.tile{padding:10px 8px}.tile span{font-size:11px}}
@media (min-width:900px){body{font-size:15.5px}}
</style></head>
<body>
<header class="appbar"><div class="in">
  <img src="icon-192.png" alt="" onerror="this.style.display='none'">
  <div><h1>Sector Rotation</h1><div class="d" id="asof"></div></div>
  <div class="sp"></div>
  <button class="iconbtn" id="theme" aria-label="Toggle dark mode">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
  </button>
</div></header>

<main>
  <section class="view on" id="v-home"></section>
  <section class="view" id="v-sectors"></section>
  <section class="view" id="v-stocks"></section>
  <section class="view" id="v-signals"></section>
  <section class="view" id="v-chart"></section>
</main>

<nav><div class="in">
  <button data-v="home" class="on"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 10.5 12 3l9 7.5V21H3z"/><path d="M9 21v-6h6v6"/></svg>Home</button>
  <button data-v="sectors"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="8" height="8" rx="2"/><rect x="13" y="3" width="8" height="8" rx="2"/><rect x="3" y="13" width="8" height="8" rx="2"/><rect x="13" y="13" width="8" height="8" rx="2"/></svg>Sectors</button>
  <button data-v="stocks"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/></svg>Stocks</button>
  <button data-v="signals"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2 4 14h7l-1 8 9-12h-7z"/></svg>Signals</button>
  <button data-v="chart"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 3v18M3 12h18"/></svg>Chart</button>
</div></nav>

<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const QC = {Leading:"var(--lead)",Weakening:"var(--weak)",Lagging:"var(--lag)",Improving:"var(--imp)"};
const fmt = (v,d=2) => v==null||isNaN(v) ? "–" : Number(v).toLocaleString("en-IN",{maximumFractionDigits:d,minimumFractionDigits:d});
const sgn = (v,d=2) => v==null||isNaN(v) ? "–" : (v>0?"+":"")+fmt(v,d);
const pc = v => v==null ? "" : v>0 ? "pos" : v<0 ? "neg" : "";
const esc = s => String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function sigTag(sig){
  const k = sig.startsWith("OUTPERFORMING")?"g":sig.startsWith("LIKELY")?"b":sig.startsWith("UNDER")?"r":
            sig.startsWith("Losing")?"y":"b";
  const txt = sig.replace("OUTPERFORMING","Outperforming").replace("LIKELY TO OUTPERFORM","Likely to outperform")
                 .replace("UNDERPERFORMING","Underperforming");
  return `<span class="tag ${k}">${esc(txt)}</span>`;
}
function setupTag(s){
  const k = /Breakout|breakout|Momentum|Pullback|Near 52W|Weekly uptrend/.test(s)?"g":/Weak|Avoid|below 50/.test(s)?"r":
            /Overbought|Extended/.test(s)?"y":/Tight base|Building base|Early trend/.test(s)?"b":"";
  return `<span class="tag ${k}">${esc(s)}</span>`;
}
function ring(v,c="var(--accent)"){return `<div class="score" style="--v:${v};--c:${c}"><span>${Math.round(v)}</span></div>`}

/* ---------- HOME ---------- */
function home(){
  const S=D.sectors, cnt=q=>S.filter(s=>s.Quadrant===q).length;
  const by=p=>S.filter(s=>s.Signal.startsWith(p));
  const out=by("OUTPERFORMING"), early=by("LIKELY"), watch=S.filter(s=>/^(Watch|Re-acc)/.test(s.Signal));
  const uniq=a=>{const seen=new Set();return a.filter(s=>!seen.has(s.Stock)&&seen.add(s.Stock))};
  const swingTop=uniq([...D.stocks].filter(s=>/Base breakout|Pullback/.test(s["Swing setup"]))
      .sort((a,b)=>b["Swing score"]-a["Swing score"])).slice(0,3);
  const posTop=uniq([...D.stocks].filter(s=>!/Avoid/.test(s["Positional setup"]))
      .sort((a,b)=>b["Positional score"]-a["Positional score"])).slice(0,3);
  const list=(arr,empty)=>arr.length?arr.map(s=>`
    <div class="mini" onclick="openSector('${esc(s.Sector)}')">
      <span class="pill q-${s.Quadrant}">${s.Quadrant}</span>
      <div class="grow"><b>${esc(s.Sector)}</b><div class="s">Rel 1M <span class="${pc(s["Rel 1M %"])}">${sgn(s["Rel 1M %"])}%</span> · 3M <span class="${pc(s["Rel 3M %"])}">${sgn(s["Rel 3M %"])}%</span></div></div>
      ${ring(s.Score,QC[s.Quadrant])}
    </div>`).join(""):`<div class="s" style="padding:6px 2px">${empty}</div>`;
  $("#v-home").innerHTML=`
    <div class="card hero">
      <div class="s">Nifty 50 benchmark · weekly rotation</div>
      <div style="font-size:20px;font-weight:800;margin:4px 0 2px">${out.length} sector${out.length==1?"":"s"} beating Nifty</div>
      <div class="s">${early.length} turning up early · ${watch.length} on watch</div>
    </div>
    ${D.strat?`<div class="card tap" onclick="show('signals')"><div class="row"><div class="grow"><div class="t">Aaj ke signals</div>
      <div class="s">Breadth: 200 DMA ${fmt(D.strat.breadth.p200,0)}% · 20 DMA ${fmt(D.strat.breadth.p20,0)}% · ${D.strat.pivot.length} pivot · ${D.strat.pos.length} positional</div></div>
      <span class="on-pill ${D.strat.pivotOn?'g':'r'}">Pivot ${D.strat.pivotOn?'ON':'OFF'}</span>
      <span class="on-pill ${D.strat.posOn?'g':'r'}">Pos ${D.strat.posOn?'ON':'OFF'}</span></div></div>`:''}
    <div class="tiles">
      ${["Leading","Improving","Weakening","Lagging"].map(q=>`<div class="tile q-${q}" onclick="goSectors('${q}')"><b>${cnt(q)}</b><span>${q}</span></div>`).join("")}
    </div>
    <h2>Outperforming Nifty now</h2><div class="card">${list(out,"No sector is in the Leading zone this week.")}</div>
    <h2>Likely to outperform next</h2><div class="card">${list(early.concat(watch),"No sector is turning up right now.")}</div>
    <h2 class="hrow">Top swing setups <a onclick="openMode('swing')">See all ›</a></h2>
    ${swingTop.length?swingTop.map(s=>stockCard(s,"swing",withRanks(D.stocks,"Swing score").get(s))).join(""):`<div class="card s">No breakout or pullback swing setups this week.</div>`}
    <h2 class="hrow">Top positional picks <a onclick="openMode('positional')">See all ›</a></h2>
    ${posTop.length?posTop.map(s=>stockCard(s,"positional",withRanks(D.stocks,"Positional score").get(s))).join(""):`<div class="card s">No stocks in a positional uptrend this week.</div>`}
    <p class="note">Generated ${esc(D.generated)}. Technical screen only — not investment advice. Always check the chart before trading.</p>`;
}

/* ---------- SECTORS ---------- */
let secQ="All";
function goSectors(q){secQ=q;show("sectors")}
function sectors(){
  const qs=["All","Leading","Improving","Weakening","Lagging"];
  const arr=D.sectors.filter(s=>secQ==="All"||s.Quadrant===secQ);
  $("#v-sectors").innerHTML=`
    <div class="filters">${qs.map(q=>`<button class="chip ${q===secQ?'on':''}" onclick="secQ='${q}';sectors()">${q}</button>`).join("")}</div>
    ${arr.length?arr.map(s=>`
    <div class="card tap" onclick="openSector('${esc(s.Sector)}')">
      <div class="row">
        <div class="grow"><div class="t">${esc(s.Sector)}</div><div class="s">${esc(s.Signal)}</div></div>
        <span class="pill q-${s.Quadrant}">${s.Quadrant}</span>
      </div>
      <div class="bar"><i style="width:${s.Score}%;background:${QC[s.Quadrant]}"></i></div>
      <div class="grid">
        <div class="m"><span>Score</span><b>${fmt(s.Score,0)}</b></div>
        <div class="m"><span>RS-Ratio</span><b>${fmt(s["RS-Ratio"])}</b></div>
        <div class="m"><span>RS-Mom</span><b>${fmt(s["RS-Mom"])}</b></div>
        <div class="m"><span>Rel 1M</span><b class="${pc(s["Rel 1M %"])}">${sgn(s["Rel 1M %"])}%</b></div>
        <div class="m"><span>Rel 3M</span><b class="${pc(s["Rel 3M %"])}">${sgn(s["Rel 3M %"])}%</b></div>
        <div class="m"><span>Rel 6M</span><b class="${pc(s["Rel 6M %"])}">${sgn(s["Rel 6M %"])}%</b></div>
        <div class="m"><span>Mansfield</span><b class="${pc(s["Mansfield RS"])}">${sgn(s["Mansfield RS"])}</b></div>
        <div class="m"><span>&gt;50 DMA</span><b>${s[">50DMA"]}</b></div>
      </div>
      <div class="tags">
        <span class="tag">Last wk: ${s["Prev wk"]}</span><span class="tag">4 wks ago: ${s["4 wks ago"]}</span>
        ${s["Fresh rotation"]?'<span class="tag b">Fresh rotation</span>':''}
        ${D.stocks.some(x=>x.Sector===s.Sector)?'<span class="tag g">View stocks ›</span>':''}
      </div>
    </div>`).join(""):`<div class="empty">No sectors in ${secQ}.</div>`}`;
}

/* ---------- STOCKS ---------- */
const MODES={
  overall:{label:"Overall",score:"Score",setup:"Setup",
    chips:["All","Breakout","Momentum","Pullback","Overbought","Neutral","Weak"]},
  swing:{label:"Swing",score:"Swing score",setup:"Swing setup",
    chips:["All","Base breakout","Pullback","Tight base","Extended","No"]},
  positional:{label:"Positional",score:"Positional score",setup:"Positional setup",
    chips:["All","Base breakout","Near 52W high","Building base","Weekly uptrend","Early trend","Avoid"]},
};
let stQ="", stSec="All", stSet="All", stTop=true, stMode="overall";
function openSector(name){stSec=name;stSet="All";stQ="";show("stocks")}
function openMode(m){stMode=m;stSec="All";stSet="All";stQ="";show("stocks")}
function withRanks(arr,key){
  const by={};arr.forEach(s=>(by[s.Sector]=by[s.Sector]||[]).push(s));
  const r=new Map();Object.values(by).forEach(g=>g.sort((a,b)=>b[key]-a[key]).forEach((s,i)=>r.set(s,i+1)));
  return r;
}
const rs=v=>"₹"+fmt(v,v>=1000?0:2);
const M=(label,val,cls="")=>`<div class="m"><span>${label}</span><b class="${cls}">${val}</b></div>`;
function stockCard(s,mode="overall",rank){
  const md=MODES[mode], sc=s[md.score], setup=s[md.setup];
  rank=rank||s["Sector rank"];
  const head=`<div class="row">
      ${ring(sc)}
      <div class="grow"><div class="t">${esc(s.Stock)}</div><div class="s">${esc(s.Sector)} · #${rank} in sector</div></div>
      <div style="text-align:right"><div class="t">₹${fmt(s.Close)}</div><div class="s ${pc(s["1W %"])}">${sgn(s["1W %"])}% 1W</div></div>
    </div>`;
  let tags, grid;
  if(mode==="swing"){
    tags=`${setupTag(setup)}
      <span class="tag ${s.Trend==='Downtrend'?'r':/uptrend/i.test(s.Trend)?'g':''}">${esc(s.Trend)}</span>
      ${s["MACD fresh cross"]==='Yes'?'<span class="tag g">Fresh MACD cross</span>':`<span class="tag ${/^Bullish/.test(s.MACD)?'g':'r'}">MACD ${esc(s.MACD)}</span>`}
      <span class="tag ${s.Supertrend==='Buy'?'g':'r'}">Supertrend ${s.Supertrend}</span>`;
    grid=M("vs Sector 1M",sgn(s["RS vs Sector 1M"])+"%",pc(s["RS vs Sector 1M"]))
      +M("vs Nifty 1M",sgn(s["RS vs Nifty 1M"])+"%",pc(s["RS vs Nifty 1M"]))
      +M("RSI",fmt(s.RSI,1))
      +M("Vol today",fmt(s["Vol today x"])+"x",s["Vol today x"]>=1.5?"pos":"")
      +M("Stop",rs(s["Swing stop"]),"neg")
      +M("Target 1 (2R)",rs(s["Swing T1"]),"pos")
      +M("Target 2 (3R)",rs(s["Swing T2"]),"pos")
      +M("Risk",fmt(s["Swing risk %"],1)+"%");
  }else if(mode==="positional"){
    tags=`${setupTag(setup)}
      <span class="tag ${s["Weekly uptrend"]==='Yes'?'g':'r'}">Weekly uptrend: ${s["Weekly uptrend"]}</span>
      <span class="tag ${s["200 EMA rising"]==='Yes'?'g':'r'}">200 EMA rising: ${s["200 EMA rising"]}</span>
      <span class="tag ${/^(OUT|LIKELY)/.test(s["Sector signal"])?'g':'b'}">Sector: ${esc(s["Sector signal"].split(" - ")[0].toLowerCase())}</span>`;
    grid=M("Mansfield RS",sgn(s["Mansfield RS"]),pc(s["Mansfield RS"]))
      +M("vs Nifty 6M",sgn(s["RS vs Nifty 6M"])+"%",pc(s["RS vs Nifty 6M"]))
      +M("52W high",fmt(s["From 52W high %"],1)+"%")
      +M("ADX",fmt(s.ADX,1),s.ADX>25?"pos":"")
      +M("Stop",rs(s["Positional stop"]),"neg")
      +M("Target 1 (2R)",rs(s["Positional T1"]),"pos")
      +M("Target 2 (3R)",rs(s["Positional T2"]),"pos")
      +M("Risk",fmt(s["Positional risk %"],1)+"%");
  }else{
    tags=`${setupTag(setup)}
      <span class="tag ${s.Trend==='Downtrend'?'r':/uptrend/i.test(s.Trend)?'g':''}">${esc(s.Trend)}</span>
      <span class="tag ${/^Bullish/.test(s.MACD)?'g':'r'}">MACD ${esc(s.MACD)}</span>
      <span class="tag ${s.Supertrend==='Buy'?'g':'r'}">Supertrend ${s.Supertrend}</span>
      ${s["RS rising"]==='Yes'?'<span class="tag g">RS rising</span>':''}`;
    grid=M("vs Sector",sgn(s["RS vs Sector 3M"])+"%",pc(s["RS vs Sector 3M"]))
      +M("vs Nifty",sgn(s["RS vs Nifty 3M"])+"%",pc(s["RS vs Nifty 3M"]))
      +M("RSI",fmt(s.RSI,1))+M("ADX",fmt(s.ADX,1))
      +M("Volume",fmt(s["Vol ratio"])+"x",s["Vol ratio"]>1.2?"pos":"")
      +M("52W high",fmt(s["From 52W high %"],1)+"%")
      +M("Stop loss",rs(s["Stop loss"]))+M("Risk",fmt(s["Risk %"],1)+"%");
  }
  return `<div class="card">${head}<div class="tags">${tags}</div><div class="grid">${grid}</div></div>`;
}
const MODE_NOTE={
  overall:'"vs Sector / vs Nifty" = 3-month relative return',
  swing:"2–20 day trades · stop = recent swing low / ATR · targets at 2R and 3R",
  positional:"1–6 month trades · stop = 4-week low (max 12%) · targets at 2R and 3R",
};
function stocks(){
  const md=MODES[stMode];
  const secs=["All",...new Set(D.stocks.map(s=>s.Sector))];
  const ranks=withRanks(D.stocks,md.score);
  let arr=D.stocks.filter(s=>(stSec==="All"||s.Sector===stSec)&&(stSet==="All"||s[md.setup].includes(stSet))
      &&(!stQ||s.Stock.toLowerCase().includes(stQ.toLowerCase()))&&(!stTop||ranks.get(s)<=D.top));
  arr.sort((a,b)=>b[md.score]-a[md.score]);
  if(stSec==="All"){const seen=new Set();arr=arr.filter(s=>!seen.has(s.Stock)&&seen.add(s.Stock));}
  $("#v-stocks").innerHTML=`
    <div class="seg">${Object.entries(MODES).map(([k,m])=>`<button class="${k===stMode?'on':''}" data-mode="${k}">${m.label}</button>`).join("")}</div>
    <div class="search">
      <input id="q" placeholder="Search stock…" value="${esc(stQ)}" autocomplete="off">
      <select id="sec">${secs.map(s=>`<option ${s===stSec?'selected':''}>${esc(s)}</option>`).join("")}</select>
    </div>
    <div class="filters">
      <button class="chip ${stTop?'on':''}" id="top">Top ${D.top} / sector</button>
      ${md.chips.map(x=>`<button class="chip ${x===stSet?'on':''}" data-set="${esc(x)}">${x==="No"?"No setup":esc(x)}</button>`).join("")}
    </div>
    <div class="s" style="margin:0 4px 8px">${arr.length} stock${arr.length==1?"":"s"} · ${MODE_NOTE[stMode]}</div>
    ${arr.length?arr.map(s=>stockCard(s,stMode,ranks.get(s))).join(""):`<div class="empty">${D.stocks.length?"No stocks match these filters.":"No sectors are rotating up, so no stocks were scanned this week."}</div>`}`;
  const q=$("#q"); q.oninput=e=>{stQ=e.target.value;stocks();const n=$("#q");n.focus();n.setSelectionRange(n.value.length,n.value.length)};
  $("#sec").onchange=e=>{stSec=e.target.value;stocks()};
  $("#top").onclick=()=>{stTop=!stTop;stocks()};
  document.querySelectorAll("[data-set]").forEach(b=>b.onclick=()=>{stSet=b.dataset.set;stocks()});
  document.querySelectorAll("[data-mode]").forEach(b=>b.onclick=()=>{stMode=b.dataset.mode;stSet="All";stocks()});
}

/* ---------- CHART ---------- */
function chart(){
  $("#v-chart").innerHTML=`
    <div class="card"><div class="chartwrap"><img src="data:image/png;base64,${D.chart}" alt="Relative Rotation Graph"></div>
    <div class="s" style="margin-top:8px">Each dot is a sector; the tail shows its last few weeks. Pinch to zoom on a phone.</div></div>
    <h2>How to read it</h2>
    <div class="card legend">
      <div class="row"><span class="pill q-Improving">Improving</span><div class="s grow">Still behind Nifty but gaining. Earliest sign a sector may start outperforming.</div></div>
      <div class="row"><span class="pill q-Leading">Leading</span><div class="s grow">Beating Nifty and still gaining strength.</div></div>
      <div class="row"><span class="pill q-Weakening">Weakening</span><div class="s grow">Still ahead of Nifty but losing momentum.</div></div>
      <div class="row"><span class="pill q-Lagging">Lagging</span><div class="s grow">Behind Nifty and losing strength.</div></div>
    </div>
    <p class="note">Sectors usually rotate clockwise: Improving → Leading → Weakening → Lagging. RS-Ratio (x-axis) measures strength vs Nifty 50; RS-Momentum (y-axis) measures whether that strength is rising.</p>`;
}

/* ---------- SIGNALS ---------- */
const LS={get:(k,d)=>{try{const v=localStorage.getItem(k);return v===null?d:v}catch(e){return d}},
          set:(k,v)=>{try{localStorage.setItem(k,v)}catch(e){}}};
let CAP=Number(LS.get("cap","1000000")), RISK=Number(LS.get("risk","1")), HOLD=LS.get("hold","");
const inr=v=>"₹"+Math.round(v).toLocaleString("en-IN");
function sizing(close,stop){
  const r=close-stop; if(r<=0) return {q:0,v:0};
  const q=Math.floor(Math.min(CAP*RISK/100/r, CAP*0.15/close)); return {q, v:q*close};
}
function gauge(label,val,on){
  const col=val>=50?"var(--pos)":"var(--neg)";
  return `<div class="gauge"><div class="lab"><span>${label}</span><b style="color:${col}">${fmt(val,1)}%</b></div>
    <div class="gbar"><i style="width:${Math.min(100,val)}%;background:${col};opacity:.75"></i><b title="50%"></b></div></div>`;
}
function breadthChart(h){
  if(!h||!h.length) return "";
  const W=760,H=170,l=34,r=8,t=10,b=22,n=h.length,x=i=>l+i/(n-1)*(W-l-r),y=v=>t+(100-v)/100*(H-t-b);
  const path=k=>h.map((p,i)=>`${i?"L":"M"}${x(i).toFixed(1)},${y(p[k]).toFixed(1)}`).join("");
  const ticks=h.map((p,i)=>({p,i})).filter(o=>o.i>0&&o.p.d.slice(5,7)!==h.at(o.i-1).d.slice(5,7)&&["01","04","07","10"].includes(o.p.d.slice(5,7)));
  const MO=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;display:block" role="img" aria-label="Breadth over the last year">
    ${[0,50,100].map(g=>`<line x1="${l}" x2="${W-r}" y1="${y(g)}" y2="${y(g)}" stroke="var(--line)" ${g===50?'stroke-dasharray="4 3"':''}/><text x="${l-6}" y="${y(g)+4}" text-anchor="end" font-size="11" fill="var(--muted)">${g}%</text>`).join("")}
    <path d="${path("p20")}" fill="none" stroke="var(--weak)" stroke-width="1.4" opacity=".8"/>
    <path d="${path("p200")}" fill="none" stroke="var(--accent)" stroke-width="2.2"/>
    ${ticks.map(o=>`<text x="${x(o.i)}" y="${H-6}" text-anchor="middle" font-size="11" fill="var(--muted)">${MO.at(Number(o.p.d.slice(5,7))-1)} ${o.p.d.slice(2,4)}</text>`).join("")}
  </svg><div class="s" style="margin-top:4px"><b style="color:var(--accent)">━</b> % above 200 DMA &nbsp; <b style="color:var(--weak)">━</b> % above 20 DMA &nbsp; dashed = 50%</div>`;
}
function sigCard(x,kind,on){
  const z=sizing(x.close,x.stop);
  const top=kind==="pivot"?`<span class="tag g">Pivot high ${fmt(x.ph)} toda</span>`:`<span class="tag g">${esc(x.setup)}</span><span class="tag b">Score ${x.score}</span>`;
  return `<div class="card ${on?'':'dim'}">
    <div class="row"><div class="grow"><div class="t">${esc(x.s)}</div><div class="s">${esc(x.ind)} · 3M RS ${sgn(x.rs,1)}%</div></div>
      <div style="text-align:right"><div class="t">₹${fmt(x.close)}</div><div class="s">aaj ka close</div></div></div>
    <div class="tags">${top}<span class="tag">50 EMA ₹${fmt(x.e50)}</span></div>
    <div class="grid">
      <div class="m"><span>Entry</span><b>Kal open</b></div>
      <div class="m"><span>Stop loss</span><b class="neg">₹${fmt(x.stop)}</b></div>
      <div class="m"><span>Risk</span><b>${fmt(x.risk,1)}%</b></div>
      <div class="m"><span>Qty (${fmt(RISK,1)}% risk)</span><b>${z.q} · ${inr(z.v)}</b></div>
    </div></div>`;
}
function exitList(){
  const X=D.strat.exits, syms=HOLD.toUpperCase().split(/[\s,]+/).filter(Boolean);
  if(!syms.length) return `<div class="s">Apne stocks ke NSE symbol likhiye (jaise TCS, INFY). Ye sirf aapke phone/browser mein save rehte hain.</div>`;
  return syms.map(s=>{const v=X[s]; if(!v) return `<div class="mini"><b class="grow">${esc(s)}</b><span class="s">data nahi mila</span></div>`;
    const ex=v[0]<v[1];
    return `<div class="mini"><div class="grow"><b>${esc(s)}</b><div class="s">Close ₹${fmt(v[0])} · 50 EMA ₹${fmt(v[1])}</div></div>
      <span class="on-pill ${ex?'r':'g'}">${ex?'Exit: kal open par becho':'Hold'}</span></div>`}).join("");
}
function signals(){
  const S=D.strat;
  if(!S){ $("#v-signals").innerHTML=`<div class="empty">Aaj ke signals nahi bane (data download nahi hua). Agle update mein aayenge.</div>`; return; }
  const B=S.breadth;
  $("#v-signals").innerHTML=`
    <div class="card">
      <div class="row"><div class="grow"><div class="t">Market breadth</div><div class="s">Nifty 500 (${S.n} stocks) · ${esc(S.asOf)}</div></div></div>
      ${gauge("% stocks above 20 DMA",B.p20)}${gauge("% stocks above 50 DMA",B.p50)}${gauge("% stocks above 200 DMA",B.p200)}
      <div class="tags" style="margin-top:12px">
        <span class="on-pill ${S.pivotOn?'g':'r'}">Pivot breakout: ${S.pivotOn?'ON':'OFF'}</span>
        <span class="on-pill ${S.posOn?'g':'r'}">Positional: ${S.posOn?'ON':'OFF'}</span>
      </div>
      <div class="s" style="margin-top:8px">Pivot ke liye 200 DMA ≥ 50%, Positional ke liye 20 DMA ≥ 50% chahiye. OFF ho to naye trade nahi; purane apne exit se chalte hain.</div>
    </div>
    <div class="card">${breadthChart(S.hist)}</div>
    <h2>Position size</h2>
    <div class="card inp">
      <label>Capital (₹)<input id="cap" type="number" inputmode="numeric" value="${CAP}"></label>
      <label>Risk per trade (%)<input id="risk" type="number" step="0.25" inputmode="decimal" value="${RISK}"></label>
    </div>
    <h2>Pivot breakout signals (${S.pivot.length})</h2>
    ${S.pivotOn?'':'<div class="card s">Filter OFF hai: ye signals sirf jaankari ke liye hain, rules ke hisaab se naya trade nahi.</div>'}
    ${S.pivot.length?S.pivot.map(x=>sigCard(x,"pivot",S.pivotOn)).join(""):'<div class="card s">Aaj koi pivot breakout nahi.</div>'}
    <h2>Positional signals (${S.pos.length})</h2>
    ${S.posOn?'':'<div class="card s">Filter OFF hai: ye signals sirf jaankari ke liye hain, rules ke hisaab se naya trade nahi.</div>'}
    ${S.pos.length?S.pos.map(x=>sigCard(x,"pos",S.posOn)).join(""):'<div class="card s">Aaj koi positional signal nahi.</div>'}
    <h2>Watchlist: pivot high se 3% neeche (${S.watch.length})</h2>
    <div class="card">${S.watch.length?S.watch.map(w=>`<div class="mini"><div class="grow"><b>${esc(w.s)}</b><div class="s">${esc(w.ind)}</div></div>
      <div style="text-align:right"><div>₹${fmt(w.close)} → ₹${fmt(w.ph)}</div><div class="s">${fmt(w.dist,1)}% door · RS ${sgn(w.rs,1)}%</div></div></div>`).join(""):'<div class="s">Koi stock breakout ke kareeb nahi.</div>'}</div>
    <h2>Exit check (mere holdings)</h2>
    <div class="card exit"><textarea id="hold" placeholder="TCS, INFY, CUPID">${esc(HOLD)}</textarea><div id="exitlist" style="margin-top:8px">${exitList()}</div></div>
    <h2>Rules</h2>
    <div class="card"><details class="rules"><summary>Pivot breakout (11 saal: 21.9% saalana, max DD −23.4%)</summary><ol>
      <li>Market: Nifty 500 ke ≥ 50% stocks 200 DMA ke upar.</li>
      <li>5-bar pivots: aakhri pivot high pichle se upar, aakhri pivot low pichle se upar.</li>
      <li>Close aakhri pivot high ke upar (har pivot par ek baar) → kal open par kharido.</li>
      <li>Stop: aakhri pivot low −0.5% (entry se 3–12%).</li>
      <li>Exit: close 50 EMA ke neeche → agle din open; ya stop.</li></ol></details>
    <details class="rules" style="margin-top:8px"><summary>Positional (11 saal: 17.4% saalana, max DD −21.8%)</summary><ol>
      <li>Market: Nifty 500 ke ≥ 50% stocks 20 DMA ke upar.</li>
      <li>Industry Leading/Improving; positional score ≥ 75; setup Base breakout / Near 52W high / Weekly uptrend.</li>
      <li>Kal open par kharido. Stop: 20-din low (max 12%; 3% se kam ho to 5%).</li>
      <li>Exit: close 50 EMA ke neeche → agle din open; ya stop.</li></ol></details>
    <p class="note">Sab trades: 1% risk, ek stock max 15% capital, max 10 stocks, ek industry max 3. Backtest 2016–2026, sirf aaj ke Nifty 500 stocks par (survivorship bias) — asli returns kam ho sakte hain. Investment advice nahi.</p></div>`;
  const upd=()=>{CAP=Number($("#cap").value)||0;RISK=Number($("#risk").value)||0;LS.set("cap",CAP);LS.set("risk",RISK)};
  $("#cap").onchange=()=>{upd();signals()}; $("#risk").onchange=()=>{upd();signals()};
  $("#hold").oninput=e=>{HOLD=e.target.value;LS.set("hold",HOLD);$("#exitlist").innerHTML=exitList()};
}

/* ---------- NAV ---------- */
const R={home,sectors,stocks,chart,signals};
function show(v){
  document.querySelectorAll(".view").forEach(e=>e.classList.toggle("on",e.id==="v-"+v));
  document.querySelectorAll("nav button").forEach(b=>b.classList.toggle("on",b.dataset.v===v));
  R[v](); window.scrollTo(0,0);
}
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{if(b.dataset.v==="stocks"){stSec="All";stSet="All";} if(b.dataset.v==="sectors")secQ="All"; show(b.dataset.v)});
$("#asof").textContent="Data as of "+D.asOf;
(function(){
  const root=document.documentElement;
  try{const t=localStorage.getItem("theme");if(t)root.dataset.theme=t;}catch(e){}
  $("#theme").onclick=()=>{
    const dark=root.dataset.theme?root.dataset.theme==="dark":matchMedia("(prefers-color-scheme:dark)").matches;
    root.dataset.theme=dark?"light":"dark";
    try{localStorage.setItem("theme",root.dataset.theme)}catch(e){}
  };
})();
home();
</script>
</body></html>"""


# ----------------------------------------------------------------------------- daily strategy signals
UNIVERSE_FILE = "nifty500_list.csv"
NIFTY500_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
POS_SETUPS = ("Base breakout", "Near 52W high", "Weekly uptrend")


def load_universe():
    """symbol -> industry for the Nifty 500 (repo file first, then niftyindices.com, then built-in lists)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for f in (UNIVERSE_FILE, os.path.join(here, UNIVERSE_FILE)):
        if os.path.exists(f):
            u = pd.read_csv(f)
            return dict(zip(u.symbol, u.industry))
    try:
        import requests
        r = requests.get(NIFTY500_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        u = pd.read_csv(io.StringIO(r.text))
        return dict(zip(u["Symbol"], u["Industry"]))
    except Exception as e:  # last resort: the hand-made sector lists
        print(f"  Nifty 500 list not found ({e}); using built-in sector lists")
        return {s: sec for sec, syms in SECTOR_STOCKS.items() for s in syms}


def ew_index(closes):
    r = closes.pct_change().clip(-0.5, 0.5)
    return 100 * (1 + r.mean(axis=1).fillna(0)).cumprod()


def pivot_state(h, l, c, n=5):
    """5-bar structure pivots, confirmed n bars later (same as the backtest).
    Returns today's break-of-structure flag, the 'near breakout' flag and the levels."""
    h, l, c = h.values, l.values, c.values
    phs, pls, used, bos_today = [], [], None, False
    for t in range(len(c)):
        i = t - n
        if i - n >= 0:
            wh, wl = h[i - n:t + 1], l[i - n:t + 1]
            if h[i] == wh.max() and (wh == h[i]).sum() == 1:
                phs.append(h[i])
            if l[i] == wl.min() and (wl == l[i]).sum() == 1:
                pls.append(l[i])
        bos_today = False
        if len(phs) >= 2 and len(pls) >= 2:
            ph = phs[-1]
            if pls[-1] > pls[-2] and phs[-1] > phs[-2] and c[t] > ph and used != ph:
                used, bos_today = ph, True
    if len(phs) < 2 or len(pls) < 2:
        return None
    ph, pl, close = phs[-1], pls[-1], c[-1]
    uptrend = phs[-1] > phs[-2] and pls[-1] > pls[-2]
    stop = max(pl * 0.995, close * 0.88)
    if close - stop < close * 0.03:
        stop = close * 0.97
    return {"bos": bos_today, "near": bool(uptrend and not bos_today and ph * 0.97 <= close <= ph),
            "uptrend": bool(uptrend), "ph": ph, "pl": pl, "stop": stop}


def strategy_signals(period="2y"):
    """Daily screen for the two back-tested strategies on the Nifty 500."""
    uni = load_universe()
    print(f"\nDaily strategy screen: {len(uni)} Nifty 500 stocks")
    data = fetch_stocks(list(uni), period)
    if len(data) < 50:
        print("  too few stocks downloaded - strategy screen skipped")
        return None
    closes = pd.DataFrame({s: d["Close"] for s, d in data.items()}).sort_index()
    bench = ew_index(closes)

    def pct_above(n):
        sma = closes.rolling(n).mean()
        return (closes > sma).sum(axis=1) / sma.notna().sum(axis=1) * 100

    p20, p50, p200 = pct_above(20), pct_above(50), pct_above(200)

    # industry rotation on equal-weight industry indices (as in the backtest)
    groups = {}
    for s in closes.columns:
        groups.setdefault(uni.get(s, "Other"), []).append(s)
    ind_idx = {k: ew_index(closes[v]) for k, v in groups.items() if len(v) >= 3}
    daily = pd.DataFrame(ind_idx)
    daily[BENCHMARK[0]] = bench
    itab, _, _ = analyse(daily)
    iq = itab.set_index("Sector")
    scan_ind = {k for k, r in iq.iterrows() if r["Signal"].startswith(SCAN_SIGNALS)}

    piv, watch, pos, exits = [], [], [], {}
    for sym, df in data.items():
        try:
            c, h, l = df["Close"], df["High"], df["Low"]
            e50 = ema(c, 50).iloc[-1]
            exits[sym] = [round(float(c.iloc[-1]), 2), round(float(e50), 2)]
            b = bench.reindex(c.index).ffill()
            rs3m = pct(c.iloc[-1], c.iloc[-64]) - pct(b.iloc[-1], b.iloc[-64])
            ind = uni.get(sym, "Other")
            ps = pivot_state(h, l, c)
            close = float(c.iloc[-1])
            if ps and ps["bos"]:
                piv.append({"s": sym, "ind": ind, "close": round(close, 2), "ph": round(ps["ph"], 2),
                            "stop": round(ps["stop"], 2), "risk": round((close - ps["stop"]) / close * 100, 1),
                            "rs": round(rs3m, 1), "e50": round(float(e50), 2)})
            elif ps and ps["near"]:
                watch.append({"s": sym, "ind": ind, "close": round(close, 2), "ph": round(ps["ph"], 2),
                              "dist": round((ps["ph"] / close - 1) * 100, 1), "rs": round(rs3m, 1)})
            if ind in scan_ind and ind in ind_idx:
                m = analyse_stock(df, ind_idx[ind], bench, iq.loc[ind, "Quadrant"])
                if m["Positional score"] >= 75 and m["Positional setup"] in POS_SETUPS:
                    stop = m["Positional stop"]
                    pos.append({"s": sym, "ind": ind, "close": round(close, 2), "score": m["Positional score"],
                                "setup": m["Positional setup"], "stop": round(stop, 2),
                                "risk": round((close - stop) / close * 100, 1), "rs": round(rs3m, 1),
                                "e50": round(float(e50), 2)})
        except Exception as e:
            print(f"  {sym}: skipped ({e})")
    piv.sort(key=lambda x: -x["rs"])
    pos.sort(key=lambda x: (-x["score"], -x["rs"]))
    watch.sort(key=lambda x: x["dist"])
    hist = pd.DataFrame({"p20": p20, "p50": p50, "p200": p200}).dropna().tail(260)
    out = {
        "asOf": f"{closes.index[-1]:%d %b %Y}",
        "n": int(closes.shape[1]),
        "breadth": {"p20": round(float(p20.iloc[-1]), 1), "p50": round(float(p50.iloc[-1]), 1),
                    "p200": round(float(p200.iloc[-1]), 1)},
        "hist": [{"d": f"{d:%Y-%m-%d}", "p20": round(r.p20, 1), "p50": round(r.p50, 1), "p200": round(r.p200, 1)}
                 for d, r in hist.iterrows()],
        "pivotOn": bool(p200.iloc[-1] >= 50), "posOn": bool(p20.iloc[-1] >= 50),
        "pivot": piv, "pos": pos, "watch": watch[:40], "exits": exits,
        "industries": [{"ind": k, "q": r["Quadrant"], "sig": r["Signal"]} for k, r in iq.iterrows()],
    }
    print(f"  breadth 20/50/200 DMA: {out['breadth']}  | pivot signals {len(piv)} ({'ON' if out['pivotOn'] else 'OFF'})"
          f" | positional signals {len(pos)} ({'ON' if out['posOn'] else 'OFF'}) | watchlist {len(watch)}")
    return out


def html_report(table, fig, as_of, full_stocks=None, top_n=5, strat=None):
    import json
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    stocks = full_stocks if full_stocks is not None and not full_stocks.empty else pd.DataFrame()
    data = {
        "asOf": f"{as_of:%d %b %Y}",
        "generated": f"{datetime.now():%d %b %Y, %I:%M %p}",
        "top": int(top_n),
        "sectors": json.loads(table.to_json(orient="records")),
        "stocks": json.loads(stocks.to_json(orient="records")) if not stocks.empty else [],
        "chart": base64.b64encode(buf.getvalue()).decode(),
        "strat": strat,
    }
    payload = json.dumps(data).replace("</", "<\\/")
    return APP_TEMPLATE.replace("__DATA__", payload)



# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="NSE sector rotation screener vs Nifty 50")
    ap.add_argument("--window", type=int, default=10, help="RRG z-score window in weeks (default 10)")
    ap.add_argument("--tail", type=int, default=6, help="weeks of tail on the RRG chart (default 6)")
    ap.add_argument("--period", default="2y", help="history to download (default 2y)")
    ap.add_argument("--out", default="sector_rotation_output", help="output folder")
    ap.add_argument("--top", type=int, default=5, help="top stocks to show per sector (default 5)")
    ap.add_argument("--all-sectors", action="store_true", help="scan stocks of every sector, not only rotating ones")
    ap.add_argument("--no-stocks", action="store_true", help="sector rotation only, skip stock scan")
    ap.add_argument("--no-signals", action="store_true", help="skip the daily Nifty 500 strategy screen")
    args = ap.parse_args()

    daily = fetch_prices(args.period)
    table, tails, as_of = analyse(daily, args.window, args.tail)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"\nSector rotation vs Nifty 50 — as of {as_of:%d %b %Y}\n")
    print(table[["Sector", "Quadrant", "Signal", "Score", "RS-Ratio", "RS-Mom",
                 "Heading°", "Rel 1M %", "Rel 3M %", "Fresh rotation"]].to_string())

    full_stocks = top_stocks = None
    if not args.no_stocks:
        target = table if args.all_sectors else table[table["Signal"].str.startswith(SCAN_SIGNALS)]
        syms = [s for sec in target["Sector"] for s in SECTOR_STOCKS.get(sec, [])]
        if syms:
            stock_data = fetch_stocks(syms, args.period)
            full_stocks, top_stocks = analyse_stocks(daily, table, stock_data, args.top, args.all_sectors)
        if top_stocks is not None and not top_stocks.empty:
            print(f"\nTop {args.top} stocks in rotating sectors\n")
            for sec, grp in top_stocks.groupby("Sector", sort=False):
                print(f"--- {sec}  [{grp['Sector signal'].iloc[0]}]")
                print(grp[["Stock", "Score", "Setup", "Close", "Swing score", "Swing setup",
                           "Positional score", "Positional setup", "RSI", "ADX", "Supertrend"]]
                      .to_string(index=False))
                print()
        else:
            print("\nNo rotating sectors with stock lists — stock scan skipped.")

    os.makedirs(args.out, exist_ok=True)
    fig = rrg_chart(tails, table, as_of)
    fig.savefig(os.path.join(args.out, "rrg_chart.png"), dpi=130)
    table.to_csv(os.path.join(args.out, "sector_rotation.csv"))
    if full_stocks is not None and not full_stocks.empty:
        full_stocks.to_csv(os.path.join(args.out, "stock_screener.csv"), index=False)
    strat = None
    if not args.no_signals:
        try:
            strat = strategy_signals(args.period)
        except Exception as e:      # the rest of the app still builds if this fails
            print(f"Strategy screen failed: {e}")
        if strat:
            with open(os.path.join(args.out, "signals.json"), "w", encoding="utf-8") as f:
                import json
                json.dump(strat, f)
    page = html_report(table, fig, as_of, full_stocks, args.top, strat)
    for name in ("sector_rotation_report.html", "index.html"):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            f.write(page)
    make_icons(args.out)
    if os.environ.get("GITHUB_ACTIONS"):      # website build: put app icon + manifest next to index.html
        os.makedirs("docs", exist_ok=True)
        make_icons("docs")
    print(f"\nSaved report, CSVs and chart to {os.path.abspath(args.out)}  (open sector_rotation_report.html)")


if __name__ == "__main__":
    main()
