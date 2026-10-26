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


def analyse_stock(df, sec_px, bench_px):
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
        "Risk %": round(pct(close, stop), 2),
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
                m = analyse_stock(df, daily[sec], bench)
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


# ----------------------------------------------------------------------------- report
STOCK_COLS = ["Stock", "Score", "Setup", "Close", "1W %", "RS vs Sector 3M", "RS vs Nifty 3M",
              "RS rising", "Trend", "RSI", "MACD", "ADX", "Supertrend", "Vol ratio",
              "From 52W high %", "Stop loss", "Risk %"]


def stocks_html(top, table):
    if top is None or top.empty:
        return "<p>No sectors are currently outperforming or turning up, so no stocks were scanned.</p>"
    good = {"Strong uptrend", "Bullish", "Buy", "Yes", "Momentum", "Breakout / near 52W high",
            "Pullback to 20 EMA"}
    bad = {"Downtrend", "Bearish", "Sell", "No", "Weak - avoid"}
    qmap = table.set_index("Sector")["Quadrant"].to_dict()
    out = ""
    for sec, grp in top.groupby("Sector", sort=False):
        head = "".join(f"<th>{c}</th>" for c in STOCK_COLS)
        body = ""
        for _, r in grp.iterrows():
            cells = ""
            for c in STOCK_COLS:
                v = r[c]
                cls = "pos" if v in good else "neg" if v in bad else ""
                if isinstance(v, (int, float)) and c in ("1W %", "RS vs Sector 3M", "RS vs Nifty 3M"):
                    cls = "pos" if v > 0 else "neg"
                cells += f'<td class="{cls}"><b>{v}</b></td>' if c == "Stock" else f'<td class="{cls}">{v}</td>'
            body += f"<tr>{cells}</tr>"
        col = QCOL[qmap.get(sec, "Lagging")]
        out += (f'<h3 style="border-left:6px solid {col};padding-left:8px">{sec} '
                f'<span class="tag">{grp["Sector signal"].iloc[0]}</span></h3>'
                f'<div class="wrap"><table><tr>{head}</tr>{body}</table></div>')
    return out


def html_report(table, fig, as_of, top_stocks=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    img = base64.b64encode(buf.getvalue()).decode()

    def color_row(r):
        return f'style="border-left:5px solid {QCOL[r["Quadrant"]]}"'

    head = "".join(f"<th>{c}</th>" for c in ["#"] + list(table.columns))
    body = ""
    for i, r in table.iterrows():
        cells = "".join(
            f'<td class="{"neg" if isinstance(v, (int, float)) and v < 0 and k.startswith("Rel") else ""}">{v}</td>'
            for k, v in r.items())
        body += f"<tr {color_row(r)}><td>{i}</td>{cells}</tr>"

    def bucket(prefix):
        names = table[table["Signal"].str.startswith(prefix)]["Sector"].tolist()
        return ", ".join(names) if names else "—"

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Sector Rotation Screener</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial;margin:24px;color:#222;background:#fafafa}}
h1{{margin-bottom:4px}} .sub{{color:#666;margin-bottom:18px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-bottom:18px}}
.card{{background:#fff;border-radius:8px;padding:12px 14px;box-shadow:0 1px 3px #0002}}
.card b{{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px}}
th,td{{padding:6px 8px;border-bottom:1px solid #eee;text-align:left;white-space:nowrap}}
th{{background:#f0f0f0;position:sticky;top:0}} .neg{{color:#c0392b}} .pos{{color:#1a7f37}}
.tag{{font-size:12px;font-weight:normal;background:#eee;border-radius:10px;padding:2px 8px;margin-left:6px}}
h3{{margin-top:22px}}
.wrap{{overflow-x:auto}} img{{max-width:100%;background:#fff;border-radius:8px}}
</style></head><body>
<h1>Indian Sector Rotation Screener</h1>
<div class="sub">NSE sector indices vs Nifty 50 · weekly RRG · data as of {as_of:%d %b %Y} · generated {datetime.now():%d %b %Y %H:%M}</div>
<div class="cards">
 <div class="card" style="border-top:4px solid {QCOL['Leading']}"><b>Outperforming Nifty</b>{bucket('OUTPERFORMING')}</div>
 <div class="card" style="border-top:4px solid {QCOL['Improving']}"><b>Likely to outperform (early)</b>{bucket('LIKELY')}</div>
 <div class="card" style="border-top:4px solid {QCOL['Improving']}"><b>Watch – turning up</b>{bucket('Watch')}</div>
 <div class="card" style="border-top:4px solid {QCOL['Weakening']}"><b>Losing leadership</b>{bucket('Losing')}</div>
 <div class="card" style="border-top:4px solid {QCOL['Lagging']}"><b>Underperforming</b>{bucket('UNDER')}</div>
</div>
<img src="data:image/png;base64,{img}">
<h2>Ranked table</h2>
<div class="wrap"><table><tr>{head}</tr>{body}</table></div>
<h2 style="margin-top:28px">Leading stocks in rotating sectors</h2>
<p style="color:#666;font-size:13px">Top stocks (by score) inside sectors that are outperforming or turning up vs Nifty.
Score 0–100 = relative strength vs sector & Nifty, trend (20/50/200 EMA), RSI, MACD, ADX, Supertrend and volume.
Stop loss = Supertrend line (or 2×ATR below price when Supertrend is Sell). Vol ratio = 5-day avg volume ÷ 50-day avg.</p>
{stocks_html(top_stocks, table)}
<p style="color:#666;font-size:12px;margin-top:16px">
Rotation usually runs clockwise: Improving → Leading → Weakening → Lagging → Improving.
Sectors in <b>Improving</b> with the tail pointing up-and-right (heading 0–90°) are the earliest candidates to start outperforming Nifty.
Heading = direction the sector moved on the RRG over the last 3 weeks. Score is a 0–100 composite of RS-Ratio, RS-Momentum, heading, 1M/3M relative returns and trend.
Technical screen only — not investment advice.</p>
</body></html>"""


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
                print(grp[["Stock", "Score", "Setup", "Close", "RS vs Sector 3M", "Trend",
                           "RSI", "MACD", "ADX", "Supertrend", "Vol ratio", "Stop loss"]]
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
    with open(os.path.join(args.out, "sector_rotation_report.html"), "w", encoding="utf-8") as f:
        f.write(html_report(table, fig, as_of, top_stocks))
    print(f"\nSaved report, CSVs and chart to {os.path.abspath(args.out)}  (open sector_rotation_report.html)")


if __name__ == "__main__":
    main()
