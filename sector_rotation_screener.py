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
nav .in{max-width:820px;margin:0 auto;display:grid;grid-template-columns:repeat(4,1fr)}
nav button{background:none;border:0;color:var(--muted);font-size:11.5px;font-weight:600;padding:6px 0;
  display:flex;flex-direction:column;align-items:center;gap:3px;cursor:pointer}
nav button svg{width:23px;height:23px}
nav button.on{color:var(--accent)}
.sl{font-size:12px;color:var(--muted);margin-top:10px}
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
  <section class="view" id="v-chart"></section>
</main>

<nav><div class="in">
  <button data-v="home" class="on"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 10.5 12 3l9 7.5V21H3z"/><path d="M9 21v-6h6v6"/></svg>Home</button>
  <button data-v="sectors"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="8" height="8" rx="2"/><rect x="13" y="3" width="8" height="8" rx="2"/><rect x="3" y="13" width="8" height="8" rx="2"/><rect x="13" y="13" width="8" height="8" rx="2"/></svg>Sectors</button>
  <button data-v="stocks"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/></svg>Stocks</button>
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
  const k = /Breakout|Momentum|Pullback/.test(s)?"g":/Weak/.test(s)?"r":/Overbought/.test(s)?"y":"";
  return `<span class="tag ${k}">${esc(s)}</span>`;
}
function ring(v,c="var(--accent)"){return `<div class="score" style="--v:${v};--c:${c}"><span>${Math.round(v)}</span></div>`}

/* ---------- HOME ---------- */
function home(){
  const S=D.sectors, cnt=q=>S.filter(s=>s.Quadrant===q).length;
  const by=p=>S.filter(s=>s.Signal.startsWith(p));
  const out=by("OUTPERFORMING"), early=by("LIKELY"), watch=S.filter(s=>/^(Watch|Re-acc)/.test(s.Signal));
  const best=[...D.stocks].filter(s=>/Breakout|Momentum|Pullback/.test(s.Setup)).sort((a,b)=>b.Score-a.Score).slice(0,6);
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
    <div class="tiles">
      ${["Leading","Improving","Weakening","Lagging"].map(q=>`<div class="tile q-${q}" onclick="goSectors('${q}')"><b>${cnt(q)}</b><span>${q}</span></div>`).join("")}
    </div>
    <h2>Outperforming Nifty now</h2><div class="card">${list(out,"No sector is in the Leading zone this week.")}</div>
    <h2>Likely to outperform next</h2><div class="card">${list(early.concat(watch),"No sector is turning up right now.")}</div>
    <h2>Best stock setups</h2>
    ${best.length?best.map(stockCard).join(""):`<div class="card s">No breakout / momentum / pullback setups found.</div>`}
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
let stQ="", stSec="All", stSet="All", stTop=true;
function openSector(name){stSec=name;stSet="All";stQ="";show("stocks")}
function stockCard(s){
  return `<div class="card">
    <div class="row">
      ${ring(s.Score)}
      <div class="grow"><div class="t">${esc(s.Stock)}</div><div class="s">${esc(s.Sector)} · #${s["Sector rank"]} in sector</div></div>
      <div style="text-align:right"><div class="t">₹${fmt(s.Close)}</div><div class="s ${pc(s["1W %"])}">${sgn(s["1W %"])}% 1W</div></div>
    </div>
    <div class="tags">${setupTag(s.Setup)}
      <span class="tag ${s.Trend==='Downtrend'?'r':/uptrend/i.test(s.Trend)?'g':''}">${esc(s.Trend)}</span>
      <span class="tag ${/^Bullish/.test(s.MACD)?'g':'r'}">MACD ${esc(s.MACD)}</span>
      <span class="tag ${s.Supertrend==='Buy'?'g':'r'}">Supertrend ${s.Supertrend}</span>
      ${s["RS rising"]==='Yes'?'<span class="tag g">RS rising</span>':''}
    </div>
    <div class="grid">
      <div class="m"><span>vs Sector</span><b class="${pc(s["RS vs Sector 3M"])}">${sgn(s["RS vs Sector 3M"])}%</b></div>
      <div class="m"><span>vs Nifty</span><b class="${pc(s["RS vs Nifty 3M"])}">${sgn(s["RS vs Nifty 3M"])}%</b></div>
      <div class="m"><span>RSI</span><b>${fmt(s.RSI,1)}</b></div>
      <div class="m"><span>ADX</span><b>${fmt(s.ADX,1)}</b></div>
      <div class="m"><span>Volume</span><b class="${s["Vol ratio"]>1.2?'pos':''}">${fmt(s["Vol ratio"])}x</b></div>
      <div class="m"><span>52W high</span><b>${fmt(s["From 52W high %"],1)}%</b></div>
      <div class="m"><span>Stop loss</span><b>₹${fmt(s["Stop loss"])}</b></div>
      <div class="m"><span>Risk</span><b>${fmt(s["Risk %"],1)}%</b></div>
    </div>
  </div>`;
}
function stocks(){
  const secs=["All",...new Set(D.stocks.map(s=>s.Sector))];
  const sets=["All","Breakout","Momentum","Pullback","Overbought","Neutral","Weak"];
  let arr=D.stocks.filter(s=>(stSec==="All"||s.Sector===stSec)&&(stSet==="All"||s.Setup.includes(stSet))
      &&(!stQ||s.Stock.toLowerCase().includes(stQ.toLowerCase()))&&(!stTop||s["Sector rank"]<=D.top));
  arr.sort((a,b)=>b.Score-a.Score);
  if(stSec==="All"){const seen=new Set();arr=arr.filter(s=>!seen.has(s.Stock)&&seen.add(s.Stock));}
  $("#v-stocks").innerHTML=`
    <div class="search">
      <input id="q" placeholder="Search stock…" value="${esc(stQ)}" autocomplete="off">
      <select id="sec">${secs.map(s=>`<option ${s===stSec?'selected':''}>${esc(s)}</option>`).join("")}</select>
    </div>
    <div class="filters">
      <button class="chip ${stTop?'on':''}" id="top">Top ${D.top} / sector</button>
      ${sets.map(x=>`<button class="chip ${x===stSet?'on':''}" data-set="${x}">${x}</button>`).join("")}
    </div>
    <div class="s" style="margin:0 4px 8px">${arr.length} stock${arr.length==1?"":"s"} · "vs Sector / vs Nifty" = 3-month relative return</div>
    ${arr.length?arr.map(stockCard).join(""):`<div class="empty">${D.stocks.length?"No stocks match these filters.":"No sectors are rotating up, so no stocks were scanned this week."}</div>`}`;
  const q=$("#q"); q.oninput=e=>{stQ=e.target.value;stocks();const n=$("#q");n.focus();n.setSelectionRange(n.value.length,n.value.length)};
  $("#sec").onchange=e=>{stSec=e.target.value;stocks()};
  $("#top").onclick=()=>{stTop=!stTop;stocks()};
  document.querySelectorAll("[data-set]").forEach(b=>b.onclick=()=>{stSet=b.dataset.set;stocks()});
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

/* ---------- NAV ---------- */
const R={home,sectors,stocks,chart};
function show(v){
  document.querySelectorAll(".view").forEach(e=>e.classList.toggle("on",e.id==="v-"+v));
  document.querySelectorAll("nav button").forEach(b=>b.classList.toggle("on",b.dataset.v===v));
  R[v](); window.scrollTo(0,0);
}
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{if(b.dataset.v==="stocks"){stSec="All";} if(b.dataset.v==="sectors")secQ="All"; show(b.dataset.v)});
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


def html_report(table, fig, as_of, full_stocks=None, top_n=5):
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
    page = html_report(table, fig, as_of, full_stocks, args.top)
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
