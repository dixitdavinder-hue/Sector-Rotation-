"""
Paper trading for the app strategies: PIVOT BREAKOUT (+ Minervini VCP first), POSITIONAL and WEDGE POP (3 books).

Uses the same signal code as the app (sector_rotation_screener.py -> strategy_signals),
so the trades here are exactly the app's Signals tab, executed by the rules of the 10-year backtest.

Put in one folder: paper_trade.py, sector_rotation_screener.py, nifty500_list.csv
Run every trading day after 6:30 PM (data must include today's close):
    python -m pip install yfinance pandas numpy matplotlib
    python paper_trade.py                     # first run: two books of Rs 5 lakh each
    python paper_trade.py --capital-pivot 500000 --capital-pos 500000   (first run only)

Rules applied by the ledger (same as the backtest)
  - Signal at today's close -> BUY at the next open. Skipped if the stock opens at/below its stop.
  - Pivot trades only for stocks that pass the Minervini Trend Template (close > 50/150/200 SMA,
    50 > 150 > 200, 200 SMA up for a month, within 25% of the 52w high, 30%+ above the 52w low).
  - New trades only when the strategy's breadth filter is ON (pivot: 200 DMA >= 50%, positional: 20 DMA >= 50%).
  - Size: 1% of the book risked per trade, max 15% of the book in one stock, max 10 stocks,
    max 3 stocks of one industry. Pivot list ranked by 3-month RS, positional by score then RS.
  - Exit: stop loss (fixed at entry, sold at the stop or at the open if it gaps below),
    or a close below the 50-day EMA -> sell at the next open.
  - Ledger cost: 0.2% per side.
If you skip a day, fills, stops and 50-EMA exits are caught up from the price data,
but that day's NEW signals are missed (the app only shows today's list).

Files
  paper/<book>.json         cash, open positions, pending orders  (edit if your real fills differ)
  paper/<book>_trades.csv   closed trades
  paper/<book>_equity.csv   daily equity
  paper/report_<date>.txt   today's report + tomorrow's orders
  paper/orders_<date>.csv
  paper/positions.json      everything above for the app's Positions tab
On GitHub the workflow runs:  python paper_trade.py --dir docs/positions
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sector_rotation_screener as S  # noqa: E402

DIR = "paper"
COST, RISK, MAXPOS, POSCAP, MAXIND = 0.002, 0.01, 10, 0.15, 3
BOOKS = {"pivot": "Pivot breakout + VCP", "positional": "Positional", "wedgepop": "Wedge Pop"}
DEFAULT_RULE = {"pivot": "ema50", "positional": "ema50", "wedgepop": "wedge"}


def get_data():
    """Run the app's screen once and keep the downloaded prices for fills."""
    cache = {}
    orig = S.fetch_stocks

    def keep(symbols, period="2y"):
        cache["d"] = orig(symbols, period)
        return cache["d"]
    S.fetch_stocks = keep
    strat = S.strategy_signals()
    S.fetch_stocks = orig
    if strat is None:
        sys.exit("Strategy screen failed (too little data). Try again later.")
    data = cache["d"]
    P = {k: pd.DataFrame({s: d[k] for s, d in data.items()}).sort_index() for k in ("Open", "High", "Low", "Close")}
    P["EMA50"] = P["Close"].ewm(span=50, adjust=False).mean()
    P["EMA10"] = P["Close"].ewm(span=10, adjust=False).mean()
    P["EMA20"] = P["Close"].ewm(span=20, adjust=False).mean()
    P["SMA50"] = P["Close"].rolling(50).mean()
    return strat, P


def exit_line(P, d, s, rule):
    """(label, level) of the close-based exit for a position."""
    if rule == "sma50":
        return "50 SMA", v(P, "SMA50", d, s)
    if rule == "wedge":
        return "10 & 20 EMA", min(v(P, "EMA10", d, s), v(P, "EMA20", d, s))
    return "50 EMA", v(P, "EMA50", d, s)


def v(P, k, d, s):
    try:
        x = P[k].at[d, s]
    except KeyError:
        return np.nan
    return float(x) if pd.notna(x) else np.nan


def last_close(P, s, d):
    if s not in P["Close"]:
        return np.nan
    x = P["Close"][s].loc[:d].dropna()
    return float(x.iloc[-1]) if len(x) else np.nan


def equity(P, b, d):
    tot = b["cash"]
    for s, p in b["positions"].items():
        lc = last_close(P, s, d)
        tot += p["qty"] * (lc if not np.isnan(lc) else p["entry"])
    return tot


def load(name, cap):
    f = os.path.join(DIR, f"{name}.json")
    if os.path.exists(f):
        with open(f) as fh:
            return json.load(fh)
    return {"cash": cap, "start_capital": cap, "positions": {}, "pending_buys": [], "pending_sells": {},
            "last_date": None}


def save(name, b):
    with open(os.path.join(DIR, f"{name}.json"), "w") as fh:
        json.dump(b, fh, indent=1)


def append(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def step(name, b, P, d, trades, notes):
    """Fills at d's open, stops during d, 50-EMA exit signals at d's close."""
    tag = str(d.date())

    def sell(s, price, why):
        p = b["positions"].pop(s)
        b["cash"] += p["qty"] * price * (1 - COST)
        trades.append(dict(book=name, symbol=s, entry_date=p["entry_date"], exit_date=tag, qty=p["qty"],
                           entry=round(p["entry"], 2), exit=round(price, 2),
                           pnl=round(p["qty"] * (price * (1 - COST) - p["entry"] * (1 + COST))),
                           ret_pct=round((price / p["entry"] - 1) * 100, 2), reason=why))

    for s, why in list(b["pending_sells"].items()):
        op = v(P, "Open", d, s)
        if s in b["positions"] and not np.isnan(op):
            sell(s, op, why)
            b["pending_sells"].pop(s)
    eq_prev = equity(P, b, d - pd.Timedelta(days=1))
    for o in b["pending_buys"]:
        s, stop = o["symbol"], o["stop"]
        if s in b["positions"] or len(b["positions"]) >= MAXPOS:
            continue
        op = v(P, "Open", d, s)
        if np.isnan(op):
            notes.append(f"{name}: {s} no price on {tag}, BUY dropped"); continue
        if o.get("type") == "stop":                       # VCP buy-stop: fill only if the pivot is taken out
            hi = v(P, "High", d, s)
            if np.isnan(hi) or hi <= o["level"]:
                notes.append(f"{name}: {s} VCP pivot {o['level']:.2f} not taken out on {tag} -> order cancelled"); continue
            op = max(op, o["level"])
            stop = round(op - min(2 * o["atr"], 0.10 * op), 2)
        if op <= stop:
            notes.append(f"{name}: {s} opened {op:.2f} at/below stop {stop:.2f} -> BUY cancelled"); continue
        qty = int(min(eq_prev * RISK / (op - stop), eq_prev * POSCAP / op, b["cash"] / (op * (1 + COST))))
        if qty <= 0:
            continue
        b["cash"] -= qty * op * (1 + COST)
        b["positions"][s] = dict(qty=qty, entry=round(op, 2), stop=stop, entry_date=tag, industry=o["industry"],
                                 rule=o.get("rule", DEFAULT_RULE[name]), setup=o.get("setup", ""))
    b["pending_buys"] = []
    for s in list(b["positions"]):
        p = b["positions"][s]
        lo, op = v(P, "Low", d, s), v(P, "Open", d, s)
        if not np.isnan(lo) and lo <= p["stop"]:
            sell(s, p["stop"] if p["entry_date"] == tag else min(op, p["stop"]), "stop loss")
            b["pending_sells"].pop(s, None)
    for s, p in b["positions"].items():
        rule = p.get("rule", DEFAULT_RULE[name])
        cl = v(P, "Close", d, s)
        lab, lvl = exit_line(P, d, s, rule)
        if not np.isnan(cl) and s not in b["pending_sells"]:
            if not np.isnan(lvl) and cl < lvl:
                b["pending_sells"][s] = f"close < {lab}"
            elif rule == "wedge" and cl > v(P, "EMA20", d, s) * 1.20:
                b["pending_sells"][s] = "exhaustion: 20% above 20 EMA (profit)"
        prev = P["Close"][s].loc[:d].dropna() if s in P["Close"] else pd.Series(dtype=float)
        if len(prev) >= 2 and prev.index[-1] == d and abs(prev.iloc[-1] / prev.iloc[-2] - 1) > 0.25:
            notes.append(f"{name}: {s} moved {100 * (prev.iloc[-1] / prev.iloc[-2] - 1):.0f}% on {tag} - "
                         f"check split/bonus and fix stop/qty in {DIR}/{name}.json")
    b["last_date"] = tag


def new_orders(name, b, P, d, cands, on):
    cands = [c for c in cands if c.get("always") or on]
    if not cands:
        return
    held = {s: p for s, p in b["positions"].items() if s not in b["pending_sells"]}
    slots = MAXPOS - len(held)
    eq = equity(P, b, d)
    cash = b["cash"] + sum(b["positions"][s]["qty"] * last_close(P, s, d) for s in b["pending_sells"]
                           if s in b["positions"])
    indc = pd.Series([p["industry"] for p in held.values()], dtype=object).value_counts().to_dict()
    for c in cands:
        s, ind, stop = c["s"], c["ind"], c["stop"]
        close = c.get("level", c["close"])                 # VCP: buy-stop level
        if slots <= 0:
            break
        if s in b["positions"] or any(o["symbol"] == s for o in b["pending_buys"]) or indc.get(ind, 0) >= MAXIND or close <= stop:
            continue
        qty = int(min(eq * RISK / (close - stop), eq * POSCAP / close, cash / (close * (1 + COST))))
        if qty <= 0:
            continue
        info = f"RS {c['rs']}%" + (f", score {c['score']} {c['setup']}" if "score" in c else "") + \
            (f", VCP {'/'.join(str(x) for x in c['cons'])}%" if "cons" in c else "") + (", Wedge Pop" if c.get("kind") == "wp" else "")
        o = dict(symbol=s, stop=stop, est_price=close, est_qty=qty, industry=ind, info=info,
                 rule=c.get("rule", DEFAULT_RULE[name]), setup=c.get("kind", name))
        if "level" in c:
            o.update(type="stop", level=c["level"], atr=c["atr"])
        b["pending_buys"].append(o)
        slots -= 1
        cash -= qty * close * (1 + COST)
        indc[ind] = indc.get(ind, 0) + 1


def report(P, books, strat, d, notes):
    tag = str(d.date())
    br = strat["breadth"]
    L = [f"PAPER TRADING  -  data up to {tag}", "=" * 64,
         f"Breadth: 20 DMA {br['p20']}% | 50 DMA {br['p50']}% | 200 DMA {br['p200']}%",
         f"Pivot breakout filter: {'ON' if strat['pivotOn'] else 'OFF'} | "
         f"Positional filter: {'ON' if strat['posOn'] else 'OFF'} | VCP + Wedge Pop: no breadth filter", ""]
    orders, tot_eq, tot_cap = [], 0, 0
    for name, b in books.items():
        eq = equity(P, b, d)
        tot_eq += eq; tot_cap += b["start_capital"]
        L.append(f"--- {BOOKS[name].upper()}: equity Rs {eq:,.0f}  ({100 * (eq / b['start_capital'] - 1):+.1f}% "
                 f"since start)  cash Rs {b['cash']:,.0f}")
        for s, why in b["pending_sells"].items():
            q = b["positions"][s]["qty"]
            orders.append(dict(book=name, action="SELL", symbol=s, qty=q, stop="", note=why))
            L.append(f"  SELL {s} x {q} at tomorrow's open  ({why})")
        for o in b["pending_buys"]:
            if o.get("type") == "stop":
                orders.append(dict(book=name, action="BUY-STOP", symbol=o["symbol"], qty=o["est_qty"], stop=o["stop"],
                                   note=f"only if price trades above {o['level']}; stop = fill - 2xATR (max 10%)"))
                L.append(f"  BUY-STOP {o['symbol']} x ~{o['est_qty']} above {o['level']} tomorrow, "
                         f"stop ~{o['stop']}  [{o['industry']}, {o['info']}]")
            else:
                orders.append(dict(book=name, action="BUY", symbol=o["symbol"], qty=o["est_qty"], stop=o["stop"],
                                   note=f"skip if it opens at/below {o['stop']}"))
                L.append(f"  BUY  {o['symbol']} x ~{o['est_qty']} at tomorrow's open (close {o['est_price']}), "
                         f"stop {o['stop']}  [{o['industry']}, {o['info']}]")
        if not b["pending_sells"] and not b["pending_buys"]:
            L.append("  No orders for tomorrow.")
        L.append(f"  Open positions: {len(b['positions'])}")
        for s, p in sorted(b["positions"].items()):
            cl = last_close(P, s, d)
            lab, e = exit_line(P, d, s, p.get("rule", DEFAULT_RULE[name]))
            L.append(f"    {s:<12} {p['qty']:>5} @ {p['entry']:>9.2f} ({p['entry_date']})  now {cl:>9.2f} "
                     f"{100 * (cl / p['entry'] - 1):+6.1f}%  stop {p['stop']:>9.2f}  {lab} {e:>9.2f}")
        tf = os.path.join(DIR, f"{name}_trades.csv")
        if os.path.exists(tf):
            t = pd.read_csv(tf)
            if len(t):
                L.append(f"  Closed trades: {len(t)}, won {100 * (t.pnl > 0).mean():.0f}%, P&L Rs {t.pnl.sum():,.0f}")
        L.append("")
    L.append(f"TOTAL: Rs {tot_eq:,.0f} on Rs {tot_cap:,.0f}  ({100 * (tot_eq / tot_cap - 1):+.1f}%)")
    if notes:
        L += ["", "Notes:"] + [f"  - {n}" for n in notes[-20:]]
    text = "\n".join(L)
    with open(os.path.join(DIR, f"report_{tag}.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    pd.DataFrame(orders, columns=["book", "action", "symbol", "qty", "stop", "note"]).to_csv(
        os.path.join(DIR, f"orders_{tag}.csv"), index=False)
    print("\n" + text)


def write_json(P, books, strat, d):
    """positions.json for the app's Positions tab."""
    tag = str(d.date())
    br = strat["breadth"]
    out = {"asOf": tag, "breadth": br, "on": {"pivot": bool(strat["pivotOn"]), "positional": bool(strat["posOn"]), "wedgepop": True},
           "books": [], "total": {}}
    tot_eq = tot_cap = 0
    for name, b in books.items():
        eq = equity(P, b, d); tot_eq += eq; tot_cap += b["start_capital"]
        opn = []
        for s, p in sorted(b["positions"].items()):
            cl = last_close(P, s, d)
            lab, e = exit_line(P, d, s, p.get("rule", DEFAULT_RULE[name]))
            cl = p["entry"] if np.isnan(cl) else cl
            opn.append(dict(s=s, qty=p["qty"], entry=p["entry"], date=p["entry_date"], now=round(cl, 2),
                            ret=round((cl / p["entry"] - 1) * 100, 2),
                            pnl=round(p["qty"] * (cl * (1 - COST) - p["entry"] * (1 + COST))),
                            stop=round(p["stop"], 2), ema=None if np.isnan(e) else round(e, 2),
                            ind=p.get("industry", ""), exit=b["pending_sells"].get(s), exl=lab,
                            setup=p.get("setup", "")))
        buys = [dict(s=o["symbol"], qty=o["est_qty"], price=o["est_price"], stop=o["stop"], ind=o["industry"],
                     info=o["info"], type=o.get("type", "open")) for o in b["pending_buys"]]
        closed, curve = [], []
        tf = os.path.join(DIR, f"{name}_trades.csv")
        if os.path.exists(tf):
            t = pd.read_csv(tf)
            t = t.tail(40).iloc[::-1]
            closed = t.astype(object).where(t.notna(), None).to_dict("records")
        ef = os.path.join(DIR, f"{name}_equity.csv")
        if os.path.exists(ef):
            q = pd.read_csv(ef)
            curve = [[r.date, int(r.equity)] for r in q.itertuples()]
        out["books"].append(dict(key=name, label=BOOKS[name], equity=round(eq), start=b["start_capital"],
                                 cash=round(b["cash"]), ret=round((eq / b["start_capital"] - 1) * 100, 2),
                                 open=opn, buys=buys, closed=closed, curve=curve))
    out["total"] = dict(equity=round(tot_eq), start=tot_cap, ret=round((tot_eq / tot_cap - 1) * 100, 2))
    with open(os.path.join(DIR, "positions.json"), "w") as fh:
        json.dump(out, fh, default=lambda x: x.item() if hasattr(x, "item") else str(x), allow_nan=False)


def main():
    global DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR, help="folder for the ledger files (default: paper)")
    ap.add_argument("--capital-pivot", type=float, default=500000)
    ap.add_argument("--capital-pos", type=float, default=500000)
    ap.add_argument("--capital-wedge", type=float, default=500000)
    a = ap.parse_args()
    DIR = a.dir
    os.makedirs(DIR, exist_ok=True)
    strat, P = get_data()
    dates = P["Close"].index
    d = dates[-1]
    # pivot: only stocks that pass the Trend Template (app shows the others dimmed)
    # pivot book: VCP buy-stops first (no breadth filter, exit 50 SMA), then pivot breakouts that pass the Trend Template
    vcp = [dict(x, always=True, rule="sma50", kind="vcp") for x in strat.get("vcp", [])]
    cands = {"pivot": vcp + [dict(x, kind="pivot") for x in strat["pivot"] if x.get("tt", True)],
             "positional": [dict(x, kind="positional") for x in strat["pos"]],
             "wedgepop": [dict(x, always=True, rule="wedge", kind="wp") for x in strat.get("wp", [])]}
    on = {"pivot": strat["pivotOn"], "positional": strat["posOn"], "wedgepop": True}
    caps = {"pivot": a.capital_pivot, "positional": a.capital_pos, "wedgepop": a.capital_wedge}
    books, notes, trades = {}, [], []
    for name in BOOKS:
        b = load(name, caps[name])
        if b["last_date"] == str(d.date()):
            print(f"{name}: already updated for {d.date()} (orders unchanged)")
            books[name] = b
            continue
        todo = dates[dates > pd.Timestamp(b["last_date"])] if b["last_date"] else dates[-1:]
        for dd in todo:
            step(name, b, P, dd, trades, notes)
            append(os.path.join(DIR, f"{name}_equity.csv"),
                   [dict(date=str(dd.date()), equity=round(equity(P, b, dd)), cash=round(b["cash"]),
                         positions=len(b["positions"]))])
        new_orders(name, b, P, d, cands[name], on[name])
        append(os.path.join(DIR, f"{name}_trades.csv"), [t for t in trades if t["book"] == name])
        save(name, b)
        books[name] = b
    report(P, books, strat, d, notes)
    write_json(P, books, strat, d)


if __name__ == "__main__":
    main()
