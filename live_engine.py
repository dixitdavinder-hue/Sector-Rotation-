"""
QUANTDIX LIVE ENGINE  -  har minute live market par positions, signals aur scanner.

Kya karta hai (market hours 9:15 - 3:30, Mon-Fri):
  * Upstox se Nifty 500 ke saare stocks ka live price (open/high/low/LTP/volume) har minute
  * live breadth (20/50/200 DMA), pivot breakout + positional signals (app wale hi rules)
  * GitHub wale Positions ka live P&L, stop loss hit alert, 50 EMA warning
  * scanner: industries (Leading/Improving...), RS leaders, aaj ke gainers/losers
  * live dashboard: PC par http://localhost:8765 , ghar ke WiFi par phone se bhi

Chalane ka tareeka:   start_live.bat  par double-click   (ya:  python live_engine.py)
  Token har roz naya: Upstox login -> access token -> yahan paste.  Token kahin save nahi hota.
  Bina token ke dekhna ho:  python live_engine.py --demo   (nakli price, sirf dashboard dekhne ke liye)

Is folder mein ye files honi chahiye: live_engine.py, live_dashboard.html,
sector_rotation_screener.py, nifty500_list.csv
"""
import argparse, gzip, io, json, os, socket, sys, threading, time, traceback
from datetime import datetime, date
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import sector_rotation_screener as S  # noqa: E402

GITHUB_POS = "https://raw.githubusercontent.com/dixitdavinder-hue/Sector-Rotation-/main/docs/positions/positions.json"
INSTR_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
QUOTE_URL = "https://api.upstox.com/v2/market-quote/quotes"
LIVE_DIR = os.path.join(HERE, "live")
PORT = 8765
CONFIRM_FROM = (15, 15)          # 3:15 PM ke baad close-signal "confirm" maana jaata hai
MARKET = ((9, 15), (15, 31))

# backtest (10 saal, Nifty 500) se nikle timing rules - dashboard par dikhaye jaate hain
RULES = {
    "pivot": "Breakout close par confirm hota hai -> kal open par BUY. "
             "(Backtest: kal open 24.0% CAGR; din mein breakout par turant entry 19.3%; 3:20 PM entry 16.0%)",
    "positional": "3:15-3:25 PM ke beech BUY (aaj ke close ke paas). "
                  "(Backtest: aaj close 20.6-21.0% CAGR vs kal open 18.8%)",
    "stop": "Stop loss live: price stop ke neeche gaya to exit (backtest mein bhi din ke andar).",
    "ema": "50 EMA: close 50 EMA ke neeche band ho to exit. 3:15 ke baad warning ko confirm maano.",
}


class TokenError(Exception):
    pass


def now():
    return datetime.now()


def market_open(t=None):
    t = t or now()
    if t.weekday() >= 5:
        return False
    hm = (t.hour, t.minute)
    return MARKET[0] <= hm < MARKET[1]


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return "PC-ka-IP"


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, default=lambda x: x.item() if hasattr(x, "item") else str(x), allow_nan=False)
    os.replace(tmp, path)


def clean(v, nd=2):
    try:
        v = float(v)
    except Exception:
        return None
    return None if not np.isfinite(v) else round(v, nd)


# ----------------------------------------------------------------------------- data
def load_history(uni):
    print("Pichle 2 saal ka daily data download ho raha hai (ek baar, 1-3 minute) ...")
    data = S.fetch_stocks(list(uni), "2y")
    today = pd.Timestamp(date.today())
    for s in list(data):
        df = data[s]
        df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.to_datetime(df.index)
        data[s] = df[df.index < today][["Open", "High", "Low", "Close", "Volume"]].copy()
    print(f"  {len(data)} stocks ka history ready (aakhri din: {max(d.index[-1] for d in data.values()):%d %b %Y})")
    return data


def load_book():
    try:
        r = requests.get(GITHUB_POS + f"?t={int(time.time())}", timeout=20)
        r.raise_for_status()
        print("  Positions GitHub se aa gayi")
        return r.json()
    except Exception as e:
        f = os.path.join(HERE, "paper", "positions.json")
        if os.path.exists(f):
            print(f"  GitHub se positions nahi aayi ({e}); local {f} use kar rahe hain")
            return json.load(open(f, encoding="utf-8"))
        print(f"  Positions nahi mili ({e}) - sirf signals/scanner dikhenge")
        return None


def load_instruments(symbols):
    print("Upstox instrument list download ho rahi hai ...")
    r = requests.get(INSTR_URL, timeout=60)
    r.raise_for_status()
    rows = json.loads(gzip.decompress(r.content))
    m = {}
    for x in rows:
        if x.get("segment") == "NSE_EQ" and x.get("instrument_type") in ("EQ", None) and x.get("trading_symbol") in symbols:
            m[x["trading_symbol"]] = x["instrument_key"]
    miss = sorted(set(symbols) - set(m))
    print(f"  {len(m)} stocks mapped" + (f" | nahi mile: {', '.join(miss[:15])}{' ...' if len(miss) > 15 else ''}" if miss else ""))
    return m


def upstox_quotes(keymap, token):
    """symbol -> dict(o,h,l,ltp,vol)  via Upstox v2 full market quote (max ~500 keys/call)."""
    inv = {v: k for k, v in keymap.items()}
    keys = list(keymap.values())
    out = {}
    hdr = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    for i in range(0, len(keys), 400):
        chunk = keys[i:i + 400]
        r = requests.get(QUOTE_URL, params={"instrument_key": ",".join(chunk)}, headers=hdr, timeout=30)
        if r.status_code == 401:
            raise TokenError("Upstox token galat ya expire ho gaya")
        r.raise_for_status()
        for k, q in (r.json().get("data") or {}).items():
            sym = inv.get(q.get("instrument_token")) or k.split(":", 1)[-1]
            o = q.get("ohlc") or {}
            ltp = q.get("last_price")
            if sym in keymap and ltp:
                out[sym] = dict(o=o.get("open") or ltp, h=o.get("high") or ltp, l=o.get("low") or ltp,
                                ltp=ltp, vol=q.get("volume") or 0)
    return out


def demo_quotes(hist, state={}):
    """--demo: nakli live price (pichle close se random chaal) - sirf dashboard test ke liye."""
    rng = np.random.default_rng(int(time.time()) // 60)
    out = {}
    for s, df in hist.items():
        pc = float(df["Close"].iloc[-1])
        st = state.setdefault(s, dict(o=pc * (1 + rng.normal(0, 0.006)), h=None, l=None, p=None))
        p = (st["p"] or st["o"]) * (1 + rng.normal(0, 0.003))
        st["p"] = p; st["h"] = max(st["h"] or st["o"], p, st["o"]); st["l"] = min(st["l"] or st["o"], p, st["o"])
        out[s] = dict(o=st["o"], h=st["h"], l=st["l"], ltp=p, vol=0)
    return out


def with_today(hist, quotes, today):
    ts = pd.Timestamp(today)
    live = {}
    for s, df in hist.items():
        q = quotes.get(s)
        if not q:
            live[s] = df
            continue
        row = pd.DataFrame({"Open": [q["o"]], "High": [max(q["h"], q["ltp"])], "Low": [min(q["l"], q["ltp"])],
                            "Close": [q["ltp"]], "Volume": [q["vol"]]}, index=[ts])
        live[s] = pd.concat([df, row])
    return live


def run_scan(data):
    S.fetch_stocks = lambda syms, period="2y": {k: v for k, v in data.items() if k in set(syms)}
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return S.strategy_signals()


# ----------------------------------------------------------------------------- snapshot
def build_snapshot(hist, live, quotes, strat, base, book, uni, alerts, seen, status):
    t = now()
    confirm = (t.hour, t.minute) >= CONFIRM_FROM or not market_open(t)
    prev = {s: float(df["Close"].iloc[-1]) for s, df in hist.items()}

    def chg(s):
        q = quotes.get(s)
        return clean((q["ltp"] / prev[s] - 1) * 100) if q and s in prev else None

    def stamp(kind, key, msg, level):
        if key in seen:
            return
        seen.add(key)
        alerts.insert(0, dict(t=f"{t:%H:%M}", kind=kind, msg=msg, level=level))
        del alerts[80:]
        print(f"  [{t:%H:%M}] {msg}")

    # ---- positions
    books, tot_eq, tot_cap, tot_day = [], 0.0, 0.0, 0.0
    for b in (book or {}).get("books", []):
        eq, day, rows = b["cash"], 0.0, []
        for p in b["open"]:
            s = p["s"]; q = quotes.get(s); df = live.get(s)
            ltp = q["ltp"] if q else p["now"]
            lo = q["l"] if q else ltp
            e50 = clean(S.ema(df["Close"], 50).iloc[-1]) if df is not None else p.get("ema")
            pc = prev.get(s, p["now"])
            val = p["qty"] * ltp; eq += val; day += p["qty"] * (ltp - pc)
            st = "HOLD"
            if lo <= p["stop"]:
                st = "STOP HIT"
                stamp("stop", f"stop-{s}", f"{s}: STOP LOSS hit (low ₹{lo:,.2f} ≤ stop ₹{p['stop']:,.2f}) → exit", "r")
            elif e50 and ltp < e50:
                st = "50 EMA ke neeche"
                if confirm:
                    stamp("ema", f"ema-{s}", f"{s}: close 50 EMA (₹{e50:,.2f}) ke neeche ja raha hai → exit signal", "a")
            if p.get("exit"):
                st = "EXIT (aaj open par)"
            rows.append(dict(s=s, book=b["label"], qty=p["qty"], entry=p["entry"], date=p["date"], ltp=clean(ltp),
                             day=chg(s), ret=clean((ltp / p["entry"] - 1) * 100),
                             pnl=clean(p["qty"] * (ltp * 0.998 - p["entry"] * 1.002), 0),
                             stop=p["stop"], dstop=clean((ltp / p["stop"] - 1) * 100, 1), ema=e50,
                             high=clean(q["h"]) if q else None, low=clean(lo), status=st))
        tot_eq += eq; tot_cap += b["start"]; tot_day += day
        today_orders = [dict(s=o["s"], qty=o["qty"], stop=o["stop"], ltp=clean(quotes[o["s"]]["ltp"]) if o["s"] in quotes else None,
                             open=clean(quotes[o["s"]]["o"]) if o["s"] in quotes else None, info=o.get("info", ""))
                        for o in b.get("buys", [])]
        books.append(dict(key=b["key"], label=b["label"], equity=clean(eq, 0), start=b["start"], cash=b["cash"],
                          ret=clean((eq / b["start"] - 1) * 100), day=clean(day, 0), open=rows, orders=today_orders))

    # ---- signals
    br, br0 = strat["breadth"], base["breadth"]
    piv_on, pos_on = br["p200"] >= 50, br["p20"] >= 50
    held = {p["s"] for b in books for p in b["open"]}
    piv = []
    for x in strat["pivot"]:
        x = dict(x, chg=chg(x["s"]), held=x["s"] in held, state="confirm" if confirm else "forming")
        piv.append(x)
        if confirm and piv_on and x.get("tt", True) and not x["held"]:
            stamp("pivot", f"piv-{x['s']}", f"PIVOT breakout confirm: {x['s']} ₹{x['close']:,.2f} (stop ₹{x['stop']:,.2f}) → kal open par BUY", "g")
    pos = []
    for x in strat["pos"]:
        x = dict(x, chg=chg(x["s"]), held=x["s"] in held, state="confirm" if confirm else "forming")
        pos.append(x)
        if confirm and pos_on and not x["held"]:
            stamp("pos", f"pos-{x['s']}", f"POSITIONAL signal: {x['s']} ₹{x['close']:,.2f} (stop ₹{x['stop']:,.2f}) → 3:15-3:25 PM BUY", "g")
    for k, old in (("p200", br0["p200"]), ("p20", br0["p20"])):
        if (old < 50) != (br[k] < 50):
            nm = "Pivot filter (200 DMA)" if k == "p200" else "Positional filter (20 DMA)"
            stamp("breadth", f"br-{k}-{br[k] >= 50}", f"{nm} {'ON' if br[k] >= 50 else 'OFF'} ho raha hai: {br[k]}% (kal {old}%)", "a")

    # ---- scanner
    sc = []
    for s, q in quotes.items():
        df = live.get(s)
        if df is None or len(df) < 70:
            continue
        c = df["Close"]
        sc.append(dict(s=s, ind=uni.get(s, "Other"), ltp=clean(q["ltp"]), chg=chg(s),
                       r3m=clean((c.iloc[-1] / c.iloc[-64] - 1) * 100, 1),
                       hi52=clean((c.iloc[-1] / df["High"].iloc[-252:].max() - 1) * 100, 1)))
    scd = pd.DataFrame(sc)
    top = lambda col, asc, n=15: scd.dropna(subset=[col]).sort_values(col, ascending=asc).head(n).to_dict("records") if len(scd) else []
    adv = int((scd.chg > 0).sum()) if len(scd) else 0
    dec = int((scd.chg < 0).sum()) if len(scd) else 0

    return {
        "time": f"{t:%d %b %Y, %I:%M:%S %p}", "epoch": int(time.time()), "status": status,
        "market": market_open(t), "confirm": confirm, "quotes": len(quotes),
        "breadth": br, "breadth0": br0, "on": {"pivot": piv_on, "positional": pos_on}, "rules": RULES,
        "total": dict(equity=clean(tot_eq, 0), start=tot_cap, ret=clean((tot_eq / tot_cap - 1) * 100) if tot_cap else None,
                      day=clean(tot_day, 0)),
        "books": books, "alerts": alerts,
        "pivot": piv, "pos": pos, "watch": [dict(w, chg=chg(w["s"])) for w in strat["watch"][:25]],
        "industries": strat["industries"],
        "scan": {"adv": adv, "dec": dec, "gainers": top("chg", False), "losers": top("chg", True),
                 "rs": top("r3m", False), "near52": top("hi52", False)},
    }


# ----------------------------------------------------------------------------- web
def serve():
    os.makedirs(LIVE_DIR, exist_ok=True)

    class H(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    srv = ThreadingHTTPServer(("0.0.0.0", PORT), partial(H, directory=LIVE_DIR))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def ask_token():
    print("\nUpstox access token paste karein (right-click = paste) aur Enter dabayein.")
    print("Bina token ke demo dekhna ho to sirf Enter dabayein.")
    return input("Token: ").strip()


def main():
    ap = argparse.ArgumentParser(description="Live engine: positions, signals, scanner every minute")
    ap.add_argument("--token", help="Upstox access token (na do to puchega)")
    ap.add_argument("--demo", action="store_true", help="nakli live price - dashboard test ke liye")
    ap.add_argument("--interval", type=int, default=60, help="seconds between updates (default 60)")
    a = ap.parse_args()

    os.makedirs(LIVE_DIR, exist_ok=True)
    src = os.path.join(HERE, "live_dashboard.html")
    if os.path.exists(src):
        with open(src, encoding="utf-8") as f, open(os.path.join(LIVE_DIR, "index.html"), "w", encoding="utf-8") as g:
            g.write(f.read())
    serve()
    ip = lan_ip()
    print("=" * 64)
    print(f" QUANTDIX LIVE:   PC par  ->  http://localhost:{PORT}")
    print(f"                  phone par (same WiFi) -> http://{ip}:{PORT}")
    print("=" * 64)

    token = a.token or os.environ.get("UPSTOX_TOKEN") or ("" if a.demo else ask_token())
    demo = a.demo or not token
    if demo:
        print("\n*** DEMO MODE: price nakli hain, sirf dashboard dekhne ke liye ***\n")

    uni = S.load_universe()
    hist = load_history(uni)
    book = load_book()
    base = run_scan(hist)
    keymap = {} if demo else load_instruments(set(hist))
    alerts, seen = [], set()
    last_fetch = 0
    while True:
        t = now()
        try:
            if demo or market_open(t) or last_fetch == 0 or time.time() - last_fetch > 600:
                quotes = demo_quotes(hist) if demo else upstox_quotes(keymap, token)
                last_fetch = time.time()
                live = with_today(hist, quotes, date.today())
                strat = run_scan(live)
                status = "DEMO (nakli price)" if demo else ("LIVE" if market_open(t) else "Market band (aakhri price)")
                snap = build_snapshot(hist, live, quotes, strat, base, book, uni, alerts, seen, status)
                write_json(os.path.join(LIVE_DIR, "live.json"), snap)
                b = snap["breadth"]
                print(f"[{t:%H:%M:%S}] {status} | {len(quotes)} stocks | breadth 20/50/200: {b['p20']}/{b['p50']}/{b['p200']}"
                      f" | pivot {len(snap['pivot'])} | positional {len(snap['pos'])}"
                      + (f" | portfolio ₹{snap['total']['equity']:,.0f} (aaj {snap['total']['day']:+,.0f})" if snap["books"] else ""))
        except TokenError as e:
            print(f"\n!!! {e}. Naya token banaiye.")
            token = ask_token()
            if not token:
                demo = True
            continue
        except Exception as e:
            print(f"[{t:%H:%M:%S}] error: {e} - agle minute phir koshish")
            traceback.print_exc(limit=1)
        time.sleep(a.interval if (demo or market_open()) else 60)


if __name__ == "__main__":
    main()
