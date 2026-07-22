#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fred_tester.py — Monetenfred-System: Validierung, Backtest, Live-Protokoll
==========================================================================
Dekodiertes Modell (Stand 21.07.2026):
  * Leiter: runde Zehnermarken ± 2 Punkte (Kern-Sprossen x02/x22/x38/x50/x58/x88)
  * dp     = Sprosse, die der Mitte (H+L)/2 der :45-Stundenkerze am nächsten liegt
  * Einstieg: Block per Limit an der Sprosse, Tranchen-Ausstiege an den nächsten Sprossen

Subkommandos:
  fetch      DAX-5-Minuten-Daten laden (yfinance, max. 60 Tage) -> SQLite
  validate   Formel-Hypothesen gegen historische Ansagen testen (Train/Test-Split)
  backtest   Handelbarkeits-Messung der Sprossen-Einstiege (Tranchen-Modell)
  live       Ansage protokollieren + aktuelle dp-Prognose ausgeben
  report     Protokoll-Statistik (Schaufenster-Quote vs. handelbare Quote)

Beispiele:
  python3 fred_tester.py fetch
  python3 fred_tester.py validate --ansagen monetenfred_ansagen.csv
  python3 fred_tester.py backtest --variante limit --tranchen 4
  python3 fred_tester.py live --text "11.45 brand new 912 dp"
"""

import argparse
import re
import sqlite3
import sys
from datetime import datetime, timedelta, time as dtime

import numpy as np
import pandas as pd

DB = "fred.db"
TZ = "Europe/Berlin"

CORE_OFFSETS = [2, 22, 38, 50, 58, 88]          # Kern-Leiter je 100er (historische Häufung)
SESSION_START = dtime(9, 45)                     # erster Intraday-Checkpoint: 9:45->10:45
SESSION_END = dtime(17, 45)


# ----------------------------------------------------------------------------
# Leiter & Snap
# ----------------------------------------------------------------------------
def ladder_core(lo: float, hi: float) -> list:
    """Kern-Leiter: 6 Sprossen je 100er-Block."""
    out = []
    for h in range(int(lo) // 100, int(hi) // 100 + 1):
        for o in CORE_OFFSETS:
            v = h * 100 + o
            if lo <= v <= hi:
                out.append(v)
    return sorted(out)


def ladder_tens(lo: float, hi: float) -> list:
    """Feine Leiter: jede Zehnermarke +/- 2 (plus glatte 50er/100er)."""
    out = set()
    for t in range(int(lo) // 10, int(hi) // 10 + 2):
        for v in (t * 10 - 2, t * 10 + 2):
            if lo <= v <= hi:
                out.add(v)
        if t * 10 % 50 == 0 and lo <= t * 10 <= hi:
            out.add(t * 10)
    return sorted(out)


def snap(price: float, ladder_fn) -> int:
    rungs = ladder_fn(price - 80, price + 80)
    return min(rungs, key=lambda r: abs(r - price))


def next_rungs(level: float, ladder_fn, n: int, direction: int) -> list:
    """n naechste Sprossen ober-(+1)/unterhalb(-1) von level."""
    rungs = ladder_fn(level - 400, level + 400)
    if direction > 0:
        cand = [r for r in rungs if r > level + 0.01]
        return cand[:n]
    cand = [r for r in rungs if r < level - 0.01]
    return cand[-n:][::-1]


# ----------------------------------------------------------------------------
# Daten
# ----------------------------------------------------------------------------
def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS bars(
        ts TEXT PRIMARY KEY, o REAL, h REAL, l REAL, c REAL, v REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS protokoll(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        erfasst TEXT, ansage_zeit TEXT, rohtext TEXT,
        dp REAL, u_ziele TEXT, o_ziele TEXT,
        prognose_dp REAL, prognose_regel TEXT,
        abweichung REAL, bewertet INTEGER DEFAULT 0)""")
    return con


def cmd_fetch(args):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("pip install yfinance --break-system-packages")
    df = yf.download(args.symbol, interval="5m", period="60d",
                     progress=False, auto_adjust=False)
    if df.empty:
        sys.exit("Keine Daten erhalten.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = df.index.tz_convert(TZ)
    con = db()
    rows = [(ts.isoformat(), r.open, r.high, r.low, r.close, r.volume)
            for ts, r in df.iterrows()]
    con.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    print(f"{len(rows)} 5-Min-Bars gespeichert ({df.index[0]} .. {df.index[-1]})")


def cmd_import_csv(args):
    """MT5-Export (GBE) einlesen: Formate 'DATE,TIME,O,H,L,C,...' oder Tab-getrennt."""
    raw = pd.read_csv(args.datei, sep=None, engine="python")
    raw.columns = [c.strip().lower().replace("<", "").replace(">", "") for c in raw.columns]
    dt_col = None
    if "date" in raw.columns and "time" in raw.columns and not pd.api.types.is_numeric_dtype(raw["time"]):
        dt = pd.to_datetime(raw["date"].astype(str) + " " + raw["time"].astype(str))
        dt = dt - pd.Timedelta(hours=args.tz_shift)
        dt = dt.dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
    else:
        dt_col = next((c for c in raw.columns if c in ("time", "date", "datetime", "timestamp")), raw.columns[0])
        col = raw[dt_col]
        if pd.api.types.is_numeric_dtype(col):
            # Unix-Timestamp (TradingView): Sekunden oder Millisekunden erkennen
            unit = "ms" if col.iloc[0] > 1e11 else "s"
            dt = pd.to_datetime(col, unit=unit, utc=True).dt.tz_convert(TZ)
            dt = dt - pd.Timedelta(hours=args.tz_shift)
        else:
            dt = pd.to_datetime(col)
            dt = dt - pd.Timedelta(hours=args.tz_shift)
            if dt.dt.tz is None:
                dt = dt.dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
    cols = {c: c[0] for c in ("open", "high", "low", "close") if c in raw.columns}
    df = raw.rename(columns=cols)
    df["v"] = raw.get("volume", raw.get("tickvol", 0))
    df.index = dt
    df = df[["o", "h", "l", "c", "v"]].dropna()
    con = db()
    rows = [(ts.isoformat(), r.o, r.h, r.l, r.c, r.v) for ts, r in df.iterrows()]
    con.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    print(f"{len(rows)} Bars importiert ({df.index[0]} .. {df.index[-1]}). "
          f"tz_shift={args.tz_shift}h (GBE-Serverzeit ist meist UTC+3 im Sommer -> shift 1)")


def load_bars() -> pd.DataFrame:
    con = db()
    df = pd.read_sql("SELECT * FROM bars ORDER BY ts", con,
                     parse_dates=["ts"], index_col="ts")
    if df.empty:
        sys.exit("Keine Bars in der DB — zuerst `fetch` ausführen (auf dem VPS).")
    df.index = pd.DatetimeIndex(df.index).tz_convert(TZ)
    return df


def hour45_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregiert 5-Min-Bars auf das :45-Stundenraster (09:45-10:45, ...)."""
    shifted = df.copy()
    shifted.index = shifted.index - pd.Timedelta(minutes=45)
    agg = shifted.resample("1h").agg(
        o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    agg.index = agg.index + pd.Timedelta(minutes=45)   # Index = Kerzen-START (:45)
    agg = agg.dropna()
    agg["checkpoint"] = agg.index + pd.Timedelta(hours=1)  # Snap-Zeitpunkt
    agg["mid"] = (agg.h + agg.l) / 2
    return agg


# ----------------------------------------------------------------------------
# Ansagen parsen
# ----------------------------------------------------------------------------
DP_PATTERNS = [
    re.compile(r"dp\D{0,25}?(\d{2}\.\d{3})", re.I),
    re.compile(r"(\d{2}\.\d{3})\D{0,15}dp", re.I),
    re.compile(r"dp\D{0,25}?(\d{3})\b", re.I),
    re.compile(r"\b(\d{3})\s+(?:brand\s+new\s+)?dp", re.I),
]


def parse_dp(text: str):
    for pat in DP_PATTERNS:
        m = pat.search(text)
        if m:
            raw = m.group(1)
            return float(raw.replace(".", "")) if "." in raw else float(raw)
    return None


def resolve_thousands(level3: float, ref_price: float) -> float:
    """3-stellige Kurznotation (z.B. 912) anhand des Marktpreises auflösen."""
    if level3 >= 1000:
        return level3
    base = int(ref_price) // 1000 * 1000
    cands = [base - 1000 + level3, base + level3, base + 1000 + level3]
    return min(cands, key=lambda c: abs(c - ref_price))


def load_ansagen(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    rows = []
    for _, r in df.iterrows():
        dp3 = parse_dp(str(r.get("nachricht", "")))
        if dp3 is None:
            continue
        try:
            ts = pd.Timestamp(datetime.strptime(
                f"{r.datum} {r.uhrzeit}", "%d.%m.%y %H:%M"), tz=TZ)
        except (ValueError, TypeError):
            continue
        rows.append({"ts": ts, "dp3": dp3, "typ": r.get("typ", ""),
                     "text": r.get("nachricht", "")})
    out = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    print(f"{len(out)} Ansagen mit dp geparst.")
    return out


# ----------------------------------------------------------------------------
# validate — Hypothesen gegen Historie
# ----------------------------------------------------------------------------
def cmd_validate(args):
    bars = load_bars()
    candles = hour45_candles(bars)
    ans = load_ansagen(args.ansagen)
    ans = ans[ans.typ.isin(["dp-update", "ovn-set", "ziele", "kommentar"])]

    hypos = {
        "mitte_kern":  ("mid", ladder_core),
        "mitte_zehner": ("mid", ladder_tens),
        "close_kern":  ("c", ladder_core),
        "close_zehner": ("c", ladder_tens),
    }
    results = []
    for _, a in ans.iterrows():
        # zugehörige :45-Kerze: letzte, deren Checkpoint <= Ansagezeit
        prior = candles[candles.checkpoint <= a.ts]
        if prior.empty:
            continue
        cd = prior.iloc[-1]
        if a.ts - cd.checkpoint > pd.Timedelta(minutes=59):
            continue  # Ansage passt zu keinem frischen Checkpoint (z.B. ovn)
        dp_true = resolve_thousands(a.dp3, cd.c)
        row = {"ts": a.ts, "dp_true": dp_true}
        for name, (ref, lfn) in hypos.items():
            pred = snap(cd[ref], lfn)
            row[name] = pred - dp_true
        results.append(row)

    res = pd.DataFrame(results)
    if res.empty:
        sys.exit("Keine bewertbaren Intraday-Ansagen im Datenzeitraum.")
    split = int(len(res) * 0.5)
    print(f"\nBewertbare Intraday-dp-Ansagen: {len(res)} "
          f"(Train: {split}, Test: {len(res)-split})")
    for teil, chunk in (("TRAIN", res.iloc[:split]), ("TEST", res.iloc[split:])):
        print(f"\n--- {teil} ---")
        for name in hypos:
            err = chunk[name].abs()
            hits = (err <= 2.5).mean() * 100
            print(f"  {name:14s}  mittl. Abw. {err.mean():6.1f} P | "
                  f"Median {err.median():5.1f} | Exakt-Treffer(±2,5P) {hits:4.0f}%")
    res.to_csv("validate_ergebnis.csv", index=False)
    print("\nDetails: validate_ergebnis.csv")


def cmd_validate_ovn(args):
    """Overnight-dp-Validierung: testet mehrere Nachtfenster-Hypothesen (braucht 24h-Daten, z.B. GBE-Import)."""
    bars = load_bars()
    ans = load_ansagen(args.ansagen)
    ovn = ans[ans.text.str.contains("ovn", case=False, na=False)]
    windows = {
        "1745_bis_0945": (dtime(17, 45), dtime(9, 45)),
        "2200_bis_0945": (dtime(22, 0),  dtime(9, 45)),
        "1730_bis_0900": (dtime(17, 30), dtime(9, 0)),
        "2200_bis_0800": (dtime(22, 0),  dtime(8, 0)),
    }
    results = []
    for _, a in ovn.iterrows():
        # Nacht NACH der Ansage (Sets werden abends 17:45-20:30 gepostet)
        d0 = a.ts.normalize()
        row = {"ts": a.ts}
        for name, (t_start, t_end) in windows.items():
            start = d0 + pd.Timedelta(hours=t_start.hour, minutes=t_start.minute)
            end_day = d0 + pd.Timedelta(days=1)
            while end_day.dayofweek >= 5:                 # Wochenende überspringen
                end_day += pd.Timedelta(days=1)
            end = end_day + pd.Timedelta(hours=t_end.hour, minutes=t_end.minute)
            win = bars[(bars.index >= start) & (bars.index <= end)]
            if len(win) < 12:
                continue
            mid = (win.h.max() + win.l.min()) / 2
            dp_true = resolve_thousands(a.dp3, mid)
            row[name] = snap(mid, ladder_tens) - dp_true
        if len(row) > 1:
            results.append(row)
    res = pd.DataFrame(results)
    if res.empty:
        sys.exit("Keine bewertbaren ovn-Sets — 24h-Daten vorhanden? (import-csv)")
    print(f"\nBewertbare ovn-Sets: {len(res)}")
    for name in windows:
        if name not in res.columns:
            continue
        err = res[name].abs().dropna()
        if err.empty:
            continue
        print(f"  Fenster {name:14s}  mittl. Abw. {err.mean():6.1f} P | "
              f"Median {err.median():5.1f} | Exakt(±2,5P) {(err <= 2.5).mean()*100:4.0f}% | n={len(err)}")
    res.to_csv("validate_ovn_ergebnis.csv", index=False)
    print("Details: validate_ovn_ergebnis.csv")


def add_indicators(df, macd_fast=12, macd_slow=26, macd_sig=9, wr_len=14):
    """MACD-Histogramm/Linien und Williams %R (0..100-Skala, 50=Mitte) ergaenzen."""
    ef = df.c.ewm(span=macd_fast, adjust=False).mean()
    es = df.c.ewm(span=macd_slow, adjust=False).mean()
    macd = ef - es
    signal = macd.ewm(span=macd_sig, adjust=False).mean()
    df = df.copy()
    df["macd"] = macd
    df["macd_sig"] = signal
    df["macd_cross_up"] = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    df["macd_cross_dn"] = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    hh = df.h.rolling(wr_len).max()
    ll = df.l.rolling(wr_len).min()
    # klassisch -100..0; hier auf 0..100 gedreht (0=Tief der Range, 100=Hoch)
    df["wr"] = (df.c - ll) / (hh - ll).replace(0, np.nan) * 100
    return df


def tag_regime(bars):
    """Pro Handelstag: True=Trendtag, False=Range-Tag.
    Trendtag wenn Tagesrange > 1.3x 10-Tage-Schnitt UND Schluss im aeussersten
    Drittel der Range (gerichtet), ODER ungefuelltes Gap > 0.3 ATR."""
    daily = bars.resample("1D").agg(o=("o","first"), h=("h","max"),
                                     l=("l","min"), c=("c","last")).dropna()
    daily["rng"] = daily.h - daily.l
    atr = daily["rng"].rolling(10, min_periods=3).mean()
    out, prev_c = {}, None
    for i,(d,r) in enumerate(daily.iterrows()):
        a = atr.iloc[i]
        if np.isnan(a) or a <= 0:
            a = daily["rng"].iloc[:i+1].mean()
        expansion = r.rng > 1.3 * a
        pos = (r.c - r.l)/r.rng if r.rng > 0 else 0.5
        directed = pos > 0.75 or pos < 0.25
        gap = abs(r.o - prev_c)/a if prev_c and a > 0 else 0
        gap_tr = prev_c is not None and gap > 0.3 and (
            (r.o > prev_c and r.c > r.o) or (r.o < prev_c and r.c < r.o))
        out[d.date()] = bool((expansion and directed) or gap_tr)
        prev_c = r.c
    return out


def cmd_reach(args):
    """Reine Ziel-Erreichung: Wie oft werden obere/untere Sprossen INNERHALB der :45-Stunde erreicht?
    Keine Trades, keine Stops — nur die Frage, ob die Levels als Beschreibung stimmen.
    Referenz: Ein Zufalls-Level gleicher Distanz wird als Vergleich mitgerechnet."""
    bars = load_bars()
    candles = hour45_candles(bars)
    lfn = ladder_tens if args.leiter == "zehner" else ladder_core
    rng = np.random.default_rng(0)
    rows = []
    for ts, cd in candles.iterrows():
        cp = cd.checkpoint
        if not (SESSION_START <= cp.time() <= SESSION_END):
            continue
        dp = snap(cd.mid, lfn)
        win = bars[(bars.index > cp) & (bars.index <= cp + pd.Timedelta(hours=1))]
        if len(win) < 6:
            continue
        entry = win.iloc[0].o
        hi, lo = win.h.max(), win.l.min()
        o1, o2 = next_rungs(dp, lfn, 2, +1)[:2] + [np.nan, np.nan][:max(0, 2-len(next_rungs(dp, lfn, 2, +1)))]
        u_list = next_rungs(dp, lfn, 2, -1)
        o_list = next_rungs(dp, lfn, 2, +1)
        rec = {"cp": cp, "dp": dp, "entry": entry,
               "o1": o_list[0] if len(o_list) > 0 else np.nan,
               "o2": o_list[1] if len(o_list) > 1 else np.nan,
               "u1": u_list[0] if len(u_list) > 0 else np.nan,
               "u2": u_list[1] if len(u_list) > 1 else np.nan,
               "hi": hi, "lo": lo}
        rec["o1_hit"] = hi >= rec["o1"] if not np.isnan(rec["o1"]) else False
        rec["o2_hit"] = hi >= rec["o2"] if not np.isnan(rec["o2"]) else False
        rec["u1_hit"] = lo <= rec["u1"] if not np.isnan(rec["u1"]) else False
        rec["u2_hit"] = lo <= rec["u2"] if not np.isnan(rec["u2"]) else False
        rec["oben"] = entry <= dp        # dp über Einstieg -> Long-Bias erwartet Oben-Ziele
        # Zufalls-Referenz: Level gleicher Distanz wie o1, aber zufällig platziert
        if not np.isnan(rec["o1"]):
            dist = rec["o1"] - dp
            rand_lvl = entry + rng.choice([-1, 1]) * dist
            rec["rand_hit"] = (hi >= rand_lvl) if rand_lvl > entry else (lo <= rand_lvl)
        rows.append(rec)

    r = pd.DataFrame(rows)
    if r.empty:
        sys.exit("Keine Kerzen im Datenbereich.")
    n = len(r)
    print(f"\nZiel-Erreichung innerhalb der :45-Stunde ({args.leiter}-Leiter, {n} Stunden)")
    print(f"  1. oberes Ziel (o1) erreicht:   {100*r.o1_hit.mean():5.1f}%")
    print(f"  2. oberes Ziel (o2) erreicht:   {100*r.o2_hit.mean():5.1f}%")
    print(f"  1. unteres Ziel (u1) erreicht:  {100*r.u1_hit.mean():5.1f}%")
    print(f"  2. unteres Ziel (u2) erreicht:  {100*r.u2_hit.mean():5.1f}%")
    print(f"  MIND. ein 1. Ziel (o1 ODER u1): {100*(r.o1_hit | r.u1_hit).mean():5.1f}%")
    print(f"  BEIDE 1. Ziele (o1 UND u1):     {100*(r.o1_hit & r.u1_hit).mean():5.1f}%")
    # Der ehrliche Test: richtungsrichtiges Ziel
    long_bias = r[r.oben]
    short_bias = r[~r.oben]
    dir_hit = pd.concat([long_bias.o1_hit, short_bias.u1_hit])
    print(f"\n  Richtungs-Ziel (dp-Bias) erreicht: {100*dir_hit.mean():5.1f}%")
    if "rand_hit" in r:
        print(f"  Zufalls-Level gleicher Distanz:    {100*r.rand_hit.mean():5.1f}%  <- Vergleichsbasis")
        print(f"\n  => Edge nur real, wenn Richtungs-Ziel DEUTLICH über Zufall liegt.")
    r.to_csv("reach_ergebnis.csv", index=False)
    print("Details: reach_ergebnis.csv")


def cmd_ruecklauf(args):
    """Backtest von Monetenfreds ECHTER Methode (aus 9 Monaten Chat rekonstruiert):
    - POI = runde 100er-Marke (A-Level) bzw. Sprosse
    - Einstieg NICHT am ersten Anlauf, sondern im RÜCKLAUF: Kurs muss die Marke
      erst durchstoßen und zurückerobern (Long: unter Marke, dann zurück drüber)
    - Enger Stop (15-20 P), bei kleinem Verlust WIEDEREINSTIEG an nächster Sprosse
    - Ziel: nächste 30er-Sprosse; SL nach +Gewinn auf Einstand nachziehen
    """
    bars = load_bars()
    lfn = ladder_tens if args.leiter == "zehner" else ladder_core
    stop_p = args.stop
    max_versuche = args.versuche
    ziel_dist = args.ziel

    # Trendtag-Regime bestimmen (fuer optionalen Filter)
    regime = tag_regime(bars)
    n_trend = sum(1 for v in regime.values() if v)
    n_range = len(regime) - n_trend

    # Runde 100er als A-POIs im Datenbereich
    lo_all, hi_all = bars.l.min(), bars.h.max()
    pois = [p for p in range(int(lo_all)//100*100, int(hi_all)//100*100 + 100, 100)]

    trades = []
    df = add_indicators(bars).reset_index()
    tscol = df.columns[0]  # Zeitspalte
    macd_lookback = args.macd_lookback
    wr_lo, wr_hi = args.wr_min, args.wr_max
    gefiltert_momentum = 0
    for poi in pois:
        # Long-Setups: Kurs faellt unter POI und erobert zurueck (Ruecklauf von unten)
        below = False
        versuche = 0
        i = 0
        while i < len(df) - 1:
            row = df.iloc[i]
            if row.l <= poi - 2 and row.c < poi:
                below = True
            # Rueckeroberung: war unter POI, schliesst jetzt wieder drueber
            if below and row.c > poi and versuche < max_versuche:
                tag = df.iloc[i][tscol].date()
                ist_trend = regime.get(tag, False)
                # Filter: nur Range-Tage handeln, wenn --filter gesetzt
                if args.filter == "range" and ist_trend:
                    below = False
                    i += 1
                    continue
                # Momentum-Filter: MACD-Kreuz in den letzten N Bars + %R im Mittelband
                if args.momentum == "an":
                    lo_i = max(0, i - macd_lookback)
                    cross = bool(df.macd_cross_up.iloc[lo_i:i+1].any())
                    wr = df.wr.iloc[i]
                    if not cross or np.isnan(wr) or not (wr_lo <= wr <= wr_hi):
                        gefiltert_momentum += 1
                        below = False
                        i += 1
                        continue
                entry = poi + args.spread
                stop = entry - stop_p
                target = entry + ziel_dist
                below = False
                versuche += 1
                # Trade auswerten ueber die naechsten Bars
                outcome = None
                for j in range(i+1, min(i+24, len(df))):
                    b = df.iloc[j]
                    if b.l <= stop:
                        outcome = -(stop_p + args.spread); break
                    if b.h >= target:
                        outcome = ziel_dist - args.spread; break
                if outcome is None:
                    outcome = df.iloc[min(i+23, len(df)-1)].c - entry
                trades.append({"poi": poi, "idx": i, "pnl": outcome,
                               "win": outcome > 0, "versuch": versuche})
                if outcome > 0:
                    versuche = 0  # nach Gewinn Zaehler zuruecksetzen
                i += 3
            i += 1

    t = pd.DataFrame(trades)
    if t.empty:
        sys.exit("Keine Rücklauf-Setups gefunden.")
    wins = t.win.sum()
    pf = t[t.pnl > 0].pnl.sum() / max(1e-9, -t[t.pnl < 0].pnl.sum())
    net = t.pnl.sum()
    # Setup-Ebene: Gruppen bis zum ersten Gewinn (Wiedereinstiegs-Logik)
    print(f"\nRücklauf-Backtest — Monetenfreds echte Methode")
    print(f"  Leiter={args.leiter}, Stop={stop_p}P, Ziel={ziel_dist}P, "
          f"max. Wiedereinstiege={max_versuche}, Spread={args.spread}P")
    print(f"  Filter={args.filter}  (Range-Tage: {n_range}, Trend-Tage: {n_trend})")
    if args.momentum == "an":
        print(f"  Momentum-Filter: MACD-Kreuz (max {macd_lookback} Bars zurück) + "
              f"%R {wr_lo}-{wr_hi}  |  verworfen: {gefiltert_momentum} Setups")
    print(f"  Einzel-Trades:            {len(t)}")
    print(f"  Trefferquote:             {100*wins/len(t):5.1f}%")
    print(f"  Ø P&L pro Trade:          {t.pnl.mean():+6.1f} P")
    print(f"  Profit-Faktor:            {pf:5.2f}")
    print(f"  Netto gesamt:             {net:+.0f} P")
    print(f"  Ø Versuche bis Gewinn:    {t[t.win].versuch.mean():.2f}")
    print(f"\n  Referenz: Zufall+Kosten liegt bei PF ~0,7 — alles unter 1,2 ist Rauschen.")
    t.to_csv("ruecklauf_ergebnis.csv", index=False)
    print("  Details: ruecklauf_ergebnis.csv")


def cmd_macd(args):
    """Backtest des MACD/%R-Setups (1-Min-Trigger, optional 5-Min-Bestaetigung).
    Vergleicht explizit: mit Stop vs. ohne Stop (Halten bis Gegenkreuz).
    Ziel/Stop wahlweise fest in Punkten oder an der Leiter-Sprosse ausgerichtet."""
    bars = load_bars()
    lfn = ladder_tens if args.leiter == "zehner" else ladder_core
    df = add_indicators(bars, wr_len=args.wr_len)

    # 5-Minuten-Bestaetigung: MACD auf resampelten Bars
    if args.mtf == "an":
        b5 = bars.resample("5min").agg(o=("o","first"), h=("h","max"),
                                        l=("l","min"), c=("c","last")).dropna()
        d5 = add_indicators(b5, wr_len=args.wr_len)
        up5 = d5.macd > d5.macd_sig
        df["mtf_up"] = up5.reindex(df.index, method="ffill")
    else:
        df["mtf_up"] = None

    d = df.reset_index()
    tscol = d.columns[0]
    max_bars = args.max_bars

    def run(mit_stop: bool):
        trades = []
        i = args.wr_len + 30
        while i < len(d) - 2:
            r = d.iloc[i]
            long_sig = bool(r.macd_cross_up)
            short_sig = bool(r.macd_cross_dn)
            if not (long_sig or short_sig):
                i += 1
                continue
            direction = 1 if long_sig else -1
            # %R-Mittelband-Filter
            if not np.isnan(r.wr) and not (args.wr_min <= r.wr <= args.wr_max):
                i += 1
                continue
            # Multi-Timeframe-Bestaetigung
            if args.mtf == "an" and r.mtf_up is not None and not pd.isna(r.mtf_up):
                if (direction > 0) != bool(r.mtf_up):
                    i += 1
                    continue
            entry = d.iloc[i+1].o + direction * args.spread
            # Ziel/Stop: Leiter-Sprossen oder feste Punkte
            if args.ziele == "leiter":
                tg = next_rungs(entry, lfn, 1, direction)
                st = next_rungs(entry, lfn, 2, -direction)
                target = tg[0] if tg else entry + direction * args.ziel
                stop = st[-1] if len(st) > 1 else entry - direction * args.stop
            else:
                target = entry + direction * args.ziel
                stop = entry - direction * args.stop
            outcome, bars_held = None, 0
            for j in range(i+1, min(i+1+max_bars, len(d))):
                b = d.iloc[j]
                bars_held = j - i
                if mit_stop:
                    hit_stop = (b.l <= stop) if direction > 0 else (b.h >= stop)
                    if hit_stop:
                        outcome = (stop - entry) * direction - args.spread
                        break
                hit_tg = (b.h >= target) if direction > 0 else (b.l <= target)
                if hit_tg:
                    outcome = (target - entry) * direction - args.spread
                    break
                # Gegenkreuz beendet den Trade
                gegen = b.macd_cross_dn if direction > 0 else b.macd_cross_up
                if bool(gegen):
                    outcome = (b.c - entry) * direction - args.spread
                    break
            if outcome is None:
                outcome = (d.iloc[min(i+max_bars, len(d)-1)].c - entry) * direction - args.spread
            trades.append({"ts": d.iloc[i][tscol], "dir": direction, "entry": entry,
                           "pnl": outcome, "bars": bars_held})
            i += 3
        return pd.DataFrame(trades)

    print(f"\nMACD/%R-Setup — {args.wr_len}er %R, Band {args.wr_min}-{args.wr_max}, "
          f"MTF={args.mtf}, Ziele={args.ziele}")
    for label, mit_stop in (("MIT Stop", True), ("OHNE Stop", False)):
        t = run(mit_stop)
        if t.empty:
            print(f"  {label}: keine Trades")
            continue
        pf = t[t.pnl > 0].pnl.sum() / max(1e-9, -t[t.pnl < 0].pnl.sum())
        tage = t.ts.dt.date.nunique()
        worst = t.pnl.min()
        # groesste Verlustserie
        streak = maxstreak = 0
        for p in t.pnl:
            streak = streak + 1 if p < 0 else 0
            maxstreak = max(maxstreak, streak)
        print(f"\n  --- {label} ---")
        print(f"    Trades: {len(t)}  ({len(t)/max(1,tage):.1f}/Tag an {tage} Tagen)")
        print(f"    Trefferquote:      {100*(t.pnl>0).mean():5.1f}%")
        print(f"    Ø P&L pro Trade:   {t.pnl.mean():+6.1f} P")
        print(f"    Profit-Faktor:     {pf:5.2f}")
        print(f"    Netto gesamt:      {t.pnl.sum():+.0f} P   (Ø {t.pnl.sum()/max(1,tage):+.0f} P/Tag)")
        print(f"    Groesster Verlust: {worst:+.0f} P")
        print(f"    Laengste Verlustserie: {maxstreak}")
        t.to_csv(f"macd_{'stop' if mit_stop else 'nostop'}.csv", index=False)
    print(f"\n  Bei 1000 Stueck / BV 0,01: 1 Punkt = 10 EUR")


# ----------------------------------------------------------------------------
# backtest — Handelbarkeit der Sprossen-Einstiege
# ----------------------------------------------------------------------------
def cmd_backtest(args):
    bars = load_bars()
    candles = hour45_candles(bars)
    lfn = ladder_tens if args.leiter == "zehner" else ladder_core
    spread = args.spread
    trades = []

    for ts, cd in candles.iterrows():
        cp = cd.checkpoint
        if not (SESSION_START <= cp.time() <= SESSION_END):
            continue
        dp = snap(cd.mid, lfn)
        window = bars[(bars.index > cp) & (bars.index <= cp + pd.Timedelta(hours=1))]
        if len(window) < 6:
            continue
        entry_bar = window.iloc[0]
        direction = 1 if entry_bar.o > dp else -1        # Seite des dp = Richtung
        entry = entry_bar.o + direction * spread          # Market am Checkpoint

        targets = next_rungs(entry, lfn, args.tranchen, direction)
        stop = next_rungs(dp, lfn, 1, -direction)
        stop = stop[0] - direction * 0.0 if stop else entry - direction * 40

        pnl, size = 0.0, 1.0 / args.tranchen
        remaining = args.tranchen
        stopped = False
        for _, b in window.iterrows():
            lo_hit = b.l <= stop if direction > 0 else b.h >= stop
            for tgt in list(targets):
                hit = b.h >= tgt if direction > 0 else b.l <= tgt
                if hit:
                    pnl += size * (tgt - entry) * direction - size * spread
                    targets.remove(tgt)
                    remaining -= 1
            if lo_hit and remaining > 0:
                pnl += remaining * size * (stop - entry) * direction - remaining * size * spread
                remaining, stopped = 0, True
                break
            if remaining == 0:
                break
        if remaining > 0:                                  # Rest zum Stundenschluss glattstellen
            pnl += remaining * size * (window.iloc[-1].c - entry) * direction
        trades.append({"cp": cp, "dp": dp, "dir": direction, "entry": entry,
                       "pnl": pnl, "stopped": stopped,
                       "tranchen_gefuellt": args.tranchen - remaining})

    t = pd.DataFrame(trades)
    wins = (t.pnl > 0).sum()
    pf = t[t.pnl > 0].pnl.sum() / max(1e-9, -t[t.pnl < 0].pnl.sum())
    print(f"\nBacktest ({args.leiter}-Leiter, {args.tranchen} Tranchen, "
          f"Spread {spread} P, {len(t)} Stunden-Setups)")
    print(f"  Gewinnquote (pro Block):   {100*wins/len(t):5.1f}%")
    print(f"  Ø P&L pro Block:           {t.pnl.mean():+6.1f} DAX-Punkte")
    print(f"  Profit-Faktor:             {pf:5.2f}")
    print(f"  Stop-Quote:                {100*t.stopped.mean():5.1f}%")
    print(f"  Ø gefüllte Ziel-Tranchen:  {t.tranchen_gefuellt.mean():4.2f} / {args.tranchen}")
    t.to_csv("backtest_ergebnis.csv", index=False)
    print("Details: backtest_ergebnis.csv")


# ----------------------------------------------------------------------------
# live & report — Protokoll
# ----------------------------------------------------------------------------
def cmd_live(args):
    con = db()
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    dp3 = parse_dp(args.text or "")
    prog, regel = None, "mitte_zehner"
    try:
        candles = hour45_candles(load_bars())
        cd = candles.iloc[-1]
        prog = snap(cd.mid, ladder_tens)
        print(f"Prognose ({regel}) für Checkpoint {cd.checkpoint:%H:%M}: dp {prog}")
    except SystemExit:
        print("(Keine Kursdaten — Prognose übersprungen; `fetch` nachholen.)")
    dp_true = None
    if dp3 is not None and prog is not None:
        dp_true = resolve_thousands(dp3, prog)
        print(f"Ansage-dp: {dp_true}  |  Abweichung: {prog - dp_true:+.0f} P")
    con.execute("""INSERT INTO protokoll
        (erfasst, ansage_zeit, rohtext, dp, prognose_dp, prognose_regel, abweichung, bewertet)
        VALUES (?,?,?,?,?,?,?,?)""",
        (now, args.zeit or now, args.text, dp_true, prog, regel,
         None if dp_true is None or prog is None else prog - dp_true,
         1 if dp_true is not None and prog is not None else 0))
    con.commit()
    print("Protokolliert.")


def cmd_report(args):
    con = db()
    df = pd.read_sql("SELECT * FROM protokoll", con)
    if df.empty:
        sys.exit("Protokoll leer.")
    b = df[df.bewertet == 1]
    print(f"Einträge: {len(df)}  |  bewertet: {len(b)}")
    if len(b):
        print(f"  Exakt-Treffer (±2,5 P): {(b.abweichung.abs() <= 2.5).mean()*100:.0f}%")
        print(f"  Mittlere Abweichung:    {b.abweichung.abs().mean():.1f} P")


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch"); f.add_argument("--symbol", default="^GDAXI")
    f.set_defaults(fn=cmd_fetch)

    v = sub.add_parser("validate")
    v.add_argument("--ansagen", default="monetenfred_ansagen.csv")
    v.set_defaults(fn=cmd_validate)

    vo = sub.add_parser("validate-ovn")
    vo.add_argument("--ansagen", default="monetenfred_ansagen.csv")
    vo.set_defaults(fn=cmd_validate_ovn)

    ic = sub.add_parser("import-csv")
    ic.add_argument("--datei", required=True, help="MT5/GBE CSV-Export (M5)")
    ic.add_argument("--tz-shift", type=float, default=1.0,
                    help="Stunden, um die die CSV-Zeit VOR Berlin liegt (GBE-Server UTC+3 im Sommer -> 1)")
    ic.set_defaults(fn=cmd_import_csv)

    b = sub.add_parser("backtest")
    b.add_argument("--leiter", choices=["kern", "zehner"], default="kern")
    b.add_argument("--tranchen", type=int, default=4)
    b.add_argument("--spread", type=float, default=1.5,
                   help="Kosten je Seite in DAX-Punkten (Spread+Slippage)")
    b.set_defaults(fn=cmd_backtest)

    md = sub.add_parser("macd")
    md.add_argument("--leiter", choices=["kern", "zehner"], default="zehner")
    md.add_argument("--ziele", choices=["fest", "leiter"], default="leiter",
                    help="Ziel/Stop an Leiter-Sprossen oder feste Punkte")
    md.add_argument("--ziel", type=float, default=25, help="festes Ziel in Punkten")
    md.add_argument("--stop", type=float, default=20, help="fester Stop in Punkten")
    md.add_argument("--wr-len", type=int, default=14, help="Williams %%R Periode")
    md.add_argument("--wr-min", type=float, default=0, help="%%R Untergrenze (0=Filter aus)")
    md.add_argument("--wr-max", type=float, default=100, help="%%R Obergrenze")
    md.add_argument("--mtf", choices=["aus", "an"], default="an",
                    help="5-Minuten-MACD als Richtungsbestaetigung verlangen")
    md.add_argument("--max-bars", type=int, default=60, help="max. Haltedauer in Bars")
    md.add_argument("--spread", type=float, default=1.5)
    md.set_defaults(fn=cmd_macd)

    rc = sub.add_parser("reach")
    rc.add_argument("--leiter", choices=["kern", "zehner"], default="zehner")
    rc.set_defaults(fn=cmd_reach)

    rl = sub.add_parser("ruecklauf")
    rl.add_argument("--leiter", choices=["kern", "zehner"], default="zehner")
    rl.add_argument("--stop", type=float, default=18, help="Stop in Punkten (Fred: 15-20)")
    rl.add_argument("--ziel", type=float, default=30, help="Ziel-Distanz in Punkten (30er-Staffel)")
    rl.add_argument("--versuche", type=int, default=3, help="max. Wiedereinstiege pro POI")
    rl.add_argument("--spread", type=float, default=1.5)
    rl.add_argument("--filter", choices=["aus", "range"], default="aus",
                    help="'range' = nur an Range-Tagen handeln (Trendtage überspringen)")
    rl.add_argument("--momentum", choices=["aus", "an"], default="aus",
                    help="'an' = zusätzlich MACD-Kreuz + Williams %%R im Mittelband verlangen")
    rl.add_argument("--macd-lookback", type=int, default=6,
                    help="MACD-Kreuz darf max. so viele Bars zurückliegen")
    rl.add_argument("--wr-min", type=float, default=35, help="Williams %%R Untergrenze (0-100)")
    rl.add_argument("--wr-max", type=float, default=70, help="Williams %%R Obergrenze (0-100)")
    rl.set_defaults(fn=cmd_ruecklauf)

    l = sub.add_parser("live")
    l.add_argument("--text", required=True, help="Rohtext der Ansage")
    l.add_argument("--zeit", help="Ansagezeit ISO, default jetzt")
    l.set_defaults(fn=cmd_live)

    r = sub.add_parser("report"); r.set_defaults(fn=cmd_report)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
