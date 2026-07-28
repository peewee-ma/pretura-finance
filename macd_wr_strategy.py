#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macd_wr_strategy.py — Backtest für Peewees MACD + Williams %R System (DAX, M1)
================================================================================
Regeln (Stand 28.07.2026):
  ENTRY LONG : MACD(12,26,9) kreuzt Signallinie nach oben
               UND Williams %R(14) steigt über -50 (Momentum-Bestätigung)
  ENTRY SHORT: spiegelverkehrt (MACD-Kreuz nach unten, %R fällt unter -50)

  FILTER 1 (Überdehnung/EMA50): Ist der Kurs mehr als EXT_MAX Punkte vom
               EMA50 entfernt, KEIN Entry mehr in Bewegungsrichtung
               (Rücklauf erwartet — nicht hinterherlaufen).
  FILTER 2 (M5-Konfluenz): Gleichgerichtetes MACD-Kreuz auf M5 innerhalb
               der letzten M5_LOOKBACK M5-Kerzen => "STRONG"-Signal (Flag,
               optional doppelte Größe via STRONG_MULT).

  REGIME (MACD-Nulllinie):
     Long  bei MACD > 0  => TREND-Modus  (skalieren: 50% bei TP1, Rest TP2,
                                          Stop nach TP1 auf Breakeven)
     Long  bei MACD < 0  => SCALP-Modus  (Konter-Trade: schneller fixer TP)
     Short spiegelverkehrt.

  STOP-VARIANTEN (alle drei werden getestet und verglichen):
     'swing'   : letztes Swing-Tief/-Hoch der letzten SWING_N M1-Kerzen
                 +/- Puffer, gedeckelt auf STOP_CAP Punkte
     'fixed'   : fix STOP_FIXED Punkte
     'recross' : Exit bei MACD-Rückkreuzung (kein Preis-Stop; Notbremse
                 STOP_CAP Punkte als Katastrophen-Stop)

  Konservative Auswertung: Liegen TP und Stop in derselben Kerze,
  wird der STOP zuerst gewertet (PESSIMISTIC=True) — vermeidet die
  bekannte Ziel-vor-Stop-Verzerrung aus dem ruecklauf-Test.

Daten:
  - MT5-Export (DE40 M1): Tab-getrennt mit <DATE> <TIME> <OPEN> ... Spalten
  - Generisches CSV: Spalten datetime/date,open,high,low,close (Groß/klein egal)

Aufruf:
  python3 macd_wr_strategy.py daten.csv
  python3 macd_wr_strategy.py daten.csv --session 08:00-17:30 --ext-max 30
"""

import argparse
import sys
import numpy as np
import pandas as pd

# ----------------------------- Parameter ------------------------------------
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
WR_PERIOD = 14
EMA_TREND = 50
WR_MID = -50.0          # Mittellinie: Long braucht %R > -50 und steigend
EXT_MAX = 30.0          # max. Abstand zum EMA50 in Punkten (Überdehnungsfilter)
M5_LOOKBACK = 3         # M5-Kreuz max. n M5-Kerzen alt => STRONG
STRONG_MULT = 2.0       # Positionsmultiplikator bei STRONG-Signal
TP_SCALP = 10.0         # fixer TP im Scalp-/Konter-Modus (Punkte)
TP1_TREND = 15.0        # Trend-Modus: 50% Teilverkauf
TP2_TREND = 35.0        # Trend-Modus: Rest-Ziel
SWING_N = 10            # Swing-Stop: Tief/Hoch der letzten n Kerzen
SWING_BUF = 2.0         # Puffer unter/über dem Swing (Punkte)
STOP_FIXED = 15.0       # fixer Stop (Punkte)
STOP_CAP = 40.0         # Deckel für Swing-Stop und Notbremse bei 'recross'
PESSIMISTIC = True      # TP & Stop in gleicher Kerze => Stop zählt zuerst
COOLDOWN = 3            # Kerzen Pause nach Trade-Exit


# ----------------------------- Daten laden ----------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Lädt MT5-Export oder generisches CSV -> DataFrame mit DatetimeIndex."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.readline()
    sep = "\t" if "\t" in head else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]

    if "date" in df.columns and "time" in df.columns:
        dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    else:
        dtcol = next(c for c in df.columns if "date" in c or "time" in c)
        dt = pd.to_datetime(df[dtcol])

    out = pd.DataFrame({
        "open":  pd.to_numeric(df["open"], errors="coerce"),
        "high":  pd.to_numeric(df["high"], errors="coerce"),
        "low":   pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
    }, index=dt).dropna().sort_index()
    return out


# ----------------------------- Indikatoren ----------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    ema_f = c.ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = c.ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd"] = ema_f - ema_s
    df["macd_sig"] = df["macd"].ewm(span=MACD_SIG, adjust=False).mean()
    hh = df["high"].rolling(WR_PERIOD).max()
    ll = df["low"].rolling(WR_PERIOD).min()
    df["wr"] = -100.0 * (hh - c) / (hh - ll).replace(0, np.nan)
    df["ema50"] = c.ewm(span=EMA_TREND, adjust=False).mean()
    df["cross_up"] = (df["macd"].shift(1) <= df["macd_sig"].shift(1)) & (df["macd"] > df["macd_sig"])
    df["cross_dn"] = (df["macd"].shift(1) >= df["macd_sig"].shift(1)) & (df["macd"] < df["macd_sig"])
    return df


def m5_confluence(df1: pd.DataFrame) -> pd.DataFrame:
    """M5-MACD-Kreuze berechnen und als 'letztes M5-Kreuz' auf M1 mappen."""
    m5 = df1[["open", "high", "low", "close"]].resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    m5 = add_indicators(m5)
    last_dir, last_ts, dirs, tss = 0, pd.NaT, [], []
    for ts, row in m5.iterrows():
        if row["cross_up"]:
            last_dir, last_ts = 1, ts
        elif row["cross_dn"]:
            last_dir, last_ts = -1, ts
        dirs.append(last_dir); tss.append(last_ts)
    m5["m5_dir"] = dirs
    m5["m5_ts"] = tss
    aligned = m5[["m5_dir", "m5_ts"]].reindex(df1.index, method="ffill")
    df1["m5_dir"] = aligned["m5_dir"].fillna(0)
    df1["m5_age_min"] = (df1.index - aligned["m5_ts"]).dt.total_seconds() / 60.0
    return df1


# ----------------------------- Backtest -------------------------------------
def run_backtest(df: pd.DataFrame, stop_mode: str, session=None) -> pd.DataFrame:
    trades = []
    pos = None
    cooldown = 0
    idx = df.index
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    macd, sig, wr, ema = df["macd"].values, df["macd_sig"].values, df["wr"].values, df["ema50"].values
    cross_up, cross_dn = df["cross_up"].values, df["cross_dn"].values
    m5dir, m5age = df["m5_dir"].values, df["m5_age_min"].values

    def in_session(ts):
        if session is None:
            return True
        t = ts.time()
        return session[0] <= t <= session[1]

    for i in range(max(MACD_SLOW, EMA_TREND, WR_PERIOD) + 5, len(df)):
        ts = idx[i]

        # ---------- offene Position managen ----------
        if pos is not None:
            p = pos
            hit_stop = (l[i] <= p["stop"]) if p["dir"] == 1 else (h[i] >= p["stop"])
            price_tp = p["tp2"] if p["scaled"] else p["tp1"]
            hit_tp = (h[i] >= price_tp) if p["dir"] == 1 else (l[i] <= price_tp)
            recross = (cross_dn[i] if p["dir"] == 1 else cross_up[i]) if stop_mode == "recross" else False

            exit_price, exit_reason, exit_frac = None, None, None
            if hit_stop and hit_tp:
                if PESSIMISTIC:
                    exit_price, exit_reason, exit_frac = p["stop"], "stop", p["size_left"]
                else:
                    exit_price, exit_reason, exit_frac = price_tp, "tp", None
            elif hit_stop:
                exit_price, exit_reason, exit_frac = p["stop"], "stop", p["size_left"]
            elif hit_tp:
                exit_price, exit_reason = price_tp, "tp"
            elif recross:
                exit_price, exit_reason, exit_frac = c[i], "recross", p["size_left"]

            if exit_reason == "tp":
                if p["mode"] == "scalp" or p["scaled"]:
                    exit_frac = p["size_left"]                      # kompletter Rest raus
                else:
                    # Trend-Modus: 50% bei TP1, Stop auf Breakeven, weiter halten
                    frac = 0.5 * p["size_left"]
                    pts = (exit_price - p["entry"]) * p["dir"]
                    p["pnl"] += pts * frac * p["mult"]
                    p["size_left"] -= frac
                    p["scaled"] = True
                    p["stop"] = p["entry"]                          # Breakeven
                    exit_price = None                               # Position läuft weiter

            if exit_price is not None and exit_frac is not None:
                pts = (exit_price - p["entry"]) * p["dir"]
                p["pnl"] += pts * exit_frac * p["mult"]
                trades.append({
                    "entry_time": p["time"], "exit_time": ts,
                    "dir": "LONG" if p["dir"] == 1 else "SHORT",
                    "mode": p["mode"], "strong": p["strong"],
                    "entry": p["entry"], "exit": exit_price,
                    "stop0": p["stop0"], "reason": exit_reason,
                    "pnl_pts": p["pnl"],
                    "hold_min": (ts - p["time"]).total_seconds() / 60.0,
                })
                pos, cooldown = None, COOLDOWN
            continue

        # ---------- Entry-Logik ----------
        if cooldown > 0:
            cooldown -= 1
            continue
        if not in_session(ts) or np.isnan(wr[i]) or np.isnan(ema[i]):
            continue

        sig_long = cross_up[i] and wr[i] > WR_MID and wr[i] > wr[i - 1]
        sig_short = cross_dn[i] and wr[i] < WR_MID and wr[i] < wr[i - 1]
        if not (sig_long or sig_short):
            continue

        d = 1 if sig_long else -1
        # Überdehnungsfilter: nicht in Bewegungsrichtung hinterherlaufen
        ext = (c[i] - ema[i]) * d
        if ext > EXT_MAX:
            continue

        strong = (m5dir[i] == d) and (m5age[i] <= M5_LOOKBACK * 5)
        mode = "trend" if macd[i] * d > 0 else "scalp"
        entry = c[i]

        if stop_mode == "swing":
            if d == 1:
                stop = max(l[i - SWING_N:i].min() - SWING_BUF, entry - STOP_CAP)
            else:
                stop = min(h[i - SWING_N:i].max() + SWING_BUF, entry + STOP_CAP)
        elif stop_mode == "fixed":
            stop = entry - d * STOP_FIXED
        else:  # recross: nur Katastrophen-Stop
            stop = entry - d * STOP_CAP

        tp1 = entry + d * (TP_SCALP if mode == "scalp" else TP1_TREND)
        tp2 = entry + d * TP2_TREND

        pos = {"time": ts, "dir": d, "entry": entry, "stop": stop, "stop0": stop,
               "tp1": tp1, "tp2": tp2, "mode": mode, "strong": strong,
               "mult": STRONG_MULT if strong else 1.0,
               "size_left": 1.0, "scaled": False, "pnl": 0.0}

    return pd.DataFrame(trades)


# ----------------------------- Auswertung -----------------------------------
def stats(tr: pd.DataFrame, label: str) -> None:
    if tr.empty:
        print(f"\n=== {label}: keine Trades ===")
        return
    wins = tr[tr["pnl_pts"] > 0]["pnl_pts"].sum()
    losses = -tr[tr["pnl_pts"] < 0]["pnl_pts"].sum()
    pf = wins / losses if losses > 0 else float("inf")
    wr_ = (tr["pnl_pts"] > 0).mean() * 100
    print(f"\n=== {label} ===")
    print(f"Trades: {len(tr)} | Trefferquote: {wr_:.1f}% | PF: {pf:.2f} | "
          f"Summe: {tr['pnl_pts'].sum():+.1f} Pkt | "
          f"Ø: {tr['pnl_pts'].mean():+.2f} Pkt | Ø Haltedauer: {tr['hold_min'].mean():.1f} Min")
    for grp, g in tr.groupby("mode"):
        gw = -g[g["pnl_pts"] < 0]["pnl_pts"].sum()
        gpf = g[g["pnl_pts"] > 0]["pnl_pts"].sum() / gw if gw > 0 else float("inf")
        print(f"   {grp:>5}: {len(g):3d} Trades | {((g['pnl_pts']>0).mean()*100):.0f}% | "
              f"PF {gpf:.2f} | {g['pnl_pts'].sum():+.1f} Pkt")
    st = tr[tr["strong"]]
    if len(st):
        print(f"  STRONG (M5-Konfluenz): {len(st)} Trades | "
              f"{((st['pnl_pts']>0).mean()*100):.0f}% | {st['pnl_pts'].sum():+.1f} Pkt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="MT5-Export oder generisches OHLC-CSV (M1 empfohlen)")
    ap.add_argument("--session", default="08:00-17:30", help="Handelsfenster HH:MM-HH:MM")
    ap.add_argument("--ext-max", type=float, default=EXT_MAX)
    args = ap.parse_args()

    global EXT_MAX
    EXT_MAX = args.ext_max
    t0, t1 = args.session.split("-")
    session = (pd.to_datetime(t0).time(), pd.to_datetime(t1).time())

    df = load_data(args.csv)
    print(f"Daten: {len(df)} Kerzen | {df.index[0]} bis {df.index[-1]}")
    df = add_indicators(df)
    df = m5_confluence(df)

    all_results = {}
    for mode in ["swing", "fixed", "recross"]:
        tr = run_backtest(df, stop_mode=mode, session=session)
        all_results[mode] = tr
        stats(tr, f"Stop-Variante: {mode.upper()}")
        tr.to_csv(f"trades_{mode}.csv", index=False)

    print("\nTrade-Listen gespeichert: trades_swing.csv, trades_fixed.csv, trades_recross.csv")
    print("Hinweis: PESSIMISTIC=True — TP & Stop in gleicher Kerze wird als Stop gewertet.")


if __name__ == "__main__":
    main()
