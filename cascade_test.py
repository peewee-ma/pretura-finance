#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cascade_test.py — Testet die "Kaskaden"-Hypothese (Monetenfred-Folklore)
================================================================================
Hypothese (aus WhatsApp-Gruppe, 06.08.26):
  "Kommt der Kurs von oben und erreicht Marke X, fällt er weiter bis
   mindestens zur nächsten Marke X-Spacing (oder weiter)."
   Spiegelbildlich für Aufwärtsbewegungen.

Wir kennen Monetenfreds exakte Zielformel NICHT (nur die DP-Formel wurde
reverse-engineered). Getestet wird deshalb die ALLGEMEINE, faire Version:
  - Levels im festen Punkte-Abstand (--spacing, Default 50)
  - "Marke erreicht" = Kurs durchbricht ein Level erstmals in eine Richtung
  - "Kaskade bestätigt" = nächstes Level in derselben Richtung wird erreicht,
    BEVOR der Kurs um --stop-buffer Punkte über das Ausgangslevel zurückläuft
  - Auswertung wie ein handelbares Setup: Entry am Ausgangslevel, Ziel = naechstes
    Level, Stop = Ausgangslevel + Puffer (Invalidierung der Kaskaden-These)

Das ist eine faire Prüfung der FOLKLORE, nicht von Monetenfreds Person —
falls es keinen Effekt gibt, sollte die Trefferquote nahe der "neutralen"
Erwartung liegen (kein Bias, aber auch kein PF-Vorteil).

Daten: gleicher Loader wie macd_wr_strategy.py / orb_strategy.py
       (MT5-Export oder TradingView Unix-Sekunden-Export, CSV).

Aufruf:
  python3 cascade_test.py "/opt/fred/IG_DAX, 1_70ea2.csv"
  python3 cascade_test.py "/opt/fred/IG_DAX, 5_5d0ca.csv" --spacing 30
  python3 cascade_test.py "/opt/fred/IG_DAX, 1_70ea2.csv" --spacing 50 --stop-buffer 10 --max-hold-min 240
"""

import argparse
import numpy as np
import pandas as pd

PESSIMISTIC = True  # Ziel & Stop in gleicher Kerze -> Stop zaehlt zuerst


# ----------------------------- Daten laden -----------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Laedt MT5-Export, TradingView-Export (Unix-Sekunden) oder generisches CSV."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.readline()
    sep = "\t" if "\t" in head else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]

    if "date" in df.columns and "time" in df.columns:
        dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    else:
        dtcol = next(c for c in df.columns if "date" in c or "time" in c)
        col = df[dtcol]
        if np.issubdtype(col.dtype, np.number):
            dt = pd.to_datetime(col, unit="s", utc=True)
            dt = dt.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
        else:
            dt = pd.to_datetime(col)

    out = pd.DataFrame({
        "open":  pd.to_numeric(df["open"], errors="coerce").to_numpy(),
        "high":  pd.to_numeric(df["high"], errors="coerce").to_numpy(),
        "low":   pd.to_numeric(df["low"], errors="coerce").to_numpy(),
        "close": pd.to_numeric(df["close"], errors="coerce").to_numpy(),
    }, index=pd.DatetimeIndex(dt)).dropna().sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


# ----------------------------- Backtest --------------------------------------
def run_backtest(df: pd.DataFrame, spacing: float, stop_buffer: float,
                  max_hold_min: int, session=None) -> pd.DataFrame:
    trades = []
    idx = df.index
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    def in_session(ts):
        if session is None:
            return True
        t = ts.time()
        return session[0] <= t <= session[1]

    last_level_hit = {}  # verhindert doppelte Signale am selben Level in Folge

    for i in range(1, len(df)):
        ts = idx[i]
        if not in_session(ts):
            continue

        lvl_prev_down = np.floor(c[i - 1] / spacing) * spacing
        lvl_now_down = np.floor(c[i] / spacing) * spacing
        lvl_prev_up = np.ceil(c[i - 1] / spacing) * spacing
        lvl_now_up = np.ceil(c[i] / spacing) * spacing

        # Abwärts-Durchbruch: Kurs faellt durch ein Level
        if lvl_now_down < lvl_prev_down:
            level = lvl_prev_down
            key = ("down", level)
            if last_level_hit.get(key) == ts:
                continue
            last_level_hit[key] = ts

            target = level - spacing
            stop = level + stop_buffer
            entry = level
            deadline = ts + pd.Timedelta(minutes=max_hold_min)

            future = df[(df.index > ts) & (df.index <= deadline)]
            exit_price, exit_time, reason = None, None, None
            for fts, row in future.iterrows():
                hit_stop = row["high"] >= stop
                hit_tgt = row["low"] <= target
                if hit_stop and hit_tgt:
                    exit_price, exit_time, reason = (stop, fts, "stop") if PESSIMISTIC else (target, fts, "target")
                    break
                elif hit_stop:
                    exit_price, exit_time, reason = stop, fts, "stop"
                    break
                elif hit_tgt:
                    exit_price, exit_time, reason = target, fts, "target"
                    break
            if exit_price is None:
                if future.empty:
                    continue
                exit_price, exit_time, reason = future["close"].iloc[-1], future.index[-1], "timeout"

            pnl = (entry - exit_price)  # Short-Richtung: Gewinn wenn Kurs faellt
            trades.append({"zeit": ts, "richtung": "ABWÄRTS-KASKADE", "level": level,
                            "ziel": target, "stop": stop, "exit": exit_price,
                            "exit_zeit": exit_time, "reason": reason, "pnl_pts": pnl})

        # Aufwärts-Durchbruch: Kurs steigt durch ein Level
        if lvl_now_up > lvl_prev_up:
            level = lvl_prev_up
            key = ("up", level)
            if last_level_hit.get(key) == ts:
                continue
            last_level_hit[key] = ts

            target = level + spacing
            stop = level - stop_buffer
            entry = level
            deadline = ts + pd.Timedelta(minutes=max_hold_min)

            future = df[(df.index > ts) & (df.index <= deadline)]
            exit_price, exit_time, reason = None, None, None
            for fts, row in future.iterrows():
                hit_stop = row["low"] <= stop
                hit_tgt = row["high"] >= target
                if hit_stop and hit_tgt:
                    exit_price, exit_time, reason = (stop, fts, "stop") if PESSIMISTIC else (target, fts, "target")
                    break
                elif hit_stop:
                    exit_price, exit_time, reason = stop, fts, "stop"
                    break
                elif hit_tgt:
                    exit_price, exit_time, reason = target, fts, "target"
                    break
            if exit_price is None:
                if future.empty:
                    continue
                exit_price, exit_time, reason = future["close"].iloc[-1], future.index[-1], "timeout"

            pnl = (exit_price - entry)  # Long-Richtung: Gewinn wenn Kurs steigt
            trades.append({"zeit": ts, "richtung": "AUFWÄRTS-KASKADE", "level": level,
                            "ziel": target, "stop": stop, "exit": exit_price,
                            "exit_zeit": exit_time, "reason": reason, "pnl_pts": pnl})

    return pd.DataFrame(trades)


# ----------------------------- Auswertung -------------------------------------
def stats(tr: pd.DataFrame, label: str) -> None:
    if tr.empty:
        print(f"\n=== {label}: keine Ereignisse ===")
        return
    valid = tr[tr["reason"] != "timeout"]
    timeouts = len(tr) - len(valid)
    wins = valid[valid["pnl_pts"] > 0]["pnl_pts"].sum()
    losses = -valid[valid["pnl_pts"] < 0]["pnl_pts"].sum()
    pf = wins / losses if losses > 0 else float("inf")
    wr_ = (valid["pnl_pts"] > 0).mean() * 100 if len(valid) else 0
    print(f"\n=== {label} ===")
    print(f"Ereignisse: {len(tr)} (davon {timeouts} Timeout, ausgeschlossen aus PF) | "
          f"Auswertbar: {len(valid)}")
    print(f"Kaskade bestätigt (Ziel erreicht vor Stop): {wr_:.1f}% | PF: {pf:.2f} | "
          f"Summe: {valid['pnl_pts'].sum():+.1f} Pkt | Ø: {valid['pnl_pts'].mean():+.2f} Pkt")
    for grp, g in valid.groupby("richtung"):
        gw = -g[g["pnl_pts"] < 0]["pnl_pts"].sum()
        gpf = g[g["pnl_pts"] > 0]["pnl_pts"].sum() / gw if gw > 0 else float("inf")
        print(f"   {grp:>18}: {len(g):4d}x | {((g['pnl_pts']>0).mean()*100):.1f}% | "
              f"PF {gpf:.2f} | {g['pnl_pts'].sum():+.1f} Pkt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="TradingView- oder MT5-Export (M1 oder M5)")
    ap.add_argument("--spacing", type=float, default=50, help="Punkte-Abstand der Marken")
    ap.add_argument("--stop-buffer", type=float, default=10,
                     help="Puffer über/unter Ausgangslevel = Invalidierung der Kaskade")
    ap.add_argument("--max-hold-min", type=int, default=240,
                     help="Max. Wartezeit in Minuten bis Ziel/Stop, sonst Timeout")
    ap.add_argument("--session", default=None, help="Handelsfenster HH:MM-HH:MM (optional)")
    args = ap.parse_args()

    session = None
    if args.session:
        t0, t1 = args.session.split("-")
        session = (pd.to_datetime(t0).time(), pd.to_datetime(t1).time())

    df = load_data(args.csv)
    print(f"Daten: {len(df)} Kerzen | {df.index[0]} bis {df.index[-1]}")
    print(f"Marken-Abstand: {args.spacing} Pkt | Stop-Puffer: {args.stop_buffer} Pkt | "
          f"Max. Haltezeit: {args.max_hold_min} Min")

    tr = run_backtest(df, args.spacing, args.stop_buffer, args.max_hold_min, session)
    stats(tr, f"Kaskaden-Test ({args.spacing}-Punkte-Marken)")
    tr.to_csv("trades_cascade.csv", index=False)
    print("\nTrade-Liste gespeichert: trades_cascade.csv")
    print("Hinweis: PESSIMISTIC=True — Ziel & Stop gleiche Kerze wird als Stop gewertet.")
    print("Hinweis: Dies testet die ALLGEMEINE Kaskaden-Folklore, nicht Monetenfreds")
    print("         exakte, unbekannte Zielformel.")


if __name__ == "__main__":
    main()
