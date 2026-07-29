#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orb_strategy.py — Opening Range Breakout Backtest fuer DAX (M1)
================================================================================
Methodik nach Zarattini, Barbon & Aziz (2023/2024, SSRN):
  1. Opening Range (OR) = erstes OR_MINUTES-Intervall der Session (Default 5 Min,
     bester Wert laut Studie), gemessen an Xetra-Handelsstart.
  2. Steigt der Kurs im OR (Close > Open), Long-Einstieg zu Beginn des naechsten
     Intervalls, sobald das OR-Hoch ueberschritten wird.
     Faellt der Kurs im OR (Close < Open), spiegelbildlich Short beim
     Unterschreiten des OR-Tiefs.
     Ist Open ~ Close im OR (Differenz < FLAT_EPS Punkte), KEIN Trade
     (per Original-Regel: "no position if open and close are about the same").
  3. Stop = Hoch/Tief des Opening-Range-Intervalls.
  4. Ziel (Stop-Gain) = TARGET_MULT x (Entry - Stop), Default 10x wie im
     Original-Paper (aggressives Chance-Risiko-Verhaeltnis von 10:1).
  5. Wird bis Sessionende weder Stop noch Ziel erreicht: Glattstellung
     zum Handelsschluss (EOD_HOUR:EOD_MINUTE).

  Konservative Auswertung: Treffen Stop und Ziel in derselben Kerze
  zusammen, zaehlt der STOP zuerst (PESSIMISTIC=True).

Daten: TradingView-Export (Unix-Sekunden, Spalten time,open,high,low,close,...)
       oder MT5-Export (<DATE> <TIME> ...). Gleicher Loader wie
       macd_wr_strategy.py.

Aufruf:
  python3 orb_strategy.py "DE40_M1.csv"
  python3 orb_strategy.py "DE40_M1.csv" --or-minutes 5 --target-mult 10 --session-start 09:00
  python3 orb_strategy.py "DE40_M1.csv" --or-minutes 15   # Alternativ-Test
"""

import argparse
import numpy as np
import pandas as pd

# ----------------------------- Parameter ------------------------------------
OR_MINUTES = 5          # Laenge der Opening Range in Minuten
TARGET_MULT = 10.0      # Ziel = TARGET_MULT x Risiko (Entry-Stop-Abstand)
FLAT_EPS = 2.0          # OR Open~Close Toleranz in Punkten -> kein Trade
SESSION_START = "09:00" # Xetra-Handelsstart
EOD_TIME = "17:30"      # Glattstellung Ende Session
PESSIMISTIC = True      # Stop & Ziel gleiche Kerze -> Stop zaehlt zuerst
MAX_RISK_PTS = 80.0     # Sicherheitsdeckel: OR-Range groesser -> kein Trade
                         # (verhindert Trades mit unrealistisch weitem Stop)


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
def run_backtest(df: pd.DataFrame, or_minutes: int, target_mult: float,
                  session_start: str, eod_time: str) -> pd.DataFrame:
    trades = []
    ss_h, ss_m = map(int, session_start.split(":"))
    eod_h, eod_m = map(int, eod_time.split(":"))

    for day, day_df in df.groupby(df.index.date):
        day_df = day_df.sort_index()
        session_open = pd.Timestamp(day).replace(hour=ss_h, minute=ss_m)
        or_end = session_open + pd.Timedelta(minutes=or_minutes)
        eod = pd.Timestamp(day).replace(hour=eod_h, minute=eod_m)

        or_bars = day_df[(day_df.index >= session_open) & (day_df.index < or_end)]
        after = day_df[day_df.index >= or_end]
        if or_bars.empty or after.empty:
            continue

        or_open = or_bars["open"].iloc[0]
        or_close = or_bars["close"].iloc[-1]
        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()
        or_range = or_high - or_low

        if abs(or_close - or_open) < FLAT_EPS:
            continue  # kein klares Richtungssignal in der Opening Range
        if or_range <= 0 or or_range > MAX_RISK_PTS:
            continue  # unrealistisch enge oder zu weite Range

        direction = 1 if or_close > or_open else -1
        trigger = or_high if direction == 1 else or_low
        stop0 = or_low if direction == 1 else or_high

        # Warten auf Durchbruch des OR-Hochs/Tiefs nach der Opening Range
        entry_price, entry_time, idx_start = None, None, None
        after_vals = after[["open", "high", "low", "close"]]
        for ts, row in after_vals.iterrows():
            if direction == 1 and row["high"] >= trigger:
                entry_price = max(trigger, row["open"])
                entry_time = ts
                break
            if direction == -1 and row["low"] <= trigger:
                entry_price = min(trigger, row["open"])
                entry_time = ts
                break
        if entry_price is None:
            continue  # kein Ausbruch an diesem Tag

        risk = abs(entry_price - stop0)
        if risk <= 0:
            continue
        target = entry_price + direction * target_mult * risk
        stop = stop0

        remaining = after_vals[after_vals.index > entry_time]
        remaining = remaining[remaining.index <= eod]
        exit_price, exit_time, reason = None, None, None
        for ts, row in remaining.iterrows():
            hit_stop = (row["low"] <= stop) if direction == 1 else (row["high"] >= stop)
            hit_tp = (row["high"] >= target) if direction == 1 else (row["low"] <= target)
            if hit_stop and hit_tp:
                if PESSIMISTIC:
                    exit_price, exit_time, reason = stop, ts, "stop"
                else:
                    exit_price, exit_time, reason = target, ts, "target"
                break
            elif hit_stop:
                exit_price, exit_time, reason = stop, ts, "stop"
                break
            elif hit_tp:
                exit_price, exit_time, reason = target, ts, "target"
                break
        if exit_price is None:
            # Glattstellung zum Handelsschluss
            eod_rows = after_vals[after_vals.index <= eod]
            if eod_rows.empty:
                continue
            exit_price = eod_rows["close"].iloc[-1]
            exit_time = eod_rows.index[-1]
            reason = "eod"

        pnl = (exit_price - entry_price) * direction
        trades.append({
            "date": day, "dir": "LONG" if direction == 1 else "SHORT",
            "or_range_pts": or_range, "entry_time": entry_time, "entry": entry_price,
            "stop": stop, "target": target, "exit_time": exit_time,
            "exit": exit_price, "reason": reason, "risk_pts": risk,
            "pnl_pts": pnl, "hold_min": (exit_time - entry_time).total_seconds() / 60.0,
        })

    return pd.DataFrame(trades)


# ----------------------------- Auswertung -------------------------------------
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
          f"Summe: {tr['pnl_pts'].sum():+.1f} Pkt | Ø: {tr['pnl_pts'].mean():+.2f} Pkt | "
          f"Ø Haltedauer: {tr['hold_min'].mean():.1f} Min | Ø Risiko: {tr['risk_pts'].mean():.1f} Pkt")
    for grp, g in tr.groupby("dir"):
        gw = -g[g["pnl_pts"] < 0]["pnl_pts"].sum()
        gpf = g[g["pnl_pts"] > 0]["pnl_pts"].sum() / gw if gw > 0 else float("inf")
        print(f"   {grp:>5}: {len(g):3d} Trades | {((g['pnl_pts']>0).mean()*100):.0f}% | "
              f"PF {gpf:.2f} | {g['pnl_pts'].sum():+.1f} Pkt")
    for grp, g in tr.groupby("reason"):
        print(f"   Exit '{grp}': {len(g):3d}x | Summe {g['pnl_pts'].sum():+.1f} Pkt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="TradingView- oder MT5-Export (M1 empfohlen)")
    ap.add_argument("--or-minutes", type=int, default=OR_MINUTES)
    ap.add_argument("--target-mult", type=float, default=TARGET_MULT)
    ap.add_argument("--session-start", default=SESSION_START)
    ap.add_argument("--eod-time", default=EOD_TIME)
    args = ap.parse_args()

    df = load_data(args.csv)
    print(f"Daten: {len(df)} Kerzen | {df.index[0]} bis {df.index[-1]}")

    tr = run_backtest(df, args.or_minutes, args.target_mult,
                       args.session_start, args.eod_time)
    stats(tr, f"ORB {args.or_minutes}-Min, Ziel {args.target_mult}x Risiko, "
              f"Start {args.session_start}")
    tr.to_csv("trades_orb.csv", index=False)
    print("\nTrade-Liste gespeichert: trades_orb.csv")
    print("Hinweis: PESSIMISTIC=True — Stop & Ziel gleiche Kerze wird als Stop gewertet.")


if __name__ == "__main__":
    main()
