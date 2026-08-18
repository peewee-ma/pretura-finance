"""
FDAX Swing Analyst - Orchestrator
====================================
Ruft alle konfigurierten Agenten parallel auf (asyncio.gather), sammelt die
Ergebnisse ein und lässt den Synthese-Agenten daraus den Gesamtreport bauen.

Modell-Wahl:
    - Fachagenten laufen auf einem günstigeren/schnelleren Modell (Standard:
      claude-haiku-4-5, analog zu eurem bestehenden Swarm-Setup).
    - Der Synthese-Agent läuft auf einem stärkeren Modell (Standard: Sonnet),
      da er die Gesamtabwägung trifft.

Wire-up TODOs (markiert im Code mit # TODO):
    - fetch_price_history_daily(): an eure bestehende Kursdatenquelle
      anbinden (aktuell yfinance als Platzhalter, analog zum
      markov-hedge-fund-method Skill).
    - GEX-Whitelist-Domains in agents_config.py verifizieren, sobald
      gex_analyzer.py-Datenquelle final geklärt ist.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from anthropic import AsyncAnthropic

from agents_config import AGENTS, SYNTHESIS_SYSTEM_PROMPT, AgentConfig
from schemas import AgentCategory, AgentResult, Direction, RiskReward, SwingAnalysisReport

logger = logging.getLogger("fdax_swing")

AGENT_MODEL = "claude-haiku-4-5"
SYNTHESIS_MODEL = "claude-sonnet-4-6"

client = AsyncAnthropic()  # erwartet ANTHROPIC_API_KEY in der Umgebung

AGENT_RESULT_JSON_INSTRUCTIONS = """
Antworte AUSSCHLIESSLICH mit einem JSON-Objekt (kein Markdown, kein Fließtext davor/danach)
in exakt folgendem Format:

{
  "direction": "bullish" | "neutral" | "bearish",
  "confidence": <int 0-100>,
  "kernaussage": "<2-4 Sätze>",
  "kursziel_low": <float oder null>,
  "kursziel_high": <float oder null>,
  "zeitrahmen": "<string oder null>",
  "quellen": ["<url oder bezeichnung>", ...],
  "fehler": "<string oder null, falls du das Thema nicht belastbar einschätzen konntest>"
}
"""


def _build_web_search_tool(agent: AgentConfig) -> Optional[dict]:
    if agent.web_access == "free":
        return {"type": "web_search_20250305", "name": "web_search"}
    if agent.web_access == "whitelist":
        return {
            "type": "web_search_20250305",
            "name": "web_search",
            "allowed_domains": agent.allowed_domains,
        }
    return None


async def _run_llm_agent(agent: AgentConfig, ticker: str, context: str) -> AgentResult:
    tool = _build_web_search_tool(agent)
    tools = [tool] if tool else []

    user_msg = (
        f"Analysiere den Titel: {ticker}\n\n"
        f"Zusätzlicher Kontext (z.B. hochgeladene Nutzerdaten, sofern vorhanden):\n"
        f"{context or '(kein zusätzlicher Kontext übergeben)'}\n\n"
        f"{AGENT_RESULT_JSON_INSTRUCTIONS}"
    )

    try:
        response = await client.messages.create(
            model=AGENT_MODEL,
            max_tokens=1000,
            system=agent.system_prompt,
            tools=tools,
            messages=[{"role": "user", "content": user_msg}],
        )

        text_parts = [b.text for b in response.content if b.type == "text"]
        raw_text = "\n".join(text_parts).strip()
        raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        data = json.loads(raw_text)
        return AgentResult(
            agent=agent.category,
            direction=Direction(data.get("direction", "neutral")),
            confidence=int(data.get("confidence", 0)),
            kernaussage=data.get("kernaussage", ""),
            kursziel_low=data.get("kursziel_low"),
            kursziel_high=data.get("kursziel_high"),
            zeitrahmen=data.get("zeitrahmen"),
            quellen=data.get("quellen", []) or [],
            fehler=data.get("fehler"),
        )
    except Exception as exc:  # noqa: BLE001 - bewusst breit, ein Agent darf den Report nicht killen
        logger.exception("Agent %s fehlgeschlagen", agent.category)
        return AgentResult(
            agent=agent.category,
            direction=Direction.NEUTRAL,
            confidence=0,
            kernaussage="Agent konnte keine Analyse liefern.",
            fehler=str(exc),
        )


def fetch_price_history_daily(ticker: str, lookback_days: int = 260):
    """
    TODO: An eure bestehende Kursdatenquelle anbinden.
    Platzhalter-Implementierung via yfinance (Daily-Daten), analog zum
    bereits genutzten markov-hedge-fund-method Skill.
    """
    import yfinance as yf

    df = yf.Ticker(ticker).history(period=f"{lookback_days}d", interval="1d")
    if df.empty:
        raise ValueError(f"Keine Kursdaten für {ticker} gefunden.")
    return df


async def _run_fibonacci_agent(ticker: str) -> AgentResult:
    try:
        df = await asyncio.to_thread(fetch_price_history_daily, ticker)
        swing_high = float(df["High"].max())
        swing_low = float(df["Low"].min())
        last_close = float(df["Close"].iloc[-1])

        diff = swing_high - swing_low
        levels = {
            "0.0": swing_high,
            "23.6": swing_high - 0.236 * diff,
            "38.2": swing_high - 0.382 * diff,
            "50.0": swing_high - 0.5 * diff,
            "61.8": swing_high - 0.618 * diff,
            "78.6": swing_high - 0.786 * diff,
            "100.0": swing_low,
        }

        nearest_level = min(levels.items(), key=lambda kv: abs(kv[1] - last_close))

        return AgentResult(
            agent=AgentCategory.FIBONACCI,
            direction=Direction.NEUTRAL,
            confidence=70,
            kernaussage=(
                f"Aktueller Kurs {last_close:.2f} liegt am nächsten am "
                f"{nearest_level[0]}%-Retracement ({nearest_level[1]:.2f}). "
                f"Swing-Range: {swing_low:.2f} - {swing_high:.2f}."
            ),
            zeitrahmen=None,
            quellen=["berechnet aus Kursdaten"],
            rohdaten={"levels": levels, "swing_high": swing_high, "swing_low": swing_low},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fibonacci-Agent fehlgeschlagen")
        return AgentResult(
            agent=AgentCategory.FIBONACCI,
            direction=Direction.NEUTRAL,
            confidence=0,
            kernaussage="Fibonacci-Level konnten nicht berechnet werden.",
            fehler=str(exc),
        )


async def _run_synthesis(ticker: str, results: list[AgentResult]) -> SwingAnalysisReport:
    payload = json.dumps([r.model_dump(mode="json") for r in results], ensure_ascii=False, indent=2)

    synth_instructions = f"""
Ticker: {ticker}

Agenten-Ergebnisse (JSON):
{payload}

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt in folgendem Format:
{{
  "gesamtbild": "bullish" | "neutral" | "bearish",
  "gesamt_konfidenz": <int 0-100>,
  "kurszielspanne": "<string>",
  "zeitrahmen": "<string>",
  "risk_reward": {{"entry": <float|null>, "stop": <float|null>, "ziel": <float|null>, "crv": <float|null>}},
  "top_pro_argumente": ["...", "..."],
  "top_contra_argumente": ["...", "..."],
  "devils_advocate_hinweis": "<string>",
  "zusammenfassung": "<3-6 Sätze>",
  "warnungen": ["...", "..."]
}}
"""

    response = await client.messages.create(
        model=SYNTHESIS_MODEL,
        max_tokens=1500,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": synth_instructions}],
    )

    text_parts = [b.text for b in response.content if b.type == "text"]
    raw_text = "\n".join(text_parts).strip()
    raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(raw_text)

    return SwingAnalysisReport(
        ticker=ticker,
        agenten_ergebnisse=results,
        gesamtbild=Direction(data["gesamtbild"]),
        gesamt_konfidenz=int(data["gesamt_konfidenz"]),
        kurszielspanne=data.get("kurszielspanne"),
        zeitrahmen=data.get("zeitrahmen"),
        risk_reward=RiskReward(**data["risk_reward"]) if data.get("risk_reward") else None,
        top_pro_argumente=data.get("top_pro_argumente", []),
        top_contra_argumente=data.get("top_contra_argumente", []),
        devils_advocate_hinweis=data.get("devils_advocate_hinweis"),
        zusammenfassung=data.get("zusammenfassung", ""),
        warnungen=data.get("warnungen", []),
    )


async def analyze_ticker(ticker: str, context: str = "") -> SwingAnalysisReport:
    """Haupteinstiegspunkt: führt alle Agenten aus und liefert den Gesamtreport."""

    tasks = []
    for agent in AGENTS:
        if agent.web_access == "computed":
            continue  # separat behandelt (aktuell nur Fibonacci)
        tasks.append(_run_llm_agent(agent, ticker, context))

    tasks.append(_run_fibonacci_agent(ticker))

    results: list[AgentResult] = await asyncio.gather(*tasks)

    report = await _run_synthesis(ticker, results)

    failed = [r.agent.value for r in results if r.fehler]
    if failed:
        report.warnungen.append(f"Agenten ohne belastbares Ergebnis: {', '.join(failed)}")

    return report
