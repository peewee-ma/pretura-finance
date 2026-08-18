"""
FDAX Swing Analyst - Orchestrator
====================================
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from anthropic import AsyncAnthropic

from swing.agents_config import AGENTS, SYNTHESIS_SYSTEM_PROMPT, AgentConfig
from swing.schemas import AgentCategory, AgentResult, Direction, RiskReward, SwingAnalysisReport

logger = logging.getLogger("fdax_swing")

AGENT_MODEL = "claude-haiku-4-5"
SYNTHESIS_MODEL = "claude-sonnet-4-6"

client = AsyncAnthropic()

MAX_WEB_SEARCHES_PER_AGENT = 2

AGENT_RESULT_JSON_INSTRUCTIONS = """
Antworte AUSSCHLIESSLICH mit einem JSON-Objekt (kein Markdown, kein Fließtext davor/danach,
keine Zitat-Tags oder Quellenverweise im Fließtext - schreibe alles in eigenen Worten)
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

Nutze maximal 2 Websuchen. Fasse dich bei der Kernaussage kurz (2-4 Sätze), damit
garantiert Platz für die abschließende JSON-Antwort bleibt.
"""


def _build_web_search_tool(agent: AgentConfig) -> Optional[dict]:
    if agent.web_access == "free":
        return {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": MAX_WEB_SEARCHES_PER_AGENT,
        }
    if agent.web_access == "whitelist":
        return {
            "type": "web_search_20250305",
            "name": "web_search",
            "allowed_domains": agent.allowed_domains,
            "max_uses": MAX_WEB_SEARCHES_PER_AGENT,
        }
    return None


def _extract_json_text(response) -> str:
    text_parts = [b.text for b in response.content if b.type == "text"]
    raw_text = "\n".join(text_parts).strip()
    raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw_text = raw_text[start : end + 1]

    return raw_text


async def _call_agent_once(agent: AgentConfig, tools: list, user_msg: str) -> AgentResult:
    response = await client.messages.create(
        model=AGENT_MODEL,
        max_tokens=8000,
        system=agent.system_prompt,
        tools=tools,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw_text = _extract_json_text(response)
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error(
            "Agent %s: JSON-Parse-Fehler. stop_reason=%s, raw_text[:500]=%r",
            agent.category, response.stop_reason, raw_text[:500],
        )
        raise

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


async def _run_llm_agent(agent: AgentConfig, ticker: str, context: str) -> AgentResult:
    tool = _build_web_search_tool(agent)
    tools = [tool] if tool else []

    user_msg = (
        f"Analysiere den Titel: {ticker}\n\n"
        f"Zusätzlicher Kontext (z.B. hochgeladene Nutzerdaten, sofern vorhanden):\n"
        f"{context or '(kein zusätzlicher Kontext übergeben)'}\n\n"
        f"{AGENT_RESULT_JSON_INSTRUCTIONS}"
    )

    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            return await _call_agent_once(agent, tools, user_msg)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Agent %s Versuch %d fehlgeschlagen: %s", agent.category, attempt + 1, exc
            )
            continue

    logger.exception("Agent %s endgültig fehlgeschlagen nach Retry", agent.category)
    return AgentResult(
        agent=agent.category,
        direction=Direction.NEUTRAL,
        confidence=0,
        kernaussage="Agent konnte keine Analyse liefern (auch nach Retry).",
        fehler=str(last_exc),
    )


def fetch_price_history_daily(ticker: str, lookback_days: int = 260):
    import yfinance as yf

    df = yf.Ticker(ticker).history(period=f"{lookback_days}d", interval="1d")
    if df.empty:
        raise ValueError(f"Keine Kursdaten für {ticker} gefunden.")
    return df


async def _run_fibonacci_agent(ticker: str) -> AgentResult:
    last_exc = None
    for attempt in range(4):
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
                    f"Aktueller Kurs {last_close:.2f} liegt am naechsten am "
                    f"{nearest_level[0]}%-Retracement ({nearest_level[1]:.2f}). "
                    f"Swing-Range: {swing_low:.2f} - {swing_high:.2f}."
                ),
                zeitrahmen=None,
                quellen=["berechnet aus Kursdaten"],
                rohdaten={"levels": levels, "swing_high": swing_high, "swing_low": swing_low},
            )
        except Exception as exc:
            last_exc = exc
            wait_s = 3 * (attempt + 1)
            logger.warning("Fibonacci-Agent Versuch %d fehlgeschlagen (%s), warte %ds", attempt + 1, exc, wait_s)
            if attempt < 3:
                await asyncio.sleep(wait_s)
    logger.exception("Fibonacci-Agent endgueltig fehlgeschlagen nach Retries")
    return AgentResult(
        agent=AgentCategory.FIBONACCI,
        direction=Direction.NEUTRAL,
        confidence=0,
        kernaussage="Fibonacci-Level konnten nicht berechnet werden (auch nach Retries).",
        fehler=str(last_exc),
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
        max_tokens=4000,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": synth_instructions}],
    )

    raw_text = _extract_json_text(response)
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
    tasks = []
    for agent in AGENTS:
        if agent.web_access == "computed":
            continue
        tasks.append(_run_llm_agent(agent, ticker, context))

    tasks.append(_run_fibonacci_agent(ticker))

    results: list[AgentResult] = await asyncio.gather(*tasks)

    report = await _run_synthesis(ticker, results)

    failed = [r.agent.value for r in results if r.fehler]
    if failed:
        report.warnungen.append(f"Agenten ohne belastbares Ergebnis: {', '.join(failed)}")

    return report
