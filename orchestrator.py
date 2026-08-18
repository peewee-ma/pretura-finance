"""
FDAX Swing Analyst - Orchestrator
====================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
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


PRICE_RANGE_CACHE_FILE = "/app/data/price_range_cache.json"
PRICE_RANGE_CACHE_TTL_SECONDS = 4 * 60 * 60  # 4 Stunden

FIBONACCI_RANGE_PROMPT = """Du bist ein Finanzdaten-Recherche-Assistent. Finde für den Titel
{ticker} folgende Werte der letzten 52 Wochen:
- Das höchste Tageshoch (52-Wochen-Hoch)
- Das niedrigste Tagestief (52-Wochen-Tief)
- Den aktuellen/letzten Schlusskurs

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt, kein Fließtext davor/danach:
{{"swing_high": <Zahl>, "swing_low": <Zahl>, "last_close": <Zahl>, "quelle": "<kurze Quellenangabe>"}}

Nutze maximal 2 Websuchen."""


def _load_price_range_cache() -> dict:
    try:
        with open(PRICE_RANGE_CACHE_FILE) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _save_price_range_cache(cache: dict) -> None:
    try:
        os.makedirs("/app/data", exist_ok=True)
        with open(PRICE_RANGE_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:  # noqa: BLE001
        pass


async def fetch_swing_range_via_llm(
    ticker: str,
    manual_swing_high: Optional[float] = None,
    manual_swing_low: Optional[float] = None,
) -> dict:
    """
    Ermittelt 52-Wochen-Hoch/-Tief und aktuellen Kurs per Web-Suche (statt
    einer externen Kursdaten-Bibliothek - konsistent mit allen anderen
    Agenten, ohne Yahoo-Finance-Rate-Limit-Problem).

    Falls manual_swing_high/manual_swing_low übergeben werden (z.B. weil
    der Nutzer die Werte aus seinem eigenen Chart präziser kennt als eine
    Websuche), werden NUR diese für die Fibonacci-Berechnung verwendet -
    der aktuelle Kurs wird trotzdem per Websuche ermittelt (und gecacht),
    da er sich laufend ändert und die manuellen Werte i.d.R. nur
    Hoch/Tief betreffen.
    """
    cache = _load_price_range_cache()
    entry = cache.get(ticker)
    if entry and (time.time() - entry.get("ts", 0)) < PRICE_RANGE_CACHE_TTL_SECONDS:
        result = dict(entry["data"])
    else:
        response = await client.messages.create(
            model=AGENT_MODEL,
            max_tokens=2000,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}],
            messages=[{"role": "user", "content": FIBONACCI_RANGE_PROMPT.format(ticker=ticker)}],
        )

        raw_text = _extract_json_text(response)
        data = json.loads(raw_text)

        result = {
            "swing_high": float(data["swing_high"]),
            "swing_low": float(data["swing_low"]),
            "last_close": float(data["last_close"]),
            "quelle": data.get("quelle", "Websuche"),
        }

        if result["swing_high"] <= result["swing_low"]:
            raise ValueError(
                f"Unplausible Werte: high={result['swing_high']} <= low={result['swing_low']}"
            )

        cache[ticker] = {"data": result, "ts": time.time()}
        _save_price_range_cache(cache)

    if manual_swing_high is not None and manual_swing_low is not None:
        if manual_swing_high <= manual_swing_low:
            raise ValueError(
                f"Unplausible manuelle Werte: high={manual_swing_high} <= low={manual_swing_low}"
            )
        result = {
            **result,
            "swing_high": manual_swing_high,
            "swing_low": manual_swing_low,
            "quelle": "manuell vom Nutzer angegeben",
        }

    return result


async def _run_fibonacci_agent(
    ticker: str,
    manual_swing_high: Optional[float] = None,
    manual_swing_low: Optional[float] = None,
) -> AgentResult:
    last_exc = None
    for attempt in range(2):
        try:
            range_data = await fetch_swing_range_via_llm(
                ticker, manual_swing_high, manual_swing_low
            )
            swing_high = range_data["swing_high"]
            swing_low = range_data["swing_low"]
            last_close = range_data["last_close"]
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
                confidence=65 if manual_swing_high is None else 80,
                kernaussage=(
                    f"Aktueller Kurs {last_close:.2f} liegt am naechsten am "
                    f"{nearest_level[0]}%-Retracement ({nearest_level[1]:.2f}). "
                    f"52-Wochen-Range: {swing_low:.2f} - {swing_high:.2f}."
                ),
                zeitrahmen=None,
                quellen=[range_data.get("quelle", "Websuche")],
                rohdaten={"levels": levels, "swing_high": swing_high, "swing_low": swing_low},
            )
        except Exception as exc:
            last_exc = exc
            logger.warning("Fibonacci-Agent Versuch %d fehlgeschlagen: %s", attempt + 1, exc)

    logger.exception("Fibonacci-Agent endgueltig fehlgeschlagen nach Retry")
    return AgentResult(
        agent=AgentCategory.FIBONACCI,
        direction=Direction.NEUTRAL,
        confidence=0,
        kernaussage="Fibonacci-Level konnten nicht berechnet werden (auch nach Retry).",
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


async def analyze_ticker(
    ticker: str,
    context: str = "",
    manual_swing_high: Optional[float] = None,
    manual_swing_low: Optional[float] = None,
) -> SwingAnalysisReport:
    tasks = []
    for agent in AGENTS:
        if agent.web_access == "computed":
            continue
        tasks.append(_run_llm_agent(agent, ticker, context))

    tasks.append(_run_fibonacci_agent(ticker, manual_swing_high, manual_swing_low))

    results: list[AgentResult] = await asyncio.gather(*tasks)

    report = await _run_synthesis(ticker, results)

    failed = [r.agent.value for r in results if r.fehler]
    if failed:
        report.warnungen.append(f"Agenten ohne belastbares Ergebnis: {', '.join(failed)}")

    return report
