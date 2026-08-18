"""
FDAX Swing Analyst - Datenmodelle
==================================
Definiert die Struktur, in der jeder Einzel-Agent antwortet, sowie den
zusammengefassten Gesamtreport, den der Synthese-Agent erzeugt.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Direction(str, Enum):
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


class AgentCategory(str, Enum):
    FUNDAMENTAL = "fundamental"
    CHARTTECHNIK = "charttechnik"
    FIBONACCI = "fibonacci"
    GEX = "gex_level"
    RELATIVE_STAERKE = "relative_staerke_sektor"
    ANALYSTEN_KURSZIEL = "analysten_kursziel"
    SAISONALITAET = "saisonalitaet"
    SWING_SETUP = "swing_setup_minervini"
    SENTIMENT = "sentiment_positionierung"
    VOLUMEN = "volumen"
    VOLATILITAET = "volatilitaet"
    MAKRO_POLITIK = "makro_politik"
    KATALYSATOREN = "katalysatoren_kalender"
    DEVILS_ADVOCATE = "devils_advocate"
    PRODUKT = "produkt_onvista"
    SYNTHESE = "synthese"


class AgentResult(BaseModel):
    """Standardisierte Antwort eines einzelnen Analyse-Agenten."""

    agent: AgentCategory
    direction: Direction
    confidence: int = Field(ge=0, le=100, description="Konfidenz des Agenten in Prozent")
    kernaussage: str = Field(description="2-4 Sätze Kernbegründung")
    kursziel_low: Optional[float] = None
    kursziel_high: Optional[float] = None
    zeitrahmen: Optional[str] = Field(
        default=None, description="z.B. '1-2 Wochen', '3-4 Wochen'"
    )
    quellen: list[str] = Field(default_factory=list, description="Verwendete Quellen/URLs")
    rohdaten: Optional[dict] = Field(
        default=None, description="Optionale strukturierte Zusatzdaten (z.B. Fibo-Level)"
    )
    fehler: Optional[str] = Field(
        default=None, description="Gesetzt, falls der Agent nicht sauber antworten konnte"
    )


class RiskReward(BaseModel):
    entry: Optional[float] = None
    stop: Optional[float] = None
    ziel: Optional[float] = None
    crv: Optional[float] = None


class SwingAnalysisReport(BaseModel):
    """Gesamtreport für einen Ticker - Output der /swing/analyze Route."""

    ticker: str
    unternehmen: Optional[str] = None
    erstellt_am: datetime = Field(default_factory=datetime.utcnow)

    agenten_ergebnisse: list[AgentResult]

    gesamtbild: Direction
    gesamt_konfidenz: int = Field(ge=0, le=100)
    kurszielspanne: Optional[str] = None
    zeitrahmen: Optional[str] = None
    risk_reward: Optional[RiskReward] = None

    top_pro_argumente: list[str] = Field(default_factory=list)
    top_contra_argumente: list[str] = Field(default_factory=list)
    devils_advocate_hinweis: Optional[str] = None

    zusammenfassung: str = Field(description="3-6 Sätze Fließtext-Fazit")

    warnungen: list[str] = Field(
        default_factory=list,
        description="z.B. fehlgeschlagene Agenten, unvollständige Datenlage",
    )
