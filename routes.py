"""
FDAX Swing Analyst - FastAPI-Route
====================================
In euer bestehendes FastAPI-Backend einhängen, z.B. in main.py:

    from fdax_swing.routes import router as swing_router
    app.include_router(swing_router, prefix="/swing", tags=["swing"])

Damit ist der Endpunkt unter POST /swing/analyze erreichbar, getrennt von
den bestehenden DAX-Intraday-Routen.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from orchestrator import analyze_ticker
from schemas import SwingAnalysisReport

router = APIRouter()


class AnalyzeRequest(BaseModel):
    ticker: str
    kontext: str = ""  # Platzhalter für spätere Upload-Integration (Ausbaustufe 2)


@router.post("/analyze", response_model=SwingAnalysisReport)
async def analyze(req: AnalyzeRequest) -> SwingAnalysisReport:
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker darf nicht leer sein.")

    try:
        return await analyze_ticker(ticker, context=req.kontext)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analyse fehlgeschlagen: {exc}") from exc
