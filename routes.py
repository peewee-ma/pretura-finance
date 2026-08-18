"""
FDAX Swing Analyst - FastAPI-Route
====================================
In euer bestehendes FastAPI-Backend einhängen (siehe README).

Endpunkte:
    POST   /swing/upload/{ticker}   - Datei hochladen (Bild/PDF/CSV), wird
                                        geparst und für diesen Ticker gespeichert
    GET    /swing/upload/{ticker}   - Zeigt, was für diesen Ticker hochgeladen ist
    DELETE /swing/upload/{ticker}   - Löscht die hochgeladenen Dateien für diesen Ticker
    POST   /swing/analyze           - Startet die Analyse; zieht automatisch den
                                        gespeicherten Upload-Kontext für den Ticker

Wiederverwendet bewusst eure bestehende Datei-Verarbeitung aus
agents/file_processor.py (analyze_chart_image, process_pdf, process_csv,
format_file_context) statt das Rad neu zu erfinden.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from swing.orchestrator import analyze_ticker
from swing.schemas import SwingAnalysisReport

from agents.file_processor import (
    analyze_chart_image,
    format_file_context,
    process_csv,
    process_pdf,
)

router = APIRouter()

# In-Memory-Speicher: Ticker -> Liste hochgeladener/geparster Dateien.
# Bewusst einfach gehalten (kein Persistenz-Layer) - reicht für den
# On-Demand-Analyse-Anwendungsfall. Geht bei Backend-Neustart verloren.
_uploaded_context: dict[str, list[dict]] = {}


class AnalyzeRequest(BaseModel):
    ticker: str
    kontext: str = ""  # Optional: zusätzlicher manueller Kontext-Text


def _get_stored_context_text(ticker: str) -> str:
    files = _uploaded_context.get(ticker, [])
    if not files:
        return ""
    return format_file_context(files)


@router.post("/upload/{ticker}")
async def upload_file_for_ticker(ticker: str, file: UploadFile = File(...)):
    ticker = ticker.strip().upper()
    content = await file.read()
    filename = file.filename.lower()

    if filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
        mime = "image/png" if filename.endswith(".png") else "image/jpeg"
        result = await analyze_chart_image(content, mime)
        file_type = "chart"
    elif filename.endswith(".pdf"):
        result = await process_pdf(content)
        file_type = "pdf"
    elif filename.endswith(".csv"):
        result = await process_csv(content)
        file_type = "csv"
    else:
        raise HTTPException(
            status_code=400,
            detail="Nicht unterstütztes Format. Erlaubt: png, jpg, jpeg, webp, pdf, csv.",
        )

    entry = {"file_type": file_type, "filename": file.filename, "data": result}
    _uploaded_context.setdefault(ticker, []).append(entry)

    return {
        "success": True,
        "ticker": ticker,
        "file_type": file_type,
        "filename": file.filename,
        "analysis": result,
        "total_files_for_ticker": len(_uploaded_context[ticker]),
    }


@router.get("/upload/{ticker}")
def get_uploaded_files_for_ticker(ticker: str):
    ticker = ticker.strip().upper()
    files = _uploaded_context.get(ticker, [])
    return {
        "ticker": ticker,
        "files": files,
        "formatted": format_file_context(files) if files else "",
    }


@router.delete("/upload/{ticker}")
def clear_uploaded_files_for_ticker(ticker: str):
    ticker = ticker.strip().upper()
    existed = ticker in _uploaded_context
    _uploaded_context.pop(ticker, None)
    return {"success": True, "cleared": existed, "ticker": ticker}


@router.post("/analyze", response_model=SwingAnalysisReport)
async def analyze(req: AnalyzeRequest) -> SwingAnalysisReport:
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker darf nicht leer sein.")

    stored_context = _get_stored_context_text(ticker)
    combined_context_parts = [p for p in [stored_context, req.kontext] if p]
    combined_context = "\n\n".join(combined_context_parts)

    try:
        return await analyze_ticker(ticker, context=combined_context)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analyse fehlgeschlagen: {exc}") from exc
