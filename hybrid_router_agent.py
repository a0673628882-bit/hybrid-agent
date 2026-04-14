"""
Hybrid Router Agent — "Трикутник" (Creator → Critic → Expert Judge)
FastAPI service that routes requests through a 3-stage LLM pipeline with
automatic model selection based on task category.
"""
import os
import logging
import time
from typing import Literal, Optional
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("hybrid-agent")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PORT = int(os.getenv("PORT", "8000"))

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Category -> Judge Model mapping (verified live on 2026-04-14)
JUDGE_MODELS = {
    "CODE": "anthropic/claude-sonnet-4.6",
    "STRATEGY": "anthropic/claude-opus-4.6",
    "DATA": "google/gemini-2.5-flash",
    "LOGIC": "openai/o1",
}

Category = Literal["CODE", "STRATEGY", "DATA", "LOGIC"]


# ---------------------------------------------------------------------------
# HTTP Client lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=120.0)
    logger.info("Hybrid Router Agent started on port %s", PORT)
    try:
        yield
    finally:
        await app.state.http.aclose()
        logger.info("Hybrid Router Agent shutdown")


app = FastAPI(
    title="Autonomous Hybrid Agent — Triangle Router",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User prompt / task")
    force_category: Optional[Category] = Field(
        None, description="Override automatic classification"
    )


class TriangleResponse(BaseModel):
    category: Category
    judge_model: str
    draft: str
    critique: str
    final_answer: str
    timings_sec: dict


# ---------------------------------------------------------------------------
# LLM Clients
# ---------------------------------------------------------------------------
async def call_gemini(http: httpx.AsyncClient, prompt: str, system: str = "") -> str:
    if not GOOGLE_API_KEY:
        raise HTTPException(500, "GOOGLE_API_KEY not configured")

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4096},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    try:
        r = await http.post(
            GEMINI_URL,
            params={"key": GOOGLE_API_KEY},
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except httpx.HTTPStatusError as e:
        logger.error("Gemini API error: %s — %s", e.response.status_code, e.response.text)
        raise HTTPException(502, f"Gemini API error: {e.response.status_code}")
    except (KeyError, IndexError) as e:
        logger.error("Malformed Gemini response: %s", e)
        raise HTTPException(502, "Malformed Gemini response")


async def call_openrouter(
    http: httpx.AsyncClient, model: str, prompt: str, system: str = ""
) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(500, "OPENROUTER_API_KEY not configured")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": model, "messages": messages, "temperature": 0.5}
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://hybrid-agent.local",
        "X-Title": "Hybrid Router Agent",
    }

    try:
        r = await http.post(OPENROUTER_URL, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as e:
        logger.error("OpenRouter error: %s — %s", e.response.status_code, e.response.text)
        raise HTTPException(502, f"OpenRouter error ({model}): {e.response.status_code}")
    except (KeyError, IndexError) as e:
        logger.error("Malformed OpenRouter response: %s", e)
        raise HTTPException(502, "Malformed OpenRouter response")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
CLASSIFIER_SYSTEM = """You are a strict classifier. Return EXACTLY ONE WORD —
one of: CODE, STRATEGY, DATA, LOGIC.
- CODE: programming, software engineering, debugging, architecture.
- STRATEGY: business planning, marketing, analytics, decision-making.
- DATA: working with documents, notes, summarization, extraction.
- LOGIC: math, complex reasoning, calculations, proofs.
Return ONLY the label, nothing else."""


async def classify(http: httpx.AsyncClient, query: str) -> Category:
    raw = await call_gemini(http, query, system=CLASSIFIER_SYSTEM)
    label = raw.strip().upper().split()[0].strip(".:,-")
    if label not in JUDGE_MODELS:
        logger.warning("Classifier returned %r, falling back to DATA", raw)
        label = "DATA"
    return label  # type: ignore


CREATOR_SYSTEM = """You are the CREATOR. Produce a detailed, thorough first
draft solution to the user's task. Be concrete, structured, and complete."""

CRITIC_SYSTEM = """You are the CRITIC. Rigorously review the draft below.
Find factual errors, logical gaps, missing edge cases, security/business risks,
weak assumptions. Be specific and actionable. Do NOT rewrite the solution —
only critique it as a numbered list of issues and recommendations."""

JUDGE_SYSTEM = """You are the EXPERT JUDGE. You receive:
(1) the original TASK, (2) the CREATOR's draft, (3) the CRITIC's review.
Synthesize the ideal final answer: incorporate valid critiques, discard wrong
ones, and deliver a polished, production-ready response to the task."""


async def run_triangle(http: httpx.AsyncClient, query: str, category: Category):
    timings = {}

    t0 = time.perf_counter()
    draft = await call_gemini(http, query, system=CREATOR_SYSTEM)
    timings["creator"] = round(time.perf_counter() - t0, 2)
    logger.info("Creator done (%ss, %d chars)", timings["creator"], len(draft))

    critic_prompt = f"TASK:\n{query}\n\nDRAFT:\n{draft}"
    t0 = time.perf_counter()
    critique = await call_gemini(http, critic_prompt, system=CRITIC_SYSTEM)
    timings["critic"] = round(time.perf_counter() - t0, 2)
    logger.info("Critic done (%ss, %d chars)", timings["critic"], len(critique))

    judge_model = JUDGE_MODELS[category]
    judge_prompt = (
        f"TASK:\n{query}\n\n"
        f"CREATOR DRAFT:\n{draft}\n\n"
        f"CRITIC REVIEW:\n{critique}\n\n"
        "Now produce the ideal final answer."
    )
    t0 = time.perf_counter()
    final = await call_openrouter(http, judge_model, judge_prompt, system=JUDGE_SYSTEM)
    timings["judge"] = round(time.perf_counter() - t0, 2)
    logger.info("Judge (%s) done (%ss, %d chars)", judge_model, timings["judge"], len(final))

    return draft, critique, final, judge_model, timings


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "google_key": bool(GOOGLE_API_KEY),
        "openrouter_key": bool(OPENROUTER_API_KEY),
        "judge_models": JUDGE_MODELS,
    }


@app.post("/classify")
async def classify_endpoint(req: QueryRequest):
    http = app.state.http
    category = req.force_category or await classify(http, req.query)
    return {"category": category, "judge_model": JUDGE_MODELS[category]}


@app.post("/triangle", response_model=TriangleResponse)
async def triangle(req: QueryRequest):
    http = app.state.http
    logger.info("Incoming /triangle request (%d chars)", len(req.query))

    category = req.force_category or await classify(http, req.query)
    logger.info("Category: %s", category)

    draft, critique, final, judge_model, timings = await run_triangle(
        http, req.query, category
    )

    return TriangleResponse(
        category=category,
        judge_model=judge_model,
        draft=draft,
        critique=critique,
        final_answer=final,
        timings_sec=timings,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("hybrid_router_agent:app", host="0.0.0.0", port=PORT, reload=False)
