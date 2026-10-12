import asyncio
import os
import sys
from typing import Any

import torch
from fastapi import FastAPI, Header, HTTPException, Request


MODEL_PATH = os.getenv("CLEF_MODEL_PATH", "/models/clef-flash")
DEVICE = os.getenv("CLEF_DEVICE", "cuda")
API_KEY = os.getenv("CLEF_API_KEY")

sys.path.insert(0, MODEL_PATH)
from joint_schema_model import load_release_model, systemone


model, processor = load_release_model(MODEL_PATH, device=DEVICE)
inference_lock = asyncio.Lock()

app = FastAPI(title="Clef-Flash API", version="0.1")


def check_authorization(authorization: str | None) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": "clef-flash",
        "device": str(next(model.parameters()).device),
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/v1/systemone")
async def system_one(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    check_authorization(authorization)
    body = await request.json()
    try:
        async with inference_lock:
            return await asyncio.to_thread(systemone, model, processor, body)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
