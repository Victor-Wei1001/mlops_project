import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import psutil
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ---------------------------
# Logging (M28)
# ---------------------------
LOG_DIR = os.getenv("API_LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "api_metrics.jsonl")

logger = logging.getLogger("api_metrics")
logger.setLevel(logging.INFO)
_file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_file_handler)


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="EN→ZH Translator API", version="0.2-m28")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Request / Response schemas
# ---------------------------
class PredictRequest(BaseModel):
    text: str
    max_new_tokens: int = 64


class PredictResponse(BaseModel):
    translation: str
    used_finetuned_weights: bool
    model_name: str


# ---------------------------
# Global model objects
# ---------------------------
MODEL_NAME = os.getenv("HF_MODEL_NAME", "t5-small")
FINETUNED_PATH = os.getenv("FINETUNED_WEIGHTS", "models/final_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer: Optional[AutoTokenizer] = None
hf_model: Optional[AutoModelForSeq2SeqLM] = None
used_finetuned = False


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _proc_metrics() -> dict:
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "rss_mb": round(mem_info.rss / (1024 * 1024), 2),
        "threads": proc.num_threads(),
    }


def _log_json(event: dict) -> None:
    logger.info(json.dumps(event, ensure_ascii=False))


def _load_model() -> None:
    """Load tokenizer + base model,
    then (optionally) load finetuned weights."""
    global tokenizer, hf_model, used_finetuned

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    if os.path.exists(FINETUNED_PATH):
        try:
            state = torch.load(FINETUNED_PATH, map_location="cpu")

            has_prefix = isinstance(state, dict) and any(
                k.startswith("t5.") for k in state.keys()
            )
            if has_prefix:
                state = {k.replace("t5.", "", 1): v for k, v in state.items()}

            missing, unexpected = hf_model.load_state_dict(state, strict=False)
            used_finetuned = True

            _log_json(
                {
                    "timestamp": _now_utc(),
                    "event": "startup_model_load",
                    "model_name": MODEL_NAME,
                    "device": DEVICE,
                    "finetuned_path": FINETUNED_PATH,
                    "used_finetuned": True,
                    "missing_keys": len(missing),
                    "unexpected_keys": len(unexpected),
                }
            )
        except Exception as exc:
            used_finetuned = False
            _log_json(
                {
                    "timestamp": _now_utc(),
                    "event": "startup_model_load_failed",
                    "model_name": MODEL_NAME,
                    "device": DEVICE,
                    "finetuned_path": FINETUNED_PATH,
                    "used_finetuned": False,
                    "error": str(exc),
                }
            )
    else:
        used_finetuned = False
        _log_json(
            {
                "timestamp": _now_utc(),
                "event": "startup_model_load",
                "model_name": MODEL_NAME,
                "device": DEVICE,
                "finetuned_path": FINETUNED_PATH,
                "used_finetuned": False,
                "note": "finetuned weights not found; using base model",
            }
        )

    hf_model.to(DEVICE)
    hf_model.eval()


@app.on_event("startup")
def startup_event() -> None:
    _load_model()


# ---------------------------
# In-memory metrics (M28)
# ---------------------------
METRICS = {
    "requests_total": 0,
    "requests_in_flight": 0,
    "predict_requests_total": 0,
    "predict_success_total": 0,
    "predict_empty_output_total": 0,
    "predict_error_total": 0,
    "latency_ms_sum": 0.0,
    "latency_ms_count": 0,
}


@app.middleware("http")
async def metrics_and_logging_middleware(request: Request, call_next):
    """Collect system metrics + request latency and write JSONL logs."""
    req_id = str(uuid.uuid4())
    start = time.perf_counter()

    METRICS["requests_total"] += 1
    METRICS["requests_in_flight"] += 1

    status_code = 500
    err: Optional[str] = None

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        err = str(exc)
        raise
    finally:
        METRICS["requests_in_flight"] -= 1

        latency_ms = (time.perf_counter() - start) * 1000.0
        METRICS["latency_ms_sum"] += latency_ms
        METRICS["latency_ms_count"] += 1

        log_event = {
            "timestamp": _now_utc(),
            "request_id": req_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
            "used_finetuned": used_finetuned,
            "model_name": MODEL_NAME,
            "device": DEVICE,
        }
        log_event.update(_proc_metrics())
        if err is not None:
            log_event["error"] = err

        _log_json(log_event)


@app.get("/health")
def health():
    if METRICS["latency_ms_count"] > 0:
        avg_latency = METRICS["latency_ms_sum"] / METRICS["latency_ms_count"]
    else:
        avg_latency = 0.0

    return {
        "status": "ok",
        "device": DEVICE,
        "model": MODEL_NAME,
        "finetuned": used_finetuned,
        "metrics": {
            "requests_total": METRICS["requests_total"],
            "requests_in_flight": METRICS["requests_in_flight"],
            "avg_latency_ms": round(avg_latency, 2),
        },
        "process": _proc_metrics(),
    }


@app.get("/metrics")
def metrics():
    if METRICS["latency_ms_count"] > 0:
        avg_latency = METRICS["latency_ms_sum"] / METRICS["latency_ms_count"]
    else:
        avg_latency = 0.0

    return {
        "app": {
            "model_name": MODEL_NAME,
            "device": DEVICE,
            "used_finetuned": used_finetuned,
        },
        "requests": {
            "total": METRICS["requests_total"],
            "in_flight": METRICS["requests_in_flight"],
            "avg_latency_ms": round(avg_latency, 2),
        },
        "predict": {
            "total": METRICS["predict_requests_total"],
            "success": METRICS["predict_success_total"],
            "empty_output": METRICS["predict_empty_output_total"],
            "errors": METRICS["predict_error_total"],
        },
        "process": _proc_metrics(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if tokenizer is None or hf_model is None:
        _load_model()

    METRICS["predict_requests_total"] += 1

    prompt = "translate: " + req.text
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    try:
        with torch.no_grad():
            output_ids = hf_model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                min_new_tokens=1,
                num_beams=1,
            )

        translation = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()

        if translation == "":
            METRICS["predict_empty_output_total"] += 1
            translation = "[Empty output after decoding]"

        METRICS["predict_success_total"] += 1
        return PredictResponse(
            translation=translation,
            used_finetuned_weights=used_finetuned,
            model_name=MODEL_NAME,
        )
    except Exception:
        METRICS["predict_error_total"] += 1
        raise
