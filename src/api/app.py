import os
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="EN→ZH Translator API", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class PredictRequest(BaseModel):
    text: str
    max_new_tokens: int = 64


class PredictResponse(BaseModel):
    translation: str
    used_finetuned_weights: bool
    model_name: str


# ---- Global (load once at startup) ----
MODEL_NAME = os.getenv("HF_MODEL_NAME", "t5-small")
FINETUNED_PATH = os.getenv("FINETUNED_WEIGHTS", "models/final_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer: Optional[AutoTokenizer] = None
hf_model: Optional[AutoModelForSeq2SeqLM] = None
used_finetuned = False


def _load_model():
    global tokenizer, hf_model, used_finetuned

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Try to load finetuned weights saved by your training script.
    # Your training saves state_dict. It should match hf_model if trained compatibly.
    if os.path.exists(FINETUNED_PATH):
        try:
            state = torch.load(FINETUNED_PATH, map_location="cpu")

            # ✅ Fix: your checkpoint keys are prefixed with "t5."
            # HF T5 expects keys without that prefix.
            if isinstance(state, dict) and any(k.startswith("t5.") for k in state.keys()):
                state = {k.replace("t5.", "", 1): v for k, v in state.items()}

            missing, unexpected = hf_model.load_state_dict(state, strict=False)
            used_finetuned = True
            print(f"Loaded finetuned weights from {FINETUNED_PATH}")
            print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

            
        except Exception as e:
            used_finetuned = False
            print(f"Warning: failed to load finetuned weights: {e}")
    else:
        used_finetuned = False
        print(f"Finetuned weights not found at {FINETUNED_PATH}, using base model.")

    hf_model.to(DEVICE)
    hf_model.eval()


@app.on_event("startup")
def startup_event():
    _load_model()


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_NAME, "finetuned": used_finetuned}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # T5 translation prompt style helps:
    # You can keep it simple:
    prompt = "translate: " + req.text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = hf_model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            min_new_tokens=1,
            num_beams=1,
        )

    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if translation == "":
        translation = "[Empty output after decoding]"

    return PredictResponse(
        translation=translation,
        used_finetuned_weights=used_finetuned,
        model_name=MODEL_NAME,
    )
