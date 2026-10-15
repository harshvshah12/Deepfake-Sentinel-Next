"""Deepfake Sentinel API.

The hosted route prefers a configured Hugging Face inference endpoint. Without a
model token it runs explicitly-labelled, low-confidence image forensics only.
It never represents those heuristics as a validated deepfake classifier.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

app = FastAPI(title="Deepfake Sentinel API", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
MAX_BYTES = 12 * 1024 * 1024
MODEL_ID = os.getenv("HF_MODEL_ID", "dima806/deepfake_vs_real_image_detection")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
_feedback_count = 0


class FeedbackPayload(BaseModel):
    prediction: str | None = None
    actual: str | None = None
    label: str | None = None
    image_hash: str | None = None
    comment: str | None = None


class GeminiPayload(BaseModel):
    api_key: str


def _read_image(raw: bytes) -> Image.Image:
    if not raw or len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Upload must be a non-empty image under 12 MB.")
    try:
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1024, 1024))
        return image
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=415, detail="Unsupported or invalid image. Upload JPEG, PNG, or WebP.")


def _hf_predict(raw: bytes) -> dict[str, Any] | None:
    if not HF_TOKEN:
        return None
    url = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
    request = urllib.request.Request(
        url, data=raw, method="POST",
        headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(request, timeout=24) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, dict) and payload.get("error"):
            return None
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            payload = payload[0]
        if not isinstance(payload, list) or not payload:
            return None
        ranked = sorted(
            [item for item in payload if isinstance(item, dict) and "label" in item and "score" in item],
            key=lambda item: float(item["score"]), reverse=True,
        )
        if not ranked:
            return None
        probs = {}
        for item in ranked:
            label = str(item["label"]).lower()
            score = float(item["score"])
            if any(word in label for word in ("fake", "deepfake", "synthetic", "ai-generated", "artificial")):
                probs["fake"] = max(probs.get("fake", 0.0), score)
            elif any(word in label for word in ("real", "authentic", "human", "original")):
                probs["real"] = max(probs.get("real", 0.0), score)
        if not probs:
            top = ranked[0]
            label = str(top["label"]).lower()
            probs["fake" if any(w in label for w in ("fake", "synthetic", "generated")) else "real"] = float(top["score"])
        fake = probs.get("fake", 0.0)
        real = probs.get("real", 0.0)
        total = fake + real
        if total <= 0:
            return None
        fake, real = fake / total, real / total
        return {
            "prediction": "Fake" if fake >= real else "Real",
            "confidence": max(fake, real),
            "real_probability": real,
            "fake_probability": fake,
            "analysis_mode": "huggingface_model",
            "model_name": MODEL_ID,
            "model_note": "Model output is a screening signal, not proof of authenticity.",
            "model_scores": ranked[:8],
        }
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None


def _forensic_fallback(image: Image.Image) -> dict[str, Any]:
    """Explainable image-statistics screen; deliberately low-confidence and non-diagnostic."""
    gray = np.asarray(image.convert("L").resize((192, 192)), dtype=np.float32) / 255.0
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2
    h, w = spectrum.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (min(h, w) / 2)
    total = float(spectrum.sum()) + 1e-9
    high_freq = float(spectrum[radius > 0.58].sum() / total)
    hist, _ = np.histogram(gray, bins=64, range=(0, 1), density=True)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum() / np.log2(64))
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    edge_energy = float((np.mean(np.abs(gx)) + np.mean(np.abs(gy))) / 2)
    # These features are not validated discriminators; keep the output neutral.
    fake_score = float(np.clip(0.5 + (0.12 - high_freq) * 0.45 + (0.62 - entropy) * 0.08, 0.35, 0.65))
    real_score = 1.0 - fake_score
    return {
        "prediction": "Fake" if fake_score >= 0.5 else "Real",
        "confidence": max(fake_score, real_score),
        "real_probability": real_score,
        "fake_probability": fake_score,
        "analysis_mode": "forensic_heuristics",
        "model_name": "Image-statistics fallback (not a trained classifier)",
        "model_note": "No HF_TOKEN configured or model endpoint unavailable. This low-confidence heuristic is not validated for deepfake detection; treat the verdict as inconclusive.",
        "spectral_anomaly_score": round(abs(high_freq - 0.12), 5),
        "high_freq_ratio": round(high_freq, 5),
        "image_entropy": round(entropy, 5),
        "edge_energy": round(edge_energy, 6),
        "feedback_applied": False,
        "adversarial_attack_applied": False,
        "epsilon": 0,
        "trajectory": [],
    }


def analyze(raw: bytes) -> dict[str, Any]:
    image = _read_image(raw)
    model_result = _hf_predict(raw)
    if model_result:
        result = model_result
        # Supplement model output with reproducible image statistics.
        fallback = _forensic_fallback(image)
        result.update({key: fallback[key] for key in ("spectral_anomaly_score", "high_freq_ratio", "image_entropy", "edge_energy")})
        result.update({"feedback_applied": False, "adversarial_attack_applied": False, "epsilon": 0, "trajectory": []})
        return result
    return _forensic_fallback(image)


@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Deepfake Sentinel API",
        "device": "remote-inference" if HF_TOKEN else "forensic-fallback",
        "model_configured": bool(HF_TOKEN),
        "model_id": MODEL_ID if HF_TOKEN else None,
        "message": "Model-backed inference is enabled." if HF_TOKEN else "API is online; configure HF_TOKEN for trained-model inference. Fallback is heuristic only.",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), attack: bool = Query(False), epsilon: float = Query(0.0), gemini_key: str | None = Query(None)):
    raw = await file.read(MAX_BYTES + 1)
    result = analyze(raw)
    result["filename"] = file.filename or "upload"
    result["content_type"] = file.content_type
    result["adversarial_attack_applied"] = False
    result["epsilon"] = 0
    result["requested_attack_ignored"] = bool(attack)
    return result


@app.post("/predict-frame")
async def predict_frame(payload: dict[str, Any]):
    value = payload.get("image", "")
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=400, detail="JSON field 'image' must contain a base64-encoded image.")
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")
    return analyze(raw)


@app.get("/model-metrics")
def model_metrics():
    # Do not publish invented ROC/AUC figures. No evaluation report is bundled here.
    return {
        "model_configured": bool(HF_TOKEN),
        "model_name": MODEL_ID if HF_TOKEN else "Image-statistics fallback",
        "analysis_mode": "huggingface_model" if HF_TOKEN else "forensic_heuristics",
        "feature_weights": {"Spatial image statistics": 100},
        "evaluation": None,
        "evaluation_note": "No independently measured ROC/AUC benchmark is published by this deployment.",
        "fft_azimuthal_curve": {"frequencies": [], "authentic_natural_decay": [], "synthetic_gan_spikes": []},
        "roc_curve": {"fpr": [], "tpr": []},
    }


@app.get("/samples")
def samples():
    return {"samples": [], "items": [], "message": "No specimen images are bundled with this deployment."}


@app.post("/feedback")
async def feedback(payload: FeedbackPayload):
    global _feedback_count
    _feedback_count += 1
    return {"success": True, "stored": False, "message": "Feedback accepted for this serverless invocation; persistent feedback storage is not configured.", "feedback_count": _feedback_count}


@app.get("/feedback-stats")
def feedback_stats():
    return {"total_feedback": _feedback_count, "persistent": False, "message": "Feedback storage is ephemeral until a database is configured."}


@app.post("/test-gemini-key")
async def test_gemini_key(payload: GeminiPayload):
    key = payload.api_key.strip()
    if not key:
        return JSONResponse(status_code=400, content={"success": False, "error": "API key is required."})
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        data=json.dumps({"contents": [{"parts": [{"text": "Reply with the single word READY."}]}]}).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            data = json.loads(response.read().decode())
        text = data["candidates"][0]["content"]["parts"][0].get("text", "READY")
        return {"success": True, "sample_verdict": text[:100]}
    except urllib.error.HTTPError as exc:
        return JSONResponse(status_code=400, content={"success": False, "error": f"Gemini rejected the key (HTTP {exc.code})."})
    except Exception:
        return JSONResponse(status_code=502, content={"success": False, "error": "Gemini could not be reached. Check the key and try again."})


# Support both ASGI path conventions used by Vercel Python Functions.
for _path, _endpoint, _methods in [
    ("health", health, ["GET"]),
    ("predict", predict, ["POST"]),
    ("predict-frame", predict_frame, ["POST"]),
    ("model-metrics", model_metrics, ["GET"]),
    ("samples", samples, ["GET"]),
    ("feedback", feedback, ["POST"]),
    ("feedback-stats", feedback_stats, ["GET"]),
    ("test-gemini-key", test_gemini_key, ["POST"]),
]:
    app.add_api_route("/api/index/" + _path, _endpoint, methods=_methods, include_in_schema=False)
