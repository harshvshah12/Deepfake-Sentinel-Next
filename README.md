# Deepfake Sentinel Next

A Vercel-hosted dashboard and Python API for the Deepfake Sentinel research project, based on [the original repository](https://github.com/harshvshah12/Deepfake-Sentinel).

## Deployment

1. Import this repository into Vercel and deploy with the default settings.
2. In **Project → Settings → Environment Variables**, add `HF_TOKEN` with a Hugging Face token that can use Inference Providers. Optional: set `HF_MODEL_ID` to a compatible image-classification model.
3. Redeploy after adding environment variables.

The API works without a token, but then uses image-statistics heuristics only. Those results are explicitly labelled **not a validated deepfake detector** and must not be treated as proof that an image is real or fake. With a configured model token, the API attempts model-backed image classification. Model output is still only a screening signal.

## Endpoints

- `GET /api/index/health` — API/model status
- `POST /api/index/predict` — multipart image upload in field `file`
- `POST /api/index/predict-frame` — JSON body with base64 image in `image`
- `GET /api/index/model-metrics` — runtime metadata; no fabricated ROC/AUC results
- `GET /api/index/samples` — bundled specimen metadata
- `POST /api/index/feedback`, `GET /api/index/feedback-stats`
- `POST /api/index/test-gemini-key` — validates a Gemini API key without persisting it

## Known boundaries

- A trained model requires `HF_TOKEN` and a compatible Hugging Face inference model.
- The fallback is not a trained detector and is intentionally low-confidence.
- Feedback is acknowledged but not persisted across serverless invocations.
- The original video workflow still needs frame extraction to analyze video clips; a video file is not an image upload.
- No independently measured benchmark scores are claimed by this deployment.

## Local checks

Install dependencies from `requirements.txt`, then run `pytest`. To run the API locally: `uvicorn api.index:app --reload`.
