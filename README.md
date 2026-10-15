# Deepfake Sentinel Next

A separate deployment-ready frontend for the Deepfake Sentinel research project, based on the original dashboard in [Deepfake-Sentinel](https://github.com/harshvshah12/Deepfake-Sentinel).

## Deployment boundary
This repository serves the dashboard as a static site on Vercel. The original analysis UI expects a FastAPI/PyTorch service at the same origin (for example, `/predict`, `/predict-video`, `/predict-frame`, `/health`, and `/samples`). Static hosting does not run that model backend, and the model checkpoint files are not included in this repository. Therefore, UI deployment alone is not evidence that model inference is available.

## Local frontend preview
Open `index.html` in a browser or run any static server from the repository root.

## Full inference backend
Use the original repository's Python backend and model assets, deploy it to a Python host that supports the required dependencies and model storage, then configure the frontend API base URL to that backend. Do not expose Gemini or other provider API keys in public client code.

## Scientific and provenance caveats
A model score is not proof of authenticity. Heuristic watermark or metadata checks must not be described as cryptographic verification; C2PA requires validating a manifest and signature, and SynthID detection requires a supported detector. Video frame aggregation is not a trained temporal model unless one is actually loaded and evaluated.

## License
See the original project's license and preserve its notices when redistributing copied assets or code.
