# MaxVit

This repository contains a small FastAPI application that uses a MaxViT-style model for image-related tasks.

## Prerequisites

- Python 3.13 (recommended)
- A virtual environment (venv) is recommended
- Install required packages (example list below)

You can create a virtual environment and install 

```
python -m pip install -r requirements.txt
```

## Run (development)

To run the application with auto-reload (development mode), use:

```
python -m uvicorn app:app --reload
```

This starts Uvicorn and serves the FastAPI app object found in `app.py`. By default it will listen on http://127.0.0.1:8000.

If you need to change host/port or use multiple workers, add flags, for example:

```powershell
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

## Files of interest

- `app.py` — FastAPI application entry point (contains `app` FastAPI instance)
- `model.py` — model architecture and loading utilities
- `maxvit_trained_with_acer.pth` — pretrained model weights

