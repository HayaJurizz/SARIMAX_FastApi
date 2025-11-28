# DCPO SARIMA Crime Forecast API

This repository contains a FastAPI application that trains SARIMA models on a 5-year monthly DCPO crime dataset and exposes endpoints for forecasting total crimes and the top offense per month.

Quick start

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the API (development):

```powershell
uvicorn Sarima_Api.Sarima:app --reload
```

Notes

- Model training runs at startup in a background thread to avoid blocking the event loop.
- Convergence warnings are handled and suppressed for smooth console output; important fit failures are logged at ERROR level.

Files of interest

- `Sarima_Api/Sarima.py` - main FastAPI app and training logic
- `data/` - CSV data files used for training

To push to GitHub

1. Create a remote repository on GitHub (via the website or `gh` CLI).
2. Add the remote and push:

```powershell
git remote add origin <REMOTE_URL>
git branch -M main
git push -u origin main
```
