"""
GlycoCast web app: Python backend (FastAPI) + website (frontend/ folder).

The backend loads the linear regression written from scratch in NumPy,
makes predictions, explains them, runs the gradient descent playground,
and draws every chart with matplotlib. The website only shows the results.

Run locally:   python -m uvicorn app:app --reload      -> http://localhost:8000
Deploy:        push to GitHub and import the repo on Vercel (see DEPLOY.md)
"""

import base64
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

# matplotlib needs a writable folder for its font cache (on Vercel only /tmp is writable)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from glycocast.linreg import LinearRegression
from glycocast import viz

ROOT = Path(__file__).parent

# ---- load everything once, when the server starts --------------------------
m = np.load(ROOT / "models" / "glycocast.npz", allow_pickle=False)
W, B, MEAN, STD = m["w"], float(m["b"]), m["mean"], m["std"]
NAMES, LABELS = [str(n) for n in m["names"]], [str(l) for l in m["labels"]]
FMIN, FMAX, FMED = m["fmin"], m["fmax"], m["fmed"]
X_TEST, Y_TEST = m["Xte"], m["yte"]
BMI_TRAIN, Y_TRAIN = m["bmi_train"], m["y_train"]
RESULTS = json.loads((ROOT / "models" / "results.json").read_text())

LR_SCALED = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 1.05]
LR_RAW = [0.0001, 0.0005, 0.001, 0.0013, 0.0015]

app = FastAPI(title="GlycoCast API", description="Linear regression from scratch for diabetes progression")


class PredictRequest(BaseModel):
    patient: Optional[int] = None              # index of a real test patient
    values: Optional[Dict[str, float]] = None  # or values for the 10 measurements


class PlaygroundRequest(BaseModel):
    scaled: bool = True
    steps: int = Field(200, ge=10, le=1000)
    lr: float = 0.05


def image_base64(draw) -> str:
    """Run a drawing function into memory and return the PNG as base64 text."""
    buf = io.BytesIO()
    draw(buf)
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/api/info")
def info():
    """Everything the website needs to build its controls and the performance tab."""
    inputs = [{"name": n, "label": l, "min": float(lo), "max": float(hi), "value": float(med)}
              for n, l, lo, hi, med in zip(NAMES, LABELS, FMIN, FMAX, FMED)]
    return {"n_test": int(len(Y_TEST)), "inputs": inputs, "average": B, "results": RESULTS,
            "lr_scaled": LR_SCALED, "lr_raw": LR_RAW}


@app.post("/api/predict")
def predict(req: PredictRequest):
    actual, x = None, FMED.copy()
    if req.patient is not None:
        if not 0 <= req.patient < len(Y_TEST):
            raise HTTPException(400, f"patient must be between 0 and {len(Y_TEST) - 1}")
        x = X_TEST[req.patient].copy()
        actual = float(Y_TEST[req.patient])
    else:
        for name, value in (req.values or {}).items():
            if name not in NAMES:
                raise HTTPException(400, f"unknown feature: {name}")
            j = NAMES.index(name)
            x[j] = float(np.clip(value, FMIN[j], FMAX[j]))

    z = (x - MEAN) / STD              # same scaling as in training
    contrib = W * z                   # each feature's push up or down
    pred = float(B + contrib.sum())   # prediction = b + sum(w * z)
    chart = image_base64(lambda buf: viz.plot_contributions(contrib, LABELS, B, pred, buf))
    return {"prediction": pred, "actual": actual, "values": dict(zip(LABELS, map(float, x))),
            "chart_png": chart}


@app.post("/api/playground")
def playground(req: PlaygroundRequest):
    allowed = LR_SCALED if req.scaled else LR_RAW
    if req.lr not in allowed:
        raise HTTPException(400, f"lr must be one of {allowed}")
    x = (BMI_TRAIN - BMI_TRAIN.mean()) / BMI_TRAIN.std() if req.scaled else BMI_TRAIN

    with np.errstate(all="ignore"):
        gd = LinearRegression(lr=req.lr, epochs=req.steps, tol=0, track_path=True).fit(x[:, None], Y_TRAIN)
    best = LinearRegression().fit_closed_form(x[:, None], Y_TRAIN)
    losses = np.array(gd.loss_history)
    if not np.all(np.isfinite(losses)) or losses[-1] > losses[0] * 10:
        return {"diverged": True}

    best_mse = float(np.mean((best.w[0] * x + best.b - Y_TRAIN) ** 2))
    chart = image_base64(lambda buf: viz.plot_playground(x, Y_TRAIN, gd.path, best.w[0], best.b, losses, buf,
                                                         req.scaled))
    return {"diverged": False, "w": float(gd.w[0]), "b": float(gd.b), "best_w": float(best.w[0]),
            "best_b": float(best.b), "gap": abs(float(losses[-1]) - best_mse), "chart_png": chart}


# The website (frontend/ folder) is served for every other path.
app.frontend("/", directory=ROOT / "frontend")
