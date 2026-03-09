"""
NetFlow AI - Intelligent Bandwidth Allocation using Reinforcement Learning
Backend API (FastAPI) — ML only, no deep learning.

Run locally with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from routes.ingestion import router as ingestion_router
from routes.training import router as training_router
from routes.evaluation import router as evaluation_router
from routes.prediction import router as prediction_router
from routes.rl_agent import router as rl_router


app = FastAPI(
    title="NetFlow AI – Intelligent Bandwidth Allocation",
    description=(
        "Reinforcement-Learning + ML backend for real-time bandwidth allocation. "
        "Uses Q-Learning with Random-Forest / Gradient-Boosting state classifiers. "
        "No deep learning."
    ),
    version="2.0.0",
)

# CORS (allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure required folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# Register API routes
app.include_router(ingestion_router, prefix="/api/ingestion", tags=["1. PCAP Ingestion"])
app.include_router(training_router, prefix="/api/training", tags=["2-3. ML Training"])
app.include_router(evaluation_router, prefix="/api/evaluation", tags=["4-5. Evaluation"])
app.include_router(prediction_router, prefix="/api/prediction", tags=["6. Prediction"])
app.include_router(rl_router, prefix="/api/rl", tags=["7. RL Agent"])


# -----------------------------
# Serve Frontend Dashboard
# -----------------------------

app.mount("/static", StaticFiles(directory="frontend"), name="frontend")


@app.get("/", tags=["Frontend"])
def dashboard():
    return FileResponse("frontend/dashboard.html")


@app.get("/dashboard")
def dashboard_page():
    return FileResponse("frontend/dashboard.html")


@app.get("/ingestion")
def ingestion_page():
    return FileResponse("frontend/DataIngestion.html")


@app.get("/training")
def training_page():
    return FileResponse("frontend/Model_training.html")


@app.get("/evaluation")
def evaluation_page():
    return FileResponse("frontend/ModelEvaluation.html")


@app.get("/prediction")
def prediction_page():
    return FileResponse("frontend/BandwidthPrediction.html")


# -----------------------------
# Health Check
# -----------------------------

@app.get("/api/health", tags=["Health"])
def health():
    return {"status": "healthy", "message": "NetFlow AI API is running"}
