"""
FastAPI ML Microservice for RL + Active Learning
Handles dataset management, model inference, and RL agent operations
"""
import os
import json
import logging
from typing import Optional, Dict, List
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.dataset_service import DatasetService
from services.model_service import ModelService
from services.rl_agent_service import RLAgentService
from services.annotation_service import AnnotationService

# ── Logging Configuration ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Device Configuration ───────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# ── Global Services ───────────────────────────────────────────────────────
dataset_service: Optional[DatasetService] = None
model_service: Optional[ModelService] = None
rl_agent_service: Optional[RLAgentService] = None
annotation_service: Optional[AnnotationService] = None

# ── Startup/Shutdown Events ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global dataset_service, model_service, rl_agent_service, annotation_service
    
    logger.info("🚀 Initializing ML services...")
    try:
        dataset_service = DatasetService(device=DEVICE)
        model_service = ModelService(device=DEVICE)
        rl_agent_service = RLAgentService(device=DEVICE)
        annotation_service = AnnotationService(
            dataset_service=dataset_service,
            model_service=model_service,
            rl_agent_service=rl_agent_service
        )
        logger.info("✅ All services initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    logger.info("🛑 Shutting down ML services...")

# ── FastAPI Application ────────────────────────────────────────────────────
app = FastAPI(
    title="MedAL ML Service",
    description="Machine Learning backend for RL + Active Learning medical annotation",
    version="1.0.0",
    lifespan=lifespan
)

# ── CORS Configuration ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data Models ────────────────────────────────────────────────────────────
class DatasetInitRequest(BaseModel):
    dataset_name: str = "PneumoniaMNIST"
    split: str = "train"
    sample_size: Optional[int] = None

class AnnotationRequest(BaseModel):
    image_id: int
    action: str  # "predict" or "request_label"
    budget_remaining: int

class StrategyComparisonRequest(BaseModel):
    query_budget: int = 50
    num_runs: int = 3
    strategies: List[str] = ["random", "entropy", "rl"]

class RLTrainRequest(BaseModel):
    episodes: int = 100
    learning_rate: float = 0.001
    gamma: float = 0.99

# ── Health & Info Endpoints ────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Check ML service health"""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "services_ready": all([
            dataset_service is not None,
            model_service is not None,
            rl_agent_service is not None,
            annotation_service is not None
        ])
    }

@app.get("/info")
async def service_info():
    """Get ML service information"""
    return {
        "name": "MedAL ML Service",
        "version": "1.0.0",
        "device": str(DEVICE),
        "available_datasets": ["PneumoniaMNIST", "BreastMNIST"],
        "available_strategies": ["random", "entropy", "rl"],
        "torch_version": torch.__version__,
    }

# ── Dataset Endpoints ──────────────────────────────────────────────────────
@app.post("/dataset/initialize")
async def initialize_dataset(request: DatasetInitRequest):
    """Initialize a dataset for annotation"""
    try:
        if dataset_service is None:
            raise HTTPException(status_code=503, detail="Dataset service not ready")
        
        dataset_service.load_dataset(
            dataset_name=request.dataset_name,
            split=request.split,
            sample_size=request.sample_size
        )
        
        return {
            "status": "success",
            "dataset_name": request.dataset_name,
            "split": request.split,
            "total_samples": dataset_service.get_dataset_info()["total_samples"],
            "image_shape": dataset_service.get_dataset_info()["image_shape"],
        }
    except Exception as e:
        logger.error(f"Dataset initialization error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/dataset/info")
async def get_dataset_info():
    """Get current dataset statistics"""
    try:
        if dataset_service is None:
            raise HTTPException(status_code=503, detail="Dataset service not ready")
        
        info = dataset_service.get_dataset_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/dataset/sample/{image_id}")
async def get_sample(image_id: int):
    """Get a specific sample for annotation"""
    try:
        if dataset_service is None:
            raise HTTPException(status_code=503, detail="Dataset service not ready")
        
        sample = dataset_service.get_sample(image_id)
        return {
            "image_id": image_id,
            "image_base64": sample["image_base64"],
            "shape": sample["shape"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/dataset/batch")
async def get_batch(count: int = 10):
    """Get a batch of samples for annotation"""
    try:
        if dataset_service is None:
            raise HTTPException(status_code=503, detail="Dataset service not ready")
        
        batch = dataset_service.get_batch(count)
        return {
            "count": len(batch),
            "samples": batch
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ── Model Endpoints ────────────────────────────────────────────────────────
@app.get("/model/metrics")
async def get_model_metrics():
    """Get current model performance metrics"""
    try:
        if model_service is None:
            raise HTTPException(status_code=503, detail="Model service not ready")
        
        metrics = model_service.get_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/confusion-matrix")
async def get_confusion_matrix():
    """Get confusion matrix"""
    try:
        if model_service is None:
            raise HTTPException(status_code=503, detail="Model service not ready")
        
        cm = model_service.get_confusion_matrix()
        return {
            "matrix": cm.tolist(),
            "labels": [0, 1],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/roc")
async def get_roc_curve():
    """Get ROC curve data"""
    try:
        if model_service is None:
            raise HTTPException(status_code=503, detail="Model service not ready")
        
        roc_data = model_service.get_roc_curve()
        return roc_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ── Annotation Endpoints ───────────────────────────────────────────────────
@app.post("/annotate")
async def process_annotation(request: AnnotationRequest):
    """Process annotation action and return reward/metrics"""
    try:
        if annotation_service is None:
            raise HTTPException(status_code=503, detail="Annotation service not ready")
        
        result = await annotation_service.process_annotation(
            image_id=request.image_id,
            action=request.action,
            budget_remaining=request.budget_remaining
        )
        
        return result
    except Exception as e:
        logger.error(f"Annotation processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# ── Strategy Endpoints ─────────────────────────────────────────────────────
@app.post("/strategy/compare")
async def compare_strategies(request: StrategyComparisonRequest):
    """Compare multiple annotation strategies"""
    try:
        if annotation_service is None:
            raise HTTPException(status_code=503, detail="Annotation service not ready")
        
        results = await annotation_service.compare_strategies(
            query_budget=request.query_budget,
            num_runs=request.num_runs,
            strategies=request.strategies
        )
        
        return results
    except Exception as e:
        logger.error(f"Strategy comparison error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/strategy/random")
async def get_random_strategy_results(queries: int = 50):
    """Get results from random sampling strategy"""
    try:
        if annotation_service is None:
            raise HTTPException(status_code=503, detail="Annotation service not ready")
        
        results = await annotation_service.run_strategy("random", queries)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/strategy/entropy")
async def get_entropy_strategy_results(queries: int = 50):
    """Get results from entropy sampling strategy"""
    try:
        if annotation_service is None:
            raise HTTPException(status_code=503, detail="Annotation service not ready")
        
        results = await annotation_service.run_strategy("entropy", queries)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/strategy/rl")
async def get_rl_strategy_results(queries: int = 50):
    """Get results from RL agent strategy"""
    try:
        if annotation_service is None:
            raise HTTPException(status_code=503, detail="Annotation service not ready")
        
        results = await annotation_service.run_strategy("rl", queries)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ── RL Agent Endpoints ─────────────────────────────────────────────────────
@app.get("/rl/state")
async def get_rl_state():
    """Get current RL agent state"""
    try:
        if rl_agent_service is None:
            raise HTTPException(status_code=503, detail="RL agent service not ready")
        
        state = rl_agent_service.get_state()
        return state
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/rl/policy")
async def get_rl_policy():
    """Get RL agent's learned policy"""
    try:
        if rl_agent_service is None:
            raise HTTPException(status_code=503, detail="RL agent service not ready")
        
        policy = rl_agent_service.get_policy()
        return policy
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rl/train-step")
async def train_rl_agent(request: RLTrainRequest):
    """Train the RL agent for multiple episodes"""
    try:
        if rl_agent_service is None:
            raise HTTPException(status_code=503, detail="RL agent service not ready")
        
        results = await rl_agent_service.train(
            episodes=request.episodes,
            learning_rate=request.learning_rate,
            gamma=request.gamma
        )
        
        return results
    except Exception as e:
        logger.error(f"RL training error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rl/reset")
async def reset_rl_agent():
    """Reset RL agent to initial state"""
    try:
        if rl_agent_service is None:
            raise HTTPException(status_code=503, detail="RL agent service not ready")
        
        rl_agent_service.reset()
        return {"status": "success", "message": "RL agent reset"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ── Root Endpoint ──────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "service": "MedAL ML Service",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
