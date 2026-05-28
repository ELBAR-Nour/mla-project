"""
FastAPI ML Microservice for RL + Active Learning
Handles dataset management, model inference, and RL agent operations
"""
import os
import json
import csv
import logging
from pathlib import Path
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
SERVICE_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = Path(os.getenv("ML_EXPERIMENTS_DIR", SERVICE_ROOT / "experiments"))

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
    strategies: List[str] = ["random", "margin", "entropy", "bald", "badge", "dqn", "double_dqn", "dueling_dqn"]

class RLTrainRequest(BaseModel):
    episodes: int = 100
    learning_rate: float = 0.001
    gamma: float = 0.99

class ModelLoadRequest(BaseModel):
    model_name: str

class CheckpointLoadRequest(BaseModel):
    checkpoint_name: str

class InferenceRunRequest(BaseModel):
    model_name: Optional[str] = None
    image_id: Optional[int] = None
    image_base64: Optional[str] = None
    dataset_name: str = "PneumoniaMNIST"
    split: str = "train"

class RLDecisionRequest(BaseModel):
    checkpoint_name: Optional[str] = None
    model_name: Optional[str] = None
    image_id: int
    dataset_name: str = "PneumoniaMNIST"
    split: str = "train"
    budget_remaining: Optional[int] = None
    max_budget: int = 200

def _ensure_dataset_loaded(dataset_name: str = "PneumoniaMNIST", split: str = "train") -> DatasetService:
    if dataset_service is None:
        raise HTTPException(status_code=503, detail="Dataset service not ready")
    if dataset_service.train_images is None or dataset_service.dataset_name != dataset_name.lower() or dataset_service.split != split:
        dataset_service.load_dataset(dataset_name=dataset_name, split=split)
    return dataset_service

def _class_labels(ds: DatasetService) -> List[str]:
    if not ds.info:
        return [str(i) for i in range(ds.n_classes or 2)]
    labels = ds.info.get("label", {})
    return [labels.get(str(i), labels.get(i, str(i))) for i in range(ds.n_classes)]

def _ensure_model_evaluated(dataset_name: str = "PneumoniaMNIST") -> DatasetService:
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model service not ready")
    ds = _ensure_dataset_loaded(dataset_name, "train")
    model = model_service.ensure_model_loaded()
    metrics, probs, labels = model_service.evaluate(model, ds.test_loader, ds.n_classes, return_raw=True)
    model_service.latest_metrics = metrics
    model_service.latest_probs = probs
    model_service.latest_labels = labels
    return ds

def _coerce_csv_value(value: str):
    if value is None:
        return value
    try:
        if value.strip() == "":
            return value
        number = float(value)
        if number.is_integer():
            return int(number)
        return number
    except (ValueError, AttributeError):
        return value

def _read_experiment_csv(filename: str) -> List[Dict]:
    path = EXPERIMENTS_DIR / filename
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]

def _read_experiment_json(filename: str) -> Dict:
    path = EXPERIMENTS_DIR / filename
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

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

@app.get("/experiments/summary")
async def get_experiment_summary():
    """Return exported notebook experiment metrics for every saved strategy/model."""
    try:
        return {
            "config": _read_experiment_json("pneumoniamnist_seed42_budget200_query10_config.json"),
            "artifact_manifest": _read_experiment_json("pneumoniamnist_seed42_budget200_query10_artifact_manifest.json"),
            "dataset_summary": _read_experiment_csv("pneumoniamnist_seed42_budget200_query10_dataset_summary.csv"),
            "main_results": _read_experiment_csv("pneumoniamnist_seed42_budget200_query10_main_results_summary.csv"),
            "clinical_metrics": _read_experiment_csv("pneumoniamnist_seed42_budget200_query10_clinical_metrics_summary.csv"),
            "learning_curves": _read_experiment_csv("pneumoniamnist_seed42_budget200_query10_learning_curves_long.csv"),
            "multiseed_summary": _read_experiment_csv("pneumoniamnist_seed42_budget200_query10_multiseed_summary.csv"),
        }
    except Exception as e:
        logger.error(f"Experiment summary loading error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/info")
async def service_info():
    """Get ML service information"""
    return {
        "name": "MedAL ML Service",
        "version": "1.0.0",
        "device": str(DEVICE),
        "available_datasets": ["PneumoniaMNIST", "BreastMNIST"],
        "available_strategies": ["random", "margin", "entropy", "bald", "badge", "dqn", "double_dqn", "dueling_dqn"],
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
@app.get("/models/artifacts")
async def list_model_artifacts():
    """List classifier and RL checkpoint .pt artifacts available for inference"""
    try:
        if model_service is None or rl_agent_service is None:
            raise HTTPException(status_code=503, detail="ML services not ready")

        return {
            "classifiers": model_service.list_model_artifacts(),
            "checkpoints": rl_agent_service.list_checkpoints(),
            "active_model": model_service.active_model_name,
            "active_checkpoint": rl_agent_service.active_checkpoint_name,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model artifact listing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/models/load")
async def load_model_artifact(request: ModelLoadRequest):
    """Load a classifier .pt artifact into memory"""
    try:
        if model_service is None:
            raise HTTPException(status_code=503, detail="Model service not ready")
        return model_service.load_model_artifact(request.model_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/inference/run")
async def run_inference(request: InferenceRunRequest):
    """Run classifier inference against a dataset sample or uploaded base64 image"""
    try:
        if model_service is None:
            raise HTTPException(status_code=503, detail="Model service not ready")
        if request.image_id is None and not request.image_base64:
            raise HTTPException(status_code=400, detail="Provide image_id or image_base64")

        ds = _ensure_dataset_loaded(request.dataset_name, request.split)
        labels = _class_labels(ds)
        image_base64 = None
        true_label = None

        if request.image_id is not None:
            if request.image_id < 0 or request.image_id >= len(ds.train_images):
                raise HTTPException(status_code=400, detail=f"image_id {request.image_id} is outside the loaded dataset")
            image_tensor = torch.tensor(ds.train_images[[request.image_id]], dtype=torch.float32)
            true_label = int(ds.train_labels[request.image_id])
            image_base64 = ds._image_to_base64(ds.train_images[request.image_id])
        else:
            image_tensor = model_service.tensor_from_base64(request.image_base64)

        prediction = model_service.predict_tensor(
            image_tensor,
            model_name=request.model_name,
            true_label=true_label,
            class_labels=labels,
        )
        return {
            "status": "success",
            "dataset_name": ds.dataset_name,
            "split": ds.split,
            "image_id": request.image_id,
            "image_base64": image_base64,
            **prediction,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/metrics")
async def get_model_metrics():
    """Get current model performance metrics"""
    try:
        _ensure_model_evaluated()
        metrics = model_service.get_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/confusion-matrix")
async def get_confusion_matrix():
    """Get confusion matrix"""
    try:
        ds = _ensure_model_evaluated()
        cm = model_service.get_confusion_matrix()
        return {
            "matrix": cm.tolist(),
            "labels": _class_labels(ds),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/roc")
async def get_roc_curve():
    """Get ROC curve data"""
    try:
        _ensure_model_evaluated()
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
@app.get("/checkpoints")
async def list_checkpoints():
    """List DQN/RL .pt checkpoints available for policy inference"""
    try:
        if rl_agent_service is None:
            raise HTTPException(status_code=503, detail="RL agent service not ready")
        return {
            "checkpoints": rl_agent_service.list_checkpoints(),
            "active_checkpoint": rl_agent_service.active_checkpoint_name,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Checkpoint listing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/checkpoints/load")
async def load_checkpoint(request: CheckpointLoadRequest):
    """Load a DQN/RL checkpoint into memory for policy inference"""
    try:
        if rl_agent_service is None:
            raise HTTPException(status_code=503, detail="RL agent service not ready")
        return rl_agent_service.load_checkpoint(request.checkpoint_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Checkpoint loading error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rl/decision")
async def get_rl_decision(request: RLDecisionRequest):
    """Run the loaded RL checkpoint on a sample state and return the recommended action"""
    try:
        if rl_agent_service is None or model_service is None:
            raise HTTPException(status_code=503, detail="ML services not ready")

        ds = _ensure_dataset_loaded(request.dataset_name, request.split)
        labels = _class_labels(ds)
        if request.model_name:
            model_service.ensure_model_loaded(request.model_name)

        state_payload = rl_agent_service.build_state_for_sample(
            dataset_service=ds,
            model_service=model_service,
            image_id=request.image_id,
            budget_remaining=request.budget_remaining,
            max_budget=request.max_budget,
            class_labels=labels,
        )
        decision = rl_agent_service.predict_action(state_payload["state"], request.checkpoint_name)
        return {
            "status": "success",
            "dataset_name": ds.dataset_name,
            "split": ds.split,
            "image_id": request.image_id,
            "image_base64": ds._image_to_base64(ds.train_images[request.image_id]),
            "state_features": state_payload["state_features"],
            "prediction": state_payload["prediction"],
            **decision,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RL decision error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

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
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
