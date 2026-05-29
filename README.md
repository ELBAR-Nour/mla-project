# MedAL: Reinforcement Learning and Active Learning for Medical Image Annotation

MedAL is a research-oriented prototype for studying how reinforcement learning (RL) can improve label efficiency in medical image annotation. The project compares classical deep active learning query strategies with learned RL policies under a constrained annotation budget, using MedMNIST benchmark datasets and an interactive annotation dashboard.

The central question is practical: when expert labels are expensive, which image should be labeled next, and when should the system rely on its current model prediction instead?

## Research Motivation

Medical image annotation is often limited by clinical expert time, privacy constraints, and the cost of high-quality labels. Active learning addresses this by selecting the most informative samples for annotation, while RL can frame annotation as a sequential decision problem where the agent learns the long-term value of spending or saving the labeling budget.

This repository explores that idea through:

- Binary medical image classification on MedMNIST datasets.
- Budget-constrained annotation policies.
- Baseline active learning strategies: random, margin, entropy, BALD, and BADGE.
- RL-based policies: DQN, Double DQN, and Dueling DQN.
- Reproducible experiment artifacts, trained checkpoints, figures, and dashboard views.

## Contributions

- Implements a full-stack research prototype for RL-guided medical annotation.
- Provides an ML service for dataset loading, inference, strategy comparison, and RL policy decisions.
- Includes trained classifier and RL checkpoint artifacts for PneumoniaMNIST experiments.
- Exports experiment summaries, learning curves, clinical metrics, ablation outputs, and publication-ready figures.
- Connects experiments to an interactive frontend for model inspection, annotation simulation, and strategy comparison.

## Methodology

### Problem Formulation

The annotation process is modeled as a sequential decision problem. At each step, the system observes model and budget state features for an unlabeled medical image and selects one of two actions:

- `request_label`: spend part of the annotation budget to acquire the true label.
- `predict`: avoid annotation cost and use the current model prediction.

The reward signal balances predictive improvement, annotation cost, uncertainty, budget urgency, and class-imbalance-aware behavior.

### Datasets

The project uses [MedMNIST](https://medmnist.com/) benchmark datasets:

| Dataset | Train | Validation | Test | Classes | Image Shape |
| --- | ---: | ---: | ---: | ---: | --- |
| PneumoniaMNIST | 4,708 | 524 | 624 | 2 | 1 x 28 x 28 |
| BreastMNIST | 546 | 78 | 156 | 2 | 1 x 28 x 28 |

PneumoniaMNIST is the primary experiment dataset in the saved artifacts. BreastMNIST is included for exploratory analysis and cross-dataset inspection.

### Query Strategies

The project compares:

- `random`: uniform random sampling.
- `margin`: samples with the smallest prediction margin.
- `entropy`: samples with highest predictive entropy.
- `bald`: Bayesian active learning by disagreement.
- `badge`: gradient-based diverse sampling.
- `DQN`: learned value-based annotation policy.
- `Double DQN`: DQN variant that reduces value overestimation.
- `Dueling DQN`: separates state value and action advantage estimation.

### Experimental Configuration

The saved PneumoniaMNIST experiment uses:

| Parameter | Value |
| --- | ---: |
| Initial labeled samples | 100 |
| Annotation budget | 200 |
| Query batch size | 10 |
| Active learning steps | 20 |
| CNN epochs | 15 |
| State dimension | 9 |
| RL episodes | 15 |
| Seed | 42 |
| Multi-seed validation | 42, 123, 456, 789, 2024 |

Full configuration is stored in `ml-service/experiments/pneumoniamnist_seed42_budget200_query10_config.json`.

## Results Snapshot

The table below summarizes the exported PneumoniaMNIST experiment results under a 200-query budget.

| Strategy | ALC | Final Validation AUC | Final Test AUC | Queries |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.9351 | 0.9871 | 0.9516 | 200 |
| Margin | 0.9272 | 0.9905 | 0.9422 | 200 |
| Entropy | 0.9169 | 0.9850 | 0.9494 | 200 |
| BALD | 0.9307 | 0.8936 | 0.9094 | 200 |
| BADGE | 0.9541 | 0.9909 | 0.9461 | 200 |
| DQN | 0.9851 | 0.9944 | 0.9561 | 200 |
| Double DQN | 0.9863 | 0.9965 | 0.9321 | 200 |
| Dueling DQN | 0.9827 | 0.9924 | 0.9497 | 200 |

Multi-seed validation shows that the RL variants maintain high annotation learning curve (ALC) scores across seeds:

| Strategy | Mean ALC | ALC Std. | Mean Validation AUC | Validation AUC Std. |
| --- | ---: | ---: | ---: | ---: |
| Entropy | 0.9523 | 0.0173 | 0.9887 | 0.0021 |
| BADGE | 0.9501 | 0.0146 | 0.9879 | 0.0026 |
| DQN | 0.9838 | 0.0047 | 0.9949 | 0.0013 |
| Double DQN | 0.9834 | 0.0053 | 0.9952 | 0.0010 |
| Dueling DQN | 0.9828 | 0.0044 | 0.9950 | 0.0009 |

These results are exported from the repository's saved experiment files and should be regenerated before formal reporting or publication.

## Repository Structure

```text
mla-project/
|-- backend/
|   |-- src/
|   |   |-- routes/          # Express API routes
|   |   |-- services/        # ML service proxy and WebSocket service
|   |   `-- server.ts
|   |-- package.json
|   `-- tsconfig.json
|-- frontend/
|   |-- src/
|   |   |-- components/      # Dashboard and UI components
|   |   |-- routes/          # TanStack Router pages
|   |   |-- lib/             # API client, store, utilities
|   |   `-- styles.css
|   |-- package.json
|   `-- vite.config.ts
|-- ml-service/
|   |-- main.py             # FastAPI ML service
|   |-- services/           # Dataset, model, annotation, and RL services
|   |-- data/               # Local MedMNIST data files
|   |-- models/             # Saved classifier artifacts
|   |-- checkpoints/        # Saved RL checkpoints
|   |-- experiments/        # CSV and JSON experiment outputs
|   |-- figures/            # Experiment and EDA figures
|   |-- results/            # Dashboard-ready result exports
|   |-- W2_data_exploration.ipynb
|   `-- w3-notebook-fixed.ipynb
`-- README.md
```

## System Architecture

```text
Frontend dashboard (React + Vite + TanStack Router)
        |
        | HTTP API
        v
Backend API (Express + TypeScript)
        |
        | Proxy requests / WebSocket events
        v
ML service (FastAPI + PyTorch)
        |
        | Dataset loading, model inference, RL decisioning
        v
MedMNIST data, trained classifiers, RL checkpoints, experiment artifacts
```

## Running the Project

Run each service from its own terminal.

### 1. ML Service

```bash
cd ml-service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Default URL: `http://127.0.0.1:8000`

Useful endpoints:

- `GET /health`
- `GET /info`
- `GET /experiments/summary`
- `GET /models/artifacts`
- `POST /rl/decision`

### 2. Backend API

```bash
cd backend
npm install
npm run dev
```

Default URL: `http://localhost:3001`

Environment variables:

- `PORT`: backend port, defaults to `3001`.
- `FRONTEND_URL`: allowed frontend origin, defaults to `http://localhost:5173`.
- `ML_SERVICE_URL`: ML service URL, defaults to `http://127.0.0.1:8000`.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Default URL: `http://localhost:5173`

Environment variables:

- `VITE_API_URL`: backend API URL, defaults to `http://localhost:3001/api`.

## Research Artifacts

Key artifacts are stored under `ml-service/`:

- `experiments/`: configuration, result summaries, learning curves, ablation outputs, and dataset summaries.
- `figures/`: generated figures for EDA, learning curves, clinical metrics, ablations, transfer behavior, and policy analysis.
- `models/`: saved classifier models for each query strategy.
- `checkpoints/`: saved DQN, Double DQN, and Dueling DQN policy checkpoints.
- `results/`: dashboard-ready CSV, JSON, and image outputs.
- `W2_data_exploration.ipynb`: exploratory data analysis notebook.
- `w3-notebook-fixed.ipynb`: experiment notebook for active learning and RL comparisons.

## Reproducibility Notes

- The saved experiment configuration uses seed `42` and validates over five seeds.
- Regenerate figures and tables from the notebooks before including them in a manuscript.
- Keep dataset, budget, query batch size, and seed fixed when comparing query strategies.
- Report both ALC and final AUC because a strategy can learn quickly early but converge differently at the final budget.
- Clinical metrics such as F1, recall, precision, and calibration should be interpreted alongside class imbalance and decision-threshold choices.

## Limitations

- The current experiments focus primarily on small 28 x 28 MedMNIST benchmark images.
- The dashboard and services are research prototypes, not clinical decision-support tools.
- Saved model artifacts may not reflect a fully tuned or externally validated medical model.
- Before publication, results should be rerun in a clean environment and documented with hardware, runtime, and dependency versions.

## License

The backend package declares an MIT license. The project README previously described the repository as educational and non-commercial. Confirm the intended repository-wide license before public release, redistribution, or publication.

## Citation

If this repository is used in a report or manuscript, cite MedMNIST and any active learning or reinforcement learning baselines used in the final experimental protocol. Add a project-specific citation entry once the work has a stable title, author list, and release tag.
