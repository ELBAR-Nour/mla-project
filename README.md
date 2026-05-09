# RL and Active Learning for Medical Data Annotation

## Project Overview
This project focuses on leveraging Reinforcement Learning (RL) and Active Learning techniques to optimize medical data annotation. The goal is to train an RL agent to prioritize which medical images should be annotated by a human expert under a strict labeling budget.

## Objective
- Train an RL agent to select which medical images a human expert should prioritize annotating under a strict labeling budget.

## Dataset
- **Dataset Name**: MedMNIST (PneumoniaMNIST or BreastMNIST)
- **Dataset Link**: [https://medmnist.com/](https://medmnist.com/)

## Open Methodology
- The RL agent can choose between requesting a label or predicting directly.
- The reward is tied to the classification models accuracy gain.

## SOTA Comparison
- Random sampling vs uncertainty sampling (entropy-based) vs RL agent.

## Suggested Metrics
- **AUC Curve**: Plotted against the number of annotation queries.

## Reference Papers (SOTA)
- State-of-the-art research on query strategies in deep active learning.

## Project Structure
```
MLA/
├── backend/
│   ├── src/
│   │   ├── routes/
│   │   ├── services/
│   │   └── server.ts
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── routes/
│   │   └── styles.css
├── ml-service/
│   ├── main.py
│   ├── services/
│   │   ├── annotation_service.py
│   │   ├── dataset_service.py
│   │   ├── model_service.py
│   │   └── rl_agent_service.py
└── RL_ActiveLearning_MedicalImageAnnotation.ipynb
```

## How to Run
### Backend
1. Navigate to the `backend` folder.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the server:
   ```bash
   npm run dev
   ```

### Frontend
1. Navigate to the `frontend` folder.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

### ML Service
1. Navigate to the `ml-service` folder.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```


## License
This project is for educational purposes and is not licensed for commercial use.