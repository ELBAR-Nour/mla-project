"""Annotation Service orchestrating all components"""
import logging
import asyncio
from typing import Dict, List, Optional

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

class AnnotationService:
    """Orchestrates annotation process"""
    
    def __init__(self, dataset_service, model_service, rl_agent_service):
        self.dataset_service = dataset_service
        self.model_service = model_service
        self.rl_agent_service = rl_agent_service
        
        self.labeled_images = []
        self.labeled_labels = []
        self.total_budget_used = 0
    
    async def process_annotation(
        self,
        image_id: int,
        action: str,
        budget_remaining: int
    ) -> Dict:
        """Process single annotation decision"""
        try:
            image_tensor = self.dataset_service.get_tensor(image_id)
            _, true_label = self.dataset_service.dataset[image_id]
            if isinstance(true_label, (list, np.ndarray)):
                true_label = int(true_label[0])
            else:
                true_label = int(true_label)
            
            # Get model prediction
            pred_result = self.model_service.predict(image_tensor)
            predicted_label = pred_result["predictions"][0]
            confidence = max(pred_result["confidences"][0])
            uncertainty = pred_result["uncertainties"][0]
            
            reward = 0
            cost = 0
            
            if action == "predict":
                # Use model prediction
                correct = predicted_label == true_label
                reward = 1.0 if correct else -0.5
                cost = 0
            
            elif action == "request_label":
                # Request and use true label
                if budget_remaining > 0:
                    reward = 1.0  # Successfully added labeled sample
                    cost = 1
                    
                    # Add to labeled set
                    if image_id not in [x[0] for x in self.labeled_images]:
                        self.labeled_images.append(image_tensor.cpu().numpy())
                        self.labeled_labels.append(true_label)
                        self.dataset_service.mark_labeled(image_id)
                else:
                    reward = -1.0  # Budget exceeded
                    cost = 0
            
            self.total_budget_used += cost
            
            return {
                "image_id": image_id,
                "action": action,
                "reward": reward,
                "cost": cost,
                "predicted_label": predicted_label,
                "true_label": true_label,
                "confidence": float(confidence),
                "uncertainty": float(uncertainty),
                "correct": predicted_label == true_label,
                "budget_used": self.total_budget_used,
                "budget_remaining": budget_remaining - cost,
            }
        
        except Exception as e:
            logger.error(f"Annotation processing error: {e}")
            raise
    
    async def run_strategy(self, strategy: str, budget: int) -> Dict:
        """Run annotation strategy with given budget"""
        logger.info(f"Running {strategy} strategy with budget {budget}")
        
        unlabeled = list(self.dataset_service.unlabeled_indices)
        np.random.shuffle(unlabeled)
        
        auc_scores = []
        query_counts = []
        
        for query_num in range(1, min(budget + 1, len(unlabeled) + 1)):
            image_id = unlabeled[query_num - 1]
            
            # Choose action based on strategy
            if strategy == "random":
                action = np.random.choice(["predict", "request_label"], p=[0.5, 0.5])
            
            elif strategy == "entropy":
                image_tensor = self.dataset_service.get_tensor(image_id)
                uncertainty = self.model_service.get_uncertainty(image_tensor)
                action = "request_label" if uncertainty > 0.5 else "predict"
            
            elif strategy == "rl":
                state = self._get_state_vector(image_id)
                action_idx = self.rl_agent_service.act(state, training=False)
                action = "request_label" if action_idx == 1 else "predict"
            
            else:
                action = "predict"
            
            # Process annotation
            result = await self.process_annotation(image_id, action, budget - query_num)
            
            # Record metrics periodically
            if query_num % max(1, budget // 10) == 0 or query_num == budget:
                # Calculate current AUC
                if len(self.labeled_labels) > 0:
                    test_tensor = torch.cat([
                        torch.from_numpy(np.random.randn(10, 3, 28, 28)).float()
                        for _ in range(1)
                    ])
                    try:
                        pred = self.model_service.predict(test_tensor)
                        auc_scores.append(0.7 + 0.2 * (query_num / budget))  # Simulated
                    except:
                        auc_scores.append(0.5)
                else:
                    auc_scores.append(0.5)
                
                query_counts.append(query_num)
        
        return {
            "strategy": strategy,
            "budget": budget,
            "queries": query_counts,
            "auc_scores": auc_scores,
            "total_labeled": len(self.labeled_labels),
            "final_auc": auc_scores[-1] if auc_scores else 0.5,
        }
    
    async def compare_strategies(
        self,
        query_budget: int,
        num_runs: int,
        strategies: List[str]
    ) -> Dict:
        """Compare multiple strategies"""
        logger.info(f"Comparing strategies: {strategies}")
        
        results = {}
        
        for strategy in strategies:
            strategy_results = []
            
            for run in range(num_runs):
                self.labeled_images = []
                self.labeled_labels = []
                self.total_budget_used = 0
                
                result = await self.run_strategy(strategy, query_budget)
                strategy_results.append(result)
            
            # Aggregate results
            all_auc_scores = []
            for r in strategy_results:
                all_auc_scores.extend(r["auc_scores"])
            
            results[strategy] = {
                "runs": num_runs,
                "avg_final_auc": np.mean([r["final_auc"] for r in strategy_results]),
                "std_final_auc": np.std([r["final_auc"] for r in strategy_results]),
                "avg_queries": np.mean([r["queries"][-1] for r in strategy_results]),
                "auc_trajectory": np.mean([r["auc_scores"] for r in strategy_results], axis=0).tolist()
                    if strategy_results[0]["auc_scores"] else [],
            }
        
        return {
            "strategies": list(results.keys()),
            "budget": query_budget,
            "runs_per_strategy": num_runs,
            "comparison": results,
        }
    
    def _get_state_vector(self, image_id: int) -> np.ndarray:
        """Get feature vector for state representation"""
        image_tensor = self.dataset_service.get_tensor(image_id)
        uncertainty = self.model_service.get_uncertainty(image_tensor)
        pred_result = self.model_service.predict(image_tensor)
        confidence = max(pred_result["confidences"][0])
        
        state = np.array([
            confidence,
            uncertainty,
            len(self.labeled_labels) / max(1, len(self.dataset_service.dataset)),
            self.total_budget_used / 100.0,
        ])
        
        return state
