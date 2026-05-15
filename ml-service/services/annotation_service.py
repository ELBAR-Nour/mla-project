import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math

def compute_entropy(probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return -np.sum(probs * np.log(probs + eps), axis=1)

class AnnotationEnvironment:
    def __init__(self, dataset_service, model_service, n_classes, max_budget=200, label_penalty=-0.05):
        self.dataset_service = dataset_service
        self.model_service = model_service
        self.n_classes = n_classes
        self.max_budget = max_budget
        self.label_penalty = label_penalty
        
        self.labeled_idx = list(dataset_service.labeled_idx)
        self.unlabeled_idx = list(dataset_service.unlabeled_idx)
        self.budget_remaining = max_budget
        self.current_sample_idx = None
        self.current_probs = None
        self.current_auc = 0.5
        self._query_log = []
        
        # Initial model training
        self.model_service.initialize_model(self.n_classes)
        loader = self.dataset_service.get_labeled_loader(batch_size=64)
        self.current_auc = self.model_service.train_model(self.model_service.model, loader, self.dataset_service.val_loader, self.n_classes)

    def reset(self) -> np.ndarray:
        self.labeled_idx = list(self.dataset_service.labeled_idx)
        self.unlabeled_idx = list(self.dataset_service.unlabeled_idx)
        self.budget_remaining = self.max_budget
        self._query_log.clear()
        
        self.model_service.initialize_model(self.n_classes)
        loader = self.dataset_service.get_labeled_loader(batch_size=64)
        self.current_auc = self.model_service.train_model(self.model_service.model, loader, self.dataset_service.val_loader, self.n_classes)
        return self._next_state()

    def _next_state(self) -> np.ndarray:
        if not self.unlabeled_idx:
            return np.zeros(8, dtype=np.float32)

        k = min(20, len(self.unlabeled_idx))
        candidates = random.sample(self.unlabeled_idx, k)
        imgs_t = torch.tensor(self.dataset_service.train_images[candidates], dtype=torch.float32).to(self.model_service.device)
        
        with torch.no_grad():
            probs = F.softmax(self.model_service.model(imgs_t), dim=-1).cpu().numpy()
            
        entropies = compute_entropy(probs)
        best_i = int(np.argmax(entropies))
        self.current_sample_idx = candidates[best_i]
        self.current_probs = probs[best_i]

        confidence = float(self.current_probs.max())
        entropy_val = float(entropies[best_i])
        sorted_p = np.sort(self.current_probs)[::-1]
        margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0
        budget_ratio = self.budget_remaining / self.max_budget
        auc_norm = self.current_auc
        labeled_ratio = len(self.labeled_idx) / (len(self.labeled_idx) + len(self.unlabeled_idx) + 1e-6)
        step_ratio = 1.0 - budget_ratio
        
        pool_labels = self.dataset_service.train_labels[self.labeled_idx]
        counts = np.bincount(pool_labels, minlength=self.n_classes).astype(float)
        imbalance = float(counts.min() / (counts.max() + 1e-6))

        return np.array([confidence, entropy_val, margin, budget_ratio, auc_norm, labeled_ratio, step_ratio, imbalance], dtype=np.float32)

    def step(self, action: int):
        prev_auc = self.current_auc
        reward = 0.0
        entropy_val = float(compute_entropy(self.current_probs[np.newaxis])[0]) if self.current_probs is not None else 0.0

        if action == 1 and self.budget_remaining > 0:
            if self.current_sample_idx in self.unlabeled_idx:
                self.unlabeled_idx.remove(self.current_sample_idx)
            self.labeled_idx.append(self.current_sample_idx)
            self.budget_remaining -= 1
            self._query_log.append(1)
            reward += 0.5 * entropy_val + self.label_penalty
            
            if len(self.labeled_idx) % 5 == 0 or self.budget_remaining == 0:
                # Update labeled loader and retrain
                loader = self.dataset_service._make_loader(self.dataset_service.train_images, self.dataset_service.train_labels, self.labeled_idx, batch_size=64, shuffle=True)
                self.current_auc = self.model_service.train_model(self.model_service.model, loader, self.dataset_service.val_loader, self.n_classes)
                delta_auc = self.current_auc - prev_auc
                reward += 10.0 * delta_auc
        else:
            if self.current_sample_idx in self.unlabeled_idx:
                self.unlabeled_idx.remove(self.current_sample_idx)
            self._query_log.append(0)
            if self.current_probs is not None:
                reward = 0.01 * float(self.current_probs.max())

        done = (self.budget_remaining <= 0) or (len(self.unlabeled_idx) == 0)
        next_state = self._next_state() if not done else np.zeros(8, dtype=np.float32)
        return next_state, reward, done


class AnnotationService:
    def __init__(self, dataset_service, model_service, rl_agent_service):
        self.dataset_service = dataset_service
        self.model_service = model_service
        self.rl_agent_service = rl_agent_service
        self.env = None

    def _ensure_env(self):
        if self.env is None:
            self.env = AnnotationEnvironment(self.dataset_service, self.model_service, self.dataset_service.n_classes)
            self.rl_agent_service.set_annotation_service(self)

    def reset_env(self):
        self._ensure_env()
        return self.env.reset()

    def step_env(self, action):
        self._ensure_env()
        return self.env.step(action)

    async def process_annotation(self, image_id: int, action: str, budget_remaining: int):
        self._ensure_env()
        action_val = 1 if action == "request_label" else 0
        
        # Override the env sample idx to match user input
        self.env.current_sample_idx = image_id
        if image_id in self.env.unlabeled_idx:
            # We need to set self.env.current_probs for the reward logic
            img_t = torch.tensor(self.dataset_service.train_images[[image_id]], dtype=torch.float32).to(self.model_service.device)
            with torch.no_grad():
                self.env.current_probs = F.softmax(self.model_service.model(img_t), dim=-1).cpu().numpy()[0]
        
        next_state, reward, done = self.env.step(action_val)
        
        pred = int(self.env.current_probs.argmax()) if self.env.current_probs is not None else -1
        true_lbl = int(self.dataset_service.train_labels[image_id])
        conf = float(self.env.current_probs.max()) if self.env.current_probs is not None else 0.0
        
        return {
            "image_id": image_id,
            "action": action,
            "reward": reward,
            "cost": 1 if action_val == 1 else 0,
            "predicted_label": pred,
            "true_label": true_lbl,
            "confidence": conf,
            "uncertainty": float(compute_entropy(self.env.current_probs[np.newaxis])[0]) if self.env.current_probs is not None else 0.0,
            "correct": pred == true_lbl,
            "budget_used": self.env.max_budget - self.env.budget_remaining,
            "budget_remaining": self.env.budget_remaining
        }

    async def run_strategy(self, strategy: str, queries: int):
        self._ensure_env()
        self.env.reset()
        aucs = [self.env.current_auc]
        queries_made = [0]
        
        labeled = list(self.env.labeled_idx)
        unlabeled = list(self.env.unlabeled_idx)
        
        # We reuse the basic AL logic from notebook
        for i in range(queries):
            if not unlabeled: break
            
            if strategy == "random":
                idx = random.choice(unlabeled)
            elif strategy == "entropy":
                imgs_t = torch.tensor(self.dataset_service.train_images[unlabeled], dtype=torch.float32).to(self.model_service.device)
                with torch.no_grad():
                    probs = F.softmax(self.model_service.model(imgs_t), dim=-1).cpu().numpy()
                entropies = compute_entropy(probs)
                best_i = int(np.argmax(entropies))
                idx = unlabeled[best_i]
            elif strategy == "rl":
                # RL agent makes selection
                state = self.env._next_state() # Need to adapt here, since step() alters state
                # For simplicity in evaluation, we let RL step through until it selects a label
                done = False
                action = 0
                while action == 0 and not done and self.env.budget_remaining > 0:
                    state = self.env._next_state()
                    action = self.rl_agent_service.agent.select_action(state)
                    # Force it to query if it loops too long?
                    next_state, r, done = self.env.step(action)
                # The step already updated the env
                aucs.append(self.env.current_auc)
                queries_made.append(i+1)
                continue
            else:
                idx = random.choice(unlabeled)
            
            # Non-RL strategy update
            unlabeled.remove(idx)
            labeled.append(idx)
            
            if (i+1) % 5 == 0 or i == queries - 1:
                loader = self.dataset_service._make_loader(self.dataset_service.train_images, self.dataset_service.train_labels, labeled, batch_size=64, shuffle=True)
                current_auc = self.model_service.train_model(self.model_service.model, loader, self.dataset_service.val_loader, self.n_classes)
                self.env.current_auc = current_auc
            aucs.append(self.env.current_auc)
            queries_made.append(i+1)

        return {
            "strategy": strategy,
            "budget": queries,
            "queries": queries_made,
            "auc_scores": aucs,
            "total_labeled": len(self.env.labeled_idx) if strategy == "rl" else len(labeled),
            "final_auc": aucs[-1]
        }

    async def compare_strategies(self, query_budget: int, num_runs: int, strategies: list):
        comparison = {}
        for strat in strategies:
            strat_aucs = []
            strat_queries = []
            for _ in range(num_runs):
                res = await self.run_strategy(strat, query_budget)
                strat_aucs.append(res["auc_scores"])
            
            avg_auc = np.mean(strat_aucs, axis=0).tolist()
            final_aucs = [a[-1] for a in strat_aucs]
            comparison[strat] = {
                "runs": num_runs,
                "avg_final_auc": float(np.mean(final_aucs)),
                "std_final_auc": float(np.std(final_aucs)),
                "avg_queries": query_budget,
                "auc_trajectory": avg_auc
            }
        
        return {
            "strategies": strategies,
            "budget": query_budget,
            "runs_per_strategy": num_runs,
            "comparison": comparison
        }
