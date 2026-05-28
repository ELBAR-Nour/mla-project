import random
import collections
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

CONFIG = {
    "dqn_lr": 1e-4,
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.88,
    "replay_buffer_size": 10000,
    "dqn_batch_size": 64,
    "tau": 0.005,
    "state_dim": 8,
    "action_dim": 2,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_end": 1.0,
    "per_eps": 1e-6,
}

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
    def forward(self, x):
        return self.net(x)

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, action_dim))

    def forward(self, x):
        feats = self.encoder(x)
        V = self.value_head(feats)
        A = self.advantage_head(feats)
        return V + (A - A.mean(dim=1, keepdim=True))

class SimpleQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)

class SimpleDuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.advantage_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, x):
        feats = self.encoder(x)
        V = self.value_head(feats)
        A = self.advantage_head(feats)
        return V + (A - A.mean(dim=1, keepdim=True))

Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritisedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = Transition(*args)
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4, device=None):
        n = len(self.buffer)
        probs = self.priorities[:n] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(n, batch_size, replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32).to(device)

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p = float(abs(err)) + CONFIG['per_eps']
            self.priorities[idx] = p
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, device):
        self.device = device
        self.q_net = DuelingQNetwork(state_dim, action_dim).to(device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=CONFIG['dqn_lr'])
        self.replay = PrioritisedReplayBuffer(CONFIG['replay_buffer_size'], CONFIG['per_alpha'])
        self.epsilon = CONFIG['epsilon_start']
        self.beta = CONFIG['per_beta_start']
        self.step_count = 0
        self.losses = []
        self.tau = CONFIG['tau']

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, CONFIG['action_dim'] - 1)
        with torch.no_grad():
            q = self.q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))
        return int(q.argmax().item())

    def decay_epsilon(self):
        self.epsilon = max(CONFIG['epsilon_end'], self.epsilon * CONFIG['epsilon_decay'])

    def anneal_beta(self, episode: int, total_episodes: int):
        frac = episode / max(total_episodes - 1, 1)
        self.beta = CONFIG['per_beta_start'] + frac * (CONFIG['per_beta_end'] - CONFIG['per_beta_start'])

    def update_target(self):
        for tp, op in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * op.data + (1 - self.tau) * tp.data)

    def train_step(self) -> float:
        if len(self.replay) < CONFIG['dqn_batch_size']:
            return 0.0
        batch, indices, is_weights = self.replay.sample(CONFIG['dqn_batch_size'], self.beta, self.device)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32).to(self.device)
        
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target_q = rewards + CONFIG['gamma'] * next_q * (1 - dones)
            
        td_errors = (current_q - target_q).detach().cpu().numpy()
        loss = (is_weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.step_count += 1
        self.update_target()
        if indices is not None:
            self.replay.update_priorities(indices, np.abs(td_errors))
        lv = loss.item()
        self.losses.append(lv)
        return lv

class RLAgentService:
    def __init__(self, device):
        self.device = device
        self.agent = DQNAgent(CONFIG['state_dim'], CONFIG['action_dim'], device)
        self.annotation_service = None
        self.latest_state = None
        service_root = Path(__file__).resolve().parents[1]
        self.checkpoints_dir = Path(os.getenv("ML_CHECKPOINTS_DIR", service_root / "checkpoints"))
        self.models_dir = Path(os.getenv("ML_MODELS_DIR", service_root / "models"))
        self.inference_net = self.agent.q_net
        self.active_checkpoint_name = None
        self.active_checkpoint_architecture = "DuelingQNetwork"
        self.active_state_dim = CONFIG["state_dim"]
        self.active_action_dim = CONFIG["action_dim"]
        self.active_epsilon = self.agent.epsilon

    def set_annotation_service(self, annotation_service):
        self.annotation_service = annotation_service

    def get_state(self):
        return {
            "epsilon": self.agent.epsilon,
            "beta": self.agent.beta,
            "replay_buffer_size": len(self.agent.replay),
            "step_count": self.agent.step_count,
            "latest_loss": self.agent.losses[-1] if self.agent.losses else 0.0,
            "latest_state_vector": self.latest_state.tolist() if self.latest_state is not None else [],
            "active_checkpoint": self.active_checkpoint_name,
            "active_state_dim": self.active_state_dim,
        }

    def get_policy(self):
        return {
            "type": self.active_checkpoint_architecture,
            "state_dim": self.active_state_dim,
            "action_dim": self.active_action_dim,
            "active_checkpoint": self.active_checkpoint_name,
            "epsilon": self.active_epsilon,
        }

    def reset(self):
        self.agent = DQNAgent(CONFIG['state_dim'], CONFIG['action_dim'], self.device)
        self.inference_net = self.agent.q_net
        self.active_checkpoint_name = None
        self.active_checkpoint_architecture = "DuelingQNetwork"
        self.active_state_dim = CONFIG["state_dim"]
        self.active_action_dim = CONFIG["action_dim"]
        self.active_epsilon = self.agent.epsilon
        self.latest_state = None

    def _load_torch_file(self, path: Path):
        return torch.load(path, map_location="cpu")

    def _module_index(self, key: str) -> int:
        parts = key.split(".")
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1

    def _extract_q_state(self, artifact) -> Optional[Dict[str, torch.Tensor]]:
        if not isinstance(artifact, dict):
            return None
        for key in ("q_net", "q_net_state", "state_dict"):
            value = artifact.get(key)
            if isinstance(value, dict):
                return value
        if any(key.startswith("net.") or key.startswith("encoder.") for key in artifact.keys()):
            return artifact
        return None

    def _infer_q_metadata(self, state_dict: Dict[str, torch.Tensor]) -> Dict:
        if any(key.startswith("encoder.") for key in state_dict.keys()):
            first_weight = state_dict["encoder.0.weight"]
            final_biases = [
                (key, value)
                for key, value in state_dict.items()
                if key.startswith("advantage_head.") and key.endswith(".bias")
            ]
            final_biases.sort(key=lambda item: self._module_index(item[0]))
            architecture = "DuelingQNetwork" if "encoder.1.weight" in state_dict else "SimpleDuelingQNetwork"
            return {
                "architecture": architecture,
                "state_dim": int(first_weight.shape[1]),
                "hidden_dim": int(first_weight.shape[0]),
                "action_dim": int(final_biases[-1][1].shape[0]),
            }

        first_weight = state_dict["net.0.weight"]
        final_biases = [
            (key, value)
            for key, value in state_dict.items()
            if key.startswith("net.") and key.endswith(".bias")
        ]
        final_biases.sort(key=lambda item: self._module_index(item[0]))
        architecture = "QNetwork" if "net.1.weight" in state_dict else "SimpleQNetwork"
        return {
            "architecture": architecture,
            "state_dim": int(first_weight.shape[1]),
            "hidden_dim": int(first_weight.shape[0]),
            "action_dim": int(final_biases[-1][1].shape[0]),
        }

    def _build_q_network(self, metadata: Dict) -> nn.Module:
        architecture = metadata["architecture"]
        if architecture == "DuelingQNetwork":
            return DuelingQNetwork(metadata["state_dim"], metadata["action_dim"], metadata["hidden_dim"])
        if architecture == "SimpleDuelingQNetwork":
            return SimpleDuelingQNetwork(metadata["state_dim"], metadata["action_dim"], metadata["hidden_dim"])
        if architecture == "QNetwork":
            return QNetwork(metadata["state_dim"], metadata["action_dim"], metadata["hidden_dim"])
        return SimpleQNetwork(metadata["state_dim"], metadata["action_dim"], metadata["hidden_dim"])

    def _resolve_checkpoint_path(self, checkpoint_name: str) -> Path:
        requested = Path(checkpoint_name).name
        candidates = [requested]
        if not requested.endswith(".pt"):
            candidates.append(f"{requested}.pt")

        for directory in (self.checkpoints_dir, self.models_dir):
            for candidate in candidates:
                path = directory / candidate
                if path.exists() and path.is_file():
                    return path
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_name}' was not found in {self.checkpoints_dir} or {self.models_dir}."
        )

    def list_checkpoints(self) -> List[Dict]:
        checkpoints = []
        sources = [
            (self.checkpoints_dir, "checkpoint"),
            (self.models_dir, "model_q_net"),
        ]
        for directory, source in sources:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.pt")):
                if source == "model_q_net" and "q_net" not in path.stem:
                    continue
                try:
                    artifact = self._load_torch_file(path)
                    state_dict = self._extract_q_state(artifact)
                    if state_dict is None:
                        continue
                    metadata = self._infer_q_metadata(state_dict)
                    checkpoints.append({
                        "name": path.name,
                        "source": source,
                        **metadata,
                        "size_bytes": path.stat().st_size,
                        "active": path.name == self.active_checkpoint_name,
                    })
                except Exception as exc:
                    checkpoints.append({
                        "name": path.name,
                        "source": source,
                        "error": str(exc),
                        "size_bytes": path.stat().st_size,
                        "active": False,
                    })
        return checkpoints

    def _default_checkpoint_name(self) -> str:
        checkpoints = [checkpoint for checkpoint in self.list_checkpoints() if "error" not in checkpoint]
        if not checkpoints:
            raise FileNotFoundError("No DQN .pt checkpoints found.")
        preferred = next((checkpoint for checkpoint in checkpoints if checkpoint["name"] == "dueling_dqn_sota.pt"), None)
        return (preferred or checkpoints[0])["name"]

    def load_checkpoint(self, checkpoint_name: str) -> Dict:
        path = self._resolve_checkpoint_path(checkpoint_name)
        artifact = self._load_torch_file(path)
        state_dict = self._extract_q_state(artifact)
        if state_dict is None:
            raise ValueError(f"Artifact '{path.name}' does not contain a DQN state dict.")

        metadata = self._infer_q_metadata(state_dict)
        net = self._build_q_network(metadata).to(self.device)
        net.load_state_dict(state_dict)
        net.eval()

        epsilon = artifact.get("epsilon", 0.0) if isinstance(artifact, dict) else 0.0
        self.inference_net = net
        self.active_checkpoint_name = path.name
        self.active_checkpoint_architecture = metadata["architecture"]
        self.active_state_dim = metadata["state_dim"]
        self.active_action_dim = metadata["action_dim"]
        self.active_epsilon = float(epsilon)
        return {
            "status": "loaded",
            "name": path.name,
            "epsilon": self.active_epsilon,
            **metadata,
        }

    def ensure_checkpoint_loaded(self, checkpoint_name: Optional[str] = None):
        if checkpoint_name and checkpoint_name != self.active_checkpoint_name:
            self.load_checkpoint(checkpoint_name)
        elif self.active_checkpoint_name is None:
            self.load_checkpoint(self._default_checkpoint_name())
        return self.inference_net

    def _adapt_state(self, state: np.ndarray) -> np.ndarray:
        target_dim = self.active_state_dim
        if len(state) == target_dim:
            return state.astype(np.float32)
        if len(state) > target_dim:
            return state[:target_dim].astype(np.float32)
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[:len(state)] = state.astype(np.float32)
        return padded

    def predict_action(self, state: np.ndarray, checkpoint_name: Optional[str] = None) -> Dict:
        net = self.ensure_checkpoint_loaded(checkpoint_name)
        state_used = self._adapt_state(state)
        with torch.no_grad():
            state_tensor = torch.tensor(state_used, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = net(state_tensor).detach().cpu().numpy()[0]

        action_index = int(np.argmax(q_values))
        action_probabilities = F.softmax(torch.tensor(q_values), dim=-1).numpy()
        action_names = ["predict", "request_label"]
        return {
            "checkpoint_name": self.active_checkpoint_name,
            "architecture": self.active_checkpoint_architecture,
            "state_dim": self.active_state_dim,
            "state_vector": state.tolist(),
            "state_used": state_used.tolist(),
            "q_values": [float(value) for value in q_values],
            "action_probabilities": [float(value) for value in action_probabilities],
            "recommended_action": action_names[action_index] if action_index < len(action_names) else str(action_index),
            "recommended_action_index": action_index,
            "policy_confidence": float(action_probabilities[action_index]),
        }

    def build_state_for_sample(
        self,
        dataset_service,
        model_service,
        image_id: int,
        budget_remaining: Optional[int] = None,
        max_budget: int = 200,
        class_labels: Optional[List[str]] = None,
    ) -> Dict:
        if dataset_service.train_images is None:
            raise ValueError("Dataset must be loaded before building an RL state.")
        if image_id < 0 or image_id >= len(dataset_service.train_images):
            raise IndexError(f"image_id {image_id} is outside the loaded dataset.")

        image_tensor = torch.tensor(dataset_service.train_images[[image_id]], dtype=torch.float32)
        true_label = int(dataset_service.train_labels[image_id])
        prediction = model_service.predict_tensor(image_tensor, true_label=true_label, class_labels=class_labels)
        probs = np.array(prediction["probabilities"], dtype=np.float32)

        confidence = float(probs.max())
        entropy_val = float(-np.sum(probs * np.log(probs + 1e-10)))
        sorted_p = np.sort(probs)[::-1]
        margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0
        remaining = max_budget if budget_remaining is None else budget_remaining
        budget_ratio = float(remaining / max(max_budget, 1))
        auc_norm = float(model_service.latest_metrics.get("auc", 0.5) or 0.5)
        labeled_total = len(dataset_service.labeled_idx)
        pool_total = labeled_total + len(dataset_service.unlabeled_idx)
        labeled_ratio = float(labeled_total / (pool_total + 1e-6))
        step_ratio = float(1.0 - budget_ratio)

        if labeled_total > 0:
            pool_labels = dataset_service.train_labels[dataset_service.labeled_idx]
            counts = np.bincount(pool_labels, minlength=dataset_service.n_classes).astype(float)
            imbalance = float(counts.min() / (counts.max() + 1e-6)) if counts.max() > 0 else 1.0
        else:
            imbalance = 1.0

        ece_raw = model_service.compute_ece(dataset_service.val_loader, CONFIG.get("ece_n_bins", 10))
        ece_norm = min(ece_raw / 0.5, 1.0)
        state = np.array([
            confidence,
            entropy_val,
            margin,
            budget_ratio,
            auc_norm,
            labeled_ratio,
            step_ratio,
            imbalance,
            ece_norm,
        ], dtype=np.float32)

        self.latest_state = state
        return {
            "state": state,
            "prediction": prediction,
            "state_features": [
                "confidence",
                "entropy",
                "margin",
                "budget_ratio",
                "auc_norm",
                "labeled_ratio",
                "step_progress",
                "imbalance",
                "ece_norm",
            ],
        }

    async def train(self, episodes=10, learning_rate=None, gamma=None):
        if learning_rate:
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = learning_rate
        if gamma:
            CONFIG['gamma'] = gamma
            
        if self.annotation_service is None:
            raise Exception("Annotation service must be set before training to access environment.")
            
        results = []
        for ep in range(episodes):
            self.agent.anneal_beta(ep, episodes)
            state = self.annotation_service.reset_env()
            self.latest_state = state
            total_reward = 0
            done = False
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done = self.annotation_service.step_env(action)
                self.agent.replay.push(state, action, reward, next_state, float(done))
                self.agent.train_step()
                state = next_state
                self.latest_state = state
                total_reward += reward
            self.agent.decay_epsilon()
            results.append({"episode": ep+1, "total_reward": total_reward})
            
        return {"status": "success", "results": results}
