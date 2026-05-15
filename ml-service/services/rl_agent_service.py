import random
import collections
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

    def set_annotation_service(self, annotation_service):
        self.annotation_service = annotation_service

    def get_state(self):
        return {
            "epsilon": self.agent.epsilon,
            "beta": self.agent.beta,
            "replay_buffer_size": len(self.agent.replay),
            "step_count": self.agent.step_count,
            "latest_loss": self.agent.losses[-1] if self.agent.losses else 0.0,
            "latest_state_vector": self.latest_state.tolist() if self.latest_state is not None else []
        }

    def get_policy(self):
        return {
            "type": "DuelingDQN",
            "state_dim": CONFIG['state_dim'],
            "action_dim": CONFIG['action_dim']
        }

    def reset(self):
        self.agent = DQNAgent(CONFIG['state_dim'], CONFIG['action_dim'], self.device)
        self.latest_state = None

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
