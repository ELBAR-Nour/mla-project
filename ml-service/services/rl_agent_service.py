"""RL Agent Service for DQN-based annotation decisions"""
import logging
from typing import Dict, List, Optional
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

class DQNNetwork(nn.Module):
    """Deep Q-Network for annotation decisions"""
    
    def __init__(self, state_size: int = 4, action_size: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RLAgentService:
    """RL Agent for learning annotation strategy"""
    
    def __init__(self, device: torch.device, state_size: int = 4, action_size: int = 2):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size  # 0: predict, 1: request label
        
        # DQN components
        self.q_network = DQNNetwork(state_size, action_size).to(device)
        self.target_network = DQNNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99  # Discount factor
        
        # Statistics
        self.episode_rewards = []
        self.episode_queries = []
        
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action (epsilon-greedy)"""
        if training and np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.max(1)[1].item()
    
    def replay(self, batch_size: int = 32):
        """Train on mini-batch from replay buffer"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q-learning update
        with torch.no_grad():
            target_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * target_q_values
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        loss = self.criterion(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_episode(self, env_step_fn, max_steps: int = 100) -> Dict:
        """Train for one episode"""
        state = np.random.rand(self.state_size)  # Random initial state
        total_reward = 0
        total_queries = 0
        
        for step in range(max_steps):
            action = self.act(state, training=True)
            reward, next_state, done = env_step_fn(action)
            
            self.remember(state, action, reward, next_state, done)
            loss = self.replay(batch_size=32)
            
            total_reward += reward
            if action == 1:  # Requesting label
                total_queries += 1
            
            state = next_state
            
            if done:
                break
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.episode_rewards.append(total_reward)
        self.episode_queries.append(total_queries)
        
        return {
            "episode_reward": total_reward,
            "episode_queries": total_queries,
            "epsilon": self.epsilon,
        }
    
    def get_policy(self, num_states: int = 1000) -> Dict:
        """Get learned policy for various states"""
        policy = {}
        
        self.q_network.eval()
        with torch.no_grad():
            for i in range(min(num_states, 100)):
                state = np.random.rand(self.state_size)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
                
                policy[str(i)] = {
                    "state": state.tolist(),
                    "q_values": q_values.tolist(),
                    "best_action": int(np.argmax(q_values)),
                    "action_names": ["predict", "request_label"],
                }
        
        return {
            "policy_samples": policy,
            "total_episodes": len(self.episode_rewards),
        }
    
    def get_state(self) -> Dict:
        """Get current agent state"""
        return {
            "epsilon": float(self.epsilon),
            "episodes_trained": len(self.episode_rewards),
            "avg_reward": float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0,
            "avg_queries": float(np.mean(self.episode_queries[-100:])) if self.episode_queries else 0,
            "memory_size": len(self.memory),
        }
    
    def reset(self):
        """Reset agent"""
        self.q_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory.clear()
        self.epsilon = 1.0
        self.episode_rewards = []
        self.episode_queries = []
        logger.info("✅ RL agent reset")
