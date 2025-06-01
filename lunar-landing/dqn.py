import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
import random
from collections import deque

# create training environment without rendering
train_env = gym.make("LunarLander-v3")
# Create separate evaluation environment with human rendering.
eval_env = gym.make("LunarLander-v3", render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def policy(observation, epsilon = 0.1, q_network = None, env = None): # random sampling from the action space.
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return max(enumerate([q_network(observation)[i] for i in range(4)]), key = lambda x: x[1])[0]

class Q_network_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, observation):
        return self.model(observation)

# experience replay buffer.
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states), 
                torch.tensor(actions, dtype=torch.int64), 
                torch.tensor(rewards, dtype=torch.float32), 
                torch.stack(next_states), 
                torch.tensor(dones, dtype=torch.float32))
    
    def __len__(self):
        return len(self.buffer)

q_network = Q_network_module().to(device)
q_target_network = Q_network_module().to(device)
q_target_network.load_state_dict(q_network.state_dict())

# hyperparameters.
gamma = 0.98
alpha = 0.0002
optimizer = torch.optim.Adam(q_network.parameters(), lr=alpha)
replay_buffer = ReplayBuffer(capacity=100000)
batch_size = 64
min_replay_size = 1000
target_update_freq = 10  # update target network every 10 episodes.

# epsilon schedule
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
epsilon = epsilon_start

# training phase.
print("Starting training phase for 1000 episodes...")
episode_rewards = []  # list to store rewards for plotting
episode_numbers = []  # List to store episode numbers for plotting.

for episodes in range(500):
    observation, _ = train_env.reset()
    observation = torch.tensor(observation, dtype=torch.float32).to(device)
    episode_over = False
    total_reward = 0
    
    while not episode_over:
        action = policy(observation, epsilon=epsilon, q_network=q_network, env=train_env)  
        next_observation, reward, terminated, truncated, _ = train_env.step(action) # the transition function.
        next_observation = torch.tensor(next_observation, dtype=torch.float32).to(device)
        episode_over = terminated or truncated
        
        # store transition in replay buffer
        replay_buffer.add(observation, action, reward, next_observation, float(episode_over))
        
        # only start learning when we have enough samples.
        if len(replay_buffer) > min_replay_size:
            # sample a batch from replay buffer
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            
            # compute target Q values.
            with torch.no_grad():
                next_q_values = q_target_network(next_states)
                max_next_q = torch.max(next_q_values, dim=1)[0]
                target_q_values = rewards + gamma * max_next_q * (1 - dones)
            
            # compute current Q values
            current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # compute loss
            loss = nn.MSELoss()(current_q, target_q_values)
            
            # optimize the model.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        observation = next_observation
        total_reward += reward
    
    # update epsilon according to schedule
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    # store episode data for plotting.
    episode_rewards.append(total_reward)
    episode_numbers.append(episodes)
    
    # update target network periodically
    if episodes % target_update_freq == 0:
        print(f"Episode {episodes}/500, Total reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")
        q_target_network.load_state_dict(q_network.state_dict())

# plot the training rewards
plt.figure(figsize=(10, 6))
plt.plot(episode_numbers, episode_rewards)
plt.xlabel('Episode Number')
plt.ylabel('Total Reward')
plt.title('Training Rewards over Episodes')
plt.grid(True)
plt.savefig('dqn_training_rewards.png')
plt.show()

print("Training completed. Starting evaluation with rendering...")

# evaluation phase.
eval_rewards = []
for eval_episode in range(10):
    observation, _ = eval_env.reset()
    observation = torch.tensor(observation, dtype=torch.float32).to(device)
    episode_over = False
    total_reward = 0
    
    while not episode_over:
        action = policy(observation, epsilon=0.0, q_network=q_network, env=eval_env)  
        next_observation, reward, terminated, truncated, _ = eval_env.step(action)
        next_observation = torch.tensor(next_observation, dtype=torch.float32).to(device)
        
        observation = next_observation
        episode_over = terminated or truncated
        total_reward += reward
        time.sleep(0.01)  # small delay to make rendering viewable.

    eval_rewards.append(total_reward)
    print(f"Evaluation Episode {eval_episode+1}/10, Reward: {total_reward:.2f}")

print(f"Evaluation complete. Average reward over 10 episodes: {np.mean(eval_rewards):.2f}")

train_env.close()
eval_env.close()