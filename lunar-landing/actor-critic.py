import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import time
import random
from collections import deque

# enable anomaly detection for debugging.
torch.autograd.set_detect_anomaly(True)

# create training environment without rendering
train_env = gym.make("LunarLander-v3")
# create separate evaluation environment with human rendering.
eval_env = gym.make("LunarLander-v3", render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# action space : discrete(4) \in {0, 1, 2, 3}
# observation space : Box(8,) : The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.

# separate Actor and Critic networks.
class ActorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)

class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
    
    def forward(self, state):
        return self.network(state)

def policy(observation, epsilon=0.1, actor_network=None, env=None):
    if np.random.rand() < epsilon:
        return env.action_space.sample(), None
    else:
        with torch.no_grad():
            action_probs = actor_network(observation)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob

# experience Replay Buffer.
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done, log_prob=None):
        self.buffer.append((state, action, reward, next_state, done, log_prob))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, log_probs = zip(*batch)
        return (torch.stack(states), 
                torch.tensor(actions, dtype=torch.int64), 
                torch.tensor(rewards, dtype=torch.float32), 
                torch.stack(next_states), 
                torch.tensor(dones, dtype=torch.float32),
                log_probs)
    
    def __len__(self):
        return len(self.buffer)

actor_network = ActorNetwork().to(device)
critic_network = CriticNetwork().to(device)
target_critic = CriticNetwork().to(device)
target_critic.load_state_dict(critic_network.state_dict())

gamma = 0.99
actor_lr = 0.0005
critic_lr = 0.001
actor_optimizer = torch.optim.Adam(actor_network.parameters(), lr=actor_lr)
critic_optimizer = torch.optim.Adam(critic_network.parameters(), lr=critic_lr)

# replay buffer setup.
replay_buffer = ReplayBuffer(capacity=10000)
batch_size = 64
min_replay_size = 500

# epsilon schedule
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
epsilon = epsilon_start

# training phase.
print("Starting training phase for 1000 episodes...")
episode_rewards = []
episode_numbers = []

for episodes in range(1000):
    observation, _ = train_env.reset()
    observation = torch.tensor(observation, dtype=torch.float32).to(device)
    episode_over = False
    total_reward = 0
    
    while not episode_over:
        action, log_prob = policy(observation, epsilon=epsilon, actor_network=actor_network, env=train_env)  
        next_observation, reward, terminated, truncated, _ = train_env.step(action)
        next_observation = torch.tensor(next_observation, dtype=torch.float32).to(device)
        episode_over = terminated or truncated
        
        # store transition in replay buffer
        replay_buffer.add(observation, action, reward, next_observation, float(episode_over), log_prob)
        
        # only start learning when we have enough samples.
        if len(replay_buffer) > min_replay_size:
            # sample a batch from replay buffer
            states, actions, rewards, next_states, dones, log_probs = replay_buffer.sample(batch_size)
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            
            # critic update
            critic_optimizer.zero_grad()
            
            # get current Q-values.
            q_values = critic_network(states)
            q_values_for_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # get target Q-values
            with torch.no_grad():
                target_q_values = target_critic(next_states)
                max_next_q = torch.max(target_q_values, dim=1)[0]
                target_q = rewards + gamma * max_next_q * (1 - dones)
            
            # compute critic loss.
            critic_loss = nn.MSELoss()(q_values_for_actions, target_q)
            
            # update critic
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_network.parameters(), 1.0)
            critic_optimizer.step()
            
            # actor update - only for non-random actions.
            valid_indices = [i for i, lp in enumerate(log_probs) if lp is not None]
            
            if valid_indices:
                actor_optimizer.zero_grad()
                
                # extract states and log_probs for valid actions
                valid_states = states[valid_indices]
                valid_log_probs = torch.stack([log_probs[i] for i in valid_indices])
                
                # get Q-values for valid states.
                with torch.no_grad():
                    valid_q_values = critic_network(valid_states)
                    valid_actions = actions[valid_indices]
                    valid_action_q_values = valid_q_values.gather(1, valid_actions.unsqueeze(1)).squeeze(1)
                
                # policy gradient loss
                actor_loss = -(valid_log_probs * valid_action_q_values).mean()
                
                # update actor.
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_network.parameters(), 1.0)
                actor_optimizer.step()
        
        observation = next_observation
        total_reward += reward
        
    # update epsilon according to schedule.
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    # soft update of target network
    if episodes % 10 == 0:
        for target_param, param in zip(target_critic.parameters(), critic_network.parameters()):
            target_param.data.copy_(0.05 * param.data + 0.95 * target_param.data)
    
    # store episode data for plotting
    episode_rewards.append(total_reward)
    episode_numbers.append(episodes)
        
    if episodes % 10 == 0:
        # calculate average reward for last 10 episodes
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        print(f"Episode {episodes}/1000, Total reward: {total_reward:.2f}, Average reward (last 10): {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

# plot the training rewards.
plt.figure(figsize=(10, 6))
plt.plot(episode_numbers, episode_rewards)
plt.xlabel('Episode Number')
plt.ylabel('Total Reward')
plt.title('Training Rewards over Episodes')
plt.grid(True)
plt.savefig('actor_critic_training_rewards.png')
plt.show()

print("Training completed. Starting evaluation with rendering...")

# evaluation phase
eval_rewards = []
for eval_episode in range(20):
    observation, _ = eval_env.reset()
    observation = torch.tensor(observation, dtype=torch.float32).to(device)
    episode_over = False
    total_reward = 0
    
    while not episode_over:
        action, _ = policy(observation, epsilon=0.0, actor_network=actor_network, env=eval_env)  
        next_observation, reward, terminated, truncated, _ = eval_env.step(action)
        next_observation = torch.tensor(next_observation, dtype=torch.float32).to(device)
        
        observation = next_observation
        episode_over = terminated or truncated
        total_reward += reward
        time.sleep(0.01)  # small delay to make rendering viewable

    eval_rewards.append(total_reward)
    print(f"Evaluation Episode {eval_episode+1}/20, Reward: {total_reward:.2f}")

print(f"Evaluation complete. Average reward over 20 episodes: {np.mean(eval_rewards):.2f}")

train_env.close()
eval_env.close()