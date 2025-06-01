import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch
import gc  # for garbage collection
import time  # for evaluation delay
import matplotlib.pyplot as plt

# create training env without rendering
train_env = gym.make("LunarLander-v3")
# create separate eval env with human rendering
eval_env = gym.make("LunarLander-v3", render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# action space : discrete(4) \in {0, 1, 2, 3}
# observation space : Box(8,) : state is 8-dim vector: coords of lander in x & y, linear velocities in x & y, angle, angular velocity, and two bools that say if legs touching ground

def policy(observation, epsilon=0.1, actor_critic_network=None): # random sampling from action space
    if np.random.rand() < epsilon:
        return train_env.action_space.sample(), None
    else:
        action_probs, _ = actor_critic_network(observation)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [next_value]  # append V(s_{T+1})
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        gc.collect()  # force garbage collection
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # shared layers
        self.shared = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(64, 4),
            nn.Softmax(dim=-1)
        )
        
        # critic head (value function)
        self.critic_head = nn.Linear(64, 1)
    
    def forward(self, observation):
        shared_features = self.shared(observation)
        action_probs = self.actor_head(shared_features)
        v_value = self.critic_head(shared_features)
        return action_probs, v_value

actor_critic_network = ActorCriticNetwork().to(device)
target_network = ActorCriticNetwork().to(device)
target_network.load_state_dict(actor_critic_network.state_dict())

# modify training params for better memory management
gamma = 0.99
alpha = 0.0003
optimizer = torch.optim.Adam(actor_critic_network.parameters(), lr=alpha)

# training params
clip_epsilon = 0.2
ppo_epochs = 10
minibatch_size = 64
value_coef = 0.5
entropy_coef = 0.02
max_episodes = 1000  # train for 1000 episodes
gradient_accumulation_steps = 4

# epsilon schedule params
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
epsilon = epsilon_start

# training phase
print("Starting training phase for 1000 episodes...")
total_reward = 0
total_steps = 0
episodes = 0
episode_rewards = []
epsilons = []

while episodes < max_episodes:
    episodes += 1
    buffer = RolloutBuffer()
    observation, _ = train_env.reset()
    observation = torch.tensor(observation, dtype=torch.float32).to(device)
    done = False
    episode_steps = 0
    episode_reward = 0
    
    # collect rollout
    with torch.no_grad():  # disable gradient tracking during rollout
        while not done:
            action, log_prob = policy(observation, epsilon=epsilon, actor_critic_network=actor_critic_network)
            next_observation, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            next_observation = torch.tensor(next_observation, dtype=torch.float32).to(device)
            
            _, v_value = actor_critic_network(observation)
            total_reward += reward
            episode_reward += reward
            
            buffer.observations.append(observation)
            buffer.actions.append(action)
            buffer.rewards.append(reward)
            buffer.dones.append(done)
            if log_prob is not None:
                buffer.log_probs.append(log_prob)
            buffer.values.append(v_value)
            
            observation = next_observation
            episode_steps += 1
            total_steps += 1

    # process collected data
    rewards = buffer.rewards
    dones = buffer.dones
    values = [v.item() for v in buffer.values]
    
    with torch.no_grad():
        _, next_value = actor_critic_network(next_observation)
        next_value = next_value.item()
    
    advantages, returns = compute_gae(rewards, values, dones, next_value)
    
    # convert to tensors and move to device
    observations = torch.stack(buffer.observations).to(device)
    actions = torch.tensor(buffer.actions, dtype=torch.int64).to(device)
    log_probs = torch.stack(buffer.log_probs).to(device)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ppo training w gradient accumulation
    dataset_size = len(observations)
    optimizer.zero_grad()  # zero gradients at start of episode
    
    for _ in range(ppo_epochs):
        indices = torch.randperm(dataset_size)
        for start in range(0, dataset_size, minibatch_size):
            end = min(start + minibatch_size, dataset_size)
            mb_idx = indices[start:end]

            obs_mb = observations[mb_idx]
            actions_mb = actions[mb_idx]
            old_log_probs_mb = log_probs[mb_idx]
            advantages_mb = advantages[mb_idx]
            returns_mb = returns[mb_idx]

            # forward pass
            action_probs, values = actor_critic_network(obs_mb)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions_mb)
            entropy = dist.entropy().mean()

            # policy ratio
            ratio = torch.exp(new_log_probs - old_log_probs_mb)

            # ppo clipped objective
            unclipped = ratio * advantages_mb
            clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_mb
            actor_loss = -torch.min(unclipped, clipped).mean()

            # critic loss
            values = values.squeeze(-1)
            critic_loss = nn.MSELoss()(values, returns_mb)

            # total loss w gradient accumulation
            total_loss = (actor_loss + value_coef * critic_loss - entropy_coef * entropy) / gradient_accumulation_steps
            total_loss.backward()

            # update weights after accumulating gradients
            if (start // minibatch_size + 1) % gradient_accumulation_steps == 0 or end == dataset_size:
                torch.nn.utils.clip_grad_norm_(actor_critic_network.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()

    # clear memory
    del observations, actions, log_probs, advantages, returns
    buffer.clear()
    
    # update epsilon according to schedule
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    epsilons.append(epsilon)
    episode_rewards.append(episode_reward)
    
    print(f"Episode {episodes}/{max_episodes}, Steps: {episode_steps}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.4f}, Total Steps: {total_steps}")
    target_network.load_state_dict(actor_critic_network.state_dict())

# plot training rewards and epsilon decay
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(range(1, episodes + 1), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Rewards')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(range(1, episodes + 1), epsilons)
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay')
plt.grid(True)

plt.tight_layout()
plt.savefig('ppo_training_results.png')
plt.show()

print(f"Training completed after {episodes} episodes with a total of {total_steps} steps.")

# evaluation phase
print("\nStarting evaluation phase with rendering...")
eval_rewards = []

for eval_episode in range(20):
    observation, _ = eval_env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        observation_tensor = torch.tensor(observation, dtype=torch.float32).to(device)
        with torch.no_grad():
            action, _ = policy(observation_tensor, epsilon=0.0, actor_critic_network=actor_critic_network)
        
        observation, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        episode_reward += reward
        time.sleep(0.01)  # small delay to make rendering viewable
    
    eval_rewards.append(episode_reward)
    print(f"Evaluation Episode {eval_episode+1}/20, Reward: {episode_reward:.2f}")

print(f"\nEvaluation complete. Average reward over 20 episodes: {np.mean(eval_rewards):.2f}")

train_env.close()
eval_env.close()