import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch

env = gym.make("HalfCheetah-v5", ctrl_cost_weight=0.0, render_mode="human")
# observation is a length 17 vector 
# action is a length 6 vector. [-1, 1]^6

def policy(observation, epsilon = 0.01):
    if np.random.rand() < epsilon:
        return np.array([2*np.random.rand() for i in range(6)])

    with torch.no_grad():
        action_logits, _ = actor_critic_network(observation)
    actions = 2 * torch.sigmoid(action_logits) - torch.tensor([1]*6)
    # actions are actually individual means of normal distributions in the range [-1, 1]. we can assume any fixed 
    return actions

class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.actor_net = nn.Linear(64, 6)
        self.critic_net = nn.Linear(64,1)
    def forward(self, observation):
        x = self.shared_net(observation)
        return self.actor_net(x), self.critic_net(x) # action logits, q-value
    
actor_critic_network = ActorCriticNetwork()
critic_criterion = nn.MSELoss()

#---------------
gamma = 0.99
alpha = 3e-3
#---------------

observation, _ = env.reset()
optimizer = torch.optim.Adam(actor_critic_network.parameters(), lr = alpha)
while True:
    action = policy(observation)
    next_observation, reward, terminated, truncated, info = env.step(action)
    _, q_value_prev = actor_critic_network(observation)
    _ , q_value = actor_critic_network(next_observation)
    q_target =  reward + gamma * q_value_prev
    delta =  q_target - q_value
    
    # optimizing actor ---------------------
    

    # optimizing critic --------------------
    critic_loss = critic_criterion(q_value, q_target)
    optimizer.zero_grad()
    critic_loss.backward()
    optimizer.step()

    observation = next_observation