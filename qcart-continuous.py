import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, hidden_size=128, input_size=4, action_space=2):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), # Added an extra ReLU for depth
            nn.Linear(hidden_size, action_space)
        )
    
    def forward(self, x):
        return self.network(x)
    
ALPHA = 1e-3
GAMMA = 0.99
ITERATIONS = 1_000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

env = gym.make("CartPole-v1")
q_net = QNetwork()

epsilon = EPSILON_START
returns = []

optimizer = torch.optim.Adam(q_net.parameters(), lr=ALPHA)

for episode in range(ITERATIONS):
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    G = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        q_values = q_net(obs)
        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1])
        else:
            action = q_values.argmax().item()

        next_obs_raw, reward, terminated, truncated, info = env.step(action)
        next_obs = torch.tensor(next_obs_raw, dtype=torch.float32)

        with torch.no_grad():
            q_next = q_net.forward(torch.tensor(next_obs, dtype=torch.float32)).max().item()
            loss = (reward + (0 if terminated else GAMMA * q_next)) 

        q_current= q_net(obs)[action]
        loss = nn.MSELoss()(q_current, torch.tensor(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs
        G += reward
    
    returns.append(G)
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    if (episode + 1) % 100 == 0:
        avg_return = np.mean(returns[-100:])
        print(f"Episode {episode + 1}/{ITERATIONS}, Avg Return (last 100): {avg_return:.2f}, Epsilon: {epsilon:.3f}")

env.close()

# Run a final evaluation episode with video recording
print("\nRunning evaluation episode with video...")
eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
eval_env = gym.wrappers.RecordVideo(eval_env, "videos", name_prefix="final_eval")
obs, info = eval_env.reset()

obs = torch.tensor(obs, dtype=torch.float32)
G = 0
terminated = False
truncated = False
while not terminated and not truncated:
    q_values = q_net(obs)
    action = q_values.argmax().item()  # Greedy policy for evaluation
    next_obs_raw, reward, terminated, truncated, info = eval_env.step(action)
    obs = torch.tensor(next_obs_raw, dtype=torch.float32)
    G += reward
eval_env.close()