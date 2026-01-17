from collections import deque
import numpy as np
import torch.nn as nn
import torch
import gymnasium as gym

class ReplayBuffer:
    def __init__(self, size: int):
        self.buffer = deque(maxlen=size)
        self.max_size = size

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, batch_size=32):
        n = len(self.buffer)
        bs = min(batch_size, n)
        idxs = np.random.randint(0, n, size=bs)
        return [self.buffer[i] for i in idxs]
    
class QNetwork(nn.Module):
    def __init__(self, hidden_size=128, input_size=4, action_space=2):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space)
        )
    
    def forward(self, x):
        return self.network(x)

criterion = nn.MSELoss()
buffer = ReplayBuffer(256)
core_model = QNetwork()

target_model = QNetwork()
target_model.load_state_dict(core_model.state_dict())

ALPHA = 1e-3
GAMMA = 0.99
ITERATIONS = 500
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

env = gym.make("CartPole-v1")

epsilon = EPSILON_START
returns = []

optimizer = torch.optim.Adam(core_model.parameters(), lr=ALPHA)


for episode in range(ITERATIONS):
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)

    G = 0
    terminated = False
    truncated = False

    num_steps = 0

    while not terminated and not truncated:
        q_values = core_model(obs)
        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1])
        else:
            action = q_values.argmax().item()
    
        next_obs_raw, reward, terminated, truncated, info = env.step(action)
        next_obs = torch.tensor(next_obs_raw, dtype=torch.float32)

        buffer.store(obs, action, reward, next_obs, terminated or truncated)

        samples = buffer.sample_batch()
        with torch.no_grad():
            y_list = []
            for sample in samples:
                s, a, r, s_, d = sample
                q_next = target_model(s_).max()
                y = torch.tensor(r, dtype=torch.float32) + (0.0 if d else GAMMA) * q_next
                y_list.append(y)
            y_batch = torch.stack(y_list)

        q_list = []
        for (s, a, r, s_, d) in samples:
            q_sa = core_model(s)[a]                             # tensor scalar
            q_list.append(q_sa)
        q_batch = torch.stack(q_list) 

        loss = criterion(q_batch, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs
        G += reward

        num_steps += 1
        if num_steps % 32 == 0:
            target_model.load_state_dict(core_model.state_dict())

    returns.append(G)
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    if (episode + 1) % 100 == 0:
        avg_return = np.mean(returns[-100:])
        print(f"Episode {episode + 1}/{ITERATIONS}, Avg Return (last 100): {avg_return:.2f}, Epsilon: {epsilon:.3f}")

env.close()


print("\nRunning evaluation episode with video...")
eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
eval_env = gym.wrappers.RecordVideo(eval_env, "dqn/videos", name_prefix="final_eval")
obs, info = eval_env.reset()

obs = torch.tensor(obs, dtype=torch.float32)
G = 0
terminated = False
truncated = False
while not terminated and not truncated:
    q_values = core_model(obs)
    action = q_values.argmax().item()  # Greedy policy for evaluation
    next_obs_raw, reward, terminated, truncated, info = eval_env.step(action)
    obs = torch.tensor(next_obs_raw, dtype=torch.float32)
    G += reward
eval_env.close()