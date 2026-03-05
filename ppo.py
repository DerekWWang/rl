import numpy as np
import torch.nn as nn
import torch
import gymnasium as gym

from torch.distributions import Categorical

# VARIABLES
OBSERVATION_DIM = 4
ACTION_DIM = 2

NUM_ENVS = 32
TRAJECTORY_WINDOW = 500

CLIP = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
EPOCHS = 10
MB_SIZE = 256

GAMMA = 0.99
LAMBDA = 0.95

NUM_EPISODES = 25

### Model
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        # "new actor"
        self.target_actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        shared_out = self.shared(obs)
        action_probs = self.actor(shared_out)
        value = self.critic(shared_out)
        return action_probs, value

vecenv =  gym.make_vec("CartPole-v1", num_envs=NUM_ENVS, vectorization_mode="sync")
acmodule = ActorCritic(OBSERVATION_DIM, ACTION_DIM, 128)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(acmodule.parameters(), lr=3e-4)

for episode in range(NUM_EPISODES):
    print(f"StartingEpisode {episode+1}/{NUM_EPISODES}")
    obs, info = vecenv.reset()

    obs_buf      = torch.zeros(TRAJECTORY_WINDOW, NUM_ENVS, OBSERVATION_DIM)
    act_buf      = torch.zeros(TRAJECTORY_WINDOW, NUM_ENVS, dtype=torch.long)
    rew_buf      = torch.zeros(TRAJECTORY_WINDOW, NUM_ENVS)
    done_buf     = torch.zeros(TRAJECTORY_WINDOW, NUM_ENVS)
    logp_buf     = torch.zeros(TRAJECTORY_WINDOW, NUM_ENVS)
    val_buf      = torch.zeros(TRAJECTORY_WINDOW, NUM_ENVS)

    obs_np, _ = vecenv.reset()
    obs = torch.from_numpy(obs_np).float()

    for t in range(TRAJECTORY_WINDOW):
        obs_buf[t] = obs

        with torch.no_grad():
            logits = acmodule.target_actor(obs) # N, act_dim
            values = acmodule.critic(obs).squeeze(-1) # N
            dist = Categorical(logits=logits)

            actions = dist.sample() # N
            logp = dist.log_prob(actions) # N

        act_buf[t] = actions
        logp_buf[t] = logp
        val_buf[t] = values
        
        next_obs_np, rewards, terminations, truncations, infos = vecenv.step(actions.numpy())
        dones = np.logical_or(terminations, truncations)

        rew_buf[t] = torch.tensor(rewards, dtype=torch.float32)
        done_buf[t] = torch.tensor(dones.astype(np.float32))

        obs = torch.tensor(next_obs_np, dtype=torch.float32)

    with torch.no_grad():
        last_values = acmodule.critic(obs).squeeze(-1) # ZN
    adv_buf = torch.zeros_like(rew_buf)
    ret_buf = torch.zeros_like(rew_buf)

    gae = torch.zeros(NUM_ENVS)
    for t in reversed(range(TRAJECTORY_WINDOW)):
        nonterminal = 1.0 - done_buf[t]
        next_values  = last_values if t == TRAJECTORY_WINDOW - 1 else val_buf[t+1]
        delta = rew_buf[t] + GAMMA * next_values * nonterminal - val_buf[t]
        gae = delta + GAMMA * LAMBDA * nonterminal * gae
        adv_buf[t] = gae
        ret_buf[t] = adv_buf[t] + val_buf[t]

    adv_flat = adv_buf.reshape(-1)
    adv_buf = (adv_buf - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    B = TRAJECTORY_WINDOW * NUM_ENVS
    obs_flat  = obs_buf.reshape(B, OBSERVATION_DIM)
    act_flat  = act_buf.reshape(B)
    logp_old  = logp_buf.reshape(B)
    adv_flat  = adv_buf.reshape(B)
    ret_flat  = ret_buf.reshape(B)
    val_old   = val_buf.reshape(B)

    idxs = torch.arange(B)

    for _ in range(EPOCHS):
        perm = idxs[torch.randperm(B)]
        for start in range(0, B, MB_SIZE):
            mb = perm[start:start+MB_SIZE]

            logits = acmodule.target_actor(obs_flat[mb])
            values = acmodule.critic(obs_flat[mb]).squeeze(-1)
            dist = Categorical(logits=logits)

            # new probabilities
            logp = dist.log_prob(act_flat[mb])
            entropy = dist.entropy().mean()

            # update the critic network
            critic_loss = criterion(values, ret_flat[mb])

            # update the actor network
            ratio = torch.exp(logp - logp_old[mb])
            surrogate_left = ratio * adv_flat[mb] # size: MB
            surrogate_right = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * adv_flat[mb]

            actor_loss = -torch.min(surrogate_left, surrogate_right).mean()
            loss = actor_loss + VF_COEF * critic_loss - ENT_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

print("\nRunning evaluation episode with video...")
eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
eval_env = gym.wrappers.RecordVideo(eval_env, "ppo/videos", name_prefix="final_eval")
obs, info = eval_env.reset()

obs = torch.tensor(obs, dtype=torch.float32)
G = 0
terminated = False
truncated = False
while not terminated and not truncated:
    with torch.no_grad():
        q_values = acmodule.target_actor(obs)
        action = q_values.argmax().item()  # Greedy policy for evaluation
        next_obs_raw, reward, terminated, truncated, info = eval_env.step(action)
        obs = torch.tensor(next_obs_raw, dtype=torch.float32)
        G += reward
eval_env.close()