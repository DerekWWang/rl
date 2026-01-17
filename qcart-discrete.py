import numpy as np
import gymnasium as gym

X_MAX = 2.4
X_MIN = -2.4
V_MAX = 3.0
V_MIN = -3.0
A_MAX = 12
A_MIN = -12
PAV_MAX = 50
PAV_MIN = -50

NUM_BINS = 50

POSITIONAL_STATES = np.linspace(X_MIN, X_MAX, NUM_BINS)
VELOCITY_STATES = np.linspace(V_MIN, V_MAX, NUM_BINS)
POLE_ANGLE_STATES = np.linspace(A_MIN, A_MAX, NUM_BINS)
POLE_ANGVL_STATES = np.linspace(PAV_MIN, PAV_MAX, NUM_BINS)

ACTION_VALUES = [0, 1]

Q_TABLE = {}

ALPHA = 1e-3
GAMMA = 0.99
ITERATIONS = 100_000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Initialize Q-table with discretized state indices (0 to NUM_BINS inclusive)
for p_idx in range(NUM_BINS + 1):
    for v_idx in range(NUM_BINS + 1):
        for pa_idx in range(NUM_BINS + 1):
            for pav_idx in range(NUM_BINS + 1):
                Q_TABLE[(p_idx, v_idx, pa_idx, pav_idx)] = {0: 0.0, 1: 0.0}

def digitize(raw_state: tuple):
    p, v, pa, pav = raw_state
    # Clip values to avoid out-of-bounds indices
    p_index = np.clip(np.digitize(p, POSITIONAL_STATES), 0, NUM_BINS)
    v_index = np.clip(np.digitize(v, VELOCITY_STATES), 0, NUM_BINS)
    pa_index = np.clip(np.digitize(pa, POLE_ANGLE_STATES), 0, NUM_BINS)
    pav_index = np.clip(np.digitize(pav, POLE_ANGVL_STATES), 0, NUM_BINS)

    return (p_index, v_index, pa_index, pav_index)

def q_update(cur_state: tuple, cur_action: int, cur_reward: float, raw_next_state: tuple):
    cur_q = Q_TABLE[cur_state][cur_action]
    next_state = digitize(raw_next_state)

    next_q = max(Q_TABLE[next_state][0], Q_TABLE[next_state][1])
    new_q = cur_q + ALPHA * (cur_reward + GAMMA * next_q - cur_q)
    Q_TABLE[cur_state][cur_action] = new_q
    return new_q

def epsilon_greedy_policy(state: tuple, epsilon: float):
    if np.random.rand() < epsilon:
        return np.random.choice(ACTION_VALUES)
    else:
        q_values = Q_TABLE[state]
        return max(q_values, key=q_values.get)

# Create environment (without video recording for training)
env = gym.make("CartPole-v1")

epsilon = EPSILON_START
returns = []

for episode in range(ITERATIONS):
    obs, info = env.reset()
    G = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        state = digitize(obs)
        action = epsilon_greedy_policy(state, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        q_update(state, action, reward, next_obs)
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
G = 0
terminated = False
truncated = False

while not terminated and not truncated:
    state = digitize(obs)
    action = epsilon_greedy_policy(state, 0.0)  # Greedy policy for evaluation
    obs, reward, terminated, truncated, info = eval_env.step(action)
    G += reward

eval_env.close()
print(f"Evaluation episode return: {G}")