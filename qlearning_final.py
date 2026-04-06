import gymnasium as gym
import numpy as np

# Increase steps for discovery
env = gym.make("MountainCar-v0", max_episode_steps=1500)

LEARNING_RATE = 0.5
DISCOUNT_RATE = 0.99
EPISODES = 50

# Keep epsilon alive for 45 episodes to prevent the -0.18 deadlock
epsilon = 1.0
epsilon_decay = 1.0 / 45

# Small grid (18x18) to ensure the reward spreads quickly
DISCRETE_OS_SIZE = [18, 18]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Initialize Q-table
q_table = np.zeros((DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(np.clip(discrete_state, 0, np.array(DISCRETE_OS_SIZE) - 1).astype(int))


for episode in range(EPISODES):
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    max_pos = -1.2

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = env.action_space.sample()

        new_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if new_state[0] > max_pos: max_pos = new_state[0]

        # --- MOMENTUM REWARD SHAPING ---
        # Reward is based on how well the car uses its current momentum
        # Moving right (+) with positive velocity (+) = High Reward
        # Moving left (-) with negative velocity (-) = High Reward
        position_from_center = new_state[0] - (-0.5)
        velocity = new_state[1]

        # This formula rewards the car for pushing in the direction of its swing
        reward = (position_from_center * velocity) * 1000

        # Add a small "Height" bonus to keep it climbing
        reward += (position_from_center ** 2) * 10

        if new_state[0] >= 0.5:
            reward += 10000
            done = True

        new_discrete_state = get_discrete_state(new_state)

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_RATE * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state

    epsilon = max(0.01, epsilon - epsilon_decay)  # Never let exploration hit true 0
    print(f"Try {episode}: Best Pos = {max_pos:.2f}, Epsilon = {epsilon:.2f}")
    if max_pos >= 0.5: print(">>> FLAG CAPTURED! <<<")

env.close()

np.save('mountain_car_brain.npy', q_table)
print("Brain saved as 'mountain_car_brain.npy'!")

