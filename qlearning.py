import gymnasium as gym
import numpy as np

# 1. Start the game with a window you can actually watch
env = gym.make("MountainCar-v0", render_mode="human")

# 2. Load the file we just saved
q_table = np.load('mountain_car_brain.npy')

# 3. Use the EXACT same grid settings as your training script
DISCRETE_OS_SIZE = [18, 18]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(np.clip(discrete_state, 0, np.array(DISCRETE_OS_SIZE) - 1).astype(int))


# 4. Run the victory lap
state, _ = env.reset()
done = False

print("Watching the trained agent...")

while not done:
    discrete_state = get_discrete_state(state)

    # No more random moves! Always take the best action from the brain
    action = np.argmax(q_table[discrete_state])

    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

print("Mission Accomplished!")
env.close()