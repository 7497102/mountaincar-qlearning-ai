import matplotlib.pyplot as plt
from qlearning_final import q_table, np, env
# 1. Collapse the 3 actions into 1 "Best Score"
# We don't care about the bad moves, we only want to see the score of the BEST move in every cell.
best_q_values = np.max(q_table, axis=2)

# 2. Set up the Plot
plt.figure(figsize=(10, 8))
plt.title("The Car's Brain: Q-Value Heatmap")
plt.xlabel("Car Position (Left <---> Right)")
plt.ylabel("Car Velocity (Moving Left <---> Moving Right)")

# 3. Draw the Heatmap
# We use .T (Transpose) to flip the axes so Position is on the bottom (X)
plt.imshow(best_q_values.T, origin='lower', cmap='viridis', aspect='auto',
           extent=[env.observation_space.low[0], env.observation_space.high[0],
                   env.observation_space.low[1], env.observation_space.high[1]])

plt.colorbar(label="Confidence Score (Q-Value)")
plt.show()