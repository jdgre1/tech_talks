import numpy as np
import matplotlib.pyplot as plt

# Objective function
def objective(pos):
    return pos[0]**2 + pos[1]**2  # Simple paraboloid (minimum at 0,0)

# PSO settings
NUM_PARTICLES = 30
DIMENSIONS = 2
ITERATIONS = 100
W = 0.9      # increase inertia (promotes exploration)
C1 = 1.9    # reduce cognitive (less personal pull)
C2 = 1.0     # reduce social (less global pull)

# Initialize particles
positions = np.random.uniform(-10, 10, (NUM_PARTICLES, DIMENSIONS))
velocities = np.random.uniform(-1, 1, (NUM_PARTICLES, DIMENSIONS))

p_best = positions.copy()
p_best_scores = np.array([objective(p) for p in positions])
g_best = p_best[np.argmin(p_best_scores)]

# For plotting
plt.ion()
fig, ax = plt.subplots()
MAX_VELOCITY = 2.0  # or even lower like 0.5
try:
    for iteration in range(ITERATIONS):
        for i in range(NUM_PARTICLES):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (W * velocities[i] +
                            C1 * r1 * (p_best[i] - positions[i]) +
                            C2 * r2 * (g_best - positions[i]))

            velocities[i] = np.clip(velocities[i], -MAX_VELOCITY, MAX_VELOCITY)
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], -10, 10)
            # Evaluate
            score = objective(positions[i])
            if score < p_best_scores[i]:
                p_best[i] = positions[i]
                p_best_scores[i] = score
                if score < objective(g_best):
                    g_best = positions[i]

        # Visualization
        ax.clear()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_title(f"Iteration {iteration} | Best: {objective(g_best):.4f}")
        ax.scatter(positions[:, 0], positions[:, 1], color='blue')
        ax.scatter(g_best[0], g_best[1], color='red', marker='*', s=200)
        plt.pause(0.5)

    plt.ioff()
    plt.show()

except KeyboardInterrupt:
    print("\nSimulation terminated by user (Ctrl+C).")
    plt.close()
