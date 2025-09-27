import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import random

# Parameters
GRID_SIZE = 50
NUM_ANTS = 50
NUM_STEPS = 600
ALPHA_INIT = 1.0
BETA_INIT = 2.0
EVAPORATION_INIT = 0.1
PHEROMONE_DEPOSIT = 1.0
SAVE_VIDEO = False  # Set to True to save as MP4 (requires ffmpeg)

np.random.seed(0)

# Grid and pheromones
pheromone = np.ones((GRID_SIZE, GRID_SIZE))
heuristic = np.zeros((GRID_SIZE, GRID_SIZE))
goal = (GRID_SIZE - 2, GRID_SIZE - 2)
start = (1, 1)

for y in range(GRID_SIZE):
    for x in range(GRID_SIZE):
        dist = np.hypot(goal[0] - x, goal[1] - y)
        heuristic[y, x] = 1.0 / (dist + 1e-6)

# Ant class with trail memory
class Ant:
    def __init__(self):
        self.path = [start]
        self.reached_goal = False

    def move(self, alpha, beta):
        if self.reached_goal:
            return
        y, x = self.path[-1]
        neighbors = [(y + dy, x + dx) for dy in [-1, 0, 1]
                     for dx in [-1, 0, 1] if (dy != 0 or dx != 0)]
        neighbors = [(ny, nx) for ny, nx in neighbors
                     if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE]
        if not neighbors:
            return

        probs = []
        for ny, nx in neighbors:
            tau = pheromone[ny, nx] ** alpha
            eta = heuristic[ny, nx] ** beta
            probs.append(tau * eta)
        probs = np.array(probs)
        probs /= probs.sum()
        choice = neighbors[np.random.choice(len(neighbors), p=probs)]
        self.path.append(choice)
        if choice == goal:
            self.reached_goal = True

# Initialize ants
ants = [Ant() for _ in range(NUM_ANTS)]

# Visualization setup
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
pheromone_im = ax.imshow(pheromone, cmap='YlOrRd', vmin=0, vmax=5)
ant_dots, = ax.plot([], [], 'bo', markersize=4, label='Ants')
goal_dot, = ax.plot(goal[1], goal[0], 'ro', markersize=6, label='Goal')
ax.legend()

# Sliders
ax_alpha = plt.axes([0.15, 0.1, 0.65, 0.03])
ax_beta = plt.axes([0.15, 0.06, 0.65, 0.03])
ax_evap = plt.axes([0.15, 0.02, 0.65, 0.03])
s_alpha = Slider(ax_alpha, 'Alpha (pheromone)', 0.1, 5.0, valinit=ALPHA_INIT)
s_beta = Slider(ax_beta, 'Beta (heuristic)', 0.1, 5.0, valinit=BETA_INIT)
s_evap = Slider(ax_evap, 'Evaporation', 0.01, 0.5, valinit=EVAPORATION_INIT)

# Animation function
def update(frame):
    alpha = s_alpha.val
    beta = s_beta.val
    evap = s_evap.val

    pheromone[:] *= (1 - evap)

    ant_positions = []
    for ant in ants:
        ant.move(alpha, beta)
        y, x = ant.path[-1]
        ant_positions.append((x, y))
        if ant.reached_goal:
            for ay, ax in ant.path:
                pheromone[ay, ax] += PHEROMONE_DEPOSIT / len(ant.path)

    if ant_positions:
        xs, ys = zip(*ant_positions)
        ant_dots.set_data(xs, ys)
    pheromone_im.set_data(pheromone)
    return pheromone_im, ant_dots

ani = animation.FuncAnimation(
    fig, update, frames=NUM_STEPS, interval=100, blit=True, repeat=False
)

if SAVE_VIDEO:
    ani.save("aco_demo.mp4", writer="ffmpeg")

plt.show()

