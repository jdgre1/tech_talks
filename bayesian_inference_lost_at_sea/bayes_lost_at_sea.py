import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns


class BayesLostAtSea:

    def __init__(self, grid_cells=40, n_paths=1000, n_steps=100,
        drift_velocity=np.array([0.05, 0.01]), noise_std=0.1):
        self.grid_cells_ = grid_cells
        self.n_paths_ = n_paths
        self.n_steps_ = n_steps
        self.drift_velocity_ = drift_velocity
        self.noise_std_ = noise_std
        self.detection_probability = 0.9
        self.start_pos_ = np.array([grid_cells / 20, 4 * grid_cells / 23])  # Approx (2, 7)
        self.prior_grid_ = np.full((grid_cells, grid_cells), 1.0 / (grid_cells * grid_cells))
        self.posterior_grid_ = self.prior_grid_.copy()
        self.all_endpoints_ = []
        self.samples = []

        self.resolution_m = 25                 # 25m per cell
        self.ocean_current = (10, -5)          # Example: 10m east, 5m south per timestep
        self.drift_std = 0.5                   # Standard deviation of random drift (in grid units)
        self.grid_size = (40, 40)              # 1km × 1km / 25m = 40x40 grid
        self.true_pos = (20.0, 20.0)           # Initial position in grid coordinates
        self.particles = []
        # Assume target is initialized here or set later in the simulation
        self.true_target_grid = (int(self.grid_size[0] // 2), int(self.grid_size[1] // 2))  # Just an example
        # This would be the center of the grid or set to a random location

    def initialize_particles(self, num_particles=1000):
        # Example: initialize uniformly across the grid
        self.particles = [
            (np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size))
            for _ in range(num_particles)
        ]

    def update_posterior(self, search_cells, found_at=None):
        """
        Update the posterior grid using Bayesian update based on where the person was searched and whether they were found.
        """
        likelihood = self.compute_likelihood(self.posterior_grid_.shape, found_at)

        # Only update the searched cells with the likelihood
        for (x, y) in search_cells:
            self.posterior_grid_[x, y] *= likelihood[x, y]

        # Normalize
        total = np.sum(self.posterior_grid_)
        if total > 0:
            self.posterior_grid_ /= total
        else:
            # Handle degenerate case where belief collapses
            self.posterior_grid_ = np.ones_like(self.posterior_grid_) / self.posterior_grid_.size

    @staticmethod
    def plot_prior_grid(prior_grid, start_pos, end_pos, actual_drift_path, title='Prior Probability Grid', show_heatmap=True, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        if show_heatmap:
            sns.heatmap(prior_grid, cmap='YlGnBu', square=True, cbar_kws={'label': 'Probability'}, ax=ax)
        ax.scatter(*start_pos, color='green', label='Start Pos', zorder=5)
        ax.plot(actual_drift_path[:, 0], actual_drift_path[:, 1], color='blue', label='Actual Drift Path', linewidth=2)
        ax.scatter(*end_pos, color='yellow', label='End Pos', zorder=5)
        ax.set_title(title)
        ax.set_xlabel('X (East)')
        ax.set_ylabel('Y (North)')
        ax.invert_yaxis()  # Origin at bottom-left
        return ax

    def drift_target(self):
        """
        Apply stochastic drift to the true position of the target.
        Uses ocean current + random Gaussian motion.
        """
        # Current in meters per time step (assume pre-defined)
        dx_current, dy_current = self.ocean_current

        # Convert meters to grid cells (25m resolution)
        dx_grid = dx_current / self.resolution_m
        dy_grid = dy_current / self.resolution_m

        # Add stochastic Gaussian noise
        noise_x = np.random.normal(0, self.drift_std)
        noise_y = np.random.normal(0, self.drift_std)

        # Update position
        x, y = self.true_pos
        x_new = np.clip(x + dx_grid + noise_x, 0, self.grid_size[0] - 1)
        y_new = np.clip(y + dy_grid + noise_y, 0, self.grid_size[1] - 1)
        self.true_pos = (x_new, y_new)

    @staticmethod
    def simulate_drift_endpoints(grid_cells=None, start_pos = np.array([2.0,7.0]), n_paths=1000, n_steps=100,
                                drift_velocity=np.array([0.05, 0.01]), noise_std=0.1):
        """
        Simulates endpoints of drifting objects over multiple paths.

        Parameters:
            grid_cells (int): Size of the square grid (grid_cells x grid_cells).
            n_paths (int): Number of simulated drift paths.
            n_steps (int): Number of steps in each path.
            drift_velocity (np.ndarray): Mean drift per step (east, north).
            noise_std (float): Standard deviation of random motion per step.

        Returns:
            list of tuple: Rounded final cell positions (row, col) of each simulated path.
        """
        all_endpoints = []

        for _ in range(n_paths):
            pos = start_pos.copy()
            for _ in range(n_steps):
                noise = np.random.normal(0, noise_std, size=2)
                pos += drift_velocity + noise
                pos = np.clip(pos, 0, grid_cells - 1)
            end_cell = tuple(np.round(pos).astype(int))
            all_endpoints.append(end_cell)

        return all_endpoints
    
    @staticmethod
    def create_prior_grid(grid_size=1000, resolution=25, all_endpoints=[]):
        n_cells = grid_size // resolution
       # Convert to DataFrame for counting
        df = pd.DataFrame(all_endpoints, columns=['x', 'y'])
        prior_counts = df.value_counts().reset_index(name='count')

        # Build prior probability grid
        prior_grid = np.zeros((n_cells, n_cells))
        for _, row in prior_counts.iterrows():
            prior_grid[row['y'], row['x']] = row['count']
        prior_grid = prior_grid / np.sum(prior_grid)

        return prior_grid
    
    def compute_likelihood(self, searched_cells, detection_radius=1, detection_prob=0.9, false_alarm_prob=0.01):
        """
        Computes a likelihood grid based on whether the true target lies within any of the searched zones.

        Parameters:
        - searched_cells: list of (x, y) tuples where a search occurred
        - detection_radius: radius around the true target where detection is likely
        - detection_prob: probability of detecting target if it's within radius
        - false_alarm_prob: probability of false detection elsewhere

        Returns:
        - likelihood_grid: 2D array same size as posterior_grid_
        """
        likelihood_grid = np.full(self.posterior_grid_.shape, false_alarm_prob)

        tx, ty = self.true_target_grid  # assume this is set at initialization or updated
        for x, y in searched_cells:
            dist = np.hypot(tx - x, ty - y)
            if dist <= detection_radius:
                likelihood_grid[x, y] = detection_prob

        return likelihood_grid

    @staticmethod
    def generate_actual_path(grid_cells=None, start_pos = np.array([2.0,7.0]), n_steps=100,
                                 drift_velocity=np.array([0.05, 0.01]), noise_std=0.1):
        # Generate the actual path (for illustration)
        actual_path = [start_pos.copy()]
        pos = start_pos.copy()
        for _ in range(n_steps):
            noise = np.random.normal(0, noise_std, size=2)
            pos += drift_velocity + noise
            pos = np.clip(pos, 0, grid_cells - 1)
            actual_path.append(pos.copy())
        actual_path = np.array(actual_path)
        end_pos =  actual_path[-1]
        return actual_path, end_pos
   
    @staticmethod
    def plot_posterior_with_zones(ax, posterior_grid, all_zone_centers, zone_centers, radius, title=None):
        """
            Plots posterior heatmap and overlays all previous and current search zones.

            Parameters:
            ax (matplotlib.axes.Axes): Axis to plot on.
            posterior_grid (np.ndarray): Grid of posterior probabilities.
            all_zone_centers (list of (int, int)): All previously searched zone centers.
            zone_centers (list of (int, int)): Current round's zone centers.
            radius (int): Search zone radius.
            title (str, optional): Plot title.
        """
        rect_width = 2 * radius + 1
        rect_height = 2 * radius + 1

        sns.heatmap(posterior_grid, cmap='YlGnBu', square=True, cbar_kws={'label': 'Posterior Probability'}, ax=ax)

        # Faint grey for all past zones
        for x, y in all_zone_centers:
            rect = patches.Rectangle((y - radius + 0.5, x - radius + 0.5), rect_width, rect_height,
                                    linewidth=1, edgecolor='lightgrey', facecolor='none')
            ax.add_patch(rect)

        # Red for current zones
        for x, y in zone_centers:
            rect = patches.Rectangle((y - radius + 0.5, x - radius + 0.5), rect_width, rect_height,
                                    linewidth=1.5, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        ax.invert_yaxis()
        if title:
            ax.set_title(title)
        ax.legend()

    def compute_prior_from_particles(self):
        grid = np.zeros(self.prior_grid_.shape)

        # Round and convert particles to integer grid indices
        particle_indices = np.rint(self.particles).astype(int)

        # Filter out-of-bounds indices
        mask = (
            (particle_indices[:, 0] >= 0) & (particle_indices[:, 0] < grid.shape[0]) &
            (particle_indices[:, 1] >= 0) & (particle_indices[:, 1] < grid.shape[1])
        )
        valid_indices = particle_indices[mask]

        for x, y in valid_indices:
            grid[x, y] += 1

        total = grid.sum()
        if total > 0:
            grid /= total
        else:
            grid = np.ones(grid.shape) / np.prod(grid.shape)

        return grid
        
    def propagate_particles(self):
        """
        Propagate each particle forward in time using the same model as the true target:
        ocean current drift + Gaussian noise.
        """
        new_particles = []

        dx_current, dy_current = self.ocean_current
        dx_grid = dx_current / self.resolution_m
        dy_grid = dy_current / self.resolution_m

        for x, y in self.particles:
            noise_x = np.random.normal(0, self.drift_std)
            noise_y = np.random.normal(0, self.drift_std)

            x_new = np.clip(x + dx_grid + noise_x, 0, self.grid_size[0] - 1)
            y_new = np.clip(y + dy_grid + noise_y, 0, self.grid_size[1] - 1)

            new_particles.append((x_new, y_new))

        self.particles = new_particles