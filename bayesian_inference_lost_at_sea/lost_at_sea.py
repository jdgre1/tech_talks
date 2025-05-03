import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns



class LostAtSea:

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
    def get_top_k_cells(prior_grid, k=10):
        flat_indices = np.argsort(prior_grid.ravel())[::-1]
        top_k = np.column_stack(np.unravel_index(flat_indices[:k], prior_grid.shape))
        return [tuple(cell) for cell in top_k]

    @staticmethod
    def get_top_k_cells_with_radius(posterior_grid, k=10, radius=1):
        # Flatten and sort the indices
        '''
            This function get_top_k_cells_with_radius takes in a 2D probability grid (posterior_grid) 
            and returns a list of cell coordinates that correspond to the top k highest-probability regions, 
            each expanded by a square neighborhood defined by a given radius
            Example: radius=1 gives a 3×3 neighborhood (including the center cell).
        '''
         
        flat_indices = np.dstack(np.unravel_index(np.argsort(posterior_grid.ravel())[::-1], posterior_grid.shape))[0]

        selected_cells = []
        seen_cells = set()

        for center in flat_indices:
            x_c, y_c = center
            local_cells = []

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    x, y = x_c + dx, y_c + dy
                    if 0 <= x < posterior_grid.shape[0] and 0 <= y < posterior_grid.shape[1]:
                        if (x, y) not in seen_cells:
                            local_cells.append((x, y))
                            seen_cells.add((x, y))

            selected_cells.extend(local_cells)
            if len(seen_cells) >= k * (2 * radius + 1) ** 2:
                break

        return selected_cells

    def bayesian_update_posterior(self, searched_cells, detection_prob=0.9):
        """
        Apply Bayesian update to posterior based on negative observations in searched cells.

        Parameters:
            searched_cells (list of tuple): List of (x, y) cell coordinates searched.
            detection_prob (float): Probability of detecting the target if it's in the searched cell.

        Returns:
            np.ndarray: Updated posterior grid.
        """
        posterior = self.posterior_grid_.copy()

        # Likelihood update: P(D | H) = 1 - detection_prob in searched cells, 1 elsewhere
        likelihood = np.ones_like(posterior)
        for x, y in searched_cells:
            likelihood[x, y] = 1 - detection_prob  # Less likely target was here if not detected

        # Apply Bayesian update
        posterior *= likelihood
        posterior /= posterior.sum()  # Renormalize to maintain valid probability distribution

        return posterior
    
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

