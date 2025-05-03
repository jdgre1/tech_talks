import numpy as np

class SearchStrategy:

    @staticmethod
    def get_top_k_cells(prior_grid, k=10):
        flat_indices = np.argsort(prior_grid.ravel())[::-1]
        top_k = np.column_stack(np.unravel_index(flat_indices[:k], prior_grid.shape))
        return [tuple(cell) for cell in top_k]

    @staticmethod
    def get_top_k_cells_with_radius(posterior_grid, k=10, radius=1):
        flat_indices = np.dstack(
            np.unravel_index(np.argsort(posterior_grid.ravel())[::-1], posterior_grid.shape)
        )[0]

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

    @staticmethod
    def select_search_area(prob_grid, k=10, radius=None):
        """
        Selects the next search area based on the probability grid.
        If radius is None or 0, selects the top k cells directly.
        Otherwise, selects k high-probability regions, each expanded by a neighborhood of given radius.
        """
        if radius is None or radius <= 0:
            return SearchStrategy.get_top_k_cells(prob_grid, k)
        else:
            return SearchStrategy.get_top_k_cells_with_radius(prob_grid, k, radius)
