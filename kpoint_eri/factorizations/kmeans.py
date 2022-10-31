import numpy as np


class KMeansCVT(object):
    def __init__(self, grid, max_iteration=100, threshold=1e-6):
        """Initialize k-means solver to find interpolating points for ISDF.

        :param grid: Real space grid of dimension [Ng,Ndim], where Ng is the number
            of (dense) real space grid points and Ndim is number of spatial
            dimensions.
        :param max_iteration: Maximum number of iterations to perform when
            classifying grid points. Default 100.
        :param threshold: Threshold for exiting classification. Default 1e-6.
        """
        self.grid = grid
        self.max_iteration = max_iteration
        self.threshold = threshold

    @staticmethod
    def classify_grid_points(grid_points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        r"""Assign grid points to centroids.

        Find centroid closest to each given grid point.

        Note we don't use instance variable self.grid as we can abuse this
        function and use it to map grid to centroid and centroid to grid point.

        :param grid_points: grid points to assign.
        :param centroids: Centroids to which grid points should be assigned,
            array of length num_interp_points.
        :returns: 1D np.array assigning grid point to centroids
        """
        # Build N_g x N_mu matrix of distances.
        # distances = np.linalg.norm(
            # grid_points[:, None, :] - centroids[None, :, :], axis=2
        # )
        num_grid_points = grid_points.shape[0]
        num_interp_points = centroids.shape[0]
        distances = np.zeros((num_grid_points, num_interp_points))
        # For loop is faster than broadcasting by 2x.
        for ig in range(num_grid_points):
            distances[ig] = np.linalg.norm(grid_points[ig]-centroids, axis=1)
        # Find shortest distance for each grid point.
        classification = np.argmin(distances, axis=1)
        return classification

    def compute_new_centroids(
        self, weighting, grid_mapping, current_centroids
    ) -> np.ndarray:
        r"""
        Centroids are defined via:

        .. math::

            c(C_\mu) = \frac{\sum_{j in C(\mu)} r_j \rho(r_j)}{\sum_{j in
            C(\mu)} \rho(r_j)},

        where :math:`\rho(r_j)` is the weighting factor.
        """
        num_interp_points = current_centroids.shape[0]
        new_centroids = np.zeros_like(current_centroids)
        for interp_indx in range(num_interp_points):
            # get grid points belonging to this centroid
            grid_indx = np.where(grid_mapping == interp_indx)[0]
            grid_points = self.grid[grid_indx]
            weight = weighting[grid_indx]
            numerator = np.einsum("Jx,J->x", grid_points, weight)
            denominator = np.einsum("J->", weight)
            if denominator < 1e-12:
                print("Warning very small denominator, something seems wrong!")
                print("{interp_indx}")
            new_centroids[interp_indx] = numerator / denominator
        return new_centroids

    def map_centroids_to_grid(self, centroids):
        grid_mapping = self.classify_grid_points(centroids, self.grid)
        return grid_mapping

    def find_interpolating_points(
        self,
        num_interp_points: int,
        weighting_factor: np.ndarray,
        centroids=None,
        verbose=True,
    ) -> np.ndarray:
        """
        """
        num_grid_points = self.grid.shape[0]
        if centroids is None:
            # Randomly select grid points as centroids.
            centroids_indx = np.random.choice(
                num_grid_points, num_interp_points, replace=False
            )
            centroids = self.grid[centroids_indx].copy()
        else:
            assert len(centroids) == num_interp_points
        # define here to avoid linter errors about possibly undefined.
        new_centroids = np.zeros_like(centroids)
        delta_grid = 1.0
        if verbose:
            print("{:<10s}  {:>13s}".format("iteration", "Error"))
        for iteration in range(self.max_iteration):
            grid_mapping = self.classify_grid_points(self.grid, centroids)
            # Global reduce
            new_centroids[:] = self.compute_new_centroids(
                weighting_factor, grid_mapping, centroids
            )
            delta_grid = np.linalg.norm(new_centroids - centroids)
            if verbose and iteration % 10 == 0:
                print(f"{iteration:<9d}  {delta_grid:13.8e}")
            if delta_grid < self.threshold:
                if verbose:
                    print("KMeansCVT successfully completed.")
                    print(f"Final error {delta_grid:13.8e}.")
                return self.map_centroids_to_grid(new_centroids)
            centroids[:] = new_centroids[:]
        print("Warning K-Means not converged.")
        print(f"Final error {delta_grid:13.8e}.")
        return self.map_centroids_to_grid(new_centroids)
