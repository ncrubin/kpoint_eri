import numpy as np


from kpoint_eri.factorizations.kmeans import KMeansCVT


def gaussian(dx, sigma):
    return np.exp(-np.einsum("Gx,Gx->G", dx, dx) / sigma**2.0) / (
        2 * (sigma * np.pi) ** 0.5
    )


def gen_gaussian(xx, yy, grid, sigma=0.125):
    x0 = np.array([0.25, 0.25])
    dx0 = grid - x0[None, :]
    weight = gaussian(dx0, sigma)
    x1 = np.array([0.25, 0.75])
    dx = grid - x1[None, :]
    weight += gaussian(dx, sigma)
    x2 = np.array([0.75, 0.25])
    dx = grid - x2[None, :]
    weight += gaussian(dx, sigma)
    x3 = np.array([0.75, 0.75])
    dx = grid - x3[None, :]
    weight += gaussian(dx, sigma)
    return weight


def test_kmeans():
    np.random.seed(7)
    # 3D spatial grid
    num_grid_x = 10
    num_grid_points = num_grid_x**2
    xs = np.linspace(0, 1, num_grid_x)
    ys = np.linspace(0, 1, num_grid_x)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.zeros((num_grid_points, 2))
    grid[:, 0] = xx.ravel()
    grid[:, 1] = yy.ravel()
    num_interp_points = 10

    centroids_indx = np.random.choice(num_grid_points, num_interp_points, replace=False)
    centroids = grid[centroids_indx].copy()

    weight = gen_gaussian(xx, yy, grid)
    kmeans = KMeansCVT(grid)
    interp_points = kmeans.find_interpolating_points(num_interp_points, weight,
                                                     verbose=False)
    interp_points_ref = [32, 81, 77, 62, 28, 22, 76, 24, 27, 73]
    assert np.allclose(interp_points, interp_points_ref)

if __name__ == "__main__":
    test_kmeans()
