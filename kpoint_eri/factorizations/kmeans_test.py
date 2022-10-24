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
    num_grid_x = 20
    num_grid_points = num_grid_x**2
    xs = np.linspace(0, 1, num_grid_x)
    ys = np.linspace(0, 1, num_grid_x)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.zeros((num_grid_points, 2))
    grid[:, 0] = xx.ravel()
    grid[:, 1] = yy.ravel()
    num_interp_points = 10
    num_dim = 2

    centroids_indx = np.random.choice(num_grid_points, num_interp_points, replace=False)
    centroids = grid[centroids_indx].copy()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)
    from scipy.spatial import Voronoi, voronoi_plot_2d

    weight = gen_gaussian(xx, yy, grid)

    vor = Voronoi(centroids)
    voronoi_plot_2d(
        vor,
        show_vertices=False,
        line_colors="blue",
        line_width=2,
        line_alpha=0.6,
        point_size=2,
        ax=ax[0],
    )

    CS = ax[0].contour(xx, yy, weight.reshape((num_grid_x, num_grid_x)))
    ax[0].clabel(CS, inline=1, fontsize=10)
    ax[0].plot(
        centroids[:, 0], centroids[:, 1], linewidth=0, marker="o", color="orange"
    )

    kmeans = KMeansCVT(grid)
    interp_points = kmeans.find_interpolating_points(num_interp_points, weight)
    vor_interp = Voronoi(interp_points)
    voronoi_plot_2d(
        vor_interp,
        show_vertices=False,
        line_colors="blue",
        line_width=2,
        line_alpha=0.6,
        point_size=0,
        ax=ax[1],
    )
    CS = ax[1].contour(xx, yy, weight.reshape((num_grid_x, num_grid_x)))
    ax[1].plot(
        interp_points[:, 0],
        interp_points[:, 1],
        linewidth=0,
        marker="o",
        color="orange",
    )
    ax[1].clabel(CS, inline=1, fontsize=10)
    plt.savefig("kmeans.png")
    plt.show()


if __name__ == "__main__":
    test_kmeans()
