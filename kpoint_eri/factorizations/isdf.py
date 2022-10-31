import numpy as np

from pyscf.pbc import tools, df


def solve_isdf(orbitals, interp_indx):
    """Solve for interpolating vectors given interpolating points and orbitals.

    Used for supercell and k-point so factor out as function.

    :param orbitals: orbitals on a grid of shape [num_grid_points, num_orbitals]
    :param interp_indx: array indexing interpolating points (subset of grid
        points to use selected by K-Means algorithm. shape is [num_interp_points].
    :returns tuple: (Interpolang vectors, interpolating orbitals) (xi_mu(r),
        phi_i(r_mu)). Note xi_mu(r) is called Theta[R, mu] in keeping with
        original ISDF notation.
    """
    interp_orbitals = orbitals[interp_indx]
    # Form pseudo-densities
    # P[R, r_mu] = \sum_{i} phi_{R,i}^* phi_{mu,i}
    pseudo_density = np.einsum(
        "Ri,mi->Rm", orbitals.conj(), interp_orbitals, optimize=True
    )
    # [Z C^]_{J, mu} = (sum_i phi_{i, J}^* phi_{i, mu}) (sum_j phi_{j, J} # phi_{i, mu})
    ZC_dag = np.einsum(
        "Rm,Rm->Rm", pseudo_density, pseudo_density.conj(), optimize=True
    )
    # Just down sample from ZC_dag
    CC_dag = ZC_dag[interp_indx].copy()
    # Solve ZC_dag = Theta CC_dag
    # Theta =  ZC_dag (CC_dag)^{-1}
    # TODO: Originally used over-determined least squares solve, does it matter?
    CC_dag_inv = np.linalg.pinv(CC_dag)
    Theta = ZC_dag @ CC_dag_inv
    return Theta, interp_orbitals


def supercell_isdf(
    mydf: df.FFTDF,
    interp_indx: np.ndarray,
    orbitals: np.ndarray,
    grid_points: np.ndarray,
    kpoint=np.zeros(3),
):
    r"""
    Build ISDF-THC tensors.

    Given the orbitals evaluated on a (dense) real space grid, and a set of
    interpolating points (indexed  by interp_indx) determine the interpolating
    orbitals (chi), central tensor (zeta), and interpolating vectors Theta (also
    called xi).

    :param mydf: instance of pyscf.pbc.df.FFTDF object.
    :param interp_indx: array indexing interpolating points determined through
        K-Means CVT procedure. Dimension [num_interp_points]
    :param orbitals: orbitals on a grid of shape [num_grid_points, num_orbitals]
    :param grid_points: Real space grid. Dimension [num_grid_points, num_dim],
        num_dim is 1, 2 or 3 for 1D, 2D, 3D.
    :returns tuple: (chi, zeta, Theta): orbitals on interpolating
        points, zeta (central tensor), and matrix of interpolating vectors Theta
        of dimension [num_grid_points, num_interp_points] (also called
        xi_mu(r)), where num_grid_points is the number of real space grid points
        and num_interp_points is the number of interpolating points.

    TODO: Note chi is not necessarily normalized (check).
    """

    cell = mydf.cell
    num_grid_points = len(grid_points)

    Theta, chi = solve_isdf(orbitals, interp_indx)

    # FFT Theta[R, mu] -> Theta[mu, G]
    # Transpose as fft expects contiguous.
    Theta_G = tools.fft(Theta.T, mydf.mesh)
    coulG = tools.get_coulG(cell, k=kpoint, mesh=mydf.mesh)
    weighted_coulG = coulG * cell.vol / num_grid_points**2.0

    # zeta_{mu,nu} = \sum_G 4pi/(omega * G^2) zeta_{mu,G} * (zeta_G*){nu, G}
    Theta_G_tilde = np.einsum("iG,G->iG", Theta_G, weighted_coulG)
    zeta = (Theta_G_tilde) @ Theta_G.conj().T
    return chi, zeta, Theta


def kpoint_isdf(
    mydf: df.FFTDF,
    interp_indx: np.ndarray,
    kpts: np.ndarray,
    orbitals: np.ndarray,
    grid_points: np.ndarray,
):
    r"""
    Build kpoint ISDF-THC tensors.

    Given the orbitals evaluated on a (dense) real space grid, and a set of
    interpolating points (indexed  by interp_indx) determine the interpolating
    orbitals (chi), central tensor (zeta), and interpolating vectors Theta (also
    called xi).

    WARNING: Not currently 100% worked out re central tensor.

    :param mydf: instance of pyscf.pbc.df.FFTDF object.
    :param interp_indx: array indexing interpolating points determined through
        K-Means CVT procedure. Dimension [num_interp_points]
    :param kpts: Array of k-points. 
    :param orbitals: orbitals on a grid of shape [num_grid_points,
        num_orbitals], note num_orbitals = N_k * m, where m is the number of
        orbitals in the unit cell and N_k is the number of k-points.
    :param grid_points: Real space grid. Dimension [num_grid_points, num_dim],
        num_dim is 1, 2 or 3 for 1D, 2D, 3D.
    :returns tuple: (chi, Theta): orbitals on interpolating
        points, and a matrix of interpolating vectors Theta
        of dimension [num_grid_points, num_interp_points] (also called
        xi_mu(r)), where num_grid_points is the number of real space grid points
        and num_interp_points is the number of interpolating points.

    """
    num_grid_points = len(grid_points)
    assert orbitals.shape[0] == num_grid_points
    Theta, chi = solve_isdf(orbitals, interp_indx)
    return chi, Theta
