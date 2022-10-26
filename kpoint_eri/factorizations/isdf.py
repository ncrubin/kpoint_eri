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
    orbitals,
    grid_points,
    kpoint=np.zeros(3),
):
    r"""
    Build ISDF-THC tensors.

    :returns tuple: (zeta, chi, Theta): central tensor, orbitals on interpolating
        points and matrix interpolating vectors Theta = [xi_1, xi_2.., xi_Nmu]
        [R, Nmu].  Note chi is not necessarily normalized (check).
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
    kpts,
    orbitals,
    grid_points,
):
    r"""
    Build ISDF-THC representation of ERI tensors.

    :returns tuple: (zeta, chi, Theta): central tensor, orbitals on interpolating
        points and matrix interpolating vectors Theta = [xi_1, xi_2.., xi_Nmu].
        Note chi is not necessarily normalized (check).
    """
    num_kpts = len(kpts)
    cell = mydf.cell

    num_grid_points = len(grid_points)
    num_interp_points = len(interp_indx)
    assert orbitals.shape[0] == num_grid_points

    Theta, chi = solve_isdf(orbitals, interp_indx)
    zeta_Q = np.zeros(
        (num_kpts, num_interp_points, num_interp_points), dtype=np.complex128
    )
    # FFT Theta[R, mu] -> Theta[mu, G]
    # Transpose as fft expects contiguous for each mu.
    Theta_G = tools.fft(Theta.T, mydf.mesh)
    for iq, q in enumerate(kpts):
        coulG = tools.get_coulG(cell, k=q, mesh=mydf.mesh)
        weighted_coulG = coulG * cell.vol / num_grid_points**2.0
        # zeta_{mu,nu} = \sum_G 4pi/(omega * G^2) zeta_{mu,G} * (zeta_G*){nu, G}
        Theta_G_tilde = np.einsum("iG,G->iG", Theta_G, weighted_coulG)
        zeta_Q[iq] = Theta_G_tilde @ Theta_G.conj().T
    return chi, zeta_Q, Theta
