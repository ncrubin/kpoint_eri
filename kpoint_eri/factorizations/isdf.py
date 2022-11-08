import itertools
import numpy as np

from pyscf.pbc import tools, df, gto
from pyscf.pbc.lib.kpts_helper import unique, get_kconserv

from kpoint_eri.resource_estimates.utils.misc_utils import (
    build_momentum_transfer_mapping,
)


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


def build_kpoint_zeta(
    df_inst,
    Q,
    delta_G,
    delta_G_prime,
    grid_points,
    xi_mu,
):
    """Build k-point THC zeta (central tensor) for given Q, delta_G,
    delta_G_prime.

    :param mydf: instance of pyscf.pbc.df.FFTDF object.
    :param Q: Momentum transfer (in 1BZ).
    :param delta_G: Reciprocal lattice vector satisfying Q - (Q-k) = delta_G
    :param delta_G_prime: Reciprocal lattice vector satisfying Q - (Q-k') = delta_G
    :param grid_points: Real space grid. Dimension [num_grid_points, num_dim],
        num_dim is 1, 2 or 3 for 1D, 2D, 3D.
    :param xi_mu: array containing interpolating vectors determined during ISDF
        procedure
    :returns zeta: central tensor of dimension [num_interp_points, num_interp_points]
    """
    cell = df_inst.cell
    num_grid_points = grid_points.shape[0]
    # delta_G - delta_G_prime because we have Gpq and Gsr and Gsr = -Grs, phase
    # = Delta G = Gpq + Grs
    phase_factor = np.exp(
        -1j
        * (np.einsum("x,Rx->R", delta_G - delta_G_prime, grid_points, optimize=True))
    )
    # Minus sign again due to we use Q = kp - kq, but we should have V(G + k_q - k_p)
    coulG = tools.get_coulG(cell, k=-(Q + delta_G), mesh=df_inst.mesh)
    weighted_coulG = coulG * cell.vol / num_grid_points
    xi_muG = tools.fft(xi_mu.T, df_inst.mesh)
    xi_muG *= weighted_coulG
    vR = tools.ifft(xi_muG, df_inst.mesh)
    zeta = np.einsum("R,Rn,mR->mn", phase_factor, xi_mu, vR, optimize=True)
    return zeta


def build_kpoint_zeta_single_tranlsation(
    df_inst,
    q,
    delta_G,
    grid_points,
    xi_mu,
):
    """Build k-point THC zeta (central tensor) for given Q, delta_G,
    delta_G_prime.

    :param mydf: instance of pyscf.pbc.df.FFTDF object.
    :param q: Momentum transfer kp-kq.
    :param delta_G: Reciprocal lattice vector satisfying Q - (Q-k) = delta_G
    :param grid_points: Real space grid. Dimension [num_grid_points, num_dim],
        num_dim is 1, 2 or 3 for 1D, 2D, 3D.
    :param xi_mu: array containing interpolating vectors determined during ISDF
        procedure
    :returns zeta: central tensor of dimension [num_interp_points, num_interp_points]
    """
    cell = df_inst.cell
    num_grid_points = grid_points.shape[0]
    # delta_G - delta_G_prime because we have Gpq and Gsr and Gsr = -Grs, phase
    # = Delta G = Gpq + Grs
    phase_factor = np.exp(
        -1j * (np.einsum("x,Rx->R", delta_G, grid_points, optimize=True))
    )
    # Minus sign again due to we use Q = kp - kq, but we should have V(G + k_q - k_p)
    coulG = tools.get_coulG(cell, k=-q, mesh=df_inst.mesh)
    weighted_coulG = coulG * cell.vol / num_grid_points
    xi_muG = tools.fft(xi_mu.T, df_inst.mesh)
    xi_muG *= weighted_coulG
    vR = tools.ifft(xi_muG, df_inst.mesh)
    zeta = np.einsum("R,Rn,mR->mn", phase_factor, xi_mu, vR, optimize=True)
    return zeta


def build_G_vectors(cell):
    """Build all 27 Gvectors

    :param cell: pyscf.pbc.gto.Cell object.
    :returns tuple: G_dict a dictionary mapping miller index to appropriate
        G_vector index and G_vectors array of 27 G_vectors shape [27, 3].
    """
    G_dict = {}
    G_vectors = np.zeros((27, 3), dtype=np.float64)
    lattice_vectors = cell.lattice_vectors()
    indx = 0
    for n1, n2, n3 in itertools.product(range(-1, 2), repeat=3):
        G_dict[(n1, n2, n3)] = indx
        G_vectors[indx] = np.einsum("x,wx->w", (n1, n2, n3), cell.reciprocal_vectors())
        miller_indx = np.rint(
            np.einsum("wx,x->w", lattice_vectors, G_vectors[indx]) / (2 * np.pi)
        )
        assert (miller_indx == (n1, n2, n3)).all()
        indx += 1
    return G_dict, G_vectors


def find_unique_G_vectors(G_vectors, G_mapping):
    """Find all unique G-vectors and build mapping to original set.

    :param G_vectors: array of 27 G-vectors.
    :param G_mapping: array of 27 G-vectors.
    :returns tuple: unique_mapping, delta_Gs. unique_mapping[iq, ik] =
        unique_G_index in range [0,...,num_unique_Gs[iq]], and delta_Gs are the
        unique G-vectors of size [num_qpoints, num_unique_Gs[iq]].
    """
    unique_mapping = np.zeros_like(G_mapping)
    num_qpoints = G_mapping.shape[0]
    delta_Gs = np.zeros((num_qpoints,), dtype=object)
    for iq in range(num_qpoints):
        unique_G = np.unique(G_mapping[iq])
        delta_Gs[iq] = G_vectors[unique_G]
        # Build map to unique index
        unique_mapping[iq] = [
            ix for el in G_mapping[iq] for ix in np.where(unique_G == el)[0]
        ]

    return unique_mapping, delta_Gs


def build_G_vector_mappings(
    cell: gto.Cell,
    kpts: np.ndarray,
    momentum_map: np.ndarray,
):
    """Build G-vector mappings that map k-point differences to 1BZ.

    :param cell: pyscf.pbc.gto.Cell object.
    :param kpts: array of kpoints.
    :param momentum_map: momentum mapping to satisfy Q = (k_p - k_q) mod G.
        momentum_map[iq, ikp] = ikq.
    :returns tuple: (G_vectors, Gpq_mapping, Gpq_mapping_unique, delta_Gs), G_vectors is a list of all 27
        G-vectors and Gpq_mapping[iq, kp] = indx_Gpq, where Gpq = kpts[ikp] -
        kpts[ikq] - kpts[iq], i.e. returns index to G-vector (consistent with
        G_vectors) satisfying this condition. Gpq_mapping_unique provides
        mapping to unique G_vector index. Delta_gs provides compressed lists of
        unique G vectors.
    """
    G_dict, G_vectors = build_G_vectors(cell)
    lattice_vectors = cell.lattice_vectors()
    num_kpts = len(kpts)
    Gpq_mapping = np.zeros((num_kpts, num_kpts), dtype=np.int32)
    num_kpts = len(kpts)
    for iq in range(num_kpts):
        for ikp in range(num_kpts):
            ikq = momentum_map[iq, ikp]
            delta_Gpq = (kpts[ikp] - kpts[ikq]) - kpts[iq]
            miller_indx = np.rint(
                np.einsum("wx,x->w", lattice_vectors, delta_Gpq) / (2 * np.pi)
            )
            Gpq_mapping[iq, ikp] = G_dict[tuple(miller_indx)]

    Gpq_mapping_unique, delta_Gs = find_unique_G_vectors(G_vectors, Gpq_mapping)
    return G_vectors, Gpq_mapping, Gpq_mapping_unique, delta_Gs


def build_G_vector_mappings_single_translation(
    cell: gto.Cell,
    kpts: np.ndarray,
    kpts_pq,
):
    """Build G-vector mappings that map k-point differences to 1BZ.

    :param cell: pyscf.pbc.gto.Cell object.
    :param kpts: array of kpoints.
    :param kpts_pq: Unique list of kp - kq indices of shape [num_unique_pq, 2].
    :returns tuple: (G_vectors, Gpqr_mapping, Gpqr_mapping_unique, delta_Gs), G_vectors is a list of all 27
        G-vectors and Gpqr_mapping[iq, kr] = indx_Gpqr, where Gpqr = kpts[ikp] -
        kpts[ikq] + kpts[ikr] - kpts[iks], i.e. returns index to G_vectors (consistent with
        G_vectors) satisfying this condition. Gpqr_mapping_unique provides
        mapping to unique G_vector index. Delta_gs provides compressed lists of
        unique G vectors.
    """
    G_dict, G_vectors = build_G_vectors(cell)
    lattice_vectors = cell.lattice_vectors()
    num_kpts = len(kpts)
    Gpqr_mapping = np.zeros((len(kpts_pq), num_kpts), dtype=np.int32)
    kconserv = get_kconserv(cell, kpts)
    for iq, (ikp, ikq) in enumerate(kpts_pq):
        q = kpts[ikp] - kpts[ikq]
        for ikr in range(num_kpts):
            iks = kconserv[ikp, ikq, ikr]
            delta_Gpqr = q + kpts[ikr] - kpts[iks]
            # delta_Gpq += kpts[0]
            miller_indx = np.rint(
                np.einsum("wx,x->w", lattice_vectors, delta_Gpqr) / (2 * np.pi)
            )
            Gpqr_mapping[iq, ikr] = G_dict[tuple(miller_indx)]

    Gpqr_mapping_unique, delta_Gs = find_unique_G_vectors(G_vectors, Gpqr_mapping)
    return G_vectors, Gpqr_mapping, Gpqr_mapping_unique, delta_Gs


def build_eri_isdf(chi, zeta, q_indx, kpts_indx, G_mapping):
    """Build (pkp qkq | rkr sks) from k-point ISDF factors.

    :param chi: array of interpolating orbitals of shape [num_kpts, num_mo, num_interp_points]
    :param zeta: central tensor of dimension [num_kpts, num_G, num_G, num_interp_points, num_interp_points].
    :param q_indx: Index of momentum transfer.
    :param kpts_indx: List of kpt indices corresponding to [kp, kq, kr, ks]
    :param G_mapping: array to map kpts to G vectors [q_indx, kp] = G_pq
    :returns eri:  (pkp qkq | rkr sks)
    """
    ikp, ikq, ikr, iks = kpts_indx
    Gpq = G_mapping[q_indx, ikp]
    Gsr = G_mapping[q_indx, iks]
    eri = np.einsum(
        "pm,qm,mn,rn,sn->pqrs",
        chi[ikp].conj(),
        chi[ikq],
        zeta[q_indx][Gpq, Gsr],
        chi[ikr].conj(),
        chi[iks],
        optimize=True,
    )
    return eri


def build_eri_isdf_single_translation(chi, zeta, q_indx, kpts_indx, G_mapping):
    """Build (pkp qkq | rkr sks) from k-point ISDF factors.

    :param chi: array of interpolating orbitals of shape [num_kpts, num_mo, num_interp_points]
    :param zeta: central tensor of dimension [num_kpts, num_G, num_G, num_interp_points, num_interp_points].
    :param q_indx: Index of momentum transfer.
    :param kpts_indx: List of kpt indices corresponding to [kp, kq, kr, ks]
    :param G_mapping: array to map kpts to G vectors [q_indx, kp] = G_pq
    :returns eri:  (pkp qkq | rkr sks)
    """
    ikp, ikq, ikr, iks = kpts_indx
    delta_G_indx = G_mapping[q_indx, ikr]
    eri = np.einsum(
        "pm,qm,mn,rn,sn->pqrs",
        chi[ikp].conj(),
        chi[ikq],
        zeta[q_indx][delta_G_indx],
        chi[ikr].conj(),
        chi[iks],
        optimize=True,
    )
    return eri


def kpoint_isdf_double_translation(
    df_inst: df.FFTDF,
    interp_indx: np.ndarray,
    kpts: np.ndarray,
    orbitals: np.ndarray,
    grid_points: np.ndarray,
    only_unique_G: bool = False,
):
    r"""
    Build kpoint ISDF-THC tensors.

    Given the orbitals evaluated on a (dense) real space grid, and a set of
    interpolating points (indexed  by interp_indx) determine the interpolating
    orbitals (chi), central tensor (zeta), and interpolating vectors Theta (also
    called xi).

    For the double translation case we build zeta[Q, G, G'] for all possible G
    and G' that satisfy Q - (Q-k) = G. If only_unique_G is True we only build
    the unique G's which satisfiy this expression rather than all 27^2.

    :param df_inst: instance of pyscf.pbc.df.FFTDF object.
    :param interp_indx: array indexing interpolating points determined through
        K-Means CVT procedure. Dimension [num_interp_points]
    :param kpts: Array of k-points.
    :param orbitals: orbitals on a grid of shape [num_grid_points,
        num_orbitals], note num_orbitals = N_k * m, where m is the number of
        orbitals in the unit cell and N_k is the number of k-points.
    :param grid_points: Real space grid. Dimension [num_grid_points, num_dim],
        num_dim is 1, 2 or 3 for 1D, 2D, 3D.
    :param only_unique_G: Only build central tensor for unique Gs which satisfy
        momentum conservation condition.
    :returns tuple: (chi, zeta, Theta, G_mapping): orbitals on interpolating
        points, and a matrix of interpolating vectors Theta
        of dimension [num_grid_points, num_interp_points] (also called
        xi_mu(r)), where num_grid_points is the number of real space grid points
        and num_interp_points is the number of interpolating points. Zeta (the
        central tensor) is of dimension [num_kpts, 27, 27, num_interp_points, num_interp_points]
        if only_unique_G is False otherwise it is of shape [num_kpts,
        num_unique[Q], num_unique[Q], 27, num_interp_points, num_interp_points].
        G_mapping maps k-points to the appropriate delta_G index, i.e.
        G_mapping[iq, ik] = i_delta_G. if only_unique_G is True the index will
        map to the appropriate index in the reduced set of G vectors.
    """
    num_grid_points = len(grid_points)
    assert orbitals.shape[0] == num_grid_points
    xi, chi = solve_isdf(orbitals, interp_indx)
    momentum_map = build_momentum_transfer_mapping(df_inst.cell, kpts)
    num_kpts = len(kpts)
    num_interp_points = xi.shape[1]
    assert xi.shape == (num_grid_points, num_interp_points)
    G_vectors, G_mapping, G_mapping_unique, delta_Gs_unique = build_G_vector_mappings(
        df_inst.cell, kpts, momentum_map
    )
    if only_unique_G:
        G_mapping = G_mapping_unique
        delta_Gs = delta_Gs_unique
    else:
        delta_Gs = [G_vectors] * num_kpts
        G_mapping = G_mapping
    zeta = np.zeros((num_kpts,), dtype=object)
    for iq in range(num_kpts):
        num_G = len(delta_Gs[iq])
        out_array = np.zeros(
            (num_G, num_G, num_interp_points, num_interp_points), dtype=np.complex128
        )
        for iG, delta_G in enumerate(delta_Gs[iq]):
            for iG_prime, delta_G_prime in enumerate(delta_Gs[iq]):
                zeta_indx = build_kpoint_zeta(
                    df_inst, kpts[iq], delta_G, delta_G_prime, grid_points, xi
                )
                out_array[iG, iG_prime] = zeta_indx
        zeta[iq] = out_array
    return chi, zeta, xi, G_mapping


def kpoint_isdf_single_translation(
    df_inst: df.FFTDF,
    interp_indx: np.ndarray,
    kpts: np.ndarray,
    orbitals: np.ndarray,
    grid_points: np.ndarray,
    only_unique_G: bool = False,
):
    r"""
    Build kpoint ISDF-THC tensors.

    Given the orbitals evaluated on a (dense) real space grid, and a set of
    interpolating points (indexed  by interp_indx) determine the interpolating
    orbitals (chi), central tensor (zeta), and interpolating vectors Theta (also
    called xi).

    For the double translation case we build zeta[Q, G, G'] for all possible G
    and G' that satisfy Q - (Q-k) = G. If only_unique_G is True we only build
    the unique G's which satisfiy this expression rather than all 27^2.

    :param df_inst: instance of pyscf.pbc.df.FFTDF object.
    :param interp_indx: array indexing interpolating points determined through
        K-Means CVT procedure. Dimension [num_interp_points]
    :param kpts: Array of k-points.
    :param orbitals: orbitals on a grid of shape [num_grid_points,
        num_orbitals], note num_orbitals = N_k * m, where m is the number of
        orbitals in the unit cell and N_k is the number of k-points.
    :param grid_points: Real space grid. Dimension [num_grid_points, num_dim],
        num_dim is 1, 2 or 3 for 1D, 2D, 3D.
    :param only_unique_G: Only build central tensor for unique Gs which satisfy
        momentum conservation condition.
    :returns tuple: (chi, zeta, Theta, G_mapping): orbitals on interpolating
        points, and a matrix of interpolating vectors Theta
        of dimension [num_grid_points, num_interp_points] (also called
        xi_mu(r)), where num_grid_points is the number of real space grid points
        and num_interp_points is the number of interpolating points. Zeta (the
        central tensor) is of dimension [num_kpts, 27, 27, num_interp_points, num_interp_points]
        if only_unique_G is False otherwise it is of shape [num_kpts,
        num_unique[Q], num_unique[Q], 27, num_interp_points, num_interp_points].
        G_mapping maps k-points to the appropriate delta_G index, i.e.
        G_mapping[iq, ik] = i_delta_G. if only_unique_G is True the index will
        map to the appropriate index in the reduced set of G vectors.
    """
    num_grid_points = len(grid_points)
    assert orbitals.shape[0] == num_grid_points
    xi, chi = solve_isdf(orbitals, interp_indx)
    num_kpts = len(kpts)
    num_interp_points = xi.shape[1]
    assert xi.shape == (num_grid_points, num_interp_points)
    kpts_pq = np.array(
        [(kp, kpts[ikq]) for ikp, kp in enumerate(kpts) for ikq in range(num_kpts)]
    )
    kpts_pq_indx = np.array(
        [(ikp, ikq) for ikp, kp in enumerate(kpts) for ikq in range(num_kpts)]
    )
    transfers = kpts_pq[:, 0] - kpts_pq[:, 1]
    unique_q, unique_indx, unique_inverse = unique(transfers)
    _, _, G_map_unique, delta_Gs = build_G_vector_mappings_single_translation(
        df_inst.cell, kpts, kpts_pq_indx[unique_indx]
    )
    num_q_vectors = len(unique_q)
    zeta = np.zeros((num_q_vectors,), dtype=object)
    for iq in range(len(unique_q)):
        num_G = len(delta_Gs[iq])
        out_array = np.zeros(
            (num_G, num_interp_points, num_interp_points), dtype=np.complex128
        )
        for iG, delta_G in enumerate(delta_Gs[iq]):
            zeta_indx = build_kpoint_zeta_single_tranlsation(
                df_inst, unique_q[iq], delta_G, grid_points, xi
            )
            out_array[iG] = zeta_indx
        zeta[iq] = out_array
    return chi, zeta, xi, G_map_unique
