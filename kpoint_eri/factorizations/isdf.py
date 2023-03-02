import itertools
import numpy as np
import scipy.linalg

from pyscf.pbc import tools, df, gto
from pyscf.pbc.lib.kpts_helper import unique, get_kconserv, conj_mapping
from pyscf.pbc.dft import numint, gen_grid

from kpoint_eri.resource_estimates.utils.misc_utils import (
    build_momentum_transfer_mapping,
)
from kpoint_eri.factorizations.kmeans import KMeansCVT


def check_isdf_solution(orbitals, interp_orbitals, xi):
    lhs = np.einsum("Ri,Rj->Rij", orbitals.conj(), orbitals, optimize=True)
    rhs = np.einsum(
        "mi,mj->mij", interp_orbitals.conj(), interp_orbitals, optimize=True
    )
    lhs_check = np.einsum("Rm,mij->Rij", xi, rhs, optimize=True)
    return np.linalg.norm(lhs - lhs_check)


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
    # Theta = ZC_dag @ CC_dag_inv
    # Solve ZC_dag = Theta CC_dag
    # -> ZC_dag^T = CC_dag^T Theta^T
    Theta_dag, res, rank, s = np.linalg.lstsq(CC_dag.conj().T, ZC_dag.conj().T)
    return Theta_dag.conj().T, interp_orbitals


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
        G_vectors[indx] = np.einsum("n,ng->g", (n1, n2, n3), cell.reciprocal_vectors())
        miller_indx = np.rint(
            np.einsum("nx,x->n", lattice_vectors, G_vectors[indx]) / (2 * np.pi)
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


def build_G_vector_mappings_double_translation(
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


def get_miller(lattice_vectors, G):
    """Convert G to miller indx"""
    miller_indx = np.rint(
        np.einsum("wx,x->w", lattice_vectors, G) / (2 * np.pi)
    ).astype(np.int32)
    return miller_indx


def build_minus_Q_G_mapping(cell: gto.Cell, kpts: np.ndarray, momentum_map: np.ndarray):
    """Build mapping for G that satisfied (-Q) + G + (Q + Gpq) = 0 (*)
    and kp - kq = Q + Gpq.

    :param cell: pyscf.pbc.gto.Cell object.
    :param kpts: array of kpoints.
    :param momentum_map: momentum mapping to satisfy Q = (k_p - k_q) mod G.
        momentum_map[iq, ikp] = ikq.
    :returns tuple: (minus_Q_mapping, minus_Q_mapping_unique),
        minus_Q_mapping[indx_minus_Q, k] yields index for G satisfying (*)
        above, where indx_minus_Q is given by indx_minus_Q = minus_k[Q], and
        minus_k = conj_mapping(cell, kpts). minus_Q_mapping_unique indexes the
        appropriate G vector given by delta_Gs[indx_minus_Q][indx] = G, where
        indx = minus_Q_mapping_unique[indx_minus_Q, k], and deltaGs is built by
        build_G_vector_mappings_double_translation.
    """
    G_vecs, G_map, G_unique, delta_Gs = build_G_vector_mappings_double_translation(
        cell, kpts, momentum_map
    )
    G_dict, _ = build_G_vectors(cell)
    num_kpts = len(kpts)
    lattice_vectors = cell.lattice_vectors()
    minus_k_map = conj_mapping(cell, kpts)
    minus_Q_mapping = np.zeros((num_kpts, num_kpts), dtype=np.int32)
    minus_Q_mapping_unique = np.zeros((num_kpts, num_kpts), dtype=np.int32)
    for iq in range(num_kpts):
        minus_iq = minus_k_map[iq]
        for ik in range(num_kpts):
            Gpq = G_vecs[G_map[iq, ik]]
            # Complementary Gpq (G in (*) in docstring)
            Gpq_comp = -(kpts[minus_iq] + kpts[iq] + Gpq)
            # find index in original set of 27
            iGpq_comp = G_dict[tuple(get_miller(lattice_vectors, Gpq_comp))]
            minus_Q_mapping[minus_iq, ik] = iGpq_comp
        indx_delta_Gs = np.array(
            [G_dict[tuple(get_miller(lattice_vectors, G))] for G in delta_Gs[minus_iq]]
        )
        minus_Q_mapping_unique[minus_iq] = [
            ix
            for el in minus_Q_mapping[minus_iq]
            for ix in np.where(el == indx_delta_Gs)[0]
        ]

    return minus_Q_mapping, minus_Q_mapping_unique


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


def inverse_G_map_double_translation(
    cell: gto.Cell,
    kpts: np.ndarray,
    momentum_map: np.ndarray,
):
    """For given Q and G figure out all k which satisfy Q - k + G = 0

    :param cell: pyscf.pbc.gto.Cell object.
    :param kpts: array of kpoints.
    :param momentum_map: momentum mapping to satisfy Q = (k_p - k_q) mod G.
        momentum_map[iq, ikp] = ikq.
    :returns inverse_map: ragged numpy array. inverse_map[iq, iG] returns array
        of size in range(0, num_kpts) and lists all k-point indices that
        satisfy G_pq[iq, ik] = iG, i.e. an array of all ik.
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

    inverse_map = np.zeros(
        (
            num_kpts,
            27,
        ),
        dtype=object,
    )
    for iq in range(num_kpts):
        for iG in range(len(G_vectors)):
            inverse_map[iq, iG] = np.array(
                [ik for ik in range(num_kpts) if Gpq_mapping[iq, ik] == iG]
            )

    return inverse_map


def build_eri_isdf_double_translation(chi, zeta, q_indx, kpts_indx, G_mapping):
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
    only_unique_G: bool = True,
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
        num_unique[Q], num_unique[Q], num_interp_points, num_interp_points].
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
    (
        G_vectors,
        G_mapping,
        G_mapping_unique,
        delta_Gs_unique,
    ) = build_G_vector_mappings_double_translation(df_inst.cell, kpts, momentum_map)
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


def build_isdf_orbital_inputs(mf_inst):
    cell = mf_inst.cell
    kpts = mf_inst.kpts
    num_kpts = len(kpts)
    grid_points = cell.gen_uniform_grids(mf_inst.with_df.mesh)
    num_mo = mf_inst.mo_coeff[0].shape[-1]  # assuming the same for each k-point
    num_grid_points = grid_points.shape[0]
    bloch_orbitals_ao = np.array(numint.eval_ao_kpts(cell, grid_points, kpts=kpts))
    bloch_orbitals_mo = np.einsum(
        "kRp,kpi->kRi", bloch_orbitals_ao, mf_inst.mo_coeff, optimize=True
    )
    exp_minus_ikr = np.exp(-1j * np.einsum("kx,Rx->kR", kpts, grid_points))
    cell_periodic_mo = np.einsum("kR,kRi->kRi", exp_minus_ikr, bloch_orbitals_mo)
    # go from kRi->Rki
    # AO ISDF
    cell_periodic_mo = cell_periodic_mo.transpose((1, 0, 2)).reshape(
        (num_grid_points, num_kpts * num_mo)
    )
    return cell_periodic_mo


def density_guess(density, grid_inst, grid_points, num_interp_points):
    """Select initial centroids based on electronic density."""
    norm_factor = np.einsum("R,R->", density, grid_inst.weights).real
    prob_dist = (density.real * grid_inst.weights) / norm_factor
    indx = np.random.choice(
        len(grid_points),
        num_interp_points,
        replace=False,
        p=prob_dist,
    )
    return grid_points[indx]


def interp_indx_from_qrcp(Z, num_interp_pts, return_diagonal=False):
    """Find interpolating points via QRCP

    Z^T P = Q R, where R has diagonal elements that are in descending order of
    magnitude. Interpolating points are then chosen by the columns of the
    permuted Z^T.

    :param orbitals_on_grid: cell-periodic part of bloch orbitals on real space grid of shape (num_grid_points, num_kpts*num_mo)
    :param num_interp_pts: integer corresponding to number of interpolating points to select from full real space grid.
    :returns interp_indx: Index of interpolating points in full real space grid.
    """

    Q, R, P = scipy.linalg.qr(Z.T, pivoting=True)
    signs_R = np.diag(np.sign(R.diagonal()))
    # Force diagonal of R to be positive which isn't strictly enforced in QR factorization.
    R = np.dot(signs_R, R)
    Q = np.dot(Q, signs_R)
    interp_indx = P[:num_interp_pts]
    if return_diagonal:
        return interp_indx, R.diagonal()
    else:
        return interp_indx


def setup_isdf(mf_inst, verbose=False):
    assert isinstance(mf_inst.with_df, df.FFTDF), "mf object must use FFTDF"
    cell = mf_inst.cell
    kpts = mf_inst.kpts
    grid_points = cell.gen_uniform_grids(mf_inst.with_df.mesh)
    grid_inst = gen_grid.UniformGrids(cell)
    num_grid_points = grid_points.shape[0]
    if verbose:
        print(
            "Real space grid shape: ({}, {})".format(
                grid_points.shape[0], grid_points.shape[1]
            )
        )
        print("Number of grid points: {}".format(num_grid_points))
    bloch_orbitals_ao = np.array(numint.eval_ao_kpts(cell, grid_points, kpts=kpts))
    bloch_orbitals_mo = np.einsum(
        "kRp,kpi->kRi", bloch_orbitals_ao, mf_inst.mo_coeff, optimize=True
    )
    nocc = cell.nelec[0]  # assuming same for each k-point
    num_mo = mf_inst.mo_coeff[0].shape[-1]  # assuming the same for each k-point
    num_kpts = len(kpts)
    # Cell periodic part
    # u = e^{-ik.r} phi(r)
    exp_minus_ikr = np.exp(-1j * np.einsum("kx,Rx->kR", kpts, grid_points))
    cell_periodic_mo = np.einsum("kR,kRi->kRi", exp_minus_ikr, bloch_orbitals_mo)
    # go from kRi->Rki
    # AO ISDF
    cell_periodic_mo = cell_periodic_mo.transpose((1, 0, 2)).reshape(
        (num_grid_points, num_kpts * num_mo)
    )
    return grid_points, cell_periodic_mo, bloch_orbitals_mo


def solve_kmeans_kpisdf(
    mf_inst,
    num_interp_points,
    max_kmeans_iteration=500,
    single_translation=True,
    use_density_guess=False,
    verbose=True,
    kmeans_weighting_function="density",
):
    print(f" K-Means weightig function: {kmeans_weighting_function}")
    # Build real space grid, and orbitals on real space grid
    grid_points, cell_periodic_mo, bloch_orbitals_mo = setup_isdf(mf_inst)
    grid_inst = gen_grid.UniformGrids(mf_inst.cell)
    # Find interpolating points using K-Means CVT algorithm
    kmeans = KMeansCVT(grid_points, max_iteration=max_kmeans_iteration)
    nocc = mf_inst.cell.nelec[0]  # assuming same for each k-point
    density = np.einsum(
        "kRi,kRi->R",
        bloch_orbitals_mo[:, :, :nocc].conj(),
        bloch_orbitals_mo[:, :, :nocc],
        optimize=True,
    )
    if use_density_guess:
        initial_centroids = density_guess(
            density, grid_inst, grid_points, num_interp_points
        )
    else:
        initial_centroids = None
    weighting_function = None
    if kmeans_weighting_function == "density":
        weighting_function = density
    elif kmeans_weighting_function == "orbital_density":
        # w(r) = sum_{ij} |phi_i(r)| |phi_j(r)|
        tmp = np.einsum(
            "kRi,pRj->Rkipj",
            abs(bloch_orbitals_mo),
            abs(bloch_orbitals_mo),
            optimize=True,
        )
        weighting_function = np.einsum("Rkipj->R", tmp)
    elif kmeans_weighting_function == "product_square_modulus":
        # w(r) = (sum_{ki} |phi_{ki}(r)|)^2
        weighting_function = np.einsum(
            "kRi,kRi->R",
            bloch_orbitals_mo.conj(),
            bloch_orbitals_mo,
            optimize=True,
        )
        weighting_function = weighting_function**2.0
    elif kmeans_weighting_function == "sum_squares":
        # w(r) = sum_{i} |phi_{ki}(r)|
        weighting_function = np.einsum(
            "kRi,kRi->R",
            bloch_orbitals_mo.conj(),
            bloch_orbitals_mo,
            optimize=True,
        )
        weighting_function = weighting_function
    else:
        raise ValueError(
            f"Unknown value for weighting function {kmeans_weighting_function}"
        )
    interp_indx = kmeans.find_interpolating_points(
        num_interp_points,
        weighting_function.real,
        verbose=verbose,
        centroids=initial_centroids,
    )
    # Solve for THC factors.
    solution = solve_for_thc_factors(
        mf_inst,
        interp_indx,
        cell_periodic_mo,
        grid_points,
        single_translation=single_translation,
        verbose=verbose,
    )
    return solution


def solve_qrcp_isdf(
    mf_inst,
    num_interp_points,
    single_translation=True,
    verbose=True,
):
    # Build real space grid, and orbitals on real space grid
    grid_points, cell_periodic_mo, bloch_orbitals_mo = setup_isdf(mf_inst)
    # Find interpolating points using K-Means CVT algorithm
    # Z_{R, (ki)(k'j)} = u_{ki}(r)* u_{k'j}(r)
    num_grid_points = len(grid_points)
    num_orbs = cell_periodic_mo.shape[1]
    Z = np.einsum(
        "Ri,Rj->Rij", cell_periodic_mo.conj(), cell_periodic_mo, optimize=True
    ).reshape((num_grid_points, num_orbs**2))
    interp_indx = interp_indx_from_qrcp(Z, num_interp_points)
    # Solve for THC factors.
    solution = solve_for_thc_factors(
        mf_inst,
        interp_indx,
        cell_periodic_mo,
        grid_points,
        single_translation=single_translation,
        verbose=verbose,
    )
    return solution


def solve_for_thc_factors(
    mf_inst,
    interp_points_index,
    cell_periodic_mo,
    grid_points,
    single_translation=True,
    verbose=True,
):
    """
    Solve for THC factors given interpolating points
    """
    assert isinstance(mf_inst.with_df, df.FFTDF), "mf object must use FFTDF"
    if single_translation:
        chi, zeta, xi, G_mapping = kpoint_isdf_single_translation(
            mf_inst.with_df,
            interp_points_index,
            mf_inst.kpts,
            cell_periodic_mo,
            grid_points,
            only_unique_G=True,
        )
    else:
        chi, zeta, xi, G_mapping = kpoint_isdf_double_translation(
            mf_inst.with_df,
            interp_points_index,
            mf_inst.kpts,
            cell_periodic_mo,
            grid_points,
        )
    num_interp_points = len(interp_points_index)
    num_kpts = len(mf_inst.kpts)
    num_mo = mf_inst.mo_coeff[0].shape[-1]  # assuming the same for each k-point
    assert chi.shape == (num_interp_points, num_kpts * num_mo)
    # go from Rki -> kiR
    chi = chi.reshape((num_interp_points, num_kpts, num_mo))
    chi = chi.transpose((1, 2, 0))

    return chi, zeta, xi, G_mapping
