"""Build k-point dependent Cholesky factorization using pyscf. Adapted from QMCPACK."""
import numpy as np

from pyscf.pbc import tools

def build_momentum_transfer_mapping(
        cell,
        kpoints
        ):
    # Define mapping momentum_transfer_map[Q][k1] = k2 that satisfies
    # k1 - k2 + G = Q.
    a = cell.lattice_vectors() / (2*np.pi)
    delta_k1_k2_Q = kpoints[:,None,None,:] - kpoints[None,:,None,:] - kpoints[None,None,:,:]
    delta_dot_a = np.einsum('wx,kpQx->kpQw', a, delta_k1_k2_Q)
    int_delta_dot_a = np.rint(delta_dot_a)
    # Should be zero if transfer is statisfied (2*pi*n)
    mapping = np.where(np.sum(np.abs(delta_dot_a-int_delta_dot_a), axis=3) < 1e-10)
    num_kpoints = len(kpoints)
    momentum_transfer_map = np.zeros((num_kpoints,)*2, dtype=np.int32)
    # Note index flip due to Q being first index in map but broadcasted last..
    momentum_transfer_map[mapping[1], mapping[0]] = mapping[2]

    return momentum_transfer_map


def generate_orbital_products(
        Q_index,
        kmf,
        Qmap,
        kpoints):
    num_kpoints = len(kpoints)
    # For given Q generate [phi_{pk}^* phi_{q k-Q}](G)
    num_G = np.prod(kmf.cell.mesh)
    # TODO: Make AO
    num_pq = [C.shape[1]*C.shape[1] for C in kmf.mo_coeff]
    assert len(np.unique(num_pq)) == 1, "Number of MO per kp differs, please account for this."
    rho_pq = np.zeros((num_kpoints, num_G, num_pq[0]), dtype=np.complex128)
    for k1 in range(num_kpoints):
        k2 = Qmap[Q_index, k1]
        rho_pq[k1] = kmf.with_df.get_ao_pairs_G(
                (kpoints[k1], kpoints[k2]),
                kpoints[k2] - kpoints[k1],
                compact=False)
        v_G = tools.get_coulG(
                kmf.cell,
                kpoints[k2] - kpoints[k1],
                mesh=kmf.cell.mesh)
        rho_pq[k1] *= np.sqrt(v_G[:, None] * kmf.cell.vol) / num_G

    return rho_pq

def locate_max_residual(
        residual,
        mom_trans_indx,
        momentum_map,
        num_mo_per_kpoint):
    assert len(np.unique(num_mo_per_kpoint)) == 1, "TODO: Fix this"
    loc_max_res = np.argmax(residual)
    num_pq = residual.shape[1]
    max_k1_indx = loc_max_res // num_pq # row index
    max_pq_indx = loc_max_res % num_pq # column index
    max_k2_indx = momentum_map[mom_trans_indx, max_kpoint_indx]
    max_res = residual[max_kpoint_index, max_pq_indx].real
    # max_p_indx = max_pq_indx // num_mo_per_kpoint[k1]
    # max_q_indx = max_pq_indx % num_mo_per_kpoint[k2]
    return max_res, (max_k1_indx, max_k2_indx), (max_pq_indx)

def build_eri_diagonal(
        mom_trans_indx,
        rho_pq,
        num_mo_per_kpoint,
        momentum_map):
    assert len(rho_pq.shape) == 3
    num_kpoints = rho_pq.shape[0]
    num_pq = rho_pq.shape[2]
    residual = np.zeros((num_kpoints, num_pq))
    for k1 in range(num_kpoints):
        k2 = momentum_map[mom_trans_indx,k1]
        residual[k1] = np.einsum(
                        'GI,GI->I',
                        rho_pq[k1],
                        rho_pq[k1].conj()).real

    return residual

def generate_eri_column(
        kmf,
        rho_pq,
        max_k3k4,
        max_pq,
        mom_trans_indx,
        mom_trans_map,
        kpoints):

    num_G = np.prod(kmf.cell.mesh)
    num_kpoints = len(kpoints)
    num_pq = rho_pq.shape[2]
    eri_column = np.zeros((num_kpoints, num_pq), dtype=np.complex128)
    k3, k4 = max_k3k4
    for ik1, k1 in enumerate(kpoints):
        ik2 = mom_trans_map[mom_trans_indx, ik1]
        # TODO only generate appropriate scalar.
        rho_34 = kmf.with_df.get_ao_pairs_G(
                (kpoints[k4], kpoints[k3]),
                kpoints[ik2] - kpoints[ik1],
                compact=False)[:,max_pq]
        v_G = tools.get_coulG(
                kmf.cell,
                kpoints[ik2] - kpoints[ik1],
                mesh=kmf.cell.mesh)
        rho_34 *= np.sqrt(v_G * kmf.cell.vol) / num_G
        eri_column[ik1] = np.einsum(
                            'GI,G->I',
                            rho_pq[ik1],
                            rho_34.conj(),
                            )
    return eri_column

def generate_kpoint_cholesky_factorization(
        kmf,
        kpoints,
        threshold=1e-5,
        max_chol_factor=10):
    num_mo_per_kpoint = [C.shape[1] for C in kmf.mo_coeff]
    mom_trans_map = build_momentum_transfer_mapping(kmf.cell, kpoints)
    for mom_trans_indx, mom_trans in enumerate(kpoints):
        max_num_chol = max_chol_factor * num_mo_per_kpoint[mom_trans_indx]
        chol_vecs_iq = np.zeros((max_num_chol, num_pq), dtype=np.complex128)
        # 1. Build (pq|G) for this Q
        rho_pq = generate_orbital_products(
                    mom_trans_indx,
                    kmf,
                    mom_transfer_map,
                    kpoints)
        num_pq = rho_pq.shape[2]
        # 2. Build Diagonal V[(pk_p qk_q), (pk_q rk_p)] = (pk+Q q k | pk'-Q qk')
        # Recall, we are factorizing matrix M[pq,sr] = \sum_X L_(pq,X) L_(sr,X)^* = (pq|rs)
        eri_diag = build_eri_diagonal(
                       mom_trans_indx,
                       rho_pq,
                       num_mo_per_kpoint,
                       mom_transfer_map)
        approx_eri_diag = np.zeros_like(eri_diag)
        max_res, max_k1k2, max_pq = locate_max_residual(
                                        eri_diag,
                                        mom_trans_indx,
                                        mom_trans_map,
                                        num_mo_per_kpoint
                                        )
        chol_vecs_iq[0] = generate_eri_column(
                             rho_pq,
                             mom_trans_indx,
                             mom_trans_map,
                             max_k1k2,
                             max_pq)
        while max_residual > threshold:
            pass
