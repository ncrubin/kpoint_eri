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
    num_pq = np.prod([C.shape[1] for C in kmf.mo_coeff])
    rho_pq = np.zeros((num_kpoints, num_G, num_pq), dtype=np.complex128)
    for k1 in range(num_kpoints):
        k2 = Qmap[Q_index, k1]
        rho_pq[k1] = kmf.with_df.get_mo_pairs_G(
                (kmf.mo_coeff[k1], kmf.mo_coeff[k2]),
                (kpoints[k1],kpoints[k2]),
                kpoints[k2] - kpoints[k1],
                compact=False)
        v_G = tools.get_coulG(
                kmf.cell,
                kpoints[k2] - kpoints[k1],
                mesh=kmf.cell.mesh)
        rho_pq[k1] *= v_G[:, None] * kmf.cell.vol / (num_G**2.0)

    return rho_pq

def build_eri_diagonal(
        mom_trans_indx,
        rho_pq,
        num_mo_per_kpoint,
        momentum_map):
    assert len(rho_pq.shape) == 3
    num_kpoints = rho_pq.shape[0]
    num_pq = rho_pq.shape[2]
    residual = np.zeros((num_kpoints, num_pq))
    max_res = -1
    for k1 in range(num_kpoints):
        k2 = momentum_map[mom_trans_indx,k1]
        residual[k1] = np.einsum(
                        'GI,GI->I',
                        rho_pq[k1],
                        rho_pq[k1].conj()).real
        max_pq_indx = np.argmax(residual[k1])
        max_res_k = residual[k1, max_pq_indx].real
        if max_res_k > max_res:
            max_res = max_res_k
            k1_max = k1
            k2_max = k2
            p_max = max_pq_indx // num_mo_per_kpoint[k1]
            q_max = max_pq_indx % num_mo_per_kpoint[k2]

    return residual, (k1_max, k2_max), (p_max, q_max)

def generate_kpoint_cholesky_factorization(
        cell,
        kpts,
        hcore,
        mo_coeff):

    # 1. Build Diagonal V[(pk_p qk_q), (sk_s rk_r)] = (pk_p q k_q | rk_r sk_s)
    pass
