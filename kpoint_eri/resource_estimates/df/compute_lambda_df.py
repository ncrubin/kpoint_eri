import numpy as np

from kpoint_eri.resource_estimates import df
from kpoint_eri.resource_estimates import utils

from kpoint_eri.resource_estimates.df.ncr_integral_helper_df import DFABKpointIntegrals

def compute_lambda(
        hcore,
        df_factors,
        kpoints,
        momentum_map,
        nmo_pk,
        ):
    nmo = sum(nmo_pk)
    eris = np.zeros((nmo,)*4, dtype=np.complex128)
    num_kpoints = momentum_map.shape[0]
    offsets = np.cumsum(nmo_pk) - nmo_pk[0]
    eigs_A = df_factors['lambda_U'] # Q, nchol, neig
    eigs_B = df_factors['lambda_V'] # Q, nchol, neig
    lambda_F  = np.sum(np.einsum('qnt->qn', np.abs(eigs_A))**2.0)
    lambda_F += np.sum(np.einsum('qnt->qn', np.abs(eigs_B))**2.0)
    lambda_F *= 0.5 # 1/8 * 4 (spin summation)
    num_eig  = np.sum(eigs_A > 0) # should be explicitly zerod before entry
    num_eig += np.sum(eigs_A > 0) # should be explicitly zerod before entry

    # one-body contribution only contains contributions from Q = 0
    Uiq = df_factors['U'][0]
    lambda_Uiq = df_factors['lambda_U'][0]
    Viq = df_factors['V'][0]
    lambda_Viq = df_factors['lambda_V'][0]
    lambda_T = 0.0
    for ik in range(num_kpoints):
        h1b = hcore[ik]
        h2b = np.zeros_like(h1b)
        for ik_prime in range(num_kpoints):
            P = slice(offsets[ik], offsets[ik] + nmo_pk[ik])
            Q = slice(offsets[ik], offsets[ik] + nmo_pk[ik])
            R = slice(offsets[ik_prime], offsets[ik_prime] + nmo_pk[ik_prime])
            S = slice(offsets[ik_prime], offsets[ik_prime] + nmo_pk[ik_prime])
            eri_pqrs = df.build_eris_kpt(
                    Uiq,
                    lambda_Uiq,
                    Viq,
                    lambda_Viq,
                    (P,Q,R,S),
                    )
            h2b += 0.5 * np.einsum('pqrr->pq', eri_pqrs, optimize=True)
        T = h1b + h2b
        e, _ = np.linalg.eigh(T)
        lambda_T = np.sum(np.abs(e))

    lambda_tot = lambda_T + lambda_F
    return lambda_tot, lambda_T, lambda_F, num_eig

def compute_lambda_ncr(hcore, df_obj: DFABKpointIntegrals):
    """
    Compute one-body and two-body lambda for qubitization of 
    single-factorized Hamiltonian.

    one-body term h_pq(k) = hcore_{pq}(k) 
                            - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk) 
                            + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
    The first term is the kinetic energy + pseudopotential (or electron-nuclear),
    second term is from rearranging two-body operator into chemist charge-charge
    type notation, and the third is from the one body term obtained when
    squaring the two-body A and B operators.

    :param hcore: List len(kpts) long of nmo x nmo complex hermitian arrays
    :param df_obj: Object of DFABKpointIntegrals
    """
    kpts = df_obj.kmf.kpts
    one_body_mat = np.empty((len(kpts)), dtype=object)
    lambda_one_body = 0.
    for kidx in range(len(kpts)):
        # matrices for - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk) 
        # and  + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
        h1_pos = np.zeros_like(hcore[kidx])
        h1_neg = np.zeros_like(hcore[kidx])
        for qidx in range(len(kpts)):
            # - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk) 
            eri_kqqk_pqrs = df_obj.get_eri([kidx, qidx, qidx, kidx]) 
            h1_neg -= np.einsum('prrq->pq', eri_kqqk_pqrs, optimize=True)
            # + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
            eri_kkqq_pqrs = df_obj.get_eri([kidx, kidx, qidx, qidx]) 
            h1_pos += np.einsum('pqrr->pq', eri_kkqq_pqrs)

        one_body_mat[kidx] = hcore[kidx] - 0.5 * h1_neg + 0.5 * h1_pos
        one_eigs, _ = np.linalg.eigh(one_body_mat[kidx])
        lambda_one_body += np.sum(np.abs(one_eigs))
    
    lambda_two_body = 0
    num_eigs = 0
    for qidx in range(len(kpts)):
        # A and B are W
        eigs_u_by_nc, eigs_v_by_nc = df_obj.df_factors['lambda_U'][qidx], df_obj.df_factors['lambda_V'][qidx]
        squared_sum_a_eigs = np.array([np.sum(np.abs(xx))**2 for xx in eigs_u_by_nc])
        squared_sum_b_eigs = np.array([np.sum(np.abs(xx))**2 for xx in eigs_v_by_nc])
        lambda_two_body += np.sum(squared_sum_a_eigs)
        lambda_two_body += np.sum(squared_sum_b_eigs)
        num_eigs += sum([len(xx) for xx in eigs_u_by_nc]) + sum([len(xx) for xx in eigs_v_by_nc])
    lambda_two_body *= 0.25

    lambda_tot = lambda_one_body + lambda_two_body
    return lambda_tot, lambda_one_body, lambda_two_body, num_eigs


    



