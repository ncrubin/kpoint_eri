import numpy as np
from itertools import product

from kpoint_eri.resource_estimates import utils, sparse
from kpoint_eri.resource_estimates.sparse.ncr_sparse_integral_helper import NCRSSparseFactorizationHelper

def compute_lambda(
        pyscf_mf,
        integrals=None,
        localization='ibo',
        threshold=1e-5):
    kpoints = pyscf_mf.kpts
    mo_coeffs = pyscf_mf.mo_coeff
    momentum_map = utils.build_momentum_transfer_mapping(pyscf_mf.cell,
                                                   pyscf_mf.kpts)
    num_kpoints = len(kpoints)
    nmo_pk = [C.shape[1] for C in mo_coeffs]
    lambda_V = 0.0
    num_nnz = 0
    for iq in range(num_kpoints):
        for ikp, iks in product(range(num_kpoints), repeat=2):
            ikq = momentum_map[iq, ikp]
            ikr = momentum_map[iq, iks]
            kpt_pqrs = [kpoints[ik] for ik in [ikp,ikq,ikr,iks]]
            mos_pqrs = [mo_coeffs[ik] for ik in [ikp,ikq,ikr,iks]]
            mos_shape = [C.shape[1] for C in mos_pqrs]
            if integrals is None:
                eri_pqrs = sparse.build_eris_kpt(
                        pyscf_mf.with_df,
                        mos_pqrs,
                        kpt_pqrs,
                        compact=False)
            else:
                eri_pqrs = integrals[(ikq,ikp,iks)]
            eri_pqrs[abs(eri_pqrs) < threshold] = 0.0
            num_nnz += np.sum(eri_pqrs > threshold)
            lambda_V += np.sum(np.abs(eri_pqrs))

    # factor of 1/4 in hamiltonian gets multiplied by factor of 4 for spin
    # summation. so no further prefactor of num_nnz or lambda

    tot_size = num_kpoints**3 * max(nmo_pk)**4
    hcore = pyscf_mf.get_hcore(pyscf_mf.cell, kpts=pyscf_mf.kpts)
    lambda_T = 0.0
    for ik in range(num_kpoints):
        h1b = mo_coeffs[ik].conj().T @ hcore[ik] @ mo_coeffs[ik]
        h2b = np.zeros_like(h1b)
        for ik_prime in range(num_kpoints):
            kpt_pqrs = [
                    kpoints[ik],
                    kpoints[ik],
                    kpoints[ik_prime],
                    kpoints[ik_prime]
                    ]
            mos_pqrs = [
                    mo_coeffs[ik],
                    mo_coeffs[ik],
                    mo_coeffs[ik_prime],
                    mo_coeffs[ik_prime]
                    ]
            mos_shape = [C.shape[1] for C in mos_pqrs]
            if integrals is None:
                eri_pqrs = sparse.build_eris_kpt(
                        pyscf_mf.with_df,
                        mos_pqrs,
                        kpt_pqrs,
                        compact=False).reshape(mos_shape)
            else:
                eri_pqrs = integrals[(0,ik,ik_prime)].reshape(mos_shape)
            eri_pqrs[abs(eri_pqrs) < threshold] = 0.0
            h2b += np.einsum('pqrr->pq', eri_pqrs, optimize=True)
        T = h1b - 0.5 * h2b
        lambda_T = np.sum(np.abs(T))

    lambda_tot = lambda_T + lambda_V
    return lambda_tot, lambda_T, lambda_V, num_nnz, 1-num_nnz/tot_size


def compute_lambda_ncr(hcore, sparse_int_obj: NCRSSparseFactorizationHelper):
    """
    Compute lambda value for sparse method

    :param hcore: array of hcore(k) by kpoint. k-point order 
                  is pyscf order generated for this problem.
    :sparse_int_obj: The sparse integral object that is used
                     to compute eris and the number of unique
                     terms.
    :returns: total-lambda, one-Body lambda, two-body lambda, 
              number of unique terms above threshold
    """
    kpts = sparse_int_obj.kmf.kpts
    nkpts = len(kpts)
    one_body_mat = np.empty((len(kpts)), dtype=object)
    lambda_one_body = 0.

    import time
    for kidx in range(len(kpts)):
        # matrices for - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk) 
        # and  + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
        h1_pos = np.zeros_like(hcore[kidx])
        h1_neg = np.zeros_like(hcore[kidx])
        for qidx in range(len(kpts)):
            # - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk) 
            # start_time = time.time()
            eri_kqqk_pqrs = sparse_int_obj.get_eri_exact([kidx, qidx, qidx, kidx]) 
            # end_time = time.time()
            # print("Time for exact exchange like integral calc ", end_time - start_time)
            h1_neg -= np.einsum('prrq->pq', eri_kqqk_pqrs, optimize=True) / nkpts
            # + sum_{Q}sum_{r}(pkqk|rQrQ)
            # start_time = time.time()
            eri_kkqq_pqrs = sparse_int_obj.get_eri_exact([kidx, kidx, qidx, qidx])  
            # end_time = time.time()
            # print("Time for exact charge like integral calc ", end_time - start_time)

            h1_pos += np.einsum('pqrr->pq', eri_kkqq_pqrs) / nkpts

        one_body_mat[kidx] = hcore[kidx] + 0.5 * h1_neg + h1_pos
        lambda_one_body += np.sum(np.abs(one_body_mat[kidx].real)) + np.sum(np.abs(one_body_mat[kidx].imag))
    
    lambda_two_body = 0
    nkpts = len(kpts)
    # recall (k, k-q|k'-q, k')
    for kidx in range(nkpts):
        for kpidx in range(nkpts):
            for qidx in range(nkpts):                 
                kmq_idx = sparse_int_obj.k_transfer_map[qidx, kidx]
                kpmq_idx = sparse_int_obj.k_transfer_map[qidx, kpidx]
                start_time = time.time()
                test_eri_block = sparse_int_obj.get_eri([kidx, kmq_idx, kpmq_idx, kpidx]) / nkpts
                end_time = time.time()
                # print("time for int block ", end_time - start_time)
                lambda_two_body += np.sum(np.abs(test_eri_block.real)) + np.sum(np.abs(test_eri_block.imag))

    lambda_tot = lambda_one_body + lambda_two_body
    return lambda_tot, lambda_one_body, lambda_two_body, sparse_int_obj.get_total_unique_terms_above_thresh()
