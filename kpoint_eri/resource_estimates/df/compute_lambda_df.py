from typing import Tuple
import numpy as np
from numpy.typing import npt

from kpoint_eri.resource_estimates.df.integral_helper_df import DFABKpointIntegrals

def compute_lambda(hcore: npt.NDArray, df_obj: DFABKpointIntegrals) -> Tuple[float, float, float, int]:
    """Compute one-body and two-body lambda for qubitization of
    single-factorized Hamiltonian.
    
    one-body term h_pq(k) = hcore_{pq}(k)
                            - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk)
                            + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
    The first term is the kinetic energy + pseudopotential (or electron-nuclear),
    second term is from rearranging two-body operator into chemist charge-charge
    type notation, and the third is from the one body term obtained when
    squaring the two-body A and B operators.

    Arguments:
      hcore: List len(kpts) long of nmo x nmo complex hermitian arrays
      df_obj: DFABKpointIntegrals integral helper.

    Returns:
        lambda_tot: Total lambda
        lambda_one_body: One-body lambda 
        lambda_two_body: Two-body lambda 
        num_eigs: Number of retained eigenvalues.
    """
    kpts = df_obj.kmf.kpts
    nkpts = len(kpts)
    one_body_mat = np.empty((len(kpts)), dtype=object)
    lambda_one_body = 0.

    for kidx in range(len(kpts)):
        # matrices for - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk) 
        # and  + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
        h1_pos = np.zeros_like(hcore[kidx])
        h1_neg = np.zeros_like(hcore[kidx])
        for qidx in range(len(kpts)):
            # - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk) 
            eri_kqqk_pqrs = df_obj.get_eri_exact([kidx, qidx, qidx, kidx]) 
            h1_neg -= np.einsum('prrq->pq', eri_kqqk_pqrs, optimize=True) / nkpts
            # + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
            eri_kkqq_pqrs = df_obj.get_eri_exact([kidx, kidx, qidx, qidx])  
            h1_pos += np.einsum('pqrr->pq', eri_kkqq_pqrs) / nkpts

        one_body_mat[kidx] = hcore[kidx] + 0.5 * h1_neg + h1_pos
        one_eigs, _ = np.linalg.eigh(one_body_mat[kidx])
        lambda_one_body += np.sum(np.abs(one_eigs))
    
    lambda_two_body = 0
    num_eigs = 0
    for qidx in range(len(kpts)):
        for nn in range(df_obj.naux):
            first_number_to_square = 0
            second_number_to_square = 0
            # sum up p,k eigenvalues
            for kidx in range(len(kpts)):
                # A and B are W
                if df_obj.amat_lambda_vecs[kidx, qidx, nn] is None:
                    continue
                eigs_a_fixed_n_q = df_obj.amat_lambda_vecs[kidx, qidx, nn] / np.sqrt(nkpts)
                eigs_b_fixed_n_q = df_obj.bmat_lambda_vecs[kidx, qidx, nn] / np.sqrt(nkpts)
                first_number_to_square += np.sum(np.abs(eigs_a_fixed_n_q)) 
                num_eigs += len(eigs_a_fixed_n_q)
                if eigs_b_fixed_n_q is not None:
                    second_number_to_square += np.sum(np.abs(eigs_b_fixed_n_q))
                    num_eigs += len(eigs_b_fixed_n_q)

            lambda_two_body += first_number_to_square**2 
            lambda_two_body += second_number_to_square**2  

    lambda_two_body *= 0.25 

    lambda_tot = lambda_one_body + lambda_two_body
    return lambda_tot, lambda_one_body, lambda_two_body, num_eigs
