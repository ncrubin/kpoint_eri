from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Tuple

from kpoint_eri.factorizations.hamiltonian_utils import HamiltonianProperties
from kpoint_eri.resource_estimates.sparse.integral_helper_sparse import (
    SparseFactorizationHelper,
)

@dataclass
class SparseHamiltonianProperties(HamiltonianProperties):
    """Light container to store return values of compute_lambda function"""
    num_sym_unique: int

def compute_lambda(
    hcore: npt.NDArray, sparse_int_obj: SparseFactorizationHelper
) -> SparseHamiltonianProperties:
    """Compute lambda value for sparse method

    Arguments:
        hcore: array of hcore(k) by kpoint. k-point order
            is pyscf order generated for this problem.
        sparse_int_obj: The sparse integral object that is used
            to compute eris and the number of unique
            terms.

    Returns:
        lambda_tot: Total lambda
        lambda_one_body: One-body lambda
        lambda_two_body: Two-body lambda
        num_sym_unique: Number of symmetry unique terms.
    """
    kpts = sparse_int_obj.kmf.kpts
    nkpts = len(kpts)
    one_body_mat = np.empty((len(kpts)), dtype=object)
    lambda_one_body = 0.0

    import time

    for kidx in range(len(kpts)):
        # matrices for - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk)
        # and  + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
        h1_pos = np.zeros_like(hcore[kidx])
        h1_neg = np.zeros_like(hcore[kidx])
        for qidx in range(len(kpts)):
            # - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk)
            eri_kqqk_pqrs = sparse_int_obj.get_eri_exact([kidx, qidx, qidx, kidx])
            h1_neg -= np.einsum("prrq->pq", eri_kqqk_pqrs, optimize=True) / nkpts
            # + sum_{Q}sum_{r}(pkqk|rQrQ)
            eri_kkqq_pqrs = sparse_int_obj.get_eri_exact([kidx, kidx, qidx, qidx])

            h1_pos += np.einsum("pqrr->pq", eri_kkqq_pqrs) / nkpts

        one_body_mat[kidx] = hcore[kidx] + 0.5 * h1_neg + h1_pos
        lambda_one_body += np.sum(np.abs(one_body_mat[kidx].real)) + np.sum(
            np.abs(one_body_mat[kidx].imag)
        )

    lambda_two_body = 0
    nkpts = len(kpts)
    # recall (k, k-q|k'-q, k')
    for kidx in range(nkpts):
        for kpidx in range(nkpts):
            for qidx in range(nkpts):
                kmq_idx = sparse_int_obj.k_transfer_map[qidx, kidx]
                kpmq_idx = sparse_int_obj.k_transfer_map[qidx, kpidx]
                test_eri_block = (
                    sparse_int_obj.get_eri([kidx, kmq_idx, kpmq_idx, kpidx]) / nkpts
                )
                lambda_two_body += np.sum(np.abs(test_eri_block.real)) + np.sum(
                    np.abs(test_eri_block.imag)
                )

    lambda_tot = lambda_one_body + lambda_two_body
    sparse_data = SparseHamiltonianProperties(
        lambda_total=lambda_tot,
        lambda_one_body=lambda_one_body,
        lambda_two_body=lambda_two_body,
        num_sym_unique=sparse_int_obj.get_total_unique_terms_above_thresh(return_nk_counter=False)
    )
    return sparse_data 
