import numpy as np
from itertools import product

from kpoint_eri.resource_estimates import utils, sparse

def compute_lambda(
        hcore,
        sparse_inst,
        momentum_map,
        ):
    """
    Compute one-body and two-body lambda for qubitization of
    single-factorized Hamiltonian.

    one-body term h_pq(k) = hcore_{pq}(k)
                            - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk)
    The first term is the kinetic energy + pseudopotential (or electron-nuclear),
    second term is from rearranging two-body operator into chemist charge-charge
    type notation.

    We assume hcore and sparse_inst have had elements < sparse threshold set to
    zero prior to entry to function.

    :param hcore: List len(kpts) long of nmo x nmo complex hermitian arrays
    :param sparse_instance: Instance of SparseKpointIntegrals
    :param momentum_map: Maps k-q to appropriate k-point index.
    """
    lambda_V = 0.0
    num_nnz = 0
    num_kpoints = len(hcore)
    for iq in range(num_kpoints):
        for ikp, iks in product(range(num_kpoints), repeat=2):
            ikq = momentum_map[iq, ikp]
            ikr = momentum_map[iq, iks]
            ikpts = [ikp, ikq, ikr, iks]
            eri_pqrs = sparse_inst.get_eri(ikpts) / num_kpoints
            num_nnz += np.sum(np.abs(eri_pqrs) > 0)
            lambda_V += np.sum(np.abs(eri_pqrs))

    # factor of 1/4 in hamiltonian gets multiplied by factor of 4 for spin
    # summation. so no further prefactor of num_nnz or lambda
    lambda_T = 0.0
    for ik in range(num_kpoints):
        h2b = np.zeros_like(hcore[ik])
        for qidx in range(num_kpoints):
            # (p k r Q | r Q q k)
            ikpts = [ik, qidx, qidx, ik]
            eri_pqrs = sparse_inst.get_eri(ikpts) / num_kpoints
            h2b += np.einsum('prrq->pq', eri_pqrs, optimize=True)
        # Could probably re-sparsify here but not sure if worth it.
        T = hcore[ik] - 0.5 * h2b
        num_nnz += np.sum(np.abs(T) > 0)
        lambda_T += np.sum(np.abs(T))

    lambda_tot = lambda_T + lambda_V
    return lambda_tot, lambda_tot, lambda_tot, num_nnz



# FDM: I think this is for SF not sparse?
def compute_lambda_ncr(hcore, sf_obj):
    kpts = sf_obj.kmf.kpts
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
            eri_kqqk_pqrs = sf_obj.get_eri([kidx, qidx, qidx, kidx]) 
            h1_neg -= np.einsum('prrq->pq', eri_kqqk_pqrs, optimize=True) / nkpts
            # + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
            eri_kkqq_pqrs = sf_obj.get_eri([kidx, kidx, qidx, qidx])  
            h1_pos += np.einsum('pqrr->pq', eri_kkqq_pqrs) / nkpts

        one_body_mat[kidx] = hcore[kidx] - 0.5 * h1_neg + 0.5 * h1_pos
        one_eigs, _ = np.linalg.eigh(one_body_mat[kidx])
        lambda_one_body += np.sum(np.abs(one_eigs))
    
    lambda_two_body = 0
    for qidx in range(len(kpts)):
        # A and B are W
        A, B = sf_obj.build_AB_from_chol(qidx) # [naux, nao * nk, nao * nk]
        A /= np.sqrt(nkpts)
        B /= np.sqrt(nkpts)
        # sum_q sum_n (sum_{pq} |Re{A_{pq}^n}| + |Im{A_{pq}^n|)^2
        lambda_two_body += np.sum(np.einsum('npq->n', np.abs(A.real) + np.abs(A.imag))**2)
        lambda_two_body += np.sum(np.einsum('npq->n', np.abs(B.real) + np.abs(B.imag))**2)
    lambda_two_body *= 0.25

    lambda_tot = lambda_one_body + lambda_two_body
    return lambda_tot, lambda_one_body, lambda_two_body
