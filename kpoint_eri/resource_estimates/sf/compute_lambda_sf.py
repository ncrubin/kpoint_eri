import numpy as np
from itertools import product
import copy

from kpoint_eri.resource_estimates import sf
from kpoint_eri.resource_estimates.sf.ncr_integral_helper import NCRSingleFactorizationHelper

def compute_lambda(
        hcore,
        chol,
        kpoints,
        momentum_map,
        nmo_pk,
        ):
    nmo = sum(nmo_pk)
    eris = np.zeros((nmo,)*4, dtype=np.complex128)
    num_kpoints = momentum_map.shape[0]
    offsets = np.cumsum(nmo_pk) - nmo_pk[0]
    nchol_pk = [L.shape[-1] for L in chol]
    lambda_W = 0.0
    for iq in range(num_kpoints):
        An, Bn = sf.build_ABn(chol[iq,:,:,], iq, momentum_map)
        # sum_q sum_n (sum_{pq} |Re{A_{pq}^n}| + |Im{A_{pq}^n|)^2
        lambda_W += np.sum(np.einsum('npq->n', np.abs(An.real)+np.abs(An.imag))**2.0)
        lambda_W += np.sum(np.einsum('npq->n', np.abs(Bn.real)+np.abs(Bn.imag))**2.0)

    # TODO check prefactor.
    lambda_W *= 0.25

    for ik in range(num_kpoints):
        h1b = hcore[ik]
        h2b = np.zeros_like(h1b)
        for ik_prime in range(num_kpoints):
            eri_pqrs = sf.build_eris_kpt(chol[0], ik, ik_prime)
            h2b += np.einsum('pqrr->pq', eri_pqrs, optimize=True)
        T = h1b - 0.5 * h2b
        lambda_T = np.sum(np.abs(T.real) + np.abs(T.imag))

    lambda_tot = lambda_T + lambda_W
    return lambda_tot, lambda_T, lambda_W, sum(nchol_pk)


def compute_lambda_ncr(hcore, sf_obj: NCRSingleFactorizationHelper):
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

    two-body term V = 0.5 sum_{Q}sum_{n}(A_{n}(Q)^2 +_ B_{n}(Q)^2)
    or V = 0.5 sum_{Q}sum_{n'}W_{n}(Q)^{2} where n' is twice the range of n.
    lambda is 0.25sum_{Q}sum_{n'}(sum_{p,q}^{N_{k}N/2}|Re[W_{p,q}(Q)^{n}]| + |Im[W_{pq}(Q)^{n}]|)^{2}
    note the n' sum implying W is A or B.  See note for why 0.25 in front.

    :param hcore: List len(kpts) long of nmo x nmo complex hermitian arrays
    :param sf_obj: SingleFactorization object.    
    """
    kpts = sf_obj.kmf.kpts
    nkpts = len(kpts)
    one_body_mat = np.empty((len(kpts)), dtype=object)
    lambda_one_body = 0.

    old_naux = sf_obj.naux # need to reset naux for one-body computation
    sf_obj.naux = sf_obj.chol[0, 0].shape[0]

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

    sf_obj.naux = old_naux  # reset naux to original value
    # this part needs to change 
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
