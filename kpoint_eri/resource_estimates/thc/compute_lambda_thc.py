import numpy as np
from itertools import product

from kpoint_eri.resource_estimates import sf
from kpoint_eri.resource_estimates.thc.integral_helper import KPTHCHelperDoubleTranslation

def compute_lambda_real(
        h1,
        etaPp,
        MPQ
        ):
    CprP = np.einsum("Pp,Pr->prP", etaPp,
                     etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    BprQ = np.tensordot(CprP, MPQ, axes=([2], [0]))

    # projecting into the THC basis requires each THC factor mu to be nrmlzd.
    # we roll the normalization constant into the central tensor zeta
    SPQ = etaPp.dot(
        etaPp.T)  # (nthc x norb)  x (norb x nthc) -> (nthc  x nthc) metric
    cP = np.diag(np.diag(
        SPQ))  # grab diagonal elements. equivalent to np.diag(np.diagonal(SPQ))
    # no sqrts because we have two normalized THC vectors (index by mu and nu)
    # on each side.
    MPQ_normalized = cP.dot(MPQ).dot(cP)  # get normalized zeta in Eq. 11 & 12

    lambda_z = np.sum(np.abs(MPQ_normalized)) * 0.5  # Eq. 13
    # NCR: originally Joonho's code add np.einsum('llij->ij', eri_thc)
    # NCR: I don't know how much this matters.
    # (il|lj) = PiPl PQ Ql Qj ->ij
    # FPQ = Pl Ql -> PQ
    # Pi PQ Qj PQ -> ij
    T  = h1
    T -= 0.5 * np.einsum(
            "Pi,Pl,PQ,Ql,Qj->ij",
            etaPp, etaPp, MPQ, etaPp, etaPp,
            optimize=True)
    T += np.einsum(
        "Pl,Pl,PQ,Qi,Qj->ij",
        etaPp, etaPp, MPQ, etaPp, etaPp,
        optimize=True)  # Eq. 3 + Eq. 18
    e = np.linalg.eigvalsh(T)  # only need eigenvalues
    lambda_T = np.sum(
        np.abs(e))  # Eq. 19. NOTE: sum over spin orbitals removes 1/2 factor

    lambda_tot = lambda_z + lambda_T  # Eq. 20

    return lambda_tot, lambda_T, lambda_z


def compute_lambda_ncr_v2(hcore, thc_obj: KPTHCHelperDoubleTranslation):
    """
    Compute one-body and two-body lambda for qubitization of 
    tensor hypercontraction LUC

    one-body term h_pq(k) = hcore_{pq}(k) 
                            - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk) 
    :param hcore: List len(kpts) long of nmo x nmo complex hermitian arrays
    :param thc_obj: Object of KPTHCHelperDoubleTranslation
    """
    kpts = thc_obj.kmf.kpts
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
            eri_kqqk_pqrs = thc_obj.get_eri_exact([kidx, qidx, qidx, kidx]) 
            h1_neg -= np.einsum('prrq->pq', eri_kqqk_pqrs, optimize=True) / nkpts
            # # + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
            # eri_kkqq_pqrs = thc_obj.get_eri_exact([kidx, kidx, qidx, qidx])  
            # h1_pos += np.einsum('pqrr->pq', eri_kkqq_pqrs) / nkpts

        one_body_mat[kidx] = hcore[kidx] + 0.5 * h1_neg # + h1_pos
        one_eigs, _ = np.linalg.eigh(one_body_mat[kidx])
        lambda_one_body += np.sum(np.abs(one_eigs))
 
    lambda_two_body = 0
    num_eigs = 0
    lambda_two_body = np.sum(np.abs(thc_obj.zeta.real))
    lambda_two_body = np.sum(np.abs(thc_obj.zeta.imag))
    lambda_two_body *= 2 * nkpts**2

    lambda_tot = lambda_one_body + lambda_two_body
    return lambda_tot, lambda_one_body, lambda_two_body
