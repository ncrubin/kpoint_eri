import numpy as np
import numpy.typing as npt
from typing import Tuple

from kpoint_eri.resource_estimates.thc.integral_helper_thc import (
    KPTHCHelperDoubleTranslation,
)


def compute_lambda_real(
    h1: npt.NDArray,
    etaPp: npt.NDArray,
    MPQ: npt.NDArray,
    chol: npt.NDArray,
) -> Tuple[float, float, float]:
    """Compute lambda assuming real THC factors (molecular way) but without
    needing a molecular object as in openfermion. Just pared down function from
    openfermion.

    Arguments:
        h1: one-body hamiltonian 
        etaPp: THC leaf tensor 
        MPQ: THC central tensor. 
        chol: Cholesky factors. 

    Returns:
        lambda_tot: Total lambda
        lambda_one_body: One-body lambda 
        lambda_two_body: Two-body lambda 
    """
    nthc = etaPp.shape[0]

    # projecting into the THC basis requires each THC factor mu to be nrmlzd.
    # we roll the normalization constant into the central tensor zeta
    SPQ = etaPp.dot(etaPp.T)  # (nthc x norb)  x (norb x nthc) -> (nthc  x nthc) metric
    cP = np.diag(
        np.diag(SPQ)
    )  # grab diagonal elements. equivalent to np.diag(np.diagonal(SPQ))
    # no sqrts because we have two normalized THC vectors (index by mu and nu)
    # on each side.
    MPQ_normalized = cP.dot(MPQ).dot(cP)  # get normalized zeta in Eq. 11 & 12

    lambda_z = np.sum(np.abs(MPQ_normalized)) * 0.5  # Eq. 13
    # NCR: originally Joonho's code add np.einsum('llij->ij', eri_thc)
    # NCR: I don't know how much this matters.
    T = (
        h1
        - 0.5 * np.einsum("nil,nlj->ij", chol, chol, optimize=True)
        + np.einsum("nll,nij->ij", chol, chol, optimize=True)
    )  # Eq. 3 + Eq. 18
    # e, v = np.linalg.eigh(T)
    e = np.linalg.eigvalsh(T)  # only need eigenvalues
    lambda_T = np.sum(
        np.abs(e)
    )  # Eq. 19. NOTE: sum over spin orbitals removes 1/2 factor

    lambda_tot = lambda_z + lambda_T  # Eq. 20

    # return nthc, np.sqrt(res), res, lambda_T, lambda_z, lambda_tot
    return lambda_tot, lambda_T, lambda_z


def compute_lambda(hcore: npt.NDArray, thc_obj: KPTHCHelperDoubleTranslation) -> Tuple[float, float, float]:
    """Compute one-body and two-body lambda for qubitization of
    tensor hypercontraction LCU.
    
    one-body term h_pq(k) = hcore_{pq}(k)
                            - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk)

    Args:
      hcore: List len(kpts) long of nmo x nmo complex hermitian arrays
      thc_obj: Object of KPTHCHelperDoubleTranslation

    Returns:
        lambda_tot: Total lambda
        lambda_one_body: One-body lambda 
        lambda_two_body: Two-body lambda 
    """
    kpts = thc_obj.kmf.kpts
    nkpts = len(kpts)
    one_body_mat = np.empty((len(kpts)), dtype=object)
    lambda_one_body = 0.0

    for kidx in range(len(kpts)):
        # matrices for - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk)
        # and  + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
        h1_pos = np.zeros_like(hcore[kidx])
        h1_neg = np.zeros_like(hcore[kidx])
        for qidx in range(len(kpts)):
            # - 0.5 * sum_{Q}sum_{r}(pkrQ|rQqk)
            eri_kqqk_pqrs = thc_obj.get_eri_exact([kidx, qidx, qidx, kidx])
            h1_neg -= np.einsum("prrq->pq", eri_kqqk_pqrs, optimize=True) / nkpts
            # # + 0.5 sum_{Q}sum_{r}(pkqk|rQrQ)
            # eri_kkqq_pqrs = thc_obj.get_eri_exact([kidx, kidx, qidx, qidx])
            # h1_pos += np.einsum('pqrr->pq', eri_kkqq_pqrs) / nkpts

        one_body_mat[kidx] = hcore[kidx] + 0.5 * h1_neg  # + h1_pos
        one_eigs, _ = np.linalg.eigh(one_body_mat[kidx])
        lambda_one_body += np.sum(np.abs(one_eigs))

    # projecting into the THC basis requires each THC factor mu to be normalized.
    # we roll the normalization constant into the central tensor zeta
    # no sqrts because we have two normalized thc vectors (index by mu and nu)
    # on each side.
    norm_kP = (
        np.einsum("kpP,kpP->kP", thc_obj.chi.conj(), thc_obj.chi, optimize=True) ** 0.5
    )
    lambda_two_body = 0
    for iq, zeta_Q in enumerate(thc_obj.zeta):
        # xy einsum subscript indexes G index.
        for ik in range(nkpts):
            ik_minus_q = thc_obj.k_transfer_map[iq, ik]
            Gpq = thc_obj.G_mapping[iq, ik]
            for ik_prime in range(nkpts):
                ik_prime_minus_q = thc_obj.k_transfer_map[iq, ik_prime]
                Gsr = thc_obj.G_mapping[iq, ik_prime]
                norm_left = norm_kP[ik] * norm_kP[ik_minus_q]
                norm_right = norm_kP[ik_prime_minus_q] * norm_kP[ik_prime]
                MPQ_normalized = (
                    np.einsum(
                        "P,PQ,Q->PQ",
                        norm_left,
                        zeta_Q[Gpq, Gsr],
                        norm_right,
                        optimize=True,
                    )
                    / nkpts
                )
                lambda_two_body += np.sum(np.abs(MPQ_normalized.real))
                lambda_two_body += np.sum(np.abs(MPQ_normalized.imag))
    lambda_two_body *= 2

    lambda_tot = lambda_one_body + lambda_two_body
    return lambda_tot, lambda_one_body, lambda_two_body
