import numpy as np
from itertools import product

from kpoint_eri.resource_estimates import sf

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
