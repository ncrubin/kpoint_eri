import numpy as np
from itertools import product

from kpoint_eri.resource_estimates.utils import build_momentum_transfer_mapping

def build_eris_kpt(chol_q, ikp, iks):
    """
    Compute (momentum conserving) kpoint-integrals (pkp qkq | rkr sks) block
    (pkp qkp-Q | rks-Q sks) = \sum_n L[Q,ikp] L[Q,iks].conj()
    """
    Lkp_minu_q = chol_q[ikp]
    Lks_minu_q = chol_q[iks]
    eri_pqrs = np.einsum(
            'pqn,srn->pqrs',
            Lkp_minu_q,
            Lks_minu_q.conj(),
            optimize=True)
    return eri_pqrs

def kpoint_cholesky_eris(
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
    for iq in range(num_kpoints):
        for ikp, iks in product(range(num_kpoints), repeat=2):
            ikq = momentum_map[iq, ikp]
            ikr = momentum_map[iq, iks]
            P = slice(offsets[ikp], offsets[ikp] + nmo_pk[ikp])
            Q = slice(offsets[ikq], offsets[ikq] + nmo_pk[ikq])
            R = slice(offsets[ikr], offsets[ikr] + nmo_pk[ikr])
            S = slice(offsets[iks], offsets[iks] + nmo_pk[iks])
            eri_pqrs = build_eris_kpt(chol[iq], ikp, iks)
            eris[P,Q,R,S] = eri_pqrs

    return eris
