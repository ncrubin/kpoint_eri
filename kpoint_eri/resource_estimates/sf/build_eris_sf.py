import numpy as np
from itertools import product

from kpoint_eri.resource_estimates.utils import build_momentum_transfer_mapping

def build_AB(Lqn, q_index, momentum_map):
    nmo = Lqn.shape[1]
    assert len(Lqn.shape) == 3
    num_kpoints = Lqn.shape[0]
    M = np.zeros((nmo*num_kpoints, nmo*num_kpoints), dtype=np.complex128)
    A = np.zeros((nmo*num_kpoints, nmo*num_kpoints), dtype=np.complex128)
    B = np.zeros((nmo*num_kpoints, nmo*num_kpoints), dtype=np.complex128)
    for kp in range(num_kpoints):
        kq = momentum_map[q_index, kp]
        for p, q in product(range(nmo), repeat=2):
            P = int(kp * nmo + p)
            Q = int(kq * nmo + q)
            M[P,Q] += Lqn[kp, p, q]
    A = 0.5  * (M + M.conj().T)
    B = 0.5j * (M - M.conj().T)
    assert np.linalg.norm(A-A.conj().T) < 1e-12
    assert np.linalg.norm(B-B.conj().T) < 1e-12
    return A, B

def build_ABn(Lq, q_index, momentum_map):
    nmo = Lq.shape[1]
    assert len(Lq.shape) == 4
    num_kpoints = Lq.shape[0]
    nchol = Lq.shape[-1]
    M = np.zeros((nchol, nmo*num_kpoints, nmo*num_kpoints), dtype=np.complex128)
    A = np.zeros((nchol, nmo*num_kpoints, nmo*num_kpoints), dtype=np.complex128)
    B = np.zeros((nchol, nmo*num_kpoints, nmo*num_kpoints), dtype=np.complex128)
    for kp in range(num_kpoints):
        kq = momentum_map[q_index, kp]
        for p, q in product(range(nmo), repeat=2):
            P = int(kp * nmo + p)
            Q = int(kq * nmo + q)
            M[:,P,Q] += Lq[kp, p, q, :]
    A = 0.5  * (M + M.transpose((0,2,1)).conj())
    B = 0.5j * (M - M.transpose((0,2,1)).conj())
    return A, B


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
