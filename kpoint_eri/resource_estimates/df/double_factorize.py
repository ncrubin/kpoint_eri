from itertools import product
import numpy as np

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

def get_df_factor(mat, thresh):
    eigs, eigv = np.linalg.eigh(mat)
    normSC = np.sum(np.abs(eigs))
    ix = np.argsort(np.abs(eigs))[::-1]
    eigs = eigs[ix]
    eigv = eigv[:,ix]
    truncation = normSC * np.abs(eigs)
    to_zero = truncation < thresh
    eigs[to_zero] = 0.0
    eigv[:,to_zero] = 0.0
    return eigs, eigv

def double_factorize(
        chol,
        momentum_map,
        nmo_pk,
        df_thresh=1e-5,
        ):
    nmo_tot = int(sum(nmo_pk))
    nchol = chol.shape[-1]
    nk = chol.shape[0]
    # Build DF factors
    Us = np.zeros((nk, nchol, nmo_tot, nmo_tot), dtype=np.complex128)
    Vs = np.zeros((nk, nchol, nmo_tot, nmo_tot), dtype=np.complex128)
    lambda_U = np.zeros((nk, nchol, nmo_tot), dtype=np.complex128)
    lambda_V = np.zeros((nk, nchol, nmo_tot), dtype=np.complex128)
    for iq in range(nk):
        # print(iq)
        for nc in range(nchol):
            LQn = chol[iq,:,:,:,nc]
            A, B = build_AB(LQn, iq, momentum_map)
            eigs, eigv = get_df_factor(A, df_thresh)
            lambda_U[iq, nc] = eigs
            Us[iq, nc] = eigv
            eigs, eigv = get_df_factor(B, df_thresh)
            lambda_V[iq, nc] = eigs
            Vs[iq, nc] = eigv

    df_factors = {
            'U': Us,
            'lambda_U': lambda_U,
            'V': Vs,
            'lambda_V': lambda_V
            }
    return df_factors
