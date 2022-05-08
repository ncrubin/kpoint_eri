from itertools import product
import numpy as np

from kpoint_eri.resource_estimates import sf

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

def get_df_factor_batched(mat, thresh):
    eigs, eigv = np.linalg.eigh(mat)
    normSC = np.sum(np.abs(eigs), axis=1)
    ix = np.argsort(np.abs(eigs), axis=1)[:,::-1]
    eigs = np.take_along_axis(eigs, ix, axis=-1)
    eigv = np.take_along_axis(eigv, ix[:,None,:], axis=-1)
    truncation = normSC[:, None] * np.abs(eigs)
    to_zero = truncation < thresh
    eigs[to_zero] = 0.0
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
        for nc in range(nchol):
            LQn = chol[iq,:,:,:,nc]
            A, B = sf.build_AB(LQn, iq, momentum_map)
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

def double_factorize_batched(
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
        LQ = chol[iq]
        An, Bn = sf.build_ABn(LQ, iq, momentum_map)
        eigs, eigv = get_df_factor_batched(An, df_thresh)
        lambda_U[iq] = eigs
        Us[iq] = eigv
        eigs, eigv = get_df_factor_batched(Bn, df_thresh)
        lambda_V[iq] = eigs
        Vs[iq] = eigv

    df_factors = {
            'U': Us,
            'lambda_U': lambda_U,
            'V': Vs,
            'lambda_V': lambda_V
            }

    return df_factors
