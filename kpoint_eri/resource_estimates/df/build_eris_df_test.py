import numpy as np
import os

from kpoint_eri.resource_estimates import df
from kpoint_eri.resource_estimates import sf
from kpoint_eri.resource_estimates import utils

_file_path = os.path.dirname(os.path.abspath(__file__))

def test_df_eris():
    ham = utils.read_cholesky_contiguous(
            _file_path + '/../sf/chol_diamond_nk4.h5',
            frac_chol_to_keep=1.0)

    df_factors = df.double_factorize(
            ham['chol'],
            ham['qk_k2'],
            ham['nmo_pk'],
            df_thresh=0.0)

    kpoints = ham['kpoints']
    momentum_map = ham['qk_k2']
    nmo_pk = ham['nmo_pk']
    chol = ham['chol']
    iq  = 1
    ikp = 3
    iks = 0
    ikq = momentum_map[iq, ikp]
    ikr = momentum_map[iq, iks]
    eri_pqrs_sf = sf.build_eris_kpt(chol[iq], ikp, iks)
    Uiq = df_factors['U'][iq]
    lambda_Uiq = df_factors['lambda_U'][iq]
    Viq = df_factors['V'][iq]
    lambda_Viq = df_factors['lambda_V'][iq]
    offsets = np.cumsum(nmo_pk) - nmo_pk[0]
    P = slice(offsets[ikp], offsets[ikp] + nmo_pk[ikp])
    Q = slice(offsets[ikq], offsets[ikq] + nmo_pk[ikq])
    R = slice(offsets[ikr], offsets[ikr] + nmo_pk[ikr])
    S = slice(offsets[iks], offsets[iks] + nmo_pk[iks])
    eri_pqrs_df = df.build_eris_kpt(
            Uiq,
            lambda_Uiq,
            Viq,
            lambda_Viq,
            (P,Q,R,S),
            )
    assert np.allclose(eri_pqrs_df, eri_pqrs_sf)
