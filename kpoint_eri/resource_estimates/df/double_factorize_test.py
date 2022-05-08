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

    import time
    start = time.time()
    df_factors = df.double_factorize(
            ham['chol'],
            ham['qk_k2'],
            ham['nmo_pk'],
            df_thresh=0.0)
    start = time.time()
    df_factors_batched = df.double_factorize_batched(
            ham['chol'],
            ham['qk_k2'],
            ham['nmo_pk'],
            df_thresh=0.0)
    assert np.allclose(df_factors['lambda_U'], df_factors_batched['lambda_U'])
    assert np.allclose(df_factors['lambda_V'], df_factors_batched['lambda_V'])
