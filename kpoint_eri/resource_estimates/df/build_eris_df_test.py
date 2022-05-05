import numpy as np
import os

from kpoint_eri.resource_estimates import df
from kpoint_eri.resource_estimates import utils

_file_path = os.path.dirname(os.path.abspath(__file__))

def test_sf_eris():
    ham = utils.read_cholesky_contiguous(
            _file_path + '/../sf/chol_diamond_nk4.h5',
            frac_chol_to_keep=0.9)

    df_factors = df.double_factorize(
            ham['chol'],
            ham['qk_k2'],
            ham['nmo_pk'])
