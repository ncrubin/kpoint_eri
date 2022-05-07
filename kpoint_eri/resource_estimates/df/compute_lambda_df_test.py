import os
import numpy as np
from itertools import product

from kpoint_eri.resource_estimates import df
from kpoint_eri.resource_estimates import sparse
from kpoint_eri.resource_estimates import utils

_file_path = os.path.dirname(os.path.abspath(__file__))

def test_sf_eris():
    ham = utils.read_cholesky_contiguous(
            _file_path + '/../sf/chol_diamond_nk4.h5',
            frac_chol_to_keep=1.0)
    cell, kmf = utils.init_from_chkfile(_file_path+'/../sparse/diamond_221.chk')
    mo_coeffs = kmf.mo_coeff
    num_kpoints = len(mo_coeffs)
    kpoints = ham['kpoints']
    momentum_map = ham['qk_k2']
    chol = ham['chol']
    df_factors = df.double_factorize(
            ham['chol'],
            ham['qk_k2'],
            ham['nmo_pk'],
            df_thresh=0.0)
    lambda_tot, lambda_T, lambda_F = df.compute_lambda(
            ham['hcore'],
            df_factors,
            kpoints,
            momentum_map,
            ham['nmo_pk']
            )

    print(lambda_tot, lambda_T, lambda_F)
