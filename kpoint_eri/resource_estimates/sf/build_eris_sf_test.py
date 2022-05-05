import os

from kpoint_eri.resource_estimates import sf
from kpoint_eri.resource_estimates import utils

def test_sf_eris():
    ham = utils.read_cholesky_contiguous(
            'chol_diamond_nk4.h5',
            frac_chol_to_keep=0.9)
    eris = sf.kpoint_cholesky_eris(
            ham['chol'],
            ham['kpoints'],
            ham['qk_k2'],
            ham['nmo_pk'])
