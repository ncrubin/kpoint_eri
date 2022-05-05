import os
import numpy as np
from itertools import product

from kpoint_eri.resource_estimates import sf
from kpoint_eri.resource_estimates import sparse
from kpoint_eri.resource_estimates import utils

_file_path = os.path.dirname(os.path.abspath(__file__))

def test_sf_eris():
    ham = utils.read_cholesky_contiguous(
            'chol_diamond_nk4.h5',
            frac_chol_to_keep=1.0)
    cell, kmf = utils.init_from_chkfile(_file_path+'/../sparse/diamond_221.chk')
    mo_coeffs = kmf.mo_coeff
    num_kpoints = len(mo_coeffs)
    kpoints = ham['kpoints']
    momentum_map = ham['qk_k2']
    chol = ham['chol']
    # for iq in range(1,2): # 4^3 is a bit slow
        # for ikp, iks in product(range(num_kpoints), repeat=2):
    iq  = 1
    ikp = 2
    iks = 0
    ikq = momentum_map[iq, ikp]
    ikr = momentum_map[iq, iks]
    eri_pqrs_sf = sf.build_eris_kpt(chol[iq], ikp, iks)
    kpt_pqrs = [kpoints[ik] for ik in [ikp,ikq,ikr,iks]]
    mos_pqrs = [mo_coeffs[ik] for ik in [ikp,ikq,ikr,iks]]
    mos_shape = [C.shape[1] for C in mos_pqrs]
    # eri_pqrs = sparse.build_eris_kpt(kmf, mos_pqrs, kpt_pqrs, compact=False)
    eri_pqrs_exact = sparse.build_eris_kpt(
            kmf,
            mos_pqrs,
            kpt_pqrs,
            compact=False).reshape(mos_shape)
    assert np.max(np.abs(eri_pqrs_sf-eri_pqrs_exact)) < 1e-5 # chol threshold
