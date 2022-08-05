import os
import numpy as np
from itertools import product


from kpoint_eri.resource_estimates import sf
from kpoint_eri.resource_estimates import sparse
from kpoint_eri.resource_estimates import utils
from kpoint_eri.resource_estimates import cc_helper

_file_path = os.path.dirname(os.path.abspath(__file__))

def test_sparse_helpers():
    cell, kmf = utils.init_from_chkfile(_file_path+'/../sparse/diamond_221.chk')
    kcc = cc_helper.build_krcc_sparse_eris(kmf, threshold=1e-2)
    kcc.max_cycle = 1
    kcc.kernel()

def test_sf_helpers():
    cell, kmf = utils.init_from_chkfile(_file_path+'/../sparse/diamond_221.chk')
    ham = utils.read_cholesky_contiguous(
            _file_path + '/../sf/chol_diamond_nk4.h5',
            frac_chol_to_keep=0.5)
    kcc = cc_helper.build_krcc_sf_eris(
            kmf,
            ham['chol'],
            ham['qk_k2'],
            ham['kpoints']
            )
    kcc.max_cycle = 1
    kcc.kernel()

def test_df_helpers():
    ham = utils.read_cholesky_contiguous(
            _file_path + '/../sf/chol_diamond_nk4.h5',
            frac_chol_to_keep=1.0)
    cell, kmf = utils.init_from_chkfile(_file_path+'/../sparse/diamond_221.chk')
    kcc = cc_helper.build_krcc_df_eris(
            kmf,
            ham['chol'],
            ham['qk_k2'],
            ham['kpoints'],
            ham['nmo_pk']
            )
    kcc.max_iter = 1
    kcc.kernel()

def test_thc_helpers():
    cell, kmf = utils.init_from_chkfile(_file_path+'/../sparse/diamond_221.chk')
    ham = utils.read_qmcpack_thc(_file_path + '/../thc/thc_4_4.h5')
    kcc = cc_helper.build_krcc_thc_eris(
            kmf,
            ham['orbs_pu'].copy(),
            ham['Muv']
            )
    kcc.max_cycle = 1
    # note supercell so divide by nk = 4
    kcc.kernel()
