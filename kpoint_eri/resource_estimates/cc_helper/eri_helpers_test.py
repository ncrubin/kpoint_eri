import os
import numpy as np
from itertools import product

from functools import reduce
import itertools
import os
from xml.dom import NOT_FOUND_ERR
import numpy as np
from pyscf.pbc import gto, scf, cc, df, mp



from kpoint_eri.resource_estimates import sf
from kpoint_eri.resource_estimates import sparse
from kpoint_eri.resource_estimates import utils
from kpoint_eri.resource_estimates import cc_helper
from kpoint_eri.resource_estimates.cc_helper.eri_helpers import NCRSingleFactorizationHelper

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


def test_ncr_sf_helpers():
    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-hf-rev'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()

    name_prefix = ''
    basname = cell.basis
    pp_name = cell.pseudo

    kmesh = [1, 2, 3]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.chkfile = 'ncr_test_C_density_fitints.chk'
    mf.with_df._cderi_to_save = 'ncr_test_C_density_fitints_gdf.h5'
    mf.init_guess = 'chkfile'
    mf.kernel()

    from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
    mymp = mp.KMP2(mf)
    nmo = mymp.nmo
    nocc = mymp.nocc
    nvir = nmo - nocc

    Luv = cholesky_from_df_ints(mymp)  # [kpt, kpt, naux, nao, nao]

    approx_cc = cc.KRCCSD(mf)
    helper = NCRSingleFactorizationHelper(cholesky_factor=Luv, kmf=mf)
    from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_cc
    from kpoint_eri.resource_estimates.cc_helper import _ERIS
    approx_cc = build_cc(approx_cc, helper)
    eris = approx_cc.ao2mo(lambda x: x)
    emp2, _, _ = approx_cc.init_amps(eris)
    approx_cc.kernel()

    exact_cc = cc.KRCCSD(mf)
    eris = exact_cc.ao2mo()
    exact_emp2, _, _ = exact_cc.init_amps(eris)
    exact_cc.kernel()
    assert np.isclose(approx_cc.e_corr, exact_cc.e_corr)
    assert np.isclose(exact_emp2, emp2)


test_ncr_sf_helpers()