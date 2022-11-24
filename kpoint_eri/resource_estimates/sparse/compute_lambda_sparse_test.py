from functools import reduce
import numpy as np
from pyscf.pbc import gto, scf, mp, cc

from kpoint_eri.resource_estimates import sparse
from kpoint_eri.resource_estimates import utils
from kpoint_eri.resource_estimates.sparse.ncr_sparse_integral_helper import NCRSSparseFactorizationHelper
from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints

from kpoint_eri.resource_estimates.sparse.compute_lambda_sparse import compute_lambda_ncr

def test_lambda_sparse():
    cell, kmf = utils.init_from_chkfile('diamond_221.chk')
    lambda_tot, lambda_T, lambda_V  = sparse.compute_lambda(kmf)
    print(lambda_tot)

def test_ncr_lambda_sparse():
    cell = gto.Cell()
    cell.atom = '''
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
    cell.verbose = 0
    cell.build()

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.chkfile = 'ncr_test_C_density_fitints.chk'
    mf.with_df._cderi_to_save = 'ncr_test_C_density_fitints_gdf.h5'
    mf.init_guess = 'chkfile'
    mf.kernel()

    mymp = mp.KMP2(mf)
    Luv = cholesky_from_df_ints(mymp)
    helper = NCRSSparseFactorizationHelper(cholesky_factor=Luv, kmf=mf)

    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mf.mo_coeff)])

    lambda_tot, lambda_one_body, lambda_two_body = compute_lambda_ncr(hcore_mo, helper)

    from pyscf.pbc.tools.k2gamma import k2gamma
    from kpoint_eri.resource_estimates.utils.k2gamma import k2gamma
    supercell_mf = k2gamma(mf)
    supercell_mf.kernel()
    assert np.isclose(mf.e_tot, supercell_mf.e_tot / np.prod(kmesh))

    supercell_mymp = mp.KMP2(supercell_mf)
    supercell_Luv = cholesky_from_df_ints(supercell_mymp)
    supercell_helper = NCRSSparseFactorizationHelper(cholesky_factor=supercell_Luv, kmf=supercell_mf)

    supercell_hcore_ao = supercell_mf.get_hcore()
    supercell_hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), supercell_hcore_ao[k], mo)) for k, mo in enumerate(supercell_mf.mo_coeff)])

    sc_lambda_tot, sc_lambda_one_body, sc_lambda_two_body = compute_lambda_ncr(hcore_mo, helper)
    assert np.isclose(sc_lambda_one_body, lambda_one_body)
    assert np.isclose(sc_lambda_two_body, lambda_two_body)

if __name__ == "__main__":
    test_ncr_lambda_sparse()