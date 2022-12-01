from functools import reduce
import os
import numpy as np
from itertools import product

from pyscf.pbc import gto, scf, mp, cc

from kpoint_eri.resource_estimates import sf
from kpoint_eri.resource_estimates import sparse
from kpoint_eri.resource_estimates import utils

from kpoint_eri.resource_estimates.sf.compute_lambda_sf import compute_lambda_ncr, compute_lambda_ncr2


_file_path = os.path.dirname(os.path.abspath(__file__))

def test_sf_eris():
    ham = utils.read_cholesky_contiguous(
            _file_path+'/chol_diamond_nk4.h5',
            frac_chol_to_keep=1.0)
    cell, kmf = utils.init_from_chkfile(_file_path+'/../sparse/diamond_221.chk')
    mo_coeffs = kmf.mo_coeff
    num_kpoints = len(mo_coeffs)
    kpoints = ham['kpoints']
    momentum_map = ham['qk_k2']
    chol = ham['chol']
    lambda_tot, lambda_T, lambda_W, nchol = sf.compute_lambda(
            ham['hcore'],
            chol,
            kpoints,
            momentum_map,
            ham['nmo_pk']
            )

    print(lambda_tot, lambda_T, lambda_W)

def lambda_calc():
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
    nkpts = len(kpts)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.chkfile = 'ncr_test_C_density_fitints.chk'
    mf.with_df._cderi_to_save = 'ncr_test_C_density_fitints_gdf.h5'
    mf.init_guess = 'chkfile'
    mf.kernel()

    from kpoint_eri.resource_estimates.sf.ncr_integral_helper import NCRSingleFactorizationHelper
    from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
    mymp = mp.KMP2(mf)
    Luv = cholesky_from_df_ints(mymp)
    helper = NCRSingleFactorizationHelper(cholesky_factor=Luv, kmf=mf)

    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mf.mo_coeff)])

    lambda_tot, lambda_one_body, lambda_two_body = compute_lambda_ncr2(hcore_mo, helper)

    from pyscf.pbc.tools.k2gamma import k2gamma
    from kpoint_eri.resource_estimates.utils.k2gamma import k2gamma
    supercell_mf = k2gamma(mf, make_real=False)
    supercell_mf.e_tot = supercell_mf.energy_tot()
    # supercell_mf.kernel()
    assert np.isclose(mf.e_tot, supercell_mf.e_tot / np.prod(kmesh))
    exit()
    # WIP

    supercell_mymp = mp.KMP2(supercell_mf)
    supercell_Luv = cholesky_from_df_ints(supercell_mymp)
    supercell_helper = NCRSingleFactorizationHelper(cholesky_factor=supercell_Luv, kmf=supercell_mf)

    supercell_hcore_ao = supercell_mf.get_hcore()
    supercell_hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), supercell_hcore_ao[k], mo)) for k, mo in enumerate(supercell_mf.mo_coeff)])

    sc_lambda_tot, sc_lambda_one_body, sc_lambda_two_body = compute_lambda_ncr2(supercell_hcore_mo, supercell_helper)

    # check l2 norm of one-body
    assert np.isclose(np.linalg.norm(supercell_hcore_mo), np.linalg.norm(np.array(hcore_mo).ravel()))
    print(np.linalg.norm(np.abs(supercell_Luv[0, 0]).ravel()))
    abs_squared_sum = 0
    for kidx, qidx in product(range(nkpts), repeat=2):
        abs_squared_sum += np.sum(np.abs(Luv[kidx, qidx])**2)
    print(np.sqrt(abs_squared_sum))
    exit()
    assert np.isclose(sc_lambda_one_body, lambda_one_body)
    assert np.isclose(sc_lambda_two_body, lambda_two_body)



if __name__ == "__main__":
    lambda_calc()
