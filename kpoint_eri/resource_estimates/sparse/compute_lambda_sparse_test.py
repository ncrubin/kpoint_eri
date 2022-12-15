from functools import reduce
import numpy as np
import h5py
from pyscf.pbc import gto, scf, mp, cc, tools
from pyscf.pbc.lib.kpts_helper import loop_kkk, get_kconserv

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
    # from pyscf.pbc.scf.chkfile import load_scf
    # _, scf_dict = load_scf(mf.chkfile)
    # mf.mo_coeff = scf_dict['mo_coeff']
    # mf.mo_occ = scf_dict['mo_occ']
    # mf.mo_energy = scf_dict['mo_energy']
    # mf.e_tot = scf_dict['e_tot']
    # mf.with_df._cderi_to_save = 'ncr_test_C_density_fitints_gdf.h5'
    mf.init_guess = 'chkfile'
    mf.kernel()

    mymp = mp.KMP2(mf)
    Luv = cholesky_from_df_ints(mymp)
    helper = NCRSSparseFactorizationHelper(cholesky_factor=Luv, kmf=mf)

    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mf.mo_coeff)])

    lambda_tot, lambda_one_body, lambda_two_body, num_unique = compute_lambda_ncr(hcore_mo, helper)

    from kpoint_eri.resource_estimates.utils.k2gamma import k2gamma, get_phase
    supercell_mf = k2gamma(mf, make_real=False)
    dm0 = supercell_mf.make_rdm1()
    # supercell_mf.kernel(dm0)
    energy_tot_sc = supercell_mf.energy_tot()
    assert np.isclose(mf.e_tot, energy_tot_sc / np.prod(kmesh))

    supercell_mymp = mp.KMP2(supercell_mf)
    supercell_Luv = cholesky_from_df_ints(supercell_mymp)
    supercell_helper = NCRSSparseFactorizationHelper(cholesky_factor=supercell_Luv, kmf=supercell_mf)

    # Sanity check to see that k2gamma supercell AO hcore is same as literal
    # supercell hcore
    _, C = get_phase(cell, kpts)
    supercell_hcore_ao = supercell_mf.get_hcore()
    nkpts = np.prod(kmesh)
    nmo = mf.mo_coeff[0].shape[-1]
    kp_sc_hcore_ao = np.einsum("Rk,kij,Sk->RiSj", C, hcore_ao,
                               C.conj()).reshape((nmo*nkpts, nmo*nkpts))
    assert np.isclose(np.abs(np.max(supercell_hcore_ao.imag)), 0.0)
    assert np.isclose(np.abs(np.max(kp_sc_hcore_ao.imag)), 0.0)
    assert np.allclose(supercell_hcore_ao, kp_sc_hcore_ao)
    supercell_hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), supercell_hcore_ao[k], mo)) for k, mo in enumerate(supercell_mf.mo_coeff)])

    sc_lambda_tot, sc_lambda_one_body, sc_lambda_two_body, sc_num_unique = compute_lambda_ncr(supercell_hcore_mo, supercell_helper)
    print(sc_lambda_one_body)
    print(lambda_one_body)
    print(np.sum(np.abs(supercell_hcore_mo.real)) + np.sum(np.abs(supercell_hcore_mo.imag))  )
    print(np.sum([np.abs(hcore_mo[kk].real) + np.abs(hcore_mo[kk].imag) for kk in range(len(kpts))]))
    print(np.linalg.norm(supercell_hcore_mo))
    print(np.linalg.norm(np.array(hcore_mo).ravel()))


    # Sanity check Tpq by itself without any eri contribution
    norm_uc_T = sum(np.sum(np.abs(hk.real)+np.abs(hk.imag)) for hk in hcore_mo)
    norm_sc_T = sum(np.sum(np.abs(hk.real)+np.abs(hk.imag)) for hk in supercell_hcore_mo)
    supcell = supercell_mf.cell
    assert np.isclose(norm_uc_T, norm_sc_T)

    # Sanity check: Build Nk^3 eris directly and sum up outside of loop
    eris_uc = np.zeros((nkpts,)*3 + (nmo,)*4, dtype=np.complex128)
    kconserv = get_kconserv(cell, kpts)
    for k1, k2, k3 in loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        kpts = [k1, k2, k3, k4]
        eris_uc[k1, k2, k3] = helper.get_eri(kpts)

    eris_uc = eris_uc / nkpts
    norm_uc = np.linalg.norm(eris_uc.ravel())  # this computs sqrt(sum_i abs(i)) which is not what we want to do.
    eris_sc = supercell_helper.get_eri([0,0,0,0])
    norm_sc = np.linalg.norm(eris_sc.ravel())
    # Test 2-norm should be invariant wrt unitary.
    assert np.isclose(norm_uc, norm_sc)
    # Test the triple loop
    direct_uc = np.sum(np.abs(eris_uc.real)+np.abs(eris_uc.imag))
    assert np.isclose(direct_uc, lambda_two_body)

    assert np.isclose(sc_lambda_one_body, lambda_one_body)
    assert np.isclose(sc_lambda_two_body, lambda_two_body)

    print(num_unique)
    print(sc_num_unique)


def test_symmetric_ortho_localization():
    cell = gto.Cell()
    cell.atom = """
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf-rev"
    cell.a = """
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000"""
    cell.unit = "B"
    cell.verbose = 0
    cell.build()

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.with_df.mesh = cell.mesh
    mf.chkfile = "scf.chk"
    # mf.init_guess = "chkfile"
    mf.kernel()

    from pyscf.lo.orth import orth_ao
    overlaps = cell.pbc_intor("cint1e_ovlp_sph", hermi=1, kpts=kpts)
    c_orth = np.array([orth_ao(mf, s=ovlp_k) for ovlp_k in overlaps])

    mymp = mp.KMP2(mf)
    Luv = cholesky_from_df_ints(mymp)
    hcore_mo = [
        C.conj().T @ hcore @ C for (C, hcore) in zip(mf.mo_coeff, mf.get_hcore())
    ]
    # Overwrite mo coeffs with X = S^{-1/2}
    mf.mo_coeff = c_orth
    mymp_oao = mp.KMP2(mf)
    Luv_oao = cholesky_from_df_ints(mymp_oao)
    hcore_oao = [C.conj().T @ hcore @ C for (C, hcore) in zip(c_orth, mf.get_hcore())]
    for thresh in [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]:
        helper = NCRSSparseFactorizationHelper(
            cholesky_factor=Luv, kmf=mf, threshold=thresh
        )
        lambda_tot, lambda_one_body, lambda_two_body = compute_lambda_ncr(
            hcore_mo, helper
        )
        helper = NCRSSparseFactorizationHelper(
            cholesky_factor=Luv_oao, kmf=mf, threshold=thresh
        )
        lambda_tot_oao, lambda_one_body_oao, lambda_two_body_oao = compute_lambda_ncr(
            hcore_oao, helper
        )
        print(thresh, lambda_tot, lambda_tot_oao, lambda_one_body,
              lambda_one_body_oao, lambda_two_body, lambda_two_body_oao)

if __name__ == "__main__":
    test_ncr_lambda_sparse()
    # test_symmetric_ortho_localization()
