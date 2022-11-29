from functools import reduce
import os
import numpy as np
from itertools import product

from pyscf.pbc import gto, scf, mp, cc
from pyscf.pbc.df import FFTDF

from kpoint_eri.resource_estimates import df
from kpoint_eri.resource_estimates import sparse
from kpoint_eri.resource_estimates import utils

from kpoint_eri.resource_estimates.df.compute_lambda_df import compute_lambda_ncr, compute_lambda_ncr_v2
from kpoint_eri.resource_estimates.df.ncr_integral_helper_df import DFABV2KpointIntegrals
from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints


_file_path = os.path.dirname(os.path.abspath(__file__))

def test_compute_lambda_df():
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
            df_thresh=1e-5)
    lambda_tot, lambda_T, lambda_F = df.compute_lambda(
            ham['hcore'],
            df_factors,
            kpoints,
            momentum_map,
            ham['nmo_pk']
            )

    df_factors = df.double_factorize_batched(
            ham['chol'],
            ham['qk_k2'],
            ham['nmo_pk'],
            df_thresh=1e-5)
    lambda_tot_batch, lambda_T_, lambda_F_ = df.compute_lambda(
            ham['hcore'],
            df_factors,
            kpoints,
            momentum_map,
            ham['nmo_pk']
            )

    assert lambda_tot - lambda_tot_batch < 1e-12

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
    cell.verbose = 4
    cell.build()

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.chkfile = 'ncr_test_C_density_fitints.chk'
    mf.with_df._cderi_to_save = 'ncr_test_C_density_fitints_gdf.h5'
    mf.init_guess = 'chkfile'
    mf.kernel()

    from kpoint_eri.resource_estimates.df.ncr_integral_helper_df import DFABKpointIntegrals
    from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
    mymp = mp.KMP2(mf)
    Luv = cholesky_from_df_ints(mymp)
    helper = DFABKpointIntegrals(cholesky_factor=Luv, kmf=mf)
    helper.double_factorize(thresh=1.0E-4)

    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mf.mo_coeff)])

    lambda_tot, lambda_one_body, lambda_two_body, num_eigs = compute_lambda_ncr(hcore_mo, helper)
    print(lambda_tot)
    print(num_eigs)

def lambda_v2_calc():
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
    cell.verbose = 4
    cell.build()

    from pyscf.pbc.scf.chkfile import load_scf
    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)

    mf = scf.KRHF(cell, kpts)#.rs_density_fit()
    mydf = FFTDF(mf.cell, kpts=kpts)
    mf.with_df = mydf
    mf.chkfile = 'fft_ncr_test_C_density_fitints.chk'
    # cell, scf_dict = load_scf(mf.chkfile)
    # mf.e_tot = scf_dict['e_tot']
    # mf.kpts = scf_dict['kpts']
    # mf.mo_coeff = scf_dict['mo_coeff']
    # mf.mo_energy = scf_dict['mo_energy']
    # mf.mo_occ = scf_dict['mo_occ']
    # mf.with_df._cderi_to_save = 'ncr_test_C_density_fitints_gdf.h5'
    mf.init_guess = 'chkfile'
    mf.kernel()

    for kidx, qidx in product(range(len(kpts)), repeat=2):
        ijG = mf.with_df.get_mo_pairs_G([mf.mo_coeff[kidx], mf.mo_coeff[qidx]], kpts=mf.kpts[[kidx, qidx], :])
        print(ijG.shape)
    exit()

    mymp = mp.KMP2(mf)
    mymp.density_fit()
    Luv = cholesky_from_df_ints(mymp)
    helper = DFABV2KpointIntegrals(cholesky_factor=Luv, kmf=mf)
    helper.double_factorize(thresh=1.0E-5)

    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mf.mo_coeff)])

    lambda_tot, lambda_one_body, lambda_two_body, num_eigs = compute_lambda_ncr_v2(hcore_mo, helper)
    print(lambda_tot)
    print(num_eigs)

    # from pyscf.pbc.tools.k2gamma import k2gamma
    # supercell_mf = k2gamma(mf)
    # # supercell_mf.kernel()
    # supercell_mf.energy_elec()
    # supercell_mf.e_tot = supercell_mf.energy_tot()
    # print(supercell_mf.e_tot / np.prod(kmesh))
    # print(mf.e_tot)
    # print()
    from kpoint_eri.resource_estimates.utils.k2gamma import k2gamma
    from pyscf.pbc.df import RSDF
    supercell_mf = k2gamma(mf)
    supercell_mf.verbose = 10
    mydf = RSDF(supercell_mf.cell, supercell_mf.kpts)
    # mydf.mesh = [7, 7, 7 * 3]
    # mydf.mesh_compact = [7, 7, 7 * 3]
    # mydf.omega = 0.522265625
    supercell_mf.with_df = mydf
    # supercell_mf.max_cycle = 1
    supercell_mf.kernel()
    supercell_mf.energy_elec()
    supercell_mf.e_tot = supercell_mf.energy_tot()
    print(supercell_mf.e_tot / np.prod(kmesh))
    print(mf.e_tot)

    non_zero_idx = np.where(supercell_mf.mo_occ[0] != 0)[0]
    zero_idx = np.where(supercell_mf.mo_occ[0] == 0)[0]
    from pyscf import lo
    lmo_occ = lo.PM(supercell_mf.cell, supercell_mf.mo_coeff[0][:,non_zero_idx]).kernel()
    lmo_virt = lo.PM(supercell_mf.cell, supercell_mf.mo_coeff[0][:,zero_idx]).kernel()

    supercell_mf.mo_coeff[0][:, non_zero_idx] = lmo_occ
    supercell_mf.mo_coeff[0][:, zero_idx] = lmo_virt
    assert np.isclose(mf.e_tot, supercell_mf.e_tot / np.prod(kmesh))

    supercell_mymp = mp.KMP2(supercell_mf)
    supercell_Luv = cholesky_from_df_ints(supercell_mymp)
    supercell_helper = DFABV2KpointIntegrals(cholesky_factor=supercell_Luv, kmf=supercell_mf)
    supercell_helper.double_factorize(thresh=1.0E-5)
    sc_nk = supercell_helper.nk
    sc_help = supercell_helper
 
    for kidx in range(sc_nk):
        for kpidx in range(sc_nk):
            for qidx in range(sc_nk):                 
                kmq_idx = supercell_helper.k_transfer_map[qidx, kidx]
                kpmq_idx = supercell_helper.k_transfer_map[qidx, kpidx]
                exact_eri_block = supercell_helper.get_eri_exact([kidx, kmq_idx, kpmq_idx, kpidx])
                test_eri_block = supercell_helper.get_eri([kidx, kmq_idx, kpmq_idx, kpidx])
                # assert np.allclose(exact_eri_block, test_eri_block)
                print(np.allclose(exact_eri_block, test_eri_block))

    supercell_hcore_ao = supercell_mf.get_hcore()
    supercell_hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), supercell_hcore_ao[k], mo)) for k, mo in enumerate(supercell_mf.mo_coeff)])

    sc_lambda_tot, sc_lambda_one_body, sc_lambda_two_body, sc_num_eigs = compute_lambda_ncr_v2(supercell_hcore_mo, sc_help)
    print(sc_lambda_one_body, lambda_one_body)
    print(sc_lambda_two_body, lambda_two_body)
    print(sc_num_eigs, num_eigs)
    exit()

    assert np.isclose(sc_lambda_one_body, lambda_one_body)
    assert np.isclose(sc_lambda_two_body, lambda_two_body)


def fftdf_reconstruct():
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
    cell.verbose = 4
    cell.mesh = [4, 4, 4]
    cell.build()

    from pyscf.pbc.scf.chkfile import load_scf
    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    mf = scf.KRHF(cell, kpts)#.rs_density_fit()
    mydf = FFTDF(mf.cell, kpts=kpts)
    mf.with_df = mydf
    mf.chkfile = 'fft_ncr_test_C_density_fitints.chk'
    mf.init_guess = 'chkfile'
    mf.kernel()
    nmo = mf.mo_coeff[0].shape[-1]
    naux = mf.with_df.get_naoaux() // 2

    from pyscf.pbc import ao2mo
    Luv = np.zeros((nkpts, nkpts, naux, nmo, nmo), dtype=np.complex128)
    for kidx, qidx in product(range(len(kpts)), repeat=2):
        ijG = mf.with_df.get_mo_pairs_G([mf.mo_coeff[kidx], mf.mo_coeff[qidx]], kpts=mf.kpts[[kidx, qidx], :])
        Luv[kidx, qidx, :, :, :] = ijG.reshape((-1, nmo, nmo))
        if kidx == qidx and kidx == 0:
            eri_true = mf.with_df.ao2mo([mf.mo_coeff[0], mf.mo_coeff[0], mf.mo_coeff[0],
            mf.mo_coeff[0]]).reshape([nmo] * 4)
            print(eri_true.shape)
            eri_test = np.einsum('nqp,nij->pqij', Luv[0, 0].conj(), Luv[0, 0], optimize=True)
            print(np.linalg.norm(eri_true - eri_test))
            exit()

        # for i, j in product(range(nmo), repeat=2):
        #     ijG_test = mf.with_df.get_mo_pairs_G([mf.mo_coeff[kidx][:, [i]], mf.mo_coeff[qidx][:, [j]]], kpts=mf.kpts[[kidx, qidx], :])
        #     ij_idx = i * nmo + j
        #     assert np.allclose(ijG[:, ij_idx], ijG_test.flatten())
        #     assert np.allclose(ijG[:, ij_idx], B[kidx, qidx, :, i, j])

    from pyscf.pbc.lib.kpts_helper import get_kconserv
    kconserv = get_kconserv(cell, kpts) 
    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):                 
                ks = kconserv[kp, kq, kr]
                eri_kpt = mf.with_df.ao2mo([mf.mo_coeff[i] for i in (kp,kq,kr,ks)],
                                            [kpts[i] for i in (kp,kq,kr,ks)])
                eri_kpt = eri_kpt.reshape([nmo]*4)

                eri_test = np.einsum('npq,nsr->pqrs', Luv[kp, kq], Luv[ks, kr].conj(), optimize=True)
                print(np.linalg.norm(eri_test - eri_kpt))
                assert np.allclose(eri_test, eri_kpt)
                # kmq_idx = dfk_inst.k_transfer_map[qidx, kidx]
                # kpmq_idx = dfk_inst.k_transfer_map[qidx, kpidx]
                # exact_eri_block = dfk_inst.get_eri_exact([kidx, kmq_idx, kpmq_idx, kpidx])
                # test_eri_block = dfk_inst.get_eri([kidx, kmq_idx, kpmq_idx, kpidx])
                # assert np.allclose(exact_eri_block, test_eri_block)



    




if __name__ == "__main__":
    fftdf_reconstruct()
    # lambda_v2_calc()
