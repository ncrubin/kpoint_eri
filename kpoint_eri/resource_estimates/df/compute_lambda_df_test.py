from functools import reduce
import numpy as np

from pyscf.pbc import gto, scf, mp

from kpoint_eri.resource_estimates.df.compute_lambda_df import compute_lambda
from kpoint_eri.resource_estimates.df.integral_helper_df import DFABKpointIntegrals
from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints

def test_lambda_calc():
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
    cell.build(parse_arg=False)

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.with_df.mesh = cell.mesh
    mf.kernel()


    mymp = mp.KMP2(mf)
    mymp.density_fit()
    Luv = cholesky_from_df_ints(mymp)
    helper = DFABKpointIntegrals(cholesky_factor=Luv, kmf=mf)
    helper.double_factorize(thresh=1.0E-13)

    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mf.mo_coeff)])

    lambda_tot, lambda_one_body, lambda_two_body, num_eigs = compute_lambda(hcore_mo, helper)

    l2norm_hcore_mo = np.linalg.norm(hcore_mo.ravel())

    from kpoint_eri.resource_estimates.utils.k2gamma import k2gamma
    supercell_mf = k2gamma(mf, make_real=False)
    supercell_mf.with_df.mesh = supercell_mf.cell.mesh
    supercell_mf.e_tot = supercell_mf.energy_tot()
    assert np.isclose(mf.e_tot, supercell_mf.e_tot / np.prod(kmesh))

    supercell_mymp = mp.KMP2(supercell_mf)
    supercell_Luv = cholesky_from_df_ints(supercell_mymp)
    supercell_helper = DFABKpointIntegrals(cholesky_factor=supercell_Luv, kmf=supercell_mf)
    supercell_helper.double_factorize(thresh=1.0E-13)
    sc_nk = supercell_helper.nk
    sc_help = supercell_helper
 
    supercell_hcore_ao = supercell_mf.get_hcore()
    supercell_hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), supercell_hcore_ao[k], mo)) for k, mo in enumerate(supercell_mf.mo_coeff)])
    l2_norm_supercell_hcore_mo = np.linalg.norm(supercell_hcore_mo.ravel())
    
    assert np.isclose(l2_norm_supercell_hcore_mo, l2norm_hcore_mo)

    sc_lambda_tot, sc_lambda_one_body, sc_lambda_two_body, sc_num_eigs = compute_lambda(supercell_hcore_mo, sc_help)

    assert np.isclose(sc_lambda_one_body, lambda_one_body)

    lambda_two_body = 0
    lambda_two_body_v2 = 0
    for qidx in range(nkpts):
        aval_to_square = np.zeros((helper.naux), dtype=np.complex128)
        bval_to_square = np.zeros((helper.naux), dtype=np.complex128)

        aval_to_square_v2 = np.zeros((helper.naux), dtype=np.complex128)
        bval_to_square_v2 = np.zeros((helper.naux), dtype=np.complex128)

        for kidx in range(nkpts):
            Amats, Bmats = helper.build_A_B_n_q_k_from_chol(qidx, kidx) 
            Amats /= np.sqrt(nkpts)
            Bmats /= np.sqrt(nkpts)
            wa, _ = np.linalg.eigh(Amats)
            wb, _ = np.linalg.eigh(Bmats)
            aval_to_square += np.einsum('npq->n', np.abs(Amats)**2)
            bval_to_square += np.einsum('npq->n', np.abs(Bmats)**2)

            aval_to_square_v2 += np.sum(np.abs(wa)**2, axis=-1)
            bval_to_square_v2 += np.sum(np.abs(wb)**2, axis=-1)
            assert np.allclose(np.sum(np.abs(wa)**2, axis=-1), 
                               np.einsum('npq->n', np.abs(Amats)**2))

            # lambda_two_body += np.sum(np.einsum('npq->n', np.abs(Amats)**2))
            # lambda_two_body += np.sum(np.einsum('npq->n', np.abs(Bmats)**2))
        lambda_two_body += np.sum(aval_to_square)
        lambda_two_body += np.sum(bval_to_square)

        lambda_two_body_v2 += np.sum(aval_to_square_v2)
        lambda_two_body_v2 += np.sum(bval_to_square_v2)
 
        
    assert np.isclose(lambda_two_body, lambda_two_body_v2)

if __name__ == "__main__":
    test_lambda_calc()