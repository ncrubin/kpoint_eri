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
    cell.verbose = 4
    cell.build()

    from pyscf.pbc.scf.chkfile import load_scf
    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.with_df.mesh = cell.mesh
    mf.chkfile = 'fft_ncr_test_C_density_fitints.chk'
    mf.init_guess = 'chkfile'
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
    l2_norm_supercell_hcore_mo = np.linalg.norm(supercell_hcore_mo.ravel())
    
    assert np.isclose(l2_norm_supercell_hcore_mo, l2norm_hcore_mo)

    sc_lambda_tot, sc_lambda_one_body, sc_lambda_two_body, sc_num_eigs = compute_lambda(supercell_hcore_mo, sc_help)

    assert np.isclose(sc_lambda_one_body, lambda_one_body)

    from kpoint_eri.resource_estimates.sf.ncr_integral_helper import NCRSingleFactorizationHelper
    sf_helper = NCRSingleFactorizationHelper(cholesky_factor=Luv, kmf=mf)
    # this part needs to change 
    lambda_two_body = 0
    for qidx in range(len(kpts)):
        # A and B are W
        A, B = sf_helper.build_AB_from_chol(qidx) # [naux, nao * nk, nao * nk]
        A /= np.sqrt(nkpts)
        B /= np.sqrt(nkpts)
        # sum_q sum_n (sum_{pq} |Re{A_{pq}^n}| + |Im{A_{pq}^n|)^2
        # lambda_two_body += np.sum(np.einsum('npq->n', np.abs(A.real) + np.abs(A.imag))**2)
        # lambda_two_body += np.sum(np.einsum('npq->n', np.abs(B.real) + np.abs(B.imag))**2)
        lambda_two_body += np.sum(np.einsum('npq->n', np.abs(A)**2))
        lambda_two_body += np.sum(np.einsum('npq->n', np.abs(B)**2))

        # _, w, _ = np.linalg.svd(A[0])
        # print(np.sum(w**2))
        # print(np.sum(np.abs(A[0])**2))
        # print(np.trace(A[0].conj().T @ A[0]).real)
        # print(np.einsum('npq->n', np.abs(A)**2)[0])
        # w, _ = np.linalg.eigh(A[0])
        # print(np.sum(np.abs(w)**2))
        # print(np.sum([np.sum(np.abs(A[x])**2) for x in range(helper.naux)]))
        # print(np.sum(np.einsum('npq->n', np.abs(A)**2)))
    print(lambda_two_body)

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
 
        
    print(lambda_two_body)
    print(lambda_two_body_v2)

    lambda_two_body_v3 = 0
    lambda_two_body_v4 = 0
    weird_quantum_lambda_two_body = 0
    for qidx in range(nkpts):
        for nn in range(helper.naux):
            # sum up frobenius norm squared for each matrix A_{n}(Q, K) and B_{n}(Q, K)
            first_number_to_square = 0
            second_number_to_square = 0

            quantum_first_number_to_square = 0
            quantum_second_number_to_square = 0

            for kidx in range(nkpts):
                Amats, Bmats = helper.build_A_B_n_q_k_from_chol(qidx, kidx) 
                Amats /= np.sqrt(nkpts)
                Bmats /= np.sqrt(nkpts)
                
                wa, _ = np.linalg.eigh(Amats[nn])
                wb, _ = np.linalg.eigh(Bmats[nn])

                first_number_to_square += np.sum(np.abs(wa)**2) 
                second_number_to_square += np.sum(np.abs(wb)**2) 

                eigs_a_fixed_n_q = helper.amat_lambda_vecs[kidx, qidx, nn] / np.sqrt(nkpts)
                eigs_b_fixed_n_q = helper.bmat_lambda_vecs[kidx, qidx, nn] / np.sqrt(nkpts)
                lambda_two_body_v4 += np.sum(np.abs(eigs_a_fixed_n_q)**2)
                lambda_two_body_v4 += np.sum(np.abs(eigs_b_fixed_n_q)**2)

                quantum_first_number_to_square += np.sum(np.abs(eigs_a_fixed_n_q))
                quantum_second_number_to_square += np.sum(np.abs(eigs_b_fixed_n_q))
           
            lambda_two_body_v3 += first_number_to_square
            lambda_two_body_v3 += second_number_to_square

            weird_quantum_lambda_two_body += quantum_first_number_to_square**2
            weird_quantum_lambda_two_body += quantum_second_number_to_square**2

    print(lambda_two_body_v3)
    print(lambda_two_body_v4)
    print(weird_quantum_lambda_two_body * 0.25)
    lambda_tot, lambda_one_body, lambda_two_body, num_eigs = compute_lambda(hcore_mo, helper)
    print(lambda_two_body)
    # YES THIS LOOKS CORRECT!

    sc_lambda_two_body = 0
    sc_weird_quantum_lambda_two_body = 0
    for qidx in range(1):
        for nn in range(supercell_helper.naux):
            quantum_first_number_to_square = 0
            quantum_second_number_to_square = 0
            for kidx in range(1):
                Amats, Bmats = supercell_helper.build_A_B_n_q_k_from_chol(qidx, kidx) 
                sc_lambda_two_body += np.sum(np.einsum('npq->n', np.abs(Amats)**2))
                sc_lambda_two_body += np.sum(np.einsum('npq->n', np.abs(Bmats)**2))
                eigs_a_fixed_n_q = supercell_helper.amat_lambda_vecs[kidx, qidx, nn] / np.sqrt(1)
                eigs_b_fixed_n_q = supercell_helper.bmat_lambda_vecs[kidx, qidx, nn] / np.sqrt(1)

                quantum_first_number_to_square += np.sum(np.abs(eigs_a_fixed_n_q))
                quantum_second_number_to_square += np.sum(np.abs(eigs_b_fixed_n_q))
 
            sc_weird_quantum_lambda_two_body += quantum_first_number_to_square**2
            sc_weird_quantum_lambda_two_body += quantum_second_number_to_square**2

if __name__ == "__main__":
    test_lambda_calc()