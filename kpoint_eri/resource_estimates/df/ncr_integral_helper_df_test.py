import itertools

import numpy as np

from pyscf.pbc import gto, scf, mp, cc

from kpoint_eri.resource_estimates.df.ncr_integral_helper_df import DFABKpointIntegrals


def test_ab_decomp():
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

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh, scaled_center=[0.2, 0.3, 0.5])

    nkpts = len(kpts)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.chkfile = 'ncr_test_C_density_fitints.chk'
    mf.init_guess = 'chkfile'
    mf.with_df._cderi_to_save = 'ncr_test_C_density_fitints_gdf.h5'
    mf.init_guess = 'chkfile'
    mf.kernel()

    from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
    mymp = mp.KMP2(mf)
    nmo = mymp.nmo
    nocc = mymp.nocc
    nvir = nmo - nocc

    Luv = cholesky_from_df_ints(mymp)  # [kpt, kpt, naux, nao, nao]
    dfk_inst = DFABKpointIntegrals(Luv.copy(), mf)
    df_factors = dfk_inst.double_factorize(1.0E-15)

    test_a_mats = np.empty((dfk_inst.nk, dfk_inst.naux), dtype=object)
    test_b_mats = np.empty((dfk_inst.nk, dfk_inst.naux), dtype=object)
    for qidx in range(dfk_inst.nk):
        for nc in range(dfk_inst.naux):
            test_a_mats[qidx, nc] = np.einsum('pk,k,qk->pq', df_factors['U'][qidx, nc],
                                                             df_factors['lambda_U'][qidx, nc],
                                                             df_factors['U'][qidx, nc].conj())
            assert np.allclose(dfk_inst.a_mats[qidx][nc], test_a_mats[qidx, nc])
            test_b_mats[qidx, nc] = np.einsum('pk,k,qk->pq', df_factors['V'][qidx, nc],
                                                             df_factors['lambda_V'][qidx, nc],
                                                             df_factors['V'][qidx, nc].conj())
            assert np.allclose(dfk_inst.b_mats[qidx][nc], test_b_mats[qidx, nc])

    Luv_test = dfk_inst.build_chol_from_AB(dfk_inst.a_mats, dfk_inst.b_mats)

    for ki, kj in itertools.product(range(dfk_inst.nk), repeat=2):
        assert np.allclose(Luv_test[ki, kj], Luv[ki, kj])
    print("PASSED TESTS")

    # new test to get integrals
    approx_cc = cc.KRCCSD(mf)
    from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_cc
    from kpoint_eri.resource_estimates.cc_helper import _ERIS
    print("Building approx cc")
    approx_cc = build_cc(approx_cc, dfk_inst)
    eris = approx_cc.ao2mo(lambda x: x)
    emp2, _, _ = approx_cc.init_amps(eris)
    approx_cc.kernel()

    exact_cc = cc.KRCCSD(mf)
    eris = exact_cc.ao2mo()
    exact_emp2, _, _ = exact_cc.init_amps(eris)
    exact_cc.kernel()
    assert np.isclose(approx_cc.e_corr, exact_cc.e_corr)
    assert np.isclose(exact_emp2, emp2)