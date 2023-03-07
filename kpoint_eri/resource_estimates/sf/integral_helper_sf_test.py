import numpy as np
from pyscf.pbc import gto, scf, mp, cc

from kpoint_eri.resource_estimates.sf.integral_helper_sf import (
    SingleFactorizationHelper,
)
from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints


def test_ncr_sf_helper_trunc():
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
    mf.chkfile = "ncr_test_C_density_fitints.chk"
    mf.with_df._cderi_to_save = "ncr_test_C_density_fitints_gdf.h5"
    mf.with_df.mesh = mf.cell.mesh
    mf.init_guess = "chkfile"
    mf.kernel()

    exact_cc = cc.KRCCSD(mf)
    eris = exact_cc.ao2mo()
    exact_emp2, _, _ = exact_cc.init_amps(eris)

    mymp = mp.KMP2(mf)

    Luv = cholesky_from_df_ints(mymp)  # [kpt, kpt, naux, nao, nao]
    naux = Luv[0, 0].shape[0]

    print(" naux  error (Eh)")
    approx_cc = cc.KRCCSD(mf)
    approx_cc.verbose = 0
    helper = SingleFactorizationHelper(cholesky_factor=Luv, kmf=mf, naux=10)
    from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_cc

    approx_cc = build_cc(approx_cc, helper)
    eris = approx_cc.ao2mo(lambda x: x)
    emp2, _, _ = approx_cc.init_amps(eris)
    assert not np.isclose(emp2, exact_emp2)
    helper = SingleFactorizationHelper(cholesky_factor=Luv, kmf=mf, naux=naux)
    from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_cc

    approx_cc = build_cc(approx_cc, helper)
    eris = approx_cc.ao2mo(lambda x: x)
    emp2, _, _ = approx_cc.init_amps(eris)
    print()
    assert np.isclose(emp2, exact_emp2)
