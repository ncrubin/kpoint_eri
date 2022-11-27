from pyscf.pbc import gto, scf, mp
import numpy as np
import h5py


def test_eri_helpers():
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
    cell.verbose = 4
    cell.build()

    kmesh = [1, 2, 2]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KROHF(cell, kpts).rs_density_fit()
    mf.chkfile = "uccsd_test.chk"
    mf.init_guess = "chkfile"
    mf.with_df._cderi_to_save = mf.chkfile
    mf.with_df.mesh = cell.mesh
    mf.kernel()

    mf.with_df._cderi = mf.chkfile
    u_from_ro = scf.addons.convert_to_uhf(mf)
    mymp = mp.KMP2(mf)

    from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints

    Luv = cholesky_from_df_ints(mymp)  # [kpt, kpt, naux, nao, nao]
    from pyscf.pbc.cc.kccsd_uhf import _make_df_eris
    from pyscf.pbc.cc import KUCCSD

    cc = KUCCSD(u_from_ro)
    eris = cc.ao2mo()
    emp2_ref, _, _ = cc.init_amps(eris)
    ref_eris = _make_df_eris(cc)
    from kpoint_eri.resource_estimates.cc_helper.custom_ao2mo import (
        _custom_make_df_eris,
    )

    from kpoint_eri.resource_estimates.sf.ncr_integral_helper import (
        NCRSingleFactorizationHelper,
    )

    helper = NCRSingleFactorizationHelper(cholesky_factor=Luv, kmf=mf)
    test_eris = _custom_make_df_eris(cc, helper)
    eri_blocks = [
        "OOOO",
        "OOOV",
        "OOVV",
        "OVOV",
        "VOOV",
        "VOVV",
        "oooo",
        "ooov",
        "oovv",
        "ovov",
        "voov",
        "vovv",
        "OOov",
        "OOvv",
        "OVov",
        "VOov",
        "VOvv",
    ]
    for block in eri_blocks:
        assert np.allclose(test_eris.__dict__[block][:], ref_eris.__dict__[block][:])

    from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_ucc

    helper = NCRSingleFactorizationHelper(cholesky_factor=Luv, kmf=mf)
    approx_cc = KUCCSD(u_from_ro)
    approx_cc = build_ucc(approx_cc, helper)
    eris = approx_cc.ao2mo(lambda x: x)
    emp2_approx, _, _ = approx_cc.init_amps(eris)

    assert abs(emp2_approx - emp2_ref) < 1e-12


if __name__ == "__main__":
    test_eri_helpers()
