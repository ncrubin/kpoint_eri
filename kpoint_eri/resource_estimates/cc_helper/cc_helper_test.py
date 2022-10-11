from pyscf.pbc import gto, scf, mp
import numpy as np


def test_eri_blocks():
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
    cell.spin = 2
    cell.build()

    kmesh = [1, 1, 1]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KROHF(cell, kpts).rs_density_fit()
    mf.chkfile = "uccsd_test.chk"
    mf.init_guess = "chkfile"
    mf.with_df._cderi_to_save = mf.chkfile
    mf.kernel()

    from pyscf.pbc.mp.kump2 import KUMP2

    u_from_ro = scf.addons.convert_to_uhf(mf)
    # mymp = KUMP2(u_from_ro)
    # mymp.kernel()
    mymp = mp.KMP2(mf)

    from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints

    Luv = cholesky_from_df_ints(
        mymp, mo_coeff=mf.mo_coeff
    )  # [kpt, kpt, naux, nao, nao]
    from pyscf.pbc.cc.kccsd_uhf import _make_df_eris
    from pyscf.pbc.cc import KUCCSD

    cc = KUCCSD(u_from_ro)
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


if __name__ == "__main__":
    test_eri_blocks()
