from pyscf.pbc import gto, scf, mp
import numpy as np
import h5py

from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_approximate_eris_rohf


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
    cell.verbose = 0
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
    # Only ROHF integrals are supported for resource estimates but only UCCSD
    # available, so convert MOs to UHF format first.
    u_from_ro = scf.addons.convert_to_uhf(mf)
    mymp = mp.KMP2(mf)

    from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints

    Luv = cholesky_from_df_ints(mymp)  # [kpt, kpt, naux, nao, nao]
    from pyscf.pbc.cc.kccsd_uhf import _make_df_eris
    from pyscf.pbc.cc import KUCCSD

    cc = KUCCSD(u_from_ro)
    eris = _make_df_eris(cc)
    emp2_ref, _, _ = cc.init_amps(eris)
    ref_eris = _make_df_eris(cc)
    from kpoint_eri.resource_estimates.sf.integral_helper_sf import (
        SingleFactorizationHelper,
    )

    helper = SingleFactorizationHelper(cholesky_factor=Luv, kmf=mf)
    test_eris = build_approximate_eris_rohf(cc, eris, helper, inplace=False)
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
    for ki in range(len(kpts)):
        for kj in range(len(kpts)):
            assert np.allclose(test_eris.Lpv[ki, kj], ref_eris.LPV[ki, kj])

    helper = SingleFactorizationHelper(cholesky_factor=Luv, kmf=mf)
    approx_cc = KUCCSD(u_from_ro)
    emp2_approx, _, _ = approx_cc.init_amps(test_eris)

    assert abs(emp2_approx - emp2_ref) < 1e-12


if __name__ == "__main__":
    test_eri_helpers()