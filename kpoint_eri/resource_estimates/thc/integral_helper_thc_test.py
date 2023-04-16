import numpy as np
import pytest

from pyscf.pbc import gto, scf, cc

from kpoint_eri.factorizations.isdf import solve_kmeans_kpisdf
from kpoint_eri.resource_estimates.thc.integral_helper_thc import (
    KPTHCHelperDoubleTranslation,
    KPTHCHelperSingleTranslation,
)


@pytest.mark.slow
def test_thc_helper():
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
    cell.mesh = [11] * 3
    cell.build()

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts)
    mf.kernel()

    exact_cc = cc.KRCCSD(mf)
    eris = exact_cc.ao2mo()
    exact_emp2, _, _ = exact_cc.init_amps(eris)

    approx_cc = cc.KRCCSD(mf)
    approx_cc.verbose = 0
    nmo_k = mf.mo_coeff[0].shape[-1]
    kpt_thc = solve_kmeans_kpisdf(
        mf,
        np.prod(cell.mesh),  # Use the whole grid to avoid any precision issues
        single_translation=False,
        use_density_guess=True,
    )

    helper = KPTHCHelperDoubleTranslation(kpt_thc.chi, kpt_thc.zeta, mf)
    from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_approximate_eris

    num_kpts = len(mf.kpts)
    for iq in range(num_kpts):
        for ik in range(num_kpts):
            ik_minus_q = helper.k_transfer_map[iq, ik]
            for ik_prime in range(num_kpts):
                ik_prime_minus_q = helper.k_transfer_map[iq, ik_prime]
                eri_thc = helper.get_eri([ik, ik_minus_q, ik_prime_minus_q, ik_prime])
                eri_exact = helper.get_eri_exact(
                    [ik, ik_minus_q, ik_prime_minus_q, ik_prime]
                )
                assert np.allclose(eri_thc, eri_exact)

    eris_approx = build_approximate_eris(approx_cc, helper)
    emp2, _, _ = approx_cc.init_amps(eris_approx)
    assert np.isclose(emp2, exact_emp2)
    kpt_thc = solve_kmeans_kpisdf(
        mf, np.prod(cell.mesh), single_translation=True
    )
    helper = KPTHCHelperSingleTranslation(kpt_thc.chi, kpt_thc.zeta, mf)
    from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_approximate_eris

    for iq in range(num_kpts):
        for ik in range(num_kpts):
            ik_minus_q = helper.k_transfer_map[iq, ik]
            for ik_prime in range(num_kpts):
                ik_prime_minus_q = helper.k_transfer_map[iq, ik_prime]
                eri_thc = helper.get_eri([ik, ik_minus_q, ik_prime_minus_q, ik_prime])
                eri_exact = helper.get_eri_exact(
                    [ik, ik_minus_q, ik_prime_minus_q, ik_prime]
                )
                assert np.allclose(eri_thc, eri_exact)

    eris_approx = build_approximate_eris(approx_cc, helper)
    emp2, _, _ = approx_cc.init_amps(eris_approx)
    assert np.isclose(emp2, exact_emp2)


if __name__ == "__main__":
    test_thc_helper()