import h5py
import numpy as np

from pyscf.pbc import gto, scf, cc, mp
from pyscf.pbc.tools import pyscf_ase
from ase.build import bulk

from kpoint_eri.factorizations.isdf import solve_kmeans_kpisdf
from kpoint_eri.factorizations.thc_jax import kpoint_thc_via_isdf
from kpoint_eri.resource_estimates.cc_helper.cc_helper import compute_emp2_approx
from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
from kpoint_eri.resource_estimates.thc.integral_helper import (
    KPTHCHelperDoubleTranslation,
    KPTHCHelperSingleTranslation,
)


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
    cell.build()

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts)
    mf.chkfile = "integrals.chk"
    mf.init_guess = "chkfile"
    mf.kernel()

    exact_cc = cc.KRCCSD(mf)
    eris = exact_cc.ao2mo()
    exact_emp2, _, _ = exact_cc.init_amps(eris)

    approx_cc = cc.KRCCSD(mf)
    approx_cc.verbose = 0
    nmo_k = mf.mo_coeff[0].shape[-1]
    num_interp_points = nmo_k * 80
    chi, zeta, xi, G_mapping = solve_kmeans_kpisdf(
        mf,
        num_interp_points,
        single_translation=False,
        use_density_guess=True,
    )

    helper = KPTHCHelperDoubleTranslation(chi, zeta, mf)
    from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_cc

    approx_cc = build_cc(approx_cc, helper)
    eris = approx_cc.ao2mo(lambda x: x)
    emp2, _, _ = approx_cc.init_amps(eris)
    assert np.isclose(emp2, exact_emp2, atol=1e-4)
    chi, zeta, xi, G_mapping = solve_kmeans_kpisdf(
        mf, num_interp_points, single_translation=True
    )
    helper = KPTHCHelperSingleTranslation(chi, zeta, mf)
    from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_cc

    approx_cc = build_cc(approx_cc, helper)
    eris = approx_cc.ao2mo(lambda x: x)
    emp2, _, _ = approx_cc.init_amps(eris)
    assert np.isclose(emp2, exact_emp2, atol=1e-4)


# def test_thc_qrcp():
#     ase_atom = bulk("H", "bcc", a=2.0, cubic=True)
#     cell = gto.Cell()
#     cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
#     cell.a = ase_atom.cell[:].copy()
#     cell.basis = "gth-szv"
#     cell.pseudo = "gth-hf-rev"
#     cell.verbose = 4
#     cell.build()

#     kmesh = [1, 1, 3]
#     kpts = cell.make_kpts(kmesh)
#     mf = scf.KRHF(cell, kpts)
#     mf.chkfile = "integrals.chk"
#     mf.init_guess = "chkfile"
#     mf.kernel()

#     exact_cc = cc.KRCCSD(mf)
#     eris = exact_cc.ao2mo()
#     exact_emp2, _, _ = exact_cc.init_amps(eris)
#     exact_emp2 += mf.e_tot

#     mymp = mp.KMP2(mf)
#     cthc = 4
#     num_thc = cthc * mf.mo_coeff[0].shape[-1]
#     from kpoint_eri.factorizations.isdf import solve_qrcp_isdf

#     # solve at a specific value of cthc
#     chi, zeta, xi, G_map = solve_qrcp_isdf(mf, num_thc, single_translation=False)

#     helper = KPTHCHelperDoubleTranslation(chi, zeta, mf)
#     emp2 = compute_emp2_approx(mf, helper)
#     print(
#         " {:4d}  {:10.4e} {:10.4e} {:10.4e}".format(
#             cthc, emp2, exact_emp2, exact_emp2 - emp2
#         )
#     )
#     from kpoint_eri.factorizations.isdf import (
#         interp_indx_from_qrcp,
#         setup_isdf,
#         solve_for_thc_factors,
#     )

#     # Reuse QRCP solution
#     grid_points, cell_periodic_on_grid, bloch_on_grid = setup_isdf(mf)
#     c_max = 50
#     num_mo = mf.mo_coeff[0].shape[-1]
#     indx = interp_indx_from_qrcp(cell_periodic_on_grid, c_max * num_mo)
#     chi, zeta, xi, G_map = solve_for_thc_factors(
#         mf,
#         indx[: cthc * num_mo],
#         cell_periodic_on_grid,
#         grid_points,
#         single_translation=False,
#     )
#     helper = KPTHCHelperDoubleTranslation(chi, zeta, mf)
#     emp2 = compute_emp2_approx(mf, helper)
#     assert np.isclose(exact_emp2, emp2)

if __name__ == "__main__":
    test_thc_helper()
    # test_thc_qrcp()
