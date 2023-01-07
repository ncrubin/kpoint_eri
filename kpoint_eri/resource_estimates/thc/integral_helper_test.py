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
    cell.verbose = 4
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
    num_interp_points = nmo_k * 50
    try:
        print("Reading THC factors from file")
        with h5py.File(mf.chkfile, "r") as fh5:
            chi = fh5["chi"][:]
            print(fh5.keys())
            qs = [int(s.split("_")[1]) for s in fh5.keys() if "zeta" in s]
            zeta = np.zeros(len(qs), dtype=object)
            for q in qs:
                zeta[q] = fh5[f"zeta_{q}"][:]
    except KeyError:
        print("Solving ISDF")
        chi, zeta, xi, G_mapping = solve_kmeans_kpisdf(
            mf, num_interp_points, single_translation=False
        )
        with h5py.File(mf.chkfile, "r+") as fh5:
            fh5["chi"] = chi
            for q in range(zeta.shape[0]):
                fh5[f"zeta_{q}"] = zeta[q]

    helper = KPTHCHelperDoubleTranslation(chi, zeta, mf)
    from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_cc

    approx_cc = build_cc(approx_cc, helper)
    eris = approx_cc.ao2mo(lambda x: x)
    emp2, _, _ = approx_cc.init_amps(eris)
    delta = abs(emp2 - exact_emp2)
    print(
        " {:4d}  {:10.4e} {:10.4e} {:10.4e}".format(
            num_interp_points, delta, emp2, exact_emp2
        )
    )
    try:
        print("Reading THC factors from file")
        with h5py.File(mf.chkfile, "r") as fh5:
            chi = fh5["chi_st"][:]
            qs = [int(s.split("_")[1]) for s in fh5.keys() if "zeta" in s]
            zeta = np.zeros(len(qs), dtype=object)
            for q in qs:
                zeta[q] = fh5[f"zeta_st_{q}"][:]
    except KeyError:
        print("Solving ISDF")
        chi, zeta, xi, G_mapping = solve_kmeans_kpisdf(
            mf, num_interp_points, single_translation=True
        )
        with h5py.File(mf.chkfile, "r+") as fh5:
            fh5["chi_st"] = chi
            for q in range(zeta.shape[0]):
                fh5[f"zeta_st_{q}"] = zeta[q]
    helper = KPTHCHelperSingleTranslation(chi, zeta, mf)
    from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_cc

    approx_cc = build_cc(approx_cc, helper)
    eris = approx_cc.ao2mo(lambda x: x)
    emp2, _, _ = approx_cc.init_amps(eris)
    delta = abs(emp2 - exact_emp2)
    print(
        " {:4d}  {:10.4e} {:10.4e} {:10.4e}".format(
            num_interp_points, delta, emp2, exact_emp2
        )
    )


def test_thc_convergence():
    ase_atom = bulk("H", "bcc", a=2.0, cubic=True)
    cell = gto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell[:].copy()
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf-rev"
    cell.verbose = 4
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
    exact_emp2 += mf.e_tot

    rsmf = scf.KRHF(mf.cell, mf.kpts).rs_density_fit()
    # Force same MOs as FFTDF at least
    rsmf.mo_occ = mf.mo_occ
    rsmf.mo_coeff = mf.mo_coeff
    rsmf.mo_energy = mf.mo_energy
    rsmf.with_df.mesh = mf.cell.mesh
    mymp = mp.KMP2(rsmf)
    Luv = cholesky_from_df_ints(mymp)
    cthc = 4
    num_thc = cthc * mf.mo_coeff[0].shape[-1]
    chi, zeta, _ = kpoint_thc_via_isdf(
        mf,
        Luv,
        num_thc,
        perform_adagrad_opt=False,
        perform_bfgs_opt=False,
    )

    helper = KPTHCHelperDoubleTranslation(chi, zeta, mf)
    emp2 = compute_emp2_approx(mf, helper)
    print(
        " {:4d}  {:10.4e} {:10.4e} {:10.4e}".format(
            cthc, emp2, exact_emp2, exact_emp2 - emp2
        )
    )
    chi, zeta, _ = kpoint_thc_via_isdf(
        mf,
        Luv,
        num_thc,
        perform_adagrad_opt=False,
        perform_bfgs_opt=True,
    )

    helper = KPTHCHelperDoubleTranslation(chi, zeta, mf)
    emp2 = compute_emp2_approx(mf, helper)
    print(
        " {:4d}  {:10.4e} {:10.4e} {:10.4e}".format(
            cthc, emp2, exact_emp2, exact_emp2 - emp2
        )
    )


def test_thc_convergence_qrcp():
    ase_atom = bulk("H", "bcc", a=2.0, cubic=True)
    cell = gto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell[:].copy()
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf-rev"
    cell.verbose = 4
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
    exact_emp2 += mf.e_tot

    mymp = mp.KMP2(mf)
    cthc = 4
    num_thc = cthc * mf.mo_coeff[0].shape[-1]
    from kpoint_eri.factorizations.isdf import solve_qrcp_isdf

    # solve at a specific value of cthc
    chi, zeta, xi, G_map = solve_qrcp_isdf(mf, num_thc, single_translation=False)

    helper = KPTHCHelperDoubleTranslation(chi, zeta, mf)
    emp2 = compute_emp2_approx(mf, helper)
    print(
        " {:4d}  {:10.4e} {:10.4e} {:10.4e}".format(
            cthc, emp2, exact_emp2, exact_emp2 - emp2
        )
    )
    from kpoint_eri.factorizations.isdf import (
        interp_indx_from_qrcp,
        setup_isdf,
        solve_for_thc_factors,
    )

    # Reuse QRCP solution
    grid_points, cell_periodic_on_grid, bloch_on_grid = setup_isdf(mf)
    c_max = 10
    num_mo = mf.mo_coeff[0].shape[-1]
    indx = interp_indx_from_qrcp(cell_periodic_on_grid, c_max * num_mo)
    for cthc in [2, 4, 6, 8]:
        chi, zeta, xi, G_map = solve_for_thc_factors(
            mf,
            indx[: cthc * num_mo],
            cell_periodic_on_grid,
            grid_points,
            single_translation=False,
        )
        helper = KPTHCHelperDoubleTranslation(chi, zeta, mf)
        emp2 = compute_emp2_approx(mf, helper)
        print(
            " {:4d}  {:10.4e} {:10.4e} {:10.4e}".format(
                cthc, emp2, exact_emp2, exact_emp2 - emp2
            )
        )


if __name__ == "__main__":
    test_thc_helper()
    test_thc_convergence()
    test_thc_convergence_qrcp()
