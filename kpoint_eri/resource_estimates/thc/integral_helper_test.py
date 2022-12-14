import h5py
import numpy as np

from pyscf.pbc import gto, scf, cc

from kpoint_eri.factorizations.isdf import solve_kmeans_kpisdf
from kpoint_eri.resource_estimates.thc.integral_helper import (
        KPTHCHelperDoubleTranslation,
        KPTHCHelperSingleTranslation
        )

def test_thc_helper():
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

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts)
    mf.chkfile = 'integrals.chk'
    mf.init_guess = 'chkfile'
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
        chi, zeta, xi, G_mapping = solve_kmeans_kpisdf(mf, num_interp_points,
                                                       single_translation=False)
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
    print(" {:4d}  {:10.4e} {:10.4e} {:10.4e}".format(num_interp_points, delta, emp2, exact_emp2))
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
        chi, zeta, xi, G_mapping = solve_kmeans_kpisdf(mf, num_interp_points,
                                                       single_translation=True)
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
    print(" {:4d}  {:10.4e} {:10.4e} {:10.4e}".format(num_interp_points, delta, emp2, exact_emp2))

if __name__ == "__main__":
    test_thc_helper()
