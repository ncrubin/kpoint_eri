import numpy as np

from pyscf.pbc import gto, scf

from kpoint_eri.factorizations.kmeans import KMeansCVT


def test_supercell_isdf_pyscf():
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
    # kpts = cell.make_kpts(kmesh, scaled_center=[0.2, 0.3, 0.5])

    mf = scf.RHF(cell, np.array([0.1, 0.001, 0]))
    mf.chkfile = "test_C_density_fitints.chk"
    mf.init_guess = "chkfile"
    mf.with_df._cderi_to_save = "test_C_density_fitints.chk"
    mf.with_df.mesh = [10, 10, 10]
    mf.init_guess = "chkfile"
    mf.kernel()


if __name__ == "__main__":
    test_supercell_isdf_pyscf()
