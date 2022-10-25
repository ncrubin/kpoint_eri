import h5py
import numpy as np
import pytest

from pyscf.pbc import gto, scf
from pyscf.pbc.dft import numint

from kpoint_eri.factorizations.kmeans import KMeansCVT


def test_supercell_isdf_gamma():
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

    mf = scf.RHF(cell)
    mf.chkfile = "test_isdf_supercell.chk"
    mf.init_guess = "chkfile"
    mf.kernel()

    from pyscf.pbc.dft import gen_grid
    grid_inst = gen_grid.UniformGrids(cell)
    grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
    orbitals = numint.eval_ao(cell, grid_points)
    orbitals_mo = np.einsum("Rp,pi->Ri", orbitals, mf.mo_coeff, optimize=True)
    num_mo = mf.mo_coeff.shape[1]
    num_interp_points = 10 * num_mo
    nocc = cell.nelec[0]
    density = np.einsum("Ri,Ri->R", orbitals_mo[:,:nocc],
                        orbitals_mo[:,:nocc].conj(), optimize=True)
    assert np.einsum("R,R->", density, grid_inst.weights) == pytest.approx(nocc)
    with h5py.File(mf.chkfile, "r+") as fh5:
        try:
            interp_indx = fh5["interp_indx"][:]
        except KeyError:
            kmeans = KMeansCVT(grid_points, max_iteration=500)
            interp_indx = kmeans.find_interpolating_points(num_interp_points,
                                                           density)
            fh5["interp_indx"] = interp_indx

    from kpoint_eri.factorizations.isdf import supercell_isdf
    chi, zeta, Theta = supercell_isdf(mf.with_df, interp_indx,
                                      orbitals=orbitals_mo,
                                      grid_points=grid_points)
    assert Theta.shape == (len(grid_points), num_interp_points)
    # Check overlap
    ovlp_ao = mf.get_ovlp()
    # should be identity matrix
    ovlp_mo = np.einsum("pi,pq,qj->ij", mf.mo_coeff.conj(), ovlp_ao,
                        mf.mo_coeff, optimize=True)
    identity = np.eye(num_mo)
    assert np.allclose(ovlp_mo, identity)
    ovlp_mu = np.einsum("Rm,R->m", Theta, grid_inst.weights, optimize=True)
    orbitals_mo_interp = orbitals_mo[interp_indx]
    ovlp_isdf = np.einsum("mi,mj,m->ij", orbitals_mo_interp.conj(), orbitals_mo_interp,
                          ovlp_mu, optimize=True)
    assert np.allclose(ovlp_mo, ovlp_isdf)
    # Check ERIs.
    eri_ref = mf.with_df.ao2mo(mf.mo_coeff)
    from pyscf import ao2mo
    eri_ref = ao2mo.restore(1, eri_ref, num_mo)
    # THC eris
    Lijn = np.einsum("mi,mj,mn->ijn", chi.conj(), chi, zeta, optimize=True)
    eri_thc = np.einsum("ijn,nk,nl->ijkl", Lijn, chi.conj(), chi, optimize=True)
    assert np.allclose(eri_thc, eri_ref)

def test_supercell_isdf_complex():
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

    mf = scf.RHF(cell, kpt=np.array([0.1, -0.001, 0.022]))
    mf.chkfile = "test_isdf_supercell_cmplx.chk"
    mf.init_guess = "chkfile"
    mf.kernel()
    assert np.max(np.abs(mf.mo_coeff.imag)) > 1e-2

    from pyscf.pbc.dft import gen_grid
    grid_inst = gen_grid.UniformGrids(cell)
    grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
    orbitals = numint.eval_ao(cell, grid_points, kpt=mf.kpt)
    orbitals_mo = np.einsum("Rp,pi->Ri", orbitals, mf.mo_coeff, optimize=True)
    num_mo = mf.mo_coeff.shape[1]
    num_interp_points = 10 * num_mo
    nocc = cell.nelec[0]
    density = np.einsum("Ri,Ri->R", orbitals_mo[:,:nocc].conj(),
                        orbitals_mo[:,:nocc], optimize=True)
    assert np.einsum("R,R->", density, grid_inst.weights) == pytest.approx(nocc)
    with h5py.File(mf.chkfile, "r+") as fh5:
        try:
            interp_indx = fh5["interp_indx"][:]
        except KeyError:
            kmeans = KMeansCVT(grid_points, max_iteration=500)
            interp_indx = kmeans.find_interpolating_points(num_interp_points,
                                                           density.real)
            fh5["interp_indx"] = interp_indx

    from kpoint_eri.factorizations.isdf import supercell_isdf
    chi, zeta, Theta = supercell_isdf(mf.with_df, interp_indx,
                                      orbitals=orbitals_mo,
                                      grid_points=grid_points)
    assert Theta.shape == (len(grid_points), num_interp_points)
    # Check overlap
    ovlp_ao = mf.get_ovlp()
    # should be identity matrix
    ovlp_mo = np.einsum("pi,pq,qj->ij", mf.mo_coeff.conj(), ovlp_ao,
                        mf.mo_coeff, optimize=True)
    identity = np.eye(num_mo)
    assert np.allclose(ovlp_mo, identity)
    ovlp_mu = np.einsum("Rm,R->m", Theta, grid_inst.weights, optimize=True)
    orbitals_mo_interp = orbitals_mo[interp_indx]
    ovlp_isdf = np.einsum("mi,mj,m->ij", orbitals_mo_interp.conj(), orbitals_mo_interp,
                          ovlp_mu, optimize=True)
    assert np.allclose(ovlp_mo, ovlp_isdf)
    # Check ERIs.
    eri_ref = mf.with_df.ao2mo(mf.mo_coeff, kpts=mf.kpt.reshape((1, -1)))
    # Check there is a complex component
    assert np.max(np.abs(eri_ref.imag)) > 1e-3
    # for complex integrals ao2mo will yield num_mo^4 elements.
    eri_ref = eri_ref.reshape((num_mo,)*4)
    # THC eris
    Lijn = np.einsum("mi,mj,mn->ijn", chi.conj(), chi, zeta, optimize=True)
    eri_thc = np.einsum("ijn,nk,nl->ijkl", Lijn, chi.conj(), chi, optimize=True)
    assert np.allclose(eri_thc, eri_ref)

if __name__ == "__main__":
    # test_supercell_isdf_gamma()
    test_supercell_isdf_complex()
