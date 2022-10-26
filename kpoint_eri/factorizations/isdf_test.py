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

def test_kpoint_isdf():
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

    kmesh = [1, 2, 3]
    # kpts = cell.make_kpts(kmesh, scaled_center=[0.2, 0.3, 0.5])
    kpts = cell.make_kpts(kmesh)

    # mf_df = scf.KRHF(cell, kpts).rs_density_fit()
    # mf_df.with_df.mesh = cell.mesh
    # mf_df.kernel()
    mf = scf.KRHF(cell, kpts)
    mf.chkfile = "test_isdf_supercell_kpts.chk"
    # mf.kernel()
    _, scf_dict = scf.chkfile.load_scf(mf.chkfile)
    mf.mo_coeff = scf_dict["mo_coeff"]
    mf.mo_occ = scf_dict["mo_occ"]

    from pyscf.pbc.dft import gen_grid

    grid_inst = gen_grid.UniformGrids(cell)
    grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
    num_grid_points = grid_points.shape[0]
    orbitals = numint.eval_ao_kpts(cell, grid_points, kpts=kpts)
    orbitals_mo = np.einsum("kRp,kpi->kRi", orbitals, mf.mo_coeff, optimize=True)
    num_mo_per_kpt = [C.shape[-1] for C in mf.mo_coeff]
    num_mo = int(np.mean(num_mo_per_kpt))
    num_interp_points = 100 * int(np.mean(num_mo_per_kpt))
    print(num_interp_points, num_grid_points)
    nocc = cell.nelec[0]
    density = np.einsum(
        "kRi,kRi->R",
        orbitals_mo[:, :, :nocc].conj(),
        orbitals_mo[:, :, :nocc],
        optimize=True,
    )
    num_kpts = len(kpts)
    assert np.einsum("R,R->", density, grid_inst.weights) == pytest.approx(
        num_kpts * nocc
    )
    # Cell periodic part
    # u = e^{-ik.r} phi(r)
    exp_ikr = np.exp(-1j * np.einsum("kx,Rx->kR", kpts, grid_points))
    cell_periodic = np.einsum("kR,kRi->kRi", exp_ikr, orbitals_mo)
    # sanity check
    exp_ik1r = np.exp(1j * np.einsum("x,Rx->R", kpts[1], grid_points))
    recon_mo = np.einsum("R,Ri->Ri", exp_ik1r, cell_periodic[1])
    assert np.allclose(recon_mo, orbitals_mo[1])
    with h5py.File(mf.chkfile, "r+") as fh5:
        try:
            interp_indx = fh5[f"interp_indx_{num_interp_points}"][:]
        except KeyError:
            kmeans = KMeansCVT(grid_points, max_iteration=500)
            interp_indx = kmeans.find_interpolating_points(
                num_interp_points, density.real
            )
            fh5[f"interp_indx_{num_interp_points}"] = interp_indx

    from kpoint_eri.factorizations.isdf import kpoint_isdf

    # go from kRi->Rki
    # AO ISDF
    orbitals = (
        np.array(orbitals)
        .transpose((1, 0, 2))
        .reshape((num_grid_points, num_kpts * num_mo))
    )
    interp_orbitals, central_tensor, Theta = kpoint_isdf(
        mf.with_df, interp_indx, kpts, orbitals, grid_points
    )
    # Check overlap
    ovlp_ao = mf.get_ovlp()
    exp_ikr = np.exp(1j * np.einsum("kx,Rx->kR", kpts, grid_points))
    # ovlp_mu = np.einsum("kR,Rm,R->km", exp_ikr, Theta, grid_inst.weights)
    ovlp_mu = np.einsum("Rm,R->m", Theta, grid_inst.weights)
    interp_orbitals = interp_orbitals.reshape((-1, num_kpts, num_mo))
    ovlp_isdf = np.einsum(
        "mki,mkj,m->kij",
        interp_orbitals.conj(),
        interp_orbitals,
        ovlp_mu,
        optimize=True,
    )
    orbitals = orbitals.reshape((-1, num_kpts, num_mo))
    exp_ikr = np.exp(-1j * np.einsum("kx,Rx->kR", kpts, grid_points))
    cell_periodic = np.einsum("kR,Rki->kRi", exp_ikr, orbitals)
    ovlp_comparison = np.einsum(
        "kRi,kRj,R->kij", cell_periodic.conj(), cell_periodic, grid_inst.weights
    )
    for ik in range(num_kpts):
        print(np.max(np.abs(ovlp_ao[ik] - ovlp_isdf[ik])))
        print(np.max(np.abs(ovlp_ao[ik] - ovlp_comparison[ik])))
    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    # ao->mo transform + (mu,k,ao) -> (k, mo, mu)
    interp_orbitals_mo = np.einsum(
        "mkp,kpi->kim", interp_orbitals, mf.mo_coeff, optimize=True
    )
    iq = 2
    k1 = 1
    k2 = 4
    zeta_q = central_tensor[iq]
    k3 = momentum_map[iq][k1]
    k4 = momentum_map[iq][k2]
    # (pk1 qk1-Q | rk2-Q s k2)
    kpt_pqrs = [kpts[k1], kpts[k3], kpts[k4], kpts[k2]]
    mos_pqrs = [mf.mo_coeff[k1], mf.mo_coeff[k3], mf.mo_coeff[k4], mf.mo_coeff[k2]]
    eri_pqrs = mf.with_df.ao2mo(mos_pqrs, kpt_pqrs, compact=False).reshape(
        (num_mo,) * 4
    )
    # u_{pk1 mu}^* u_{qk1-Q} zeta_Q_{mu,nu}
    Lpqnu = np.einsum(
        "pm,qm,mn->pqn", interp_orbitals_mo[k1].conj(), interp_orbitals_mo[k3], zeta_q
    )
    # u_{rk2-Q nu}^* u_{sk1} L_{pqnu}
    eri_thc = np.einsum(
        "rn,sn,pqn->pqrs", interp_orbitals_mo[k4].conj(), interp_orbitals_mo[k2], Lpqnu
    )
    print(np.max(np.abs(eri_pqrs - eri_thc)))

if __name__ == "__main__":
    # test_supercell_isdf_gamma()
    test_supercell_isdf_complex()
