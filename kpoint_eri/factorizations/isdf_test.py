import itertools
import h5py
import numpy as np
import pytest

from pyscf.pbc import gto, scf, tools
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint
from pyscf.pbc.lib.kpts_helper import unique, get_kconserv, member

from kpoint_eri.factorizations.kmeans import KMeansCVT
from kpoint_eri.factorizations.isdf import (
    inverse_G_map_double_translation,
    build_kpoint_zeta,
    # build_G_vector_mappings,
    build_G_vectors,
    build_G_vector_mappings_single_translation,
    build_G_vector_mappings_double_translation,
    kpoint_isdf_double_translation,
    kpoint_isdf_single_translation,
    build_eri_isdf_double_translation,
    build_eri_isdf_single_translation,
)
from kpoint_eri.resource_estimates.utils import build_momentum_transfer_mapping

def build_kisdf_helper(mf):
    cell = mf.cell
    kpts = mf.kpts
    try:
        _, scf_dict = scf.chkfile.load_scf(mf.chkfile)
        mf.mo_coeff = scf_dict["mo_coeff"]
        mf.with_df.max_memory = 1e9
        mf.mo_occ = scf_dict["mo_occ"]
    except:
        mf.kernel()

    grid_inst = gen_grid.UniformGrids(cell)
    grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
    num_grid_points = grid_points.shape[0]
    bloch_orbitals_ao = np.array(numint.eval_ao_kpts(cell, grid_points, kpts=kpts))
    bloch_orbitals_mo = np.einsum(
        "kRp,kpi->kRi", bloch_orbitals_ao, mf.mo_coeff, optimize=True
    )
    nocc = cell.nelec[0]  # assuming same for each k-point
    density = np.einsum(
        "kRi,kRi->R",
        bloch_orbitals_mo[:, :, :nocc].conj(),
        bloch_orbitals_mo[:, :, :nocc],
        optimize=True,
    )
    num_mo = mf.mo_coeff[0].shape[-1]  # assuming the same for each k-point
    num_interp_points = 100 * num_mo
    with h5py.File(mf.chkfile, "r+") as fh5:
        try:
            interp_indx = fh5[f"interp_indx_{num_interp_points}"][:]
        except KeyError:
            kmeans = KMeansCVT(grid_points, max_iteration=500)
            interp_indx = kmeans.find_interpolating_points(
                num_interp_points, density.real
            )
            fh5[f"interp_indx_{num_interp_points}"] = interp_indx
    num_kpts = len(kpts)
    # Cell periodic part
    # u = e^{-ik.r} phi(r)
    exp_minus_ikr = np.exp(-1j * np.einsum("kx,Rx->kR", kpts, grid_points))
    cell_periodic_mo = np.einsum("kR,kRi->kRi", exp_minus_ikr, bloch_orbitals_mo)
    # go from kRi->Rki
    # AO ISDF
    cell_periodic_mo = cell_periodic_mo.transpose((1, 0, 2)).reshape(
        (num_grid_points, num_kpts * num_mo)
    )
    try:
        with h5py.File(mf.chkfile, "r") as fh5:
            chi = fh5["chi"][:]
            xi = fh5["xi"][:]
            G_mapping = fh5["G_mapping"][:]
            zeta = np.zeros((num_kpts,), dtype=object)
            for iq in range(num_kpts):
                zeta[iq] = fh5[f"zeta_{iq}"][:]
        print(chi.shape)
    except KeyError:
        chi, zeta, xi, G_mapping = kpoint_isdf_double_translation(
            mf.with_df,
            interp_indx,
            kpts,
            cell_periodic_mo,
            grid_points,
            only_unique_G=True,
        )
        chi = chi.reshape((num_interp_points, num_kpts, num_mo)).transpose((1, 2, 0))
        with h5py.File(mf.chkfile, "r+") as fh5:
            # go from Rki->kiR
            fh5["chi"] = chi
            fh5["xi"] = xi
            fh5["G_mapping"] = G_mapping
            for iq in range(num_kpts):
                fh5[f"zeta_{iq}"] = zeta[iq]

    return chi, zeta, xi, G_mapping


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
    cell.verbose = 0
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
    density = np.einsum(
        "Ri,Ri->R", orbitals_mo[:, :nocc], orbitals_mo[:, :nocc].conj(), optimize=True
    )
    assert np.einsum("R,R->", density, grid_inst.weights) == pytest.approx(nocc)
    with h5py.File(mf.chkfile, "r+") as fh5:
        try:
            interp_indx = fh5["interp_indx"][:]
        except KeyError:
            kmeans = KMeansCVT(grid_points, max_iteration=500)
            interp_indx = kmeans.find_interpolating_points(num_interp_points, density)
            fh5["interp_indx"] = interp_indx

    from kpoint_eri.factorizations.isdf import supercell_isdf

    chi, zeta, Theta = supercell_isdf(
        mf.with_df, interp_indx, orbitals=orbitals_mo, grid_points=grid_points
    )
    assert Theta.shape == (len(grid_points), num_interp_points)
    # Check overlap
    ovlp_ao = mf.get_ovlp()
    # should be identity matrix
    ovlp_mo = np.einsum(
        "pi,pq,qj->ij", mf.mo_coeff.conj(), ovlp_ao, mf.mo_coeff, optimize=True
    )
    identity = np.eye(num_mo)
    assert np.allclose(ovlp_mo, identity)
    ovlp_mu = np.einsum("Rm,R->m", Theta, grid_inst.weights, optimize=True)
    orbitals_mo_interp = orbitals_mo[interp_indx]
    ovlp_isdf = np.einsum(
        "mi,mj,m->ij",
        orbitals_mo_interp.conj(),
        orbitals_mo_interp,
        ovlp_mu,
        optimize=True,
    )
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
    cell.verbose = 0
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
    density = np.einsum(
        "Ri,Ri->R", orbitals_mo[:, :nocc].conj(), orbitals_mo[:, :nocc], optimize=True
    )
    assert np.einsum("R,R->", density, grid_inst.weights) == pytest.approx(nocc)
    with h5py.File(mf.chkfile, "r+") as fh5:
        try:
            interp_indx = fh5["interp_indx"][:]
        except KeyError:
            kmeans = KMeansCVT(grid_points, max_iteration=500)
            interp_indx = kmeans.find_interpolating_points(
                num_interp_points, density.real
            )
            fh5["interp_indx"] = interp_indx

    from kpoint_eri.factorizations.isdf import supercell_isdf

    chi, zeta, Theta = supercell_isdf(
        mf.with_df, interp_indx, orbitals=orbitals_mo, grid_points=grid_points
    )
    assert Theta.shape == (len(grid_points), num_interp_points)
    # Check overlap
    ovlp_ao = mf.get_ovlp()
    # should be identity matrix
    ovlp_mo = np.einsum(
        "pi,pq,qj->ij", mf.mo_coeff.conj(), ovlp_ao, mf.mo_coeff, optimize=True
    )
    identity = np.eye(num_mo)
    assert np.allclose(ovlp_mo, identity)
    ovlp_mu = np.einsum("Rm,R->m", Theta, grid_inst.weights, optimize=True)
    orbitals_mo_interp = orbitals_mo[interp_indx]
    ovlp_isdf = np.einsum(
        "mi,mj,m->ij",
        orbitals_mo_interp.conj(),
        orbitals_mo_interp,
        ovlp_mu,
        optimize=True,
    )
    assert np.allclose(ovlp_mo, ovlp_isdf)
    # Check ERIs.
    eri_ref = mf.with_df.ao2mo(mf.mo_coeff, kpts=mf.kpt.reshape((1, -1)))
    # Check there is a complex component
    assert np.max(np.abs(eri_ref.imag)) > 1e-3
    # for complex integrals ao2mo will yield num_mo^4 elements.
    eri_ref = eri_ref.reshape((num_mo,) * 4)
    # THC eris
    Lijn = np.einsum("mi,mj,mn->ijn", chi.conj(), chi, zeta, optimize=True)
    eri_thc = np.einsum("ijn,nk,nl->ijkl", Lijn, chi.conj(), chi, optimize=True)
    assert np.allclose(eri_thc, eri_ref)


# GPW evaluation of ERIs via pair densities
def eri_from_orb_product(mf, ijR, klR, q):
    cell = mf.cell
    grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
    num_grid_points = grid_points.shape[0]
    coulG = tools.get_coulG(cell, k=q, mesh=mf.with_df.mesh)
    weighted_coulG = coulG * cell.vol / num_grid_points
    nmo = ijR.shape[0]
    phase = np.exp(-1j * (np.einsum("x,Rx->R", q, grid_points)))
    ijG = tools.fft(ijR.reshape((nmo * nmo, -1)) * phase, mf.with_df.mesh)
    ijG *= weighted_coulG
    vR = tools.ifft(ijG, mf.with_df.mesh).reshape((nmo, nmo, -1))
    vR *= phase.conj()
    eri_q = np.einsum("ijR,klR->ijkl", vR, klR, optimize=True)
    return eri_q


# build (ij|R) from THC factors.
def build_isdf_orb_product(interp_orbitals, Theta, kpt_indx_ij, exp_ikr):
    ki, kj = kpt_indx_ij[0], kpt_indx_ij[1]
    cell_orb_product = np.einsum(
        "Rm,im,jm->Rij",
        Theta,
        interp_orbitals[ki].conj(),
        interp_orbitals[kj],
        optimize=True,
    )
    bloch_orb_product = np.einsum(
        "R,Rij->ijR", exp_ikr[ki].conj() * exp_ikr[kj], cell_orb_product, optimize=True
    )
    return bloch_orb_product


# Use ISDF for |kl) and build rho_{ij}(r) using ISDF.
def eri_from_isdf_half(
    mf, interp_orbitals, xi_mu, q, kpts, kpts_indx_pqrs, ijR, exp_ikr
):
    cell = mf.cell
    num_interp_points = interp_orbitals.shape[-1]
    ikp, ikq, ikr, iks = kpts_indx_pqrs
    q1 = kpts[ikq] - kpts[ikp]
    q2 = kpts[ikr] - kpts[iks]
    from pyscf.pbc.lib.kpts_helper import get_kconserv

    kconserv = get_kconserv(cell, kpts)
    assert xi_mu.shape[1] == num_interp_points
    cell = mf.cell
    grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
    num_grid_points = grid_points.shape[0]
    coulG = tools.get_coulG(cell, k=q, mesh=mf.with_df.mesh)
    weighted_coulG = coulG * cell.vol / num_grid_points
    nmo = ijR.shape[0]
    phase = np.exp(-1j * (np.einsum("x,Rx->R", q, grid_points)))
    ijG = tools.fft(ijR.reshape((nmo * nmo, -1)) * phase, mf.with_df.mesh)
    ijG *= weighted_coulG
    vR = tools.ifft(ijG, mf.with_df.mesh).reshape((nmo, nmo, -1))
    vR *= phase.conj()
    f_v = np.einsum(
        "R,Rv,ijR->vij", exp_ikr[ikr].conj() * exp_ikr[iks], xi_mu, vR, optimize=True
    )
    # print(f_v.shape, interp_orbitals[ikr].shape)
    eri_pqrs = np.einsum(
        "rv,sv,vpq->pqrs",
        interp_orbitals[ikr].conj(),
        interp_orbitals[iks],
        f_v,
        optimize=True,
    )
    return eri_pqrs


def eri_from_isdf(mf, interp_orbitals, xi_mu, kpts_tuple):
    cell = mf.cell
    num_interp_points = interp_orbitals.shape[-1]
    assert xi_mu.shape[1] == num_interp_points
    grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
    num_grid_points = grid_points.shape[0]
    kpts_indx, kpts = kpts_tuple
    # k_q - k_p
    q = kpts[1] - kpts[0]
    delta_G = kpts[0] - kpts[1] + kpts[2] - kpts[3]
    print("delta G: ", delta_G, q)
    phase_factor = np.exp(-1j * (np.einsum("x,Rx->R", delta_G, grid_points)))
    coulG = tools.get_coulG(cell, k=q, mesh=mf.with_df.mesh)
    weighted_coulG = coulG * cell.vol / num_grid_points
    xi_muG = tools.fft(xi_mu.T, mf.with_df.mesh)
    xi_muG *= weighted_coulG
    vR = tools.ifft(xi_muG, mf.with_df.mesh)
    zeta = np.einsum("R,Rn,mR->mn", phase_factor, xi_mu, vR, optimize=True)
    ikp, ikq, ikr, iks = kpts_indx
    eri = np.einsum(
        "rn,sn,mn,pm,qm->pqrs",
        interp_orbitals[ikr].conj(),
        interp_orbitals[iks],
        zeta,
        interp_orbitals[ikp].conj(),
        interp_orbitals[ikq],
        optimize=True,
    )
    return eri, zeta


def eri_from_isdf_2(mf, interp_orbitals, xi_mu, q, kpts_indx):
    cell = mf.cell
    num_interp_points = interp_orbitals.shape[-1]
    assert xi_mu.shape[1] == num_interp_points
    grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
    num_grid_points = grid_points.shape[0]
    coulG = tools.get_coulG(cell, k=q, mesh=mf.with_df.mesh)
    weighted_coulG = coulG * cell.vol / num_grid_points
    xi_muG = tools.fft(xi_mu.T, mf.with_df.mesh)
    xi_muG *= weighted_coulG
    vR = tools.ifft(xi_muG, mf.with_df.mesh)
    zeta = np.einsum("Rn,mR->mn", xi_mu, vR, optimize=True)
    print(np.linalg.norm(zeta - zeta.conj().T))
    ikp, ikq, ikr, iks = kpts_indx
    eri = np.einsum(
        "rn,sn,mn,pm,qm->pqrs",
        interp_orbitals[ikr].conj(),
        interp_orbitals[iks],
        zeta,
        interp_orbitals[ikp].conj(),
        interp_orbitals[ikq],
        optimize=True,
    )
    return eri


# def test_kpoint_isdf():
# cell = gto.Cell()
# cell.atom = """
# C 0.000000000000   0.000000000000   0.000000000000
# C 1.685068664391   1.685068664391   1.685068664391
# """
# cell.basis = "gth-szv"
# cell.pseudo = "gth-hf-rev"
# cell.a = """
# 0.000000000, 3.370137329, 3.370137329
# 3.370137329, 0.000000000, 3.370137329
# 3.370137329, 3.370137329, 0.000000000"""
# cell.unit = "B"
# cell.verbose = 0
# cell.build()

# kmesh = [1, 2, 3]
# kpts = cell.make_kpts(kmesh, symmorphic=False)

# mf = scf.KRHF(cell, kpts)
# mf.chkfile = "test_isdf_supercell_kpts.chk"
# # mf.kernel()
# _, scf_dict = scf.chkfile.load_scf(mf.chkfile)
# mf.mo_coeff = scf_dict["mo_coeff"]
# mf.with_df.max_memory = 1e9
# mf.mo_occ = scf_dict["mo_occ"]

# from pyscf.pbc.dft import gen_grid

# grid_inst = gen_grid.UniformGrids(cell)
# grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
# num_grid_points = grid_points.shape[0]
# bloch_orbitals_ao = np.array(numint.eval_ao_kpts(cell, grid_points, kpts=kpts))
# bloch_orbitals_mo = np.einsum(
# "kRp,kpi->kRi", bloch_orbitals_ao, mf.mo_coeff, optimize=True
# )
# num_mo_per_kpt = [C.shape[-1] for C in mf.mo_coeff]
# # Dangerous very RHF/no lin-dep dependent
# num_mo = int(np.mean(num_mo_per_kpt))
# num_interp_points = 100 * num_mo
# nocc = cell.nelec[0]
# density = np.einsum(
# "kRi,kRi->R",
# bloch_orbitals_mo[:, :, :nocc].conj(),
# bloch_orbitals_mo[:, :, :nocc],
# optimize=True,
# )
# num_kpts = len(kpts)
# assert np.einsum("R,R->", density, grid_inst.weights) == pytest.approx(
# num_kpts * nocc
# )
# # Cell periodic part
# # u = e^{-ik.r} phi(r)
# exp_minus_ikr = np.exp(-1j * np.einsum("kx,Rx->kR", kpts, grid_points))
# exp_ikr = exp_minus_ikr.conj()
# cell_periodic_mo = np.einsum("kR,kRi->kRi", exp_minus_ikr, bloch_orbitals_mo)
# cell_periodic_ao = np.einsum("kR,kRi->kRi", exp_minus_ikr, bloch_orbitals_ao)
# # sanity check
# exp_ik1r = np.exp(1j * np.einsum("x,Rx->R", kpts[1], grid_points))
# recon_mo = np.einsum("R,Ri->Ri", exp_ik1r, cell_periodic_mo[1])
# assert np.allclose(recon_mo, bloch_orbitals_mo[1])
# # Generate interpolating points.
# with h5py.File(mf.chkfile, "r+") as fh5:
# try:
# interp_indx = fh5[f"interp_indx_{num_interp_points}"][:]
# except KeyError:
# kmeans = KMeansCVT(grid_points, max_iteration=500)
# interp_indx = kmeans.find_interpolating_points(
# num_interp_points, density.real
# )
# fh5[f"interp_indx_{num_interp_points}"] = interp_indx

# from kpoint_eri.factorizations.isdf import kpoint_isdf

# # # go from kRi->Rki
# # # AO ISDF
# cell_periodic_ao = cell_periodic_ao.transpose((1, 0, 2)).reshape(
# (num_grid_points, num_kpts * num_mo)
# )
# interp_orbitals_ao, Theta = kpoint_isdf(
# mf.with_df, interp_indx, kpts, cell_periodic_ao, grid_points
# )
# # # Test ISDF solve is reproducing orbital products
# # RHS = np.einsum(
# # "Rm,mI,mJ->RIJ",
# # Theta,
# # interp_orbitals_ao.conj(),
# # interp_orbitals_ao,
# # optimize=True,
# # )
# # LHS = np.einsum(
# # "RI,RJ->RIJ", cell_periodic_ao.conj(), cell_periodic_ao, optimize=True
# # )
# # print("delta THC: ", np.abs(np.max(LHS - RHS)))
# # LHS = LHS.reshape((num_grid_points, num_kpts, num_mo, num_kpts, num_mo))
# # # Just checking bloch <-> cell periodic transformation.
# # bloch = np.einsum("kR,pR,Rkipj->Rkipj", exp_ikr.conj(), exp_ikr, LHS, optimize=True)
# # ref = np.einsum(
# # "kRi,pRj->Rkipj", bloch_orbitals_ao.conj(), bloch_orbitals_ao, optimize=True
# # )
# # assert np.allclose(ref, bloch)

# # # Test overlap <u_ik | u_jk'> = delta_{kk'} S_{ikjk'}
# # ovlp_ao = mf.get_ovlp()
# # ovlp_mu = np.einsum("Rm,R->m", Theta, grid_inst.weights)
# interp_orbitals_ao = interp_orbitals_ao.reshape((-1, num_kpts, num_mo))
# # ovlp_isdf = np.einsum(
# # "mki,mkj,m->kij",
# # interp_orbitals_ao.conj(),
# # interp_orbitals_ao,
# # ovlp_mu,
# # optimize=True,
# # )
# # # Sanity check for computing overlap via numerical integration.
# # cell_periodic_ao = cell_periodic_ao.reshape((-1, num_kpts, num_mo))
# # ovlp_comparison = np.einsum(
# # "Rki,Rkj,R->kij", cell_periodic_ao.conj(), cell_periodic_ao, grid_inst.weights
# # )
# # for ik in range(num_kpts):
# # assert np.max(np.abs(ovlp_ao[ik] - ovlp_isdf[ik])) < 1e-3
# # assert (np.max(np.abs(ovlp_ao[ik] - ovlp_comparison[ik]))) < 1e-10

# from pyscf.pbc.lib.kpts_helper import get_kconserv, member, unique, conj_mapping

# # momentum_map = build_momentum_transfer_mapping(cell, kpts)

# # kconserv = get_kconserv(cell, kpts)
# # iq = 1
# # ikp = 2
# # iks = 3
# # ikq = momentum_map[iq][ikp]
# # ikr = momentum_map[iq][iks]
# # # iks = kconserv[ikp, ikq, ikr]
# # print(kpts, len(kpts), type(kpts))
# # q1 = kpts[ikq] - kpts[ikp]
# # q2 = kpts[iks] - kpts[ikr]
# # print(q1, q2, member(q1, kpts), member(q2, kpts))
# # # print(ikq, ikr)
# # # (pk1 qk1-Q | rk2-Q s k2)
# # kpt_pqrs = [kpts[ikp], kpts[ikq], kpts[ikr], kpts[iks]]
# # mos_pqrs = [mf.mo_coeff[ikp], mf.mo_coeff[ikq], mf.mo_coeff[ikr], mf.mo_coeff[iks]]
# # eri_pqrs = mf.with_df.ao2mo(mos_pqrs, kpt_pqrs, compact=False).reshape(
# # (num_mo,) * 4
# # )
# # kpt_rspq = [kpts[ikr], kpts[iks], kpts[ikp], kpts[ikq]]
# # mos_rspq = [mf.mo_coeff[ikr], mf.mo_coeff[iks], mf.mo_coeff[ikp], mf.mo_coeff[ikq]]
# # eri_rspq = mf.with_df.ao2mo(mos_rspq, kpt_rspq, compact=False).reshape(
# # (num_mo,) * 4
# # )
# # # (pq|rs) = (rs|pq)
# # print("transpose: ", np.linalg.norm(eri_pqrs - eri_rspq))
# # # (pq|rs) = (qp|sr)*
# # kpt_rspq = [kpts[ikq], kpts[ikp], kpts[iks], kpts[ikr]]
# # mos_rspq = [mf.mo_coeff[ikq], mf.mo_coeff[ikp], mf.mo_coeff[iks], mf.mo_coeff[ikr]]
# # eri_rspq = mf.with_df.ao2mo(mos_rspq, kpt_rspq, compact=False).reshape(
# # (num_mo,) * 4
# # )
# # print("conj: ", np.linalg.norm(eri_pqrs - eri_rspq.conj()))
# # # (pq|rs) = (sr|qp)*
# # # Sanity check and rebuild ERIs using pair densities, both "exact" and from
# # # ISDF solve
# # ao_ijkl = numint.eval_ao_kpts(cell, grid_points, kpt_pqrs)
# # mo_ijkl = np.einsum("kRp,kpi->kRi", ao_ijkl, mos_pqrs)
# # # (ij|R)
# # ijR = np.einsum("Ri,Rj->ijR", mo_ijkl[0].conj(), mo_ijkl[1])
# # # (kl|R)
# # klR = np.einsum("Rk,Rl->klR", mo_ijkl[2].conj(), mo_ijkl[3])
# # # q = kpt_pqrs[2] #kpt_pqrs[1] - kpt_pqrs[0]
# # q = kpt_pqrs[1] - kpt_pqrs[0]
# # from pyscf.pbc import tools

# # eri_tmp = eri_from_orb_product(mf, ijR, klR, q)
# # print("delta eri: ", np.max(np.abs(eri_pqrs - eri_tmp)))
# # # Do it GPW way, build (ij|R) = phi_i*(r) phi_j(r) = e^{i(k_j-k_i).r} \sum_m Theta[r, m] u_{ki,m}^* u_{kj,m}
# interp_orbitals_mo = np.einsum(
# "mkp,kpi->kim", interp_orbitals_ao, mf.mo_coeff, optimize=True
# )
# # ijR_isdf = build_isdf_orb_product(interp_orbitals_mo, Theta, [ikp, ikq], exp_ikr)
# # klR_isdf = build_isdf_orb_product(interp_orbitals_mo, Theta, [ikr, iks], exp_ikr)
# # eri_isdf = eri_from_orb_product(mf, ijR_isdf, klR_isdf, q)
# # print("delta isdf orb products: ", np.max(np.abs(eri_pqrs - eri_isdf)))
# # eri_isdf_direct = eri_from_isdf(
# # mf, interp_orbitals_mo, Theta, ([ikp, ikq, ikr, iks], kpt_pqrs)
# # )
# # print("delta isdf: ", np.max(np.abs(eri_pqrs - eri_isdf_direct)))

# momentum_map = build_momentum_transfer_mapping(cell, kpts)
# mapping = []
# qvectors = []
# for ikp in range(num_kpts):
# for ikq in range(num_kpts):
# for ikr in range(num_kpts):
# for iks in range(num_kpts):
# q1 = kpts[ikp] - kpts[ikq]
# q2 = kpts[iks] - kpts[ikr]
# if (q1 - q2).dot(q1 - q2) < 1e-12:
# if (ikp, ikq) not in mapping:
# mapping.append((ikp, ikq))
# qvectors.append(q1)
# unique_transfers, unique_indx, _ = unique(qvectors)
# num_unique = len(unique_transfers)
# # enlarged_momentum_map = np.zeros((num_unique,)*2, dtype=np.int32)
# enlarged_momentum_map = -1 * np.ones((num_unique, num_kpts), dtype=np.int32)
# for iq, q in enumerate(unique_transfers):
# for ik1 in range(num_kpts):
# for ik2 in range(num_kpts):
# delta_q = q - qvectors[ik1 * num_kpts + ik2]
# if np.dot(delta_q, delta_q) < 1e-12:
# enlarged_momentum_map[iq, ik1] = ik2
# continue
# # shift center
# qpts = kpts - kpts[0]
# a = cell.lattice_vectors() / (2 * np.pi)
# minus_q = -1 * np.ones(len(kpts), dtype=np.int32)
# minus_k = conj_mapping(cell, kpts)
# print(kpts.shape)
# print(minus_q.shape)
# for iq, q in enumerate(qpts):
# for iqm, qm in enumerate(qpts):
# delta_q = q + qm
# delta_q_dot_a = np.einsum("ix,x->i", a, delta_q)
# int_delta_q_dot_a = np.rint(delta_q_dot_a)
# if np.all(np.abs(delta_q_dot_a - int_delta_q_dot_a) < 1e-10):
# minus_q[iq] = iqm
# for i, j in zip(minus_q, minus_k):
# print("q-minus: ", i, j)
# # ikq = enlarged_momentum_map[iq,ikp]
# # ikr = enlarged_momentum_map[iq,iks]
# # assert ikq > -1 and ikr > -1
# # eri_isdf_direct = eri_from_isdf_2(
# # mf, interp_orbitals_mo, Theta, unique_transfers[iq], [ikp, ikq, ikr,
# # iks]
# # )
# # print(unique_transfers[iq], kpts[ikp]-kpts[ikq], kpts[ikr]-kpts[iks])
# # # iq = 1
# # kpt_pqrs = [kpts[ikp], kpts[ikq], kpts[ikr], kpts[iks]]
# # mos_pqrs = [mf.mo_coeff[ikp], mf.mo_coeff[ikq], mf.mo_coeff[ikr], mf.mo_coeff[iks]]
# # eri_pqrs = mf.with_df.ao2mo(mos_pqrs, kpt_pqrs, compact=False).reshape(
# # (num_mo,) * 4
# # )
# # print("delta isdf: ", np.max(np.abs(eri_pqrs - eri_isdf_direct)))
# for iq in range(1, num_kpts):  # q = 0 is boring
# for ik in range(num_kpts):
# for ik_prime in range(ik + 1, num_kpts):  # ik = ik_prime is boring
# ik_minus_q = momentum_map[iq][ik]
# ik_prime_minus_q = momentum_map[iq][ik_prime]
# kpt_pqrs = [
# kpts[ik],
# kpts[ik_minus_q],
# kpts[ik_prime_minus_q],
# kpts[ik_prime],
# ]
# mos_pqrs = [
# mf.mo_coeff[ik],
# mf.mo_coeff[ik_minus_q],
# mf.mo_coeff[ik_prime_minus_q],
# mf.mo_coeff[ik_prime],
# ]
# eri_pqrs = mf.with_df.ao2mo(mos_pqrs, kpt_pqrs, compact=False).reshape(
# (num_mo,) * 4
# )
# eri_isdf_direct, zeta_q = eri_from_isdf(
# mf,
# interp_orbitals_mo,
# Theta,
# ([ik, ik_minus_q, ik_prime_minus_q, ik_prime], kpt_pqrs),
# )
# kpt_pqrs = [
# -kpts[ik],
# -kpts[ik_minus_q],
# -kpts[ik_prime_minus_q],
# -kpts[ik_prime],
# ]
# eri_pqrs_minus = mf.with_df.ao2mo(
# mos_pqrs, kpt_pqrs, compact=False
# ).reshape((num_mo,) * 4)
# eri_isdf_direct_minus, zeta_minus_q = eri_from_isdf(
# mf, interp_orbitals_mo, Theta, ([ikp, ikq, ikr, iks], kpt_pqrs)
# )
# print(
# (iq, ik, ik_minus_q, ik_prime_minus_q, ik_prime),
# np.linalg.norm(zeta_q - zeta_minus_q.conj()),
# np.linalg.norm(eri_pqrs - eri_isdf_direct),
# )


def test_G_vector_mapping():
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

    nk = 2
    kmesh = [nk, nk, nk]
    kpts = cell.make_kpts(kmesh)

    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    G_vecs, G_map, G_unique, delta_Gs = build_G_vector_mappings_double_translation(
        cell, kpts, momentum_map
    )
    num_kpts = len(kpts)
    for iq in range(num_kpts):
        for ikp in range(num_kpts):
            ikq = momentum_map[iq, ikp]
            q = kpts[ikp] - kpts[ikq]
            G_shift = G_vecs[G_map[iq, ikp]]
            assert np.allclose(q, kpts[iq] + G_shift)
    for iq in range(num_kpts):
        unique_G = np.unique(G_map[iq])
        for i, G in enumerate(G_map[iq]):
            assert unique_G[G_unique[iq][i]] == G

    inv_G_map = inverse_G_map_double_translation(cell, kpts, momentum_map)
    for iq in range(num_kpts):
        for ik in range(num_kpts):
            ix_G_qk = G_map[iq, ik]
            assert ik in inv_G_map[iq, ix_G_qk]


def test_G_vector_mapping_single_translation():
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

    nk = 3
    kmesh = [nk, nk, nk]
    kpts = cell.make_kpts(kmesh)
    num_kpts = len(kpts)

    momentum_map = build_momentum_transfer_mapping(cell, kpts)

    kpts_pq = np.array(
        [(kp, kpts[ikq]) for ikp, kp in enumerate(kpts) for ikq in range(num_kpts)]
    )

    kpts_pq_indx = np.array(
        [(ikp, ikq) for ikp, kp in enumerate(kpts) for ikq in range(num_kpts)]
    )
    transfers = kpts_pq[:, 0] - kpts_pq[:, 1]
    assert len(transfers) == (nk**3) ** 2
    unique_q, unique_indx, unique_inverse = unique(transfers)
    G_vecs, G_map, G_unique, delta_Gs = build_G_vector_mappings_single_translation(
        cell, kpts, kpts_pq_indx[unique_indx]
    )
    kconserv = get_kconserv(cell, kpts)
    for ikp in range(num_kpts):
        for ikq in range(num_kpts):
            for ikr in range(num_kpts):
                iks = kconserv[ikp, ikq, ikr]
                delta_G_expected = kpts[ikp] - kpts[ikq] + kpts[ikr] - kpts[iks]
                q = kpts[ikp] - kpts[ikq]
                qindx = member(q, transfers[unique_indx])[0]
                # print(q, len(transfers[unique_indx]), len(transfers))
                dG_indx = G_unique[qindx, ikr]
                # print(qindx, dG_indx, delta_Gs[qindx].shape)
                # print(qindx)
                # print(len(delta_Gs[qindx]), dG_indx)
                delta_G = delta_Gs[qindx][dG_indx]
                assert np.allclose(delta_G_expected, delta_G)


def test_kpoint_isdf_build():
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

    kmesh = [1, 2, 1]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts)
    mf.chkfile = "test_isdf_kpoint_build.chk"
    try:
        _, scf_dict = scf.chkfile.load_scf(mf.chkfile)
        mf.mo_coeff = scf_dict["mo_coeff"]
        mf.with_df.max_memory = 1e9
        mf.mo_occ = scf_dict["mo_occ"]
    except:
        mf.kernel()

    grid_inst = gen_grid.UniformGrids(cell)
    grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
    num_grid_points = grid_points.shape[0]
    bloch_orbitals_ao = np.array(numint.eval_ao_kpts(cell, grid_points, kpts=kpts))
    bloch_orbitals_mo = np.einsum(
        "kRp,kpi->kRi", bloch_orbitals_ao, mf.mo_coeff, optimize=True
    )
    nocc = cell.nelec[0]  # assuming same for each k-point
    density = np.einsum(
        "kRi,kRi->R",
        bloch_orbitals_mo[:, :, :nocc].conj(),
        bloch_orbitals_mo[:, :, :nocc],
        optimize=True,
    )
    num_mo = mf.mo_coeff[0].shape[-1]  # assuming the same for each k-point
    num_interp_points = 100 * num_mo
    with h5py.File(mf.chkfile, "r+") as fh5:
        try:
            interp_indx = fh5[f"interp_indx_{num_interp_points}"][:]
        except KeyError:
            kmeans = KMeansCVT(grid_points, max_iteration=500)
            interp_indx = kmeans.find_interpolating_points(
                num_interp_points, density.real
            )
            fh5[f"interp_indx_{num_interp_points}"] = interp_indx
    num_kpts = len(kpts)
    # Cell periodic part
    # u = e^{-ik.r} phi(r)
    exp_minus_ikr = np.exp(-1j * np.einsum("kx,Rx->kR", kpts, grid_points))
    cell_periodic_mo = np.einsum("kR,kRi->kRi", exp_minus_ikr, bloch_orbitals_mo)
    # go from kRi->Rki
    # AO ISDF
    cell_periodic_mo = cell_periodic_mo.transpose((1, 0, 2)).reshape(
        (num_grid_points, num_kpts * num_mo)
    )
    try:
        with h5py.File(mf.chkfile, "r") as fh5:
            chi = fh5["chi"][:]
            xi = fh5["xi"][:]
            G_mapping = fh5["G_mapping"][:]
            zeta = np.zeros((num_kpts,), dtype=object)
            for iq in range(num_kpts):
                zeta[iq] = fh5[f"zeta_{iq}"][:]
        print(chi.shape)
    except KeyError:
        chi, zeta, xi, G_mapping = kpoint_isdf_double_translation(
            mf.with_df,
            interp_indx,
            kpts,
            cell_periodic_mo,
            grid_points,
            only_unique_G=True,
        )
        chi = chi.reshape((num_interp_points, num_kpts, num_mo)).transpose((1, 2, 0))
        with h5py.File(mf.chkfile, "r+") as fh5:
            # go from Rki->kiR
            fh5["chi"] = chi
            fh5["xi"] = xi
            fh5["G_mapping"] = G_mapping
            for iq in range(num_kpts):
                fh5[f"zeta_{iq}"] = zeta[iq]
    # chi, zeta, xi, G_mapping = kpoint_isdf_double_translation(
    # mf.with_df,
    # interp_indx,
    # kpts,
    # cell_periodic_mo,
    # grid_points,
    # only_unique_G=True,
    # )
    # chi = chi.reshape((num_interp_points, num_kpts, num_mo)).transpose((1, 2, 0))
    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    for iq in range(1, num_kpts):
        for ikp in range(num_kpts):
            ikq = momentum_map[iq, ikp]
            Gpq = G_mapping[iq, ikp]
            for iks in range(num_kpts):
                ikr = momentum_map[iq, iks]
                Gsr = G_mapping[iq, iks]
                kpt_pqrs = [kpts[ikp], kpts[ikq], kpts[ikr], kpts[iks]]
                mos_pqrs = [
                    mf.mo_coeff[ikp],
                    mf.mo_coeff[ikq],
                    mf.mo_coeff[ikr],
                    mf.mo_coeff[iks],
                ]
                eri_pqrs = mf.with_df.ao2mo(mos_pqrs, kpt_pqrs, compact=False).reshape(
                    (num_mo,) * 4
                )
                eri_pqrs_isdf = build_eri_isdf_double_translation(
                    chi, zeta, iq, [ikp, ikq, ikr, iks], G_mapping
                )
                eri_from_isdf_old, zeta_ = eri_from_isdf(
                    mf, chi, xi, ([ikp, ikq, ikr, iks], kpt_pqrs)
                )
                print("delta new: ", np.linalg.norm(eri_pqrs - eri_pqrs_isdf))
                print("delta old: ", np.linalg.norm(eri_pqrs - eri_from_isdf_old))
                print("dzeta: ", np.linalg.norm(zeta_ - zeta[iq][Gpq, Gsr]))


def test_kpoint_isdf_build_single_translation():
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

    kmesh = [1, 2, 1]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts)
    mf.chkfile = "test_isdf_kpoint_build_single_translation.chk"
    try:
        _, scf_dict = scf.chkfile.load_scf(mf.chkfile)
        mf.mo_coeff = scf_dict["mo_coeff"]
        mf.with_df.max_memory = 1e9
        mf.mo_occ = scf_dict["mo_occ"]
    except:
        mf.kernel()

    grid_inst = gen_grid.UniformGrids(cell)
    grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
    num_grid_points = grid_points.shape[0]
    bloch_orbitals_ao = np.array(numint.eval_ao_kpts(cell, grid_points, kpts=kpts))
    bloch_orbitals_mo = np.einsum(
        "kRp,kpi->kRi", bloch_orbitals_ao, mf.mo_coeff, optimize=True
    )
    nocc = cell.nelec[0]  # assuming same for each k-point
    density = np.einsum(
        "kRi,kRi->R",
        bloch_orbitals_mo[:, :, :nocc].conj(),
        bloch_orbitals_mo[:, :, :nocc],
        optimize=True,
    )
    num_mo = mf.mo_coeff[0].shape[-1]  # assuming the same for each k-point
    num_interp_points = 100 * num_mo
    with h5py.File(mf.chkfile, "r+") as fh5:
        try:
            interp_indx = fh5[f"interp_indx_{num_interp_points}"][:]
        except KeyError:
            kmeans = KMeansCVT(grid_points, max_iteration=500)
            interp_indx = kmeans.find_interpolating_points(
                num_interp_points, density.real
            )
            fh5[f"interp_indx_{num_interp_points}"] = interp_indx
    num_kpts = len(kpts)
    # Cell periodic part
    # u = e^{-ik.r} phi(r)
    exp_minus_ikr = np.exp(-1j * np.einsum("kx,Rx->kR", kpts, grid_points))
    cell_periodic_mo = np.einsum("kR,kRi->kRi", exp_minus_ikr, bloch_orbitals_mo)
    # go from kRi->Rki
    # AO ISDF
    cell_periodic_mo = cell_periodic_mo.transpose((1, 0, 2)).reshape(
        (num_grid_points, num_kpts * num_mo)
    )
    try:
        with h5py.File(mf.chkfile, "r") as fh5:
            chi = fh5["chi"][:]
            xi = fh5["xi"][:]
            G_mapping = fh5["G_mapping"][:]
            num_qpoints = len(G_mapping)
            zeta = np.zeros((num_qpoints,), dtype=object)
            for iq in range(G_mapping.shape[0]):
                zeta[iq] = fh5[f"zeta_{iq}"][:]
        print(chi.shape)
    except KeyError:
        chi, zeta, xi, G_mapping = kpoint_isdf_single_translation(
            mf.with_df,
            interp_indx,
            kpts,
            cell_periodic_mo,
            grid_points,
            only_unique_G=True,
        )
        chi = chi.reshape((num_interp_points, num_kpts, num_mo)).transpose((1, 2, 0))
        with h5py.File(mf.chkfile, "r+") as fh5:
            # go from Rki->kiR
            fh5["chi"] = chi
            fh5["xi"] = xi
            fh5["G_mapping"] = G_mapping
            assert G_mapping.shape[0] == zeta.shape[0]
            for iq in range(zeta.shape[0]):
                fh5[f"zeta_{iq}"] = zeta[iq]
    kconserv = get_kconserv(cell, kpts)
    kpts_pq = np.array(
        [(kp, kpts[ikq]) for ikp, kp in enumerate(kpts) for ikq in range(num_kpts)]
    )

    kpts_pq_indx = np.array(
        [(ikp, ikq) for ikp, kp in enumerate(kpts) for ikq in range(num_kpts)]
    )
    transfers = kpts_pq[:, 0] - kpts_pq[:, 1]
    # assert len(transfers) == (nk**3)**2
    unique_q, unique_indx, unique_inverse = unique(transfers)
    for ikp in range(num_kpts):
        for ikq in range(num_kpts):
            for ikr in range(num_kpts):
                iks = kconserv[ikp, ikq, ikr]
                kpt_pqrs = [kpts[ikp], kpts[ikq], kpts[ikr], kpts[iks]]
                mos_pqrs = [
                    mf.mo_coeff[ikp],
                    mf.mo_coeff[ikq],
                    mf.mo_coeff[ikr],
                    mf.mo_coeff[iks],
                ]
                eri_pqrs = mf.with_df.ao2mo(mos_pqrs, kpt_pqrs, compact=False).reshape(
                    (num_mo,) * 4
                )
                q = kpts[ikp] - kpts[ikq]
                qindx = member(q, transfers[unique_indx])[0]
                print("Gmapping: ", G_mapping[qindx, ikr], qindx, q)
                eri_pqrs_isdf = build_eri_isdf_single_translation(
                    chi, zeta, qindx, [ikp, ikq, ikr, iks], G_mapping
                )
                dg_indx = G_mapping[qindx, ikr]
                eri_from_isdf_old, zeta_ = eri_from_isdf(
                    mf, chi, xi, ([ikp, ikq, ikr, iks], kpt_pqrs)
                )
                print("delta new: ", np.linalg.norm(eri_pqrs - eri_pqrs_isdf))
                print("delta old: ", np.linalg.norm(eri_pqrs - eri_from_isdf_old))
                print("dzeta: ", np.linalg.norm(zeta_ - zeta[qindx][dg_indx]))


def build_eri(mf, kpt_pqrs):
    p, q, r, s = kpt_pqrs
    kpts = mf.kpts
    kpt_pqrs = [kpts[p], kpts[q], kpts[r], kpts[s]]
    num_mo = mf.mo_coeff[0].shape[-1]
    mos_pqrs = [
        mf.mo_coeff[p],
        mf.mo_coeff[q],
        mf.mo_coeff[r],
        mf.mo_coeff[s],
    ]
    eri_pqrs = mf.with_df.ao2mo(mos_pqrs, kpt_pqrs, compact=False).reshape(
        (num_mo,) * 4
    )
    return eri_pqrs

def get_miller(lattice_vectors, G):
    miller_indx = np.rint(
        np.einsum("wx,x->w", lattice_vectors, G) / (2 * np.pi)
    ).astype(np.int32)
    return miller_indx

def test_kpoint_isdf_symmetries():
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
    mf.chkfile = "test_isdf_kpoint_build_symmetries.chk"
    chi, zeta, xi, _ = build_kisdf_helper(mf)
    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    G_vecs, G_map, G_unique, delta_Gs = build_G_vector_mappings_double_translation(
        cell, kpts, momentum_map
    )
    G_dict, _ = build_G_vectors(cell)
    num_kpts = len(kpts)
    # Test symmetries from F30-F33
    # Test LHS for sanity too (need to uncomment)
    grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
    lattice_vectors = cell.lattice_vectors()
    from pyscf.pbc.lib.kpts_helper import conj_mapping
    minus_k_map = conj_mapping(cell, kpts)
    # Sanity check xi is real
    print("max xi.imag: ", np.max(np.abs(xi.imag)))
    zero = np.zeros(3)
    for iq in range(1, num_kpts):
        for ik in range(num_kpts):
            ik_minus_q = momentum_map[iq, ik]
            Gpq = G_vecs[G_map[iq, ik]]
            for ik_prime in range(num_kpts):
                Gsr = G_vecs[G_map[iq, ik_prime]]
                ik_prime_minus_q = momentum_map[iq, ik_prime]
                # Sanity check G mappings
                assert np.allclose(kpts[ik] - kpts[ik_minus_q] - kpts[iq], Gpq)
                assert np.allclose(kpts[ik_prime] - kpts[ik_prime_minus_q] - kpts[iq], Gsr)
                # F30. (pk qk-Q | rk'-Q sk') = (q k-Q p k | sk' rk'-Q)*
                ik_prime_minus_q = momentum_map[iq, ik_prime]
                # uncomment to check normal eris
                # kpt_pqrs = [ik, ik_minus_q, ik_prime_minus_q, ik_prime]
                # eri_pqrs = build_eri(mf, kpt_pqrs)
                # kpt_pqrs = [ik, ik_minus_q, ik_prime_minus_q, ik_prime]
                # kpt_pqrs = [ik_minus_q, ik, ik_prime, ik_prime_minus_q]
                # eri_qpsr = build_eri(mf, kpt_pqrs).transpose((1, 0, 3, 2))
                # Sanity check relationship
                # assert np.allclose(eri_pqrs, eri_qpsr.conj())
                # Check zeta symmetry: expect zeta[Q,G1,G2,m,n] = zeta[-Q,G1_comp,G2_comp,m, n].conj()
                # Build refernce point zeta[Q,G1,G2,m,n]
                zeta_ref = build_kpoint_zeta(mf.with_df, kpts[iq], Gpq, Gsr, grid_points, xi)
                # Get -Q index
                minus_iq = minus_k_map[iq]
                # This flips the sign of the miller index corresponding to the G
                # vectors
                overleaf_Gsr_comp_tuple = ~get_miller(lattice_vectors, Gsr)
                overleaf_Gpq_comp_tuple = ~get_miller(lattice_vectors, Gpq)
                # Want to find -Q + G_pq_comp + (Q + Gpq) = 0, Q + Gpq = kp - kq = q
                # so G_pq_comp = -((-Q) - (Q+Gpq))
                Gpq_comp = -(kpts[minus_iq] + kpts[iq] + Gpq)
                Gsr_comp = -(kpts[minus_iq] + kpts[iq] + Gsr)
                assert np.allclose(kpts[minus_iq] + kpts[iq] + Gpq + Gpq_comp, zero)
                assert np.allclose(kpts[minus_iq] + kpts[iq] + Gsr + Gsr_comp, zero)
                # Compare this "complement G" to overleaf
                print("iq = {}, ik = {}, ik_prime = {}".format(iq, ik, ik_prime))
                print("G {} new !G: {}".format(get_miller(lattice_vectors, Gpq), get_miller(lattice_vectors, Gpq_comp)))
                print("G' {} new !G': {}".format(get_miller(lattice_vectors, Gsr), get_miller(lattice_vectors, Gpq_comp)))
                print("G {} ovleaf !G: {}".format(get_miller(lattice_vectors, Gpq), overleaf_Gpq_comp_tuple))
                print("G' {} ovleaf !G': {}".format(get_miller(lattice_vectors, Gsr), overleaf_Gsr_comp_tuple))
                print()
                zeta_test = build_kpoint_zeta(mf.with_df,
                                                      kpts[minus_iq], Gpq_comp,
                                                      Gsr_comp, grid_points, xi)
                # F31 (pk qk-Q | rk'-Q sk') = (rk'-Q s k'| pk qk-Q)
                assert np.allclose(zeta_ref, zeta_test.conj())
                # Sanity check do literal minus signs (should be complex
                # conjugate)
                zeta_test = build_kpoint_zeta(mf.with_df, -kpts[iq], -Gpq, -Gsr, grid_points, xi)
                assert np.allclose(zeta_ref, zeta_test.conj())
                # F32 (pk qk-Q | rk'-Q sk') = (rk'-Q s k'| pk qk-Q)
                # uncomment to check normal eris
                # kpt_pqrs = [ik_prime_minus_q, ik_prime, ik, ik_minus_q]
                # eri_rspq = build_eri(mf, kpt_pqrs).transpose((2, 3, 0, 1))
                # assert np.allclose(eri_pqrs, eri_rspq)
                # Check zeta symmetry: expect zeta[Q,G1,G2,m,n] = # zeta[-Q,G2_comp,G1_comp,m, n]
                zeta_test = build_kpoint_zeta(mf.with_df,
                                                      kpts[minus_iq], Gsr_comp,
                                                      Gpq_comp, grid_points, xi)
                assert np.allclose(zeta_ref, zeta_test.T)
                # F33 (pk qk-Q | rk'-Q sk') = (sk' r k'-Q| qk-Q pk)
                # uncomment to check normal eris
                # kpt_pqrs = [ik_prime, ik_prime_minus_q, ik_minus_q, ik]
                # eri_srqp = build_eri(mf, kpt_pqrs).transpose((3, 2, 1, 0))
                # assert np.allclose(eri_pqrs, eri_srqp.conj())
                # Check zeta symmetry: expect zeta[Q,G1,G2,m,n] = zeta[Q,G2,G1,n, m].conj()
                zeta_test = build_kpoint_zeta(mf.with_df, kpts[iq],
                                                 Gsr, Gpq, grid_points, xi)
                assert np.allclose(zeta_ref, zeta_test.conj().T)


def test_symmetry_of_G_maps():
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

    kmesh = [4, 3, 3]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts)
    # mf.chkfile = "test_isdf_kpoint_build_symmetries.chk"
    # chi, zeta, xi, _ = build_kisdf_helper(mf)
    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    G_vecs, G_map, G_unique, delta_Gs = build_G_vector_mappings_double_translation(
        cell, kpts, momentum_map
    )
    G_dict, _ = build_G_vectors(cell)
    num_kpts = len(kpts)
    # Test symmetries from F30-F33
    # Test LHS for sanity too (need to uncomment)
    # grid_points = cell.gen_uniform_grids(mf.with_df.mesh)
    lattice_vectors = cell.lattice_vectors()
    from pyscf.pbc.lib.kpts_helper import conj_mapping
    minus_k_map = conj_mapping(cell, kpts)
    # Sanity check xi is real
    # print("max xi.imag: ", np.max(np.abs(xi.imag)))
    zero = np.zeros(3)
    for iq in range(1, num_kpts):
        for ik in range(num_kpts):
            ik_minus_q = momentum_map[iq, ik]
            Gpq = G_vecs[G_map[iq, ik]]
            for ik_prime in range(num_kpts):
                Gsr = G_vecs[G_map[iq, ik_prime]]
                ik_prime_minus_q = momentum_map[iq, ik_prime]
                minus_iq = minus_k_map[iq]
                Gpq_comp = -(kpts[minus_iq] + kpts[iq] + Gpq)
                Gsr_comp = -(kpts[minus_iq] + kpts[iq] + Gsr)
                # Get indx of "complement" G in original set of 27
                iGpq_comp = G_dict[tuple(get_miller(lattice_vectors, Gpq_comp))]
                iGsr_comp = G_dict[tuple(get_miller(lattice_vectors, Gpq_comp))]
                # Get index of unique Gs in original set of 27
                G_indx_unique = [G_dict[tuple(get_miller(lattice_vectors, G))] for G in delta_Gs[iq]]
                # Check complement is in set corresponding to zeta[-Q]
                assert iGpq_comp in G_indx_unique
                assert iGsr_comp in G_indx_unique


def test_G_vector_mapping_double():
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

    import itertools
    from pyscf import lib
    nk = 4
    kmesh = [nk, 1, 1]
    # kmesh = [3, 2, 1]
    kpts = cell.make_kpts(kmesh)
    scaled_kpts = cell.get_scaled_kpts(kpts)
    ks_each_axis = []
    ks_int_each_axis = []
    for n in kmesh:
        ks = np.arange(n, dtype=float) / n
        ks_each_axis.append(ks)
        ks_int_each_axis.append(np.arange(n, dtype=float))
    scaled_kpts = lib.cartesian_prod(ks_each_axis)
    int_scaled_kpts = lib.cartesian_prod(ks_int_each_axis)
    assert np.allclose(scaled_kpts[:, 0] * kmesh[0], int_scaled_kpts[:, 0])
    assert np.allclose(scaled_kpts[:, 1] * kmesh[1], int_scaled_kpts[:, 1])
    assert np.allclose(scaled_kpts[:, 2] * kmesh[2], int_scaled_kpts[:, 2])

    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    # print(momentum_map)

    kpoints = kpts
    delta_k1_k2_Q = kpoints[:,None,None,:] - kpoints[None,:,None,:] - kpoints[None,None,:,:]
    delta_k1_k2_Q += kpoints[0][None, None, None, :]  # shift to center
    delta_k1_k2_Q_int = int_scaled_kpts[:, None, None, :] - int_scaled_kpts[None, :, None, :] - int_scaled_kpts[None, None, :, :]

    import itertools
    for (kpidx, kqidx, qidx) in itertools.product(range(len(kpts)), repeat=3):
        assert np.allclose(kpoints[kpidx] - kpoints[kqidx] - kpoints[qidx], delta_k1_k2_Q[kpidx, kqidx, qidx])
    delta_dot_a = np.einsum('wx,kpQx->kpQw', cell.lattice_vectors() / (2 * np.pi), delta_k1_k2_Q)
    delta_dot_a_v2 = np.zeros_like(delta_dot_a)  # fraction [0, 1) representation of the momentum mode
    delta_dot_a_int_version = np.zeros_like(delta_dot_a) # integer representation [[0, Nk_{x}-1], [0, Nk_{y}-1], [0, Nk_{z}- 1]]
    for (kpidx, kqidx, qidx) in itertools.product(range(len(kpts)), repeat=3):
        delta_dot_a_v2[kpidx, kqidx, qidx] = cell.get_scaled_kpts(np.array(delta_k1_k2_Q[kpidx, kqidx, qidx]))
        delta_dot_a_int_version[kpidx, kqidx, qidx] = cell.get_scaled_kpts(np.array(delta_k1_k2_Q[kpidx, kqidx, qidx])) * np.array(kmesh)

        assert np.allclose(delta_dot_a_v2[kpidx, kqidx, qidx], 
                           delta_dot_a[kpidx, kqidx, qidx])
        # print(delta_dot_a_v2[kpidx, kqidx, qidx])
        # print(delta_dot_a_int_version[kpidx, kqidx, qidx])
        # print(delta_k1_k2_Q_int[kpidx, kqidx, qidx])
        assert np.allclose(delta_k1_k2_Q_int[kpidx, kqidx, qidx], delta_dot_a_int_version[kpidx, kqidx, qidx])


    int_delta_dot_a = np.rint(delta_dot_a)
    # Should be zero if transfer is statisfied (2*pi*n)
    test_transfer_map = np.zeros((len(kpts), len(kpts)), dtype=np.int32)
    ncr_transfer_map = np.zeros((len(kpts), len(kpts)), dtype=np.int32)
    for (kpidx, kqidx, qidx) in itertools.product(range(len(kpts)), repeat=3):
        # explicitly build my transfer matrix
        if np.sum(np.abs(delta_dot_a_v2[kpidx, kqidx, qidx] - np.rint(delta_dot_a_v2[kpidx, kqidx, qidx]))) < 1.0E-10:
            test_transfer_map[kqidx, kpidx] = qidx

        # build transfer matrix based on integer version.  Pretty much we need to know 
        # if the mod is zero. rint just forces this to be the integer value beacuse we are coming
        # from float land where we aren't precise integers. 
        # print(delta_dot_a_int_version[kpidx, kqidx, qidx],             
        #      (np.rint(delta_dot_a_int_version[kpidx, kqidx, qidx][0])) % kmesh[0],
        #      (np.rint(delta_dot_a_int_version[kpidx, kqidx, qidx][1])) % kmesh[1],
        #      (np.rint(delta_dot_a_int_version[kpidx, kqidx, qidx][2])) % kmesh[2]
        #      )
        if np.allclose([np.rint(delta_dot_a_int_version[kpidx, kqidx, qidx][0]) % kmesh[0],
                        np.rint(delta_dot_a_int_version[kpidx, kqidx, qidx][1]) % kmesh[1],
                        np.rint(delta_dot_a_int_version[kpidx, kqidx, qidx][2]) % kmesh[2],
                       ], 0
                      ):
            ncr_transfer_map[kqidx, kpidx] = qidx

    # print("kmesh")
    # print(kmesh)
    mapping = np.where(np.sum(np.abs(delta_dot_a-int_delta_dot_a), axis=3) < 1e-10)
    num_kpoints = len(kpoints)
    momentum_transfer_map = np.zeros((num_kpoints,)*2, dtype=np.int32)
    # Note index flip due to Q being first index in map but broadcasted last..
    momentum_transfer_map[mapping[1], mapping[0]] = mapping[2]
    # print(momentum_transfer_map)
    # print(test_transfer_map)
    assert np.allclose(momentum_transfer_map, test_transfer_map)
    # print(ncr_transfer_map)
    assert np.allclose(ncr_transfer_map, momentum_transfer_map)

    
    # now test building G_vecs
    from kpoint_eri.factorizations.isdf import build_G_vectors
    # G_dict, G_vectors = build_G_vectors(cell)
    G_vecs, G_map, G_unique, delta_Gs = build_G_vector_mappings_double_translation(
        cell, kpts, momentum_map
    )
    num_kpts = len(kpts)
    for iq in range(num_kpts):
        for ikp in range(num_kpts):
            ikq = momentum_map[iq, ikp]
            q = kpts[ikp] - kpts[ikq]
            # print(cell.get_scaled_kpts(q))
            G_shift = G_vecs[G_map[iq, ikp]]
            # print(cell.get_scaled_kpts(G_shift))
            # print(cell.get_scaled_kpts(kpts[iq]))
            # print(cell.get_scaled_kpts(G_shift)[0], # * kmesh[0],
            #       cell.get_scaled_kpts(G_shift)[1], # * kmesh[1],
            #       cell.get_scaled_kpts(G_shift)[2],)#  * kmesh[2])
            # print(cell.get_scaled_kpts(kpts[iq] + G_shift))
            print(cell.get_scaled_kpts(q))
            print(cell.get_scaled_kpts(G_shift) + 
                  cell.get_scaled_kpts(kpts[iq]))
            print()
            assert np.allclose(q, kpts[iq] + G_shift)
    for iq in range(num_kpts):
        unique_G = np.unique(G_map[iq])
        for i, G in enumerate(G_map[iq]):
            assert unique_G[G_unique[iq][i]] == G


if __name__ == "__main__":
    # test_G_vector_mapping_double()
    # test_supercell_isdf_gamma()
    # test_supercell_isdf_complex()
    # test_kpoint_isdf_build()
    # test_kpoint_isdf_symmetries()
    test_symmetry_of_G_maps()
    # test_kpoint_isdf_build_single_translation()
    # test_G_vector_mapping()
    # test_G_vector_mapping_single_translation()
