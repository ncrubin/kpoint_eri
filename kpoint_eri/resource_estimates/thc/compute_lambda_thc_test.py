import numpy as np
import pytest
import h5py
from pyscf.pbc import gto, scf, mp
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint
from functools import reduce

from kpoint_eri.factorizations.kmeans import KMeansCVT
from openfermion.resource_estimates.thc.utils.thc_factorization import (
    lbfgsb_opt_thc_l2reg,
)
from openfermion.resource_estimates.thc.utils.thc_factorization import (
    thc_objective_regularized as thc_obj_mol,
)

from kpoint_eri.resource_estimates.thc.integral_helper import (
        KPTHCHelperDoubleTranslation,
        )

from kpoint_eri.factorizations.testing_utils import thc_test_chkpoint_helper
from kpoint_eri.resource_estimates.utils import build_momentum_transfer_mapping
from kpoint_eri.resource_estimates.thc.compute_lambda_thc import (
        compute_lambda_ncr_v2,
        compute_lambda_real
        )
from kpoint_eri.resource_estimates.utils import build_momentum_transfer_mapping
from kpoint_eri.factorizations.thc_jax import (
    adagrad_opt_kpthc_batched,
    unpack_thc_factors,
    get_zeta_size,
    lbfgsb_opt_kpthc_l2reg_batched,
)


def test_kpoint_thc_lambda():
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

    kmesh = [1, 1, 2]
    kpts = cell.make_kpts(kmesh)
    num_kpts = len(kpts)
    mf = scf.KRHF(cell, kpts)
    mf.chkfile = "test_thc_kpoint_build.chk"
    cthc = 4
    #
    # Build kpoint eris from supercell
    #
    #
    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    num_interp_points = cthc * mf.cell.nao
    chi, zeta, G_mapping, Luv_cont = thc_test_chkpoint_helper(mf, num_interp_points)
    opt_param = lbfgsb_opt_kpthc_l2reg_batched(
        chi,
        zeta,
        momentum_map,
        G_mapping,
        Luv_cont,
        chkfile_name="thc_opt.h5",
        maxiter=3000,
        penalty_param=1e-3,
    )
    num_G_per_Q = [z.shape[0] for z in zeta]
    num_mo = mf.mo_coeff[0].shape[-1]
    chi_unpacked, zeta_unpacked = unpack_thc_factors(
        opt_param, num_interp_points, num_mo, num_kpts, num_G_per_Q
    )
    opt_param = adagrad_opt_kpthc_batched(
        chi,
        zeta,
        momentum_map,
        G_mapping,
        Luv_cont,
        chkfile_name="thc_adagrad.h5",
        maxiter=3000,
    )
    chi_unpacked, zeta_unpacked = unpack_thc_factors(
        opt_param, num_interp_points, num_mo, num_kpts, num_G_per_Q
    )
    helper = KPTHCHelperDoubleTranslation(chi_unpacked, zeta_unpacked, mf)
    for q in range(num_kpts):
        print(np.sum(np.abs(zeta[q])))
    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mf.mo_coeff)])
    lambda_tot_kp, lambda_one_body_kp, lambda_two_body_kp = compute_lambda_ncr_v2(
        hcore_mo, helper,
    )
    print(lambda_tot_kp, lambda_one_body_kp, lambda_two_body_kp)
    #
    #
    # Build "molecular" eris from supercell THC
    #
    #
    #
    from kpoint_eri.resource_estimates.utils.k2gamma import k2gamma, get_phase
    supercell_mf = k2gamma(mf)
    supercell_mf.chkfile = "sc_mf_thc.chk"
    dm = supercell_mf.make_rdm1()
    supercell_mf.kernel(dm)
    num_interp_points = cthc * supercell_mf.cell.nao
    _mesh = supercell_mf.with_df.mesh.copy()
    cell = supercell_mf.cell
    # cell.mesh = [_mesh[0], _mesh[1], _mesh[2]+1]
    grid_inst = gen_grid.UniformGrids(cell)
    # supercell_mf.with_df.mesh = [_mesh[0], _mesh[1], _mesh[2]+1]
    grid_points = cell.gen_uniform_grids(supercell_mf.with_df.mesh)
    orbitals = numint.eval_ao(cell, grid_points)
    orbitals_mo = np.einsum("Rp,pi->Ri", orbitals, supercell_mf.mo_coeff[0], optimize=True)
    num_mo = supercell_mf.mo_coeff[0].shape[1]
    num_interp_points = 4 * num_mo
    nocc = cell.nelec[0]
    density = np.einsum(
        "Ri,Ri->R", orbitals_mo[:, :nocc], orbitals_mo[:, :nocc].conj(), optimize=True
    )
    assert np.einsum("R,R->", density, grid_inst.weights) == pytest.approx(nocc)
    with h5py.File(supercell_mf.chkfile, "r+") as fh5:
        try:
            interp_indx = fh5["interp_indx"][:]
        except KeyError:
            kmeans = KMeansCVT(grid_points, max_iteration=500)
            interp_indx = kmeans.find_interpolating_points(num_interp_points, density)
            fh5["interp_indx"] = interp_indx

    from kpoint_eri.factorizations.isdf import supercell_isdf

    with h5py.File(supercell_mf.chkfile, "r+") as fh5:
        try:
            interp_indx = fh5["interp_indx"][:]
        except KeyError:
            kmeans = KMeansCVT(grid_points, max_iteration=500)
            interp_indx = kmeans.find_interpolating_points(num_interp_points, density)
            fh5["interp_indx"] = interp_indx

    with h5py.File(supercell_mf.chkfile, "r+") as fh5:
        try:
            chi_sc = fh5["chi"][:]
            zeta_sc = fh5["zeta"][:]
        except KeyError:
            chi_sc, zeta_sc, Theta = supercell_isdf(
                supercell_mf.with_df, interp_indx, orbitals=orbitals_mo, grid_points=grid_points
            )
            fh5["chi"] = chi_sc
            fh5["zeta"] = zeta_sc
    num_mo = supercell_mf.mo_coeff[0].shape[-1]
    eri = supercell_mf.with_df.ao2mo([supercell_mf.mo_coeff[0]]*4, [[0,0,0]]*4,
                                     compact=False).reshape((num_mo,)*4)
    buffer = np.zeros((chi_sc.size + zeta_sc.size), dtype=np.float64)
    eri_thc = np.einsum(
        "np,nq,nm,mr,ms->pqrs",
        chi_sc,
        chi_sc,
        zeta_sc,
        chi_sc,
        chi_sc,
        optimize=True,
    )
    buffer[: chi_sc.size] = chi_sc.real.ravel()
    buffer[chi_sc.size :] = zeta_sc.real.ravel()
    np.random.seed(7)
    opt_param = lbfgsb_opt_thc_l2reg(
        eri,
        num_interp_points,
        chkfile_name="thc_opt_gamma.h5",
        maxiter=30,
        initial_guess=buffer,
        penalty_param=1e-3,
    )
    chi_unpacked_mol = opt_param[: chi_sc.size].reshape((num_interp_points, num_mo))
    zeta_unpacked_mol = opt_param[chi_sc.size :].reshape(zeta_sc.shape)
    hcore_ao = supercell_mf.get_hcore()
    hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(supercell_mf.mo_coeff)])
    lambda_tot, lambda_one_body, lambda_two_body = compute_lambda_real(
        hcore_mo, chi_unpacked_mol, zeta_unpacked_mol, eri
    )
    num_interp_points = cthc * mf.mo_coeff[0].shape[-1]
    print(
        lambda_tot,
        lambda_tot_kp,
        lambda_one_body,
        lambda_one_body_kp,
        lambda_two_body,
        lambda_two_body_kp,
    )

if __name__ == "__main__":
    test_kpoint_thc_lambda()
