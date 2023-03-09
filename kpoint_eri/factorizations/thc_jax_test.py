import numpy as np
import jax.numpy as jnp
import jax
import os

from pyscf.pbc import gto, scf, mp

from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
from kpoint_eri.resource_estimates.utils.misc_utils import (
    build_momentum_transfer_mapping,
)
from kpoint_eri.factorizations.isdf import solve_kmeans_kpisdf
from kpoint_eri.factorizations.thc_jax import (
    kpoint_thc_via_isdf,
    pack_thc_factors,
    thc_objective_regularized,
    thc_objective_regularized_batched,
    unpack_thc_factors,
    get_zeta_size,
    make_contiguous_cholesky,
    lbfgsb_opt_kpthc_l2reg,
    lbfgsb_opt_kpthc_l2reg_batched,
    adagrad_opt_kpthc_batched,
    prepare_batched_data_indx_arrays,
)

from openfermion.resource_estimates.thc.utils.thc_factorization import (
    lbfgsb_opt_thc_l2reg,
)
from openfermion.resource_estimates.thc.utils.thc_factorization import (
    thc_objective_regularized as thc_obj_mol,
)


def test_kpoint_thc_reg_gamma():
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
    cell.mesh = [11] * 3
    cell.verbose = 0
    cell.build()

    kmesh = [1, 1, 1]
    kpts = cell.make_kpts(kmesh)
    num_kpts = len(kpts)
    mf = scf.KRHF(cell, kpts)
    mf.kernel()
    num_mo = mf.mo_coeff[0].shape[-1]
    num_interp_points = 10 * mf.mo_coeff[0].shape[-1]
    kpt_thc = solve_kmeans_kpisdf(mf, num_interp_points, single_translation=False)
    chi, zeta, G_mapping = kpt_thc.chi, kpt_thc.zeta, kpt_thc.G_mapping
    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    buffer = np.zeros(2 * (chi.size + get_zeta_size(zeta)), dtype=np.float64)
    pack_thc_factors(chi, zeta, buffer)
    num_G_per_Q = [z.shape[0] for z in zeta]
    chi_unpacked, zeta_unpacked = unpack_thc_factors(
        buffer, num_interp_points, num_mo, num_kpts, num_G_per_Q
    )
    assert np.allclose(chi_unpacked, chi)
    for iq in range(num_kpts):
        assert np.allclose(zeta[iq], zeta_unpacked[iq])
    # force contiguous
    rsmf = scf.KRHF(mf.cell, mf.kpts).rs_density_fit()
    rsmf.verbose = 0
    rsmf.mo_occ = mf.mo_occ
    rsmf.mo_coeff = mf.mo_coeff
    rsmf.mo_energy = mf.mo_energy
    rsmf.with_df.mesh = mf.cell.mesh
    mymp = mp.KMP2(rsmf)
    Luv = cholesky_from_df_ints(mymp)
    Luv_cont = make_contiguous_cholesky(Luv)
    eri = np.einsum("npq,nrs->pqrs", Luv_cont[0, 0], Luv_cont[0, 0]).real
    buffer = np.zeros((chi.size + get_zeta_size(zeta)), dtype=np.float64)
    eri_thc = np.einsum(
        "pn,qn,nm,rm,sm->pqrs",
        chi[0],
        chi[0],
        zeta[0][0, 0],
        chi[0],
        chi[0],
        optimize=True,
    )
    # transposed in openfermion
    buffer[: chi.size] = chi.T.real.ravel()
    buffer[chi.size :] = zeta[iq].real.ravel()
    np.random.seed(7)
    opt_param = lbfgsb_opt_thc_l2reg(
        eri,
        num_interp_points,
        chkfile_name="thc_opt_gamma.h5",
        maxiter=10,
        initial_guess=buffer,
        penalty_param=None,
    )
    chi_unpacked_mol = opt_param[: chi.size].reshape((num_interp_points, num_mo)).T
    zeta_unpacked_mol = opt_param[chi.size :].reshape(zeta[0].shape)
    opt_param, loss = lbfgsb_opt_kpthc_l2reg(
        chi,
        zeta,
        momentum_map,
        G_mapping,
        jnp.array(Luv_cont),
        chkfile_name="thc_opt.h5",
        maxiter=10,
        penalty_param=None,
    )
    chi_unpacked, zeta_unpacked = unpack_thc_factors(
        opt_param, num_interp_points, num_mo, num_kpts, num_G_per_Q
    )
    assert np.allclose(chi_unpacked[0], chi_unpacked_mol)
    assert np.allclose(zeta_unpacked[0], zeta_unpacked_mol)
    mol_obj = thc_obj_mol(buffer, num_mo, num_interp_points, eri, 1e-3)
    buffer = np.zeros(2 * (chi.size + get_zeta_size(zeta)), dtype=np.float64)
    pack_thc_factors(chi, zeta, buffer)
    gam_obj = thc_objective_regularized(
        buffer, num_mo, num_interp_points, momentum_map, G_mapping, Luv_cont, 1e-3
    )
    assert mol_obj - gam_obj < 1e-12


def test_kpoint_thc_reg_batched():
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
    cell.mesh = [11] * 3
    cell.verbose = 0
    cell.build()

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    num_kpts = len(kpts)
    mf = scf.KRHF(cell, kpts)
    mf.kernel()
    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    num_interp_points = 10 * cell.nao
    kpt_thc = solve_kmeans_kpisdf(
        mf, num_interp_points, single_translation=False, verbose=False
    )
    chi, zeta, G_mapping = kpt_thc.chi, kpt_thc.zeta, kpt_thc.G_mapping
    rsmf = scf.KRHF(mf.cell, mf.kpts).rs_density_fit()
    rsmf.mo_occ = mf.mo_occ
    rsmf.mo_coeff = mf.mo_coeff
    rsmf.mo_energy = mf.mo_energy
    rsmf.with_df.mesh = mf.cell.mesh
    mymp = mp.KMP2(rsmf)
    Luv = cholesky_from_df_ints(mymp)
    Luv_cont = make_contiguous_cholesky(Luv)
    buffer = np.zeros(2 * (chi.size + get_zeta_size(zeta)), dtype=np.float64)
    # Pack THC factors into flat array
    pack_thc_factors(chi, zeta, buffer)
    num_G_per_Q = [z.shape[0] for z in zeta]
    num_mo = mf.mo_coeff[0].shape[-1]
    chi_unpacked, zeta_unpacked = unpack_thc_factors(
        buffer, num_interp_points, num_mo, num_kpts, num_G_per_Q
    )
    # Test packing/unpacking operation
    assert np.allclose(chi_unpacked, chi)
    for iq in range(num_kpts):
        assert np.allclose(zeta[iq], zeta_unpacked[iq])
    # Test objective is the same batched/non-batched
    penalty = 1e-3
    obj_ref = thc_objective_regularized(
        buffer, num_mo, num_interp_points, momentum_map, G_mapping, Luv_cont, penalty
    )
    # # Test gradient is the same
    indx_arrays = prepare_batched_data_indx_arrays(momentum_map, G_mapping)
    batch_size = num_kpts**2
    obj_batched = thc_objective_regularized_batched(
        buffer,
        num_mo,
        num_interp_points,
        momentum_map,
        G_mapping,
        Luv_cont,
        indx_arrays,
        batch_size,
        penalty_param=penalty,
    )
    assert abs(obj_ref - obj_batched) < 1e-12
    grad_ref_fun = jax.grad(thc_objective_regularized)
    grad_ref = grad_ref_fun(
        buffer, num_mo, num_interp_points, momentum_map, G_mapping, Luv_cont, penalty
    )
    # Test gradient is the same
    grad_batched_fun = jax.grad(thc_objective_regularized_batched)
    grad_batched = grad_batched_fun(
        buffer,
        num_mo,
        num_interp_points,
        momentum_map,
        G_mapping,
        Luv_cont,
        indx_arrays,
        batch_size,
        penalty,
    )
    assert np.allclose(grad_batched, grad_ref)
    opt_param, _ = lbfgsb_opt_kpthc_l2reg(
        chi,
        zeta,
        momentum_map,
        G_mapping,
        jnp.array(Luv_cont),
        chkfile_name="thc_opt.h5",
        maxiter=2,
        penalty_param=1e-3,
    )
    opt_param_batched, _ = lbfgsb_opt_kpthc_l2reg_batched(
        chi,
        zeta,
        momentum_map,
        G_mapping,
        jnp.array(Luv_cont),
        chkfile_name="thc_opt.h5",
        maxiter=2,
        penalty_param=1e-3,
    )
    assert np.allclose(opt_param, opt_param_batched)
    batch_size = 7
    opt_param_batched_diff_batch, _ = lbfgsb_opt_kpthc_l2reg_batched(
        chi,
        zeta,
        momentum_map,
        G_mapping,
        jnp.array(Luv_cont),
        batch_size=batch_size,
        chkfile_name="thc_opt.h5",
        maxiter=2,
        penalty_param=1e-3,
    )
    assert np.allclose(opt_param_batched, opt_param_batched_diff_batch)
    ada_param, _ = adagrad_opt_kpthc_batched(
        chi,
        zeta,
        momentum_map,
        G_mapping,
        jnp.array(Luv_cont),
        chkfile_name="thc_opt.h5",
        maxiter=2,
        batch_size=1,
    )
    assert np.allclose(opt_param, opt_param_batched)
    batch_size = 7
    ada_param_diff_batch, _ = adagrad_opt_kpthc_batched(
        chi,
        zeta,
        momentum_map,
        G_mapping,
        jnp.array(Luv_cont),
        batch_size=batch_size,
        chkfile_name="thc_opt.h5",
        maxiter=2,
    )
    assert np.allclose(ada_param, ada_param_diff_batch)


def test_kpoint_thc_helper():
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
    cell.mesh = [11] * 3
    cell.verbose = 0
    cell.build()

    kmesh = [1, 1, 2]
    kpts = cell.make_kpts(kmesh)
    num_kpts = len(kpts)
    mf = scf.KRHF(cell, kpts)
    mf.kernel()
    rsmf = scf.KRHF(mf.cell, mf.kpts).rs_density_fit()
    rsmf.verbose = 0
    rsmf.mo_occ = mf.mo_occ
    rsmf.mo_coeff = mf.mo_coeff
    rsmf.mo_energy = mf.mo_energy
    rsmf.with_df.mesh = mf.cell.mesh
    mymp = mp.KMP2(rsmf)
    Luv = cholesky_from_df_ints(mymp)
    cthc = 5
    num_mo = mf.mo_coeff[0].shape[-1]
    # Just testing function runs
    kpt_thc, loss_isdf = kpoint_thc_via_isdf(
        mf, Luv, cthc * num_mo, perform_adagrad_opt=False, perform_bfgs_opt=False
    )
    kpt_thc_bfgs, loss_bfgs = kpoint_thc_via_isdf(
        mf,
        Luv,
        cthc * num_mo,
        perform_adagrad_opt=False,
        perform_bfgs_opt=True,
        bfgs_maxiter=10,
        initial_guess=kpt_thc,
    )
    kpt_thc_ada, loss_ada = kpoint_thc_via_isdf(
        mf,
        Luv,
        cthc * num_mo,
        perform_adagrad_opt=True,
        perform_bfgs_opt=True,
        bfgs_maxiter=10,
        adagrad_maxiter=10,
        initial_guess=kpt_thc_bfgs,
    )
