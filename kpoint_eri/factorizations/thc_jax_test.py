from numpy.linalg import solve
import h5py
import numpy as np
import jax.numpy as jnp
import jax

from pyscf.pbc import gto, scf, mp

from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
from kpoint_eri.factorizations.isdf import (
    solve_kmeans_kpisdf,
)
from kpoint_eri.resource_estimates.utils import (
        build_momentum_transfer_mapping,
        k2gamma
        )
from kpoint_eri.factorizations.thc_jax import (
    pack_thc_factors,
    load_thc_factors,
    thc_objective_regularized,
    thc_objective_regularized_batched,
    unpack_thc_factors,
    get_zeta_size,
    lbfgsb_opt_kpthc_l2reg,
    lbfgsb_opt_kpthc_l2reg_batched,
    prepare_batched_data_indx_arrays,
    kpoint_thc_via_isdf,
)

from openfermion.resource_estimates.thc.utils.thc_factorization import (
    lbfgsb_opt_thc_l2reg,
)
from openfermion.resource_estimates.thc.utils.thc_factorization import (
    thc_objective_regularized as thc_obj_mol,
)


def chkpoint_helper(mf, num_interp_points):
    num_kpts = len(mf.kpts)
    try:
        _, scf_dict = scf.chkfile.load_scf(mf.chkfile)
        mf.mo_coeff = scf_dict["mo_coeff"]
        mf.with_df.max_memory = 1e9
        mf.mo_occ = scf_dict["mo_occ"]
        Luv = np.empty(
            (
                num_kpts,
                num_kpts,
            ),
            dtype=object,
        )
        with h5py.File(mf.chkfile, "r+") as fh5:
            for k1 in range(num_kpts):
                for k2 in range(num_kpts):
                    Luv[k1, k2] = fh5[f"chol_{k1}_{k2}"][:]
    except:
        mf.kernel()
        rsmf = scf.KRHF(mf.cell, mf.kpts).rs_density_fit()
        mf.with_df._cderi_to_save = mf.chkfile
        rsmf.mo_occ = mf.mo_occ
        rsmf.mo_coeff = mf.mo_coeff
        rsmf.mo_energy = mf.mo_energy
        rsmf.with_df.mesh = mf.cell.mesh
        mymp = mp.KMP2(rsmf)
        Luv = cholesky_from_df_ints(mymp)
        with h5py.File(mf.chkfile, "r+") as fh5:
            for k1 in range(num_kpts):
                for k2 in range(num_kpts):
                    fh5[f"chol_{k1}_{k2}"] = Luv[k1, k2]

    num_mo = mf.mo_coeff[0].shape[-1]
    try:
        with h5py.File(mf.chkfile, "r") as fh5:
            chi = fh5["chi"][:]
            xi = fh5["xi"][:]
            G_mapping = fh5["G_mapping"][:]
            zeta = np.zeros((num_kpts,), dtype=object)
            for iq in range(G_mapping.shape[0]):
                zeta[iq] = fh5[f"zeta_{iq}"][:]
    except KeyError:
        chi, zeta, xi, G_mapping = solve_kmeans_kpisdf(
            mf, num_interp_points, single_translation=False
        )
        with h5py.File(mf.chkfile, "r+") as fh5:
            fh5["chi"] = chi
            fh5["xi"] = xi
            fh5["G_mapping"] = G_mapping
            assert G_mapping.shape[0] == zeta.shape[0]
            for iq in range(zeta.shape[0]):
                fh5[f"zeta_{iq}"] = zeta[iq]

    naux = min([Luv[k1, k1].shape[0] for k1 in range(num_kpts)])
    # force contiguous
    Luv_cont = np.zeros(
        (
            num_kpts,
            num_kpts,
            naux,
            num_mo,
            num_mo,
        ),
        dtype=np.complex128,
    )
    for ik1 in range(num_kpts):
        for ik2 in range(num_kpts):
            Luv_cont[ik1, ik2] = Luv[ik1, ik2][:naux]
    return chi, zeta, G_mapping, Luv_cont


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
    cell.verbose = 4
    cell.build()

    kmesh = [1, 1, 1]
    kpts = cell.make_kpts(kmesh)
    num_kpts = len(kpts)
    mf = scf.KRHF(cell, kpts)
    mf.chkfile = "test_thc_kpoint_build.chk"
    mf.init_guess = "chkfile"
    mf.kernel()
    num_mo = mf.mo_coeff[0].shape[-1]
    num_interp_points = 10 * mf.mo_coeff[0].shape[-1]
    chi, zeta, G_mapping, Luv_cont = chkpoint_helper(mf, num_interp_points)
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
        penalty_param=1e-3,
    )
    chi_unpacked_mol = opt_param[: chi.size].reshape((num_interp_points, num_mo)).T
    zeta_unpacked_mol = opt_param[chi.size :].reshape(zeta[0].shape)
    # print("delta mol: ", np.linalg.norm(chi_unpacked_mol_2-chi_unpacked_mol))
    # np.random.seed(7)
    opt_param = lbfgsb_opt_kpthc_l2reg(
        chi,
        zeta,
        momentum_map,
        G_mapping,
        jnp.array(Luv_cont),
        chkfile_name="thc_opt.h5",
        maxiter=10,
        penalty_param=1e-3,
    )
    chi_unpacked, zeta_unpacked = unpack_thc_factors(
        opt_param, num_interp_points, num_mo, num_kpts, num_G_per_Q
    )
    print(chi_unpacked[0] - chi_unpacked_mol)
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
    cell.verbose = 4
    cell.build()

    kmesh = [1, 2, 2]
    kpts = cell.make_kpts(kmesh)
    num_kpts = len(kpts)
    mf = scf.KRHF(cell, kpts)
    mf.chkfile = "test_thc_kpoint_build_batched.chk"
    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    num_interp_points = 10 * cell.nao
    chi, zeta, G_mapping, Luv = chkpoint_helper(mf, num_interp_points)
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
        buffer, num_mo, num_interp_points, momentum_map, G_mapping, Luv, penalty
    )
    # Test gradient is the same
    indx_arrays = prepare_batched_data_indx_arrays(momentum_map, G_mapping,
                                                   num_mo, num_interp_points)
    batch_size = num_kpts ** 2
    obj_batched = thc_objective_regularized_batched(
        buffer, num_mo, num_interp_points, momentum_map, G_mapping, Luv,
        indx_arrays, batch_size, penalty_param=penalty
    )
    assert abs(obj_ref - obj_batched) < 1e-12
    grad_ref_fun = jax.grad(thc_objective_regularized)
    grad_ref = grad_ref_fun(
        buffer, num_mo, num_interp_points, momentum_map, G_mapping, Luv,
        penalty
    )
    # Test gradient is the same
    grad_batched_fun = jax.grad(thc_objective_regularized_batched)
    grad_batched = grad_batched_fun(
        buffer, num_mo, num_interp_points, momentum_map, G_mapping, Luv,
        indx_arrays, batch_size, penalty
    )
    assert np.allclose(grad_batched, grad_ref)
    opt_param = lbfgsb_opt_kpthc_l2reg(
        chi,
        zeta,
        momentum_map,
        G_mapping,
        jnp.array(Luv),
        chkfile_name="thc_opt.h5",
        maxiter=2,
        penalty_param=1e-3,
    )
    opt_param_batched = lbfgsb_opt_kpthc_l2reg_batched(
        chi,
        zeta,
        momentum_map,
        G_mapping,
        jnp.array(Luv),
        chkfile_name="thc_opt.h5",
        maxiter=2,
        penalty_param=1e-3,
    )
    assert np.allclose(opt_param, opt_param_batched)
    batch_size = 7
    opt_param_batched_diff_batch = lbfgsb_opt_kpthc_l2reg_batched(
        chi,
        zeta,
        momentum_map,
        G_mapping,
        jnp.array(Luv),
        batch_size=batch_size,
        chkfile_name="thc_opt.h5",
        maxiter=2,
        penalty_param=1e-3,
    )
    assert np.allclose(opt_param_batched, opt_param_batched_diff_batch)

def test_kpoint_thc_utility_function():
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

    kmesh = [1, 2, 2]
    kpts = cell.make_kpts(kmesh)
    num_kpts = len(kpts)
    mf = scf.KRHF(cell, kpts)
    mf.chkfile = "test_thc_kpoint_build_batched.chk"
    mf.init_guess = "chkfile"
    mf.kernel()
    rsmf = scf.KRHF(mf.cell, mf.kpts).rs_density_fit()
    # Force same MOs as FFTDF at least
    rsmf.mo_occ = mf.mo_occ
    rsmf.mo_coeff = mf.mo_coeff
    rsmf.mo_energy = mf.mo_energy
    rsmf.with_df.mesh = mf.cell.mesh
    mymp = mp.KMP2(rsmf)
    Luv = cholesky_from_df_ints(mymp)
    num_thc = 4 * mf.mo_coeff[0].shape[-1]
    # ISDF only
    chi, zeta = kpoint_thc_via_isdf(mf, Luv, num_thc,
                                    perform_adagrad_opt=False,
                                    perform_bfgs_opt=False,
                                    bfgs_maxiter=100,
                                    )
    chi_load, zeta_load, G_load = load_thc_factors("thc_isdf.h5")
    assert np.allclose(chi, chi_load)
    eri = np.einsum("xpq,xrs->pqrs", Luv[0,0], Luv[0,0])
    buffer = np.zeros((chi.size + get_zeta_size(zeta)), dtype=np.float64)
    buffer[: chi.size] = chi.T.real.ravel()
    buffer[chi.size :] = zeta[0][0,0].real.ravel()
    lbfgsb_opt_thc_l2reg(
        eri,
        num_thc,
        maxiter=10,
        initial_guess=buffer,
        )

def test_kpoint_thc_utility_function_k2gamma():
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

    kmesh = [1, 1, 2]
    kpts = cell.make_kpts(kmesh)
    num_kpts = len(kpts)
    mf = scf.KRHF(cell, kpts)
    mf.chkfile = "test_thc_kpoint_build_batched.chk"
    mf.init_guess = "chkfile"
    mf.kernel()
    # print(mf.cell.lattice_vectors())
    # print(mf.cell.reciprocal_vectors())
    sc_mf = k2gamma(mf)
    # print(mf.cell.lattice_vectors())
    # print(mf.cell.reciprocal_vectors())
    G = sc_mf.cell.reciprocal_vectors()
    R = sc_mf.cell.lattice_vectors()
    print(np.einsum("Gx,Rx->GR", G, R)/(2*np.pi))
    rsmf = scf.KRHF(sc_mf.cell, sc_mf.kpts).rs_density_fit()
    # Force same MOs as FFTDF at least
    rsmf.mo_occ = sc_mf.mo_occ
    rsmf.mo_coeff = sc_mf.mo_coeff
    rsmf.mo_energy = sc_mf.mo_energy
    rsmf.with_df.mesh = sc_mf.cell.mesh
    mymp = mp.KMP2(rsmf)
    Luv = cholesky_from_df_ints(mymp)
    num_thc = 4 * sc_mf.mo_coeff[0].shape[-1]
    # ISDF only
    print(sc_mf.kpts)
    print(len(Luv))
    chi, zeta, G_mapping = kpoint_thc_via_isdf(sc_mf, Luv, num_thc,
                                    perform_adagrad_opt=False,
                                    perform_bfgs_opt=False,
                                    bfgs_maxiter=100,
                                    max_kmeans_iteration=1,
                                    verbose=True
                                    )
    print(len(zeta))
    print(chi.shape, zeta[0].shape, chi.size, zeta[0].size)
    chi_load, zeta_load, G_load = load_thc_factors("thc_isdf.h5")
    assert np.allclose(chi, chi_load)
    eri = np.einsum("xpq,xrs->pqrs", Luv[0,0], Luv[0,0])
    buffer = np.zeros((chi.size + get_zeta_size(zeta)), dtype=np.float64)
    buffer[: chi.size] = chi.T.real.ravel()
    buffer[chi.size :] = zeta[0][0,0].real.ravel()
    lbfgsb_opt_thc_l2reg(
        eri,
        num_thc,
        maxiter=2,
        initial_guess=buffer,
        )
    momentum_map = build_momentum_transfer_mapping(sc_mf.cell, sc_mf.kpts)
    from kpoint_eri.factorizations.thc_jax import make_contiguous_cholesky
    Luv_cont = make_contiguous_cholesky(Luv)
    opt_param = lbfgsb_opt_kpthc_l2reg(
        chi,
        zeta,
        momentum_map,
        G_mapping,
        jnp.array(Luv_cont),
        chkfile_name="thc_opt.h5",
        maxiter=2,
    )
    opt_param_batched = lbfgsb_opt_kpthc_l2reg_batched(
        chi,
        zeta,
        momentum_map,
        G_mapping,
        jnp.array(Luv_cont),
        chkfile_name="thc_opt.h5",
        maxiter=2,
    )

if __name__ == "__main__":
    test_kpoint_thc_reg_gamma()
    test_kpoint_thc_reg_batched()
    test_kpoint_thc_utility_function_k2gamma()
