from numpy.linalg import solve
import h5py
import numpy as np

from pyscf.pbc import gto, scf, mp

from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
from kpoint_eri.factorizations.isdf import (
    solve_kmeans_kpisdf,
)
from kpoint_eri.resource_estimates.utils import build_momentum_transfer_mapping
from kpoint_eri.factorizations.thc_jax import (
    compute_eri_error,
    pack_thc_factors,
    thc_objective_regularized,
    unpack_thc_factors,
    get_zeta_size,
    lbfgsb_opt_thc_l2reg,
)


def test_kpoint_thc_reg():
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
        rsmf = scf.KRHF(cell, kpts).rs_density_fit()
        mf.with_df._cderi_to_save = mf.chkfile
        rsmf.mo_occ = mf.mo_occ
        rsmf.mo_coeff = mf.mo_coeff
        rsmf.mo_energy = mf.mo_energy
        rsmf.with_df.mesh = cell.mesh
        mymp = mp.KMP2(rsmf)
        Luv = cholesky_from_df_ints(mymp)
        with h5py.File(mf.chkfile, "r+") as fh5:
            for k1 in range(num_kpts):
                for k2 in range(num_kpts):
                    fh5[f"chol_{k1}_{k2}"] = Luv[k1, k2]

    num_mo = mf.mo_coeff[0].shape[-1]
    num_interp_points = 10 * mf.mo_coeff[0].shape[-1]
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
    momentum_map = build_momentum_transfer_mapping(cell, kpts)
    # print(mf.mo_coeff[0].shape)
    # kpt_pqrs = [kpts[0], kpts[0], kpts[0], kpts[0]]
    # mos_pqrs = [
    # mf.mo_coeff[0],
    # mf.mo_coeff[0],
    # mf.mo_coeff[0],
    # mf.mo_coeff[0],
    # ]
    # eri_pqrs = mf.with_df.ao2mo(mos_pqrs, kpt_pqrs, compact=False).reshape(
    # (num_mo,) * 4
    # )
    buffer = np.zeros(2*(chi.size + get_zeta_size(zeta)), dtype=np.float64)
    pack_thc_factors(chi, zeta, buffer)
    num_G_per_Q = [z.shape[0] for z in zeta]
    chi_unpacked, zeta_unpacked = unpack_thc_factors(
        buffer, num_interp_points, num_mo, num_kpts, num_G_per_Q
    )
    assert np.allclose(chi_unpacked, chi)
    for iq in range(num_kpts):
        assert np.allclose(zeta[iq], zeta_unpacked[iq])
    error = compute_eri_error(chi, zeta, momentum_map, G_mapping, Luv, mf)
    res = thc_objective_regularized(
        buffer, num_mo, num_interp_points, momentum_map, G_mapping, Luv, penalty_param=0.0
    )
    assert error == res
    opt_param = lbfgsb_opt_thc_l2reg(
            chi,
            zeta,
            momentum_map,
            G_mapping,
            Luv,
            chkfile_name="thc_opt.h5",
            maxiter=100,
            penalty_param=1e-3,
            )


if __name__ == "__main__":
    test_kpoint_thc_reg()
