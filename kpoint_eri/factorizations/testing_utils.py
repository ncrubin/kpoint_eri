import h5py
import numpy as np

from pyscf.pbc import gto, scf, mp

from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
from kpoint_eri.factorizations.isdf import (
    solve_kmeans_kpisdf,
)

def thc_test_chkpoint_helper(mf, num_interp_points):
    """Some boilerplate to make running THC tests simpler."""
    num_kpts = len(mf.kpts)
    try:
        _, scf_dict = scf.chkfile.load_scf(mf.chkfile)
        mf.mo_coeff = scf_dict["mo_coeff"]
        mf.with_df.max_memory = 1e9
        mf.mo_occ = scf_dict["mo_occ"]
        mf.mo_energy = scf_dict["mo_energy"]
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


def thc_test_chkpoint_helper_supercell(mf, num_interp_points):
    """Some boilerplate to make running THC tests simpler."""
    num_kpts = len(mf.kpts)
    try:
        _, scf_dict = scf.chkfile.load_scf(mf.chkfile)
        mf.mo_coeff = scf_dict["mo_coeff"]
        mf.with_df.max_memory = 1e9
        mf.mo_occ = scf_dict["mo_occ"]
        mf.mo_energy = scf_dict["mo_energy"]
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
