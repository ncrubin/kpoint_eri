import numpy as np
from pyscf.pbc import gto, scf, mp
from functools import reduce

from kpoint_eri.factorizations.thc_jax import kpoint_thc_via_isdf


from kpoint_eri.resource_estimates.thc.integral_helper import (
    KPTHCHelperDoubleTranslation,
)

from kpoint_eri.resource_estimates.thc.compute_lambda_thc import (
    compute_lambda,
    compute_lambda_real,
)
from kpoint_eri.resource_estimates.utils.misc_utils import build_momentum_transfer_mapping
from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints


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
    mf.kernel()
    #
    # Build kpoint THC eris
    #
    #
    rsmf = scf.KRHF(mf.cell, mf.kpts).rs_density_fit()
    # Force same MOs as FFTDF at least
    rsmf.mo_occ = mf.mo_occ
    rsmf.mo_coeff = mf.mo_coeff
    rsmf.mo_energy = mf.mo_energy
    rsmf.with_df.mesh = mf.cell.mesh
    mymp = mp.KMP2(rsmf)
    Luv = cholesky_from_df_ints(mymp)
    cthc = 4
    num_thc = cthc * mf.mo_coeff[0].shape[-1]
    chi, zeta, mapping, info = kpoint_thc_via_isdf(
        mf,
        Luv,
        num_thc,
        perform_adagrad_opt=False,
        perform_bfgs_opt=True,
        bfgs_maxiter=10,
        verbose=False
    )
    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray(
        [
            reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo))
            for k, mo in enumerate(mf.mo_coeff)
        ]
    )
    helper = KPTHCHelperDoubleTranslation(chi, zeta, mf)
    lambda_tot_kp, lambda_one_body_kp, lambda_two_body_kp = compute_lambda(
        hcore_mo,
        helper,
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
    rsmf = scf.KRHF(supercell_mf.cell, supercell_mf.kpts).rs_density_fit()
    # Force same MOs as FFTDF at least
    rsmf.mo_occ = supercell_mf.mo_occ
    rsmf.mo_coeff = supercell_mf.mo_coeff
    rsmf.mo_energy = supercell_mf.mo_energy
    rsmf.with_df.mesh = supercell_mf.cell.mesh
    mymp = mp.KMP2(rsmf)
    Luv_sc = cholesky_from_df_ints(mymp)
    chi, zeta, mapping, info = kpoint_thc_via_isdf(
        supercell_mf,
        Luv_sc,
        num_thc,
        perform_adagrad_opt=False,
        perform_bfgs_opt=False,
    )
    chi_mol = chi[0].T.real.copy()
    zeta_mol = zeta[0][0, 0].real.copy()
    hcore_ao = supercell_mf.get_hcore()
    hcore_mo = np.asarray(
        [
            reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo))
            for k, mo in enumerate(supercell_mf.mo_coeff)
        ]
    )
    lambda_tot, lambda_one_body, lambda_two_body = compute_lambda_real(
        hcore_mo, chi_mol, zeta_mol, Luv_sc[0, 0].real
    )
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
