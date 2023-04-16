from functools import reduce
import numpy as np
from pyscf.pbc import gto, scf, mp
from pyscf.pbc.lib.kpts_helper import loop_kkk, get_kconserv
import pytest

from kpoint_eri.resource_estimates.sparse.integral_helper_sparse import (
    SparseFactorizationHelper,
)
from kpoint_eri.factorizations.hamiltonian_utils import cholesky_from_df_ints

from kpoint_eri.resource_estimates.sparse.compute_lambda_sparse import compute_lambda


@pytest.mark.slow
def test_lambda_sparse():
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

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.with_df.mesh = cell.mesh
    mf.kernel()

    mymp = mp.KMP2(mf)
    Luv = cholesky_from_df_ints(mymp)
    helper = SparseFactorizationHelper(cholesky_factor=Luv, kmf=mf)

    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray(
        [
            reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo))
            for k, mo in enumerate(mf.mo_coeff)
        ]
    )

    lambda_data = compute_lambda(hcore_mo, helper)

    from kpoint_eri.resource_estimates.utils.k2gamma import k2gamma, get_phase

    supercell_mf = k2gamma(mf, make_real=False)
    dm0 = supercell_mf.make_rdm1()
    # supercell_mf.kernel(dm0)
    energy_tot_sc = supercell_mf.energy_tot()
    assert np.isclose(mf.e_tot, energy_tot_sc / np.prod(kmesh))

    supercell_mymp = mp.KMP2(supercell_mf)
    supercell_Luv = cholesky_from_df_ints(supercell_mymp)
    supercell_helper = SparseFactorizationHelper(
        cholesky_factor=supercell_Luv, kmf=supercell_mf
    )

    # Sanity check to see that k2gamma supercell AO hcore is same as literal
    # supercell hcore
    _, C = get_phase(cell, kpts)
    supercell_hcore_ao = supercell_mf.get_hcore()
    nkpts = np.prod(kmesh)
    nmo = mf.mo_coeff[0].shape[-1]
    kp_sc_hcore_ao = np.einsum("Rk,kij,Sk->RiSj", C, hcore_ao, C.conj()).reshape(
        (nmo * nkpts, nmo * nkpts)
    )
    assert np.isclose(np.abs(np.max(supercell_hcore_ao.imag)), 0.0)
    assert np.isclose(np.abs(np.max(kp_sc_hcore_ao.imag)), 0.0)
    assert np.allclose(supercell_hcore_ao, kp_sc_hcore_ao)
    supercell_hcore_mo = np.asarray(
        [
            reduce(np.dot, (mo.T.conj(), supercell_hcore_ao[k], mo))
            for k, mo in enumerate(supercell_mf.mo_coeff)
        ]
    )

    sc_lambda_data = compute_lambda(supercell_hcore_mo, supercell_helper)
    # Sanity check Tpq by itself without any eri contribution
    norm_uc_T = sum(np.sum(np.abs(hk.real) + np.abs(hk.imag)) for hk in hcore_mo)
    norm_sc_T = sum(
        np.sum(np.abs(hk.real) + np.abs(hk.imag)) for hk in supercell_hcore_mo
    )
    supcell = supercell_mf.cell
    assert np.isclose(norm_uc_T, norm_sc_T)

    # Sanity check: Build Nk^3 eris directly and sum up outside of loop
    eris_uc = np.zeros((nkpts,) * 3 + (nmo,) * 4, dtype=np.complex128)
    kconserv = get_kconserv(cell, kpts)
    for k1, k2, k3 in loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        kpts = [k1, k2, k3, k4]
        eris_uc[k1, k2, k3] = helper.get_eri(kpts)

    eris_uc = eris_uc / nkpts
    norm_uc = np.linalg.norm(
        eris_uc.ravel()
    )  # this computs sqrt(sum_i abs(i)) which is not what we want to do.
    eris_sc = supercell_helper.get_eri([0, 0, 0, 0])
    norm_sc = np.linalg.norm(eris_sc.ravel())
    # Test 2-norm should be invariant wrt unitary.
    assert np.isclose(norm_uc, norm_sc)
    # Test the triple loop
    direct_uc = np.sum(np.abs(eris_uc.real) + np.abs(eris_uc.imag))
    assert np.isclose(direct_uc, lambda_data.lambda_two_body)

    assert np.isclose(sc_lambda_data.lambda_one_body, lambda_data.lambda_one_body)
    assert np.isclose(sc_lambda_data.lambda_two_body, lambda_data.lambda_two_body)


if __name__ == "__main__":
    test_lambda_sparse()
