from functools import reduce
import numpy as np
import pytest

from ase.build import bulk

from pyscf.pbc import gto, scf, mp
from pyscf.pbc.tools import pyscf_ase

from kpoint_eri.resource_estimates.sf.compute_lambda_sf import compute_lambda
from kpoint_eri.resource_estimates.sf.integral_helper_sf import SingleFactorizationHelper


@pytest.mark.slow
def test_lambda_calc():
    cell = gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-hf-rev'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 0
    cell.build()

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.with_df.mesh = mf.cell.mesh
    mf.kernel()

    from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
    mymp = mp.KMP2(mf)
    Luv = cholesky_from_df_ints(mymp)
    helper = SingleFactorizationHelper(cholesky_factor=Luv, kmf=mf)

    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mf.mo_coeff)])

    lambda_tot, lambda_one_body, lambda_two_body = compute_lambda(hcore_mo, helper)

    from pyscf.pbc.tools.k2gamma import k2gamma
    from kpoint_eri.resource_estimates.utils.k2gamma import k2gamma
    supercell_mf = k2gamma(mf, make_real=False)
    supercell_mf.e_tot = supercell_mf.energy_tot()
    # supercell_mf.kernel()
    assert np.isclose(mf.e_tot, supercell_mf.e_tot / np.prod(kmesh))

    supercell_mymp = mp.KMP2(supercell_mf)
    supercell_Luv = cholesky_from_df_ints(supercell_mymp)
    supercell_helper = SingleFactorizationHelper(cholesky_factor=supercell_Luv, kmf=supercell_mf)

    supercell_hcore_ao = supercell_mf.get_hcore()
    supercell_hcore_mo = np.asarray([reduce(np.dot, (mo.T.conj(), supercell_hcore_ao[k], mo)) for k, mo in enumerate(supercell_mf.mo_coeff)])

    sc_lambda_tot, sc_lambda_one_body, sc_lambda_two_body = compute_lambda(supercell_hcore_mo, supercell_helper)

    # check l2 norm of one-body
    assert np.isclose(np.linalg.norm(supercell_hcore_mo), np.linalg.norm(np.array(hcore_mo).ravel()))

    # this part needs to change 
    lambda_two_body = 0
    for qidx in range(len(kpts)):
        # A and B are W
        A, B = helper.build_AB_from_chol(qidx) # [naux, nao * nk, nao * nk]
        A /= np.sqrt(nkpts)
        B /= np.sqrt(nkpts)
        # sum_q sum_n (sum_{pq} |Re{A_{pq}^n}| + |Im{A_{pq}^n|)^2
        # lambda_two_body += np.sum(np.einsum('npq->n', np.abs(A.real) + np.abs(A.imag))**2)
        # lambda_two_body += np.sum(np.einsum('npq->n', np.abs(B.real) + np.abs(B.imag))**2)
        lambda_two_body += np.sum(np.einsum('npq->n', np.abs(A)**2))
        lambda_two_body += np.sum(np.einsum('npq->n', np.abs(B)**2))

    lambda_two_body *= 0.5

    # this part needs to change 
    lambda_two_body_v2 = 0
    for qidx in range(1):
        # A and B are W
        A, B = supercell_helper.build_AB_from_chol(qidx) # [naux, nao * nk, nao * nk]
        A /= np.sqrt(1)
        B /= np.sqrt(1)
        # sum_q sum_n (sum_{pq} |Re{A_{pq}^n}| + |Im{A_{pq}^n|)^2
        # lambda_two_body += np.sum(np.einsum('npq->n', np.abs(A.real) + np.abs(A.imag))**2)
        # lambda_two_body += np.sum(np.einsum('npq->n', np.abs(B.real) + np.abs(B.imag))**2)
        lambda_two_body_v2 += np.sum(np.einsum('npq->n', np.abs(A)**2))
        lambda_two_body_v2 += np.sum(np.einsum('npq->n', np.abs(B)**2))

    lambda_two_body_v2 *= 0.5
    assert np.isclose(lambda_two_body_v2, lambda_two_body)

@pytest.mark.slow
def test_padding():
    ase_atom = bulk("H", "bcc", a=2.0, cubic=True)
    cell = gto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell[:].copy()
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf-rev"
    cell.verbose = 0
    cell.build()

    kmesh = [1, 2, 2]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.with_df.mesh = cell.mesh
    mf.kernel()

    from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
    from pyscf.pbc.mp.kmp2 import _add_padding

    mymp = mp.KMP2(mf)
    Luv_padded = cholesky_from_df_ints(mymp)
    mo_coeff_padded = _add_padding(mymp, mymp.mo_coeff, mymp.mo_energy)[0]
    helper = SingleFactorizationHelper(cholesky_factor=Luv_padded, kmf=mf)
    assert mf.mo_coeff[0].shape[-1] != mo_coeff_padded[0].shape[-1]

    hcore_ao = mf.get_hcore()
    hcore_no_padding= np.asarray([reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo)) for k, mo in enumerate(mf.mo_coeff)])
    hcore_padded = np.asarray([reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo))
                               for k, mo in enumerate(mo_coeff_padded)])
    assert hcore_no_padding[0].shape != hcore_padded[0].shape
    assert np.isclose(np.sum(hcore_no_padding), np.sum(hcore_padded))
    Luv_no_padding = cholesky_from_df_ints(mymp, pad_mos_with_zeros=False)
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            assert np.isclose(np.sum(Luv_padded[k1, k2]),
                              np.sum(Luv_no_padding[k1, k2]))

    helper_no_padding = SingleFactorizationHelper(cholesky_factor=Luv_no_padding, kmf=mf)
    lambda_tot_pad, lambda_one_body_pad, lambda_two_body_pad = compute_lambda(hcore_no_padding, helper_no_padding)
    helper = SingleFactorizationHelper(cholesky_factor=Luv_padded, kmf=mf)
    lambda_tot_no_pad, lambda_one_body_no_pad, lambda_two_body_no_pad = compute_lambda(hcore_padded, helper)
    assert np.isclose(lambda_tot_no_pad, lambda_tot_pad)


if __name__ == "__main__":
    test_lambda_calc()
    test_padding()
