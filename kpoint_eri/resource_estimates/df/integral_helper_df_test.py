from functools import reduce
import itertools

import numpy as np

from pyscf.pbc import gto, scf, mp

from kpoint_eri.resource_estimates.df.integral_helper_df import DFABKpointIntegrals
from kpoint_eri.factorizations.hamiltonian_utils import cholesky_from_df_ints


def test_df_amat_bmat():
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
    cell.build(parse_arg=False)

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.with_df.mesh = cell.mesh
    mf.kernel()

    mymp = mp.KMP2(mf)
    nmo = mymp.nmo

    Luv = cholesky_from_df_ints(mymp)  # [kpt, kpt, naux, nao, nao]
    dfk_inst = DFABKpointIntegrals(Luv.copy(), mf)
    naux = dfk_inst.naux

    dfk_inst.double_factorize()

    import openfermion as of

    for qidx, kidx in itertools.product(range(nkpts), repeat=2):
        Amats, Bmats = dfk_inst.build_A_B_n_q_k_from_chol(qidx, kidx)
        # check if Amats and Bmats have the correct size
        assert Amats.shape == (naux, 2 * nmo, 2 * nmo)
        assert Bmats.shape == (naux, 2 * nmo, 2 * nmo)

        # check if Amats and Bmats have the correct symmetry--Hermitian
        assert np.allclose(Amats, Amats.conj().transpose(0, 2, 1))
        assert np.allclose(Bmats, Bmats.conj().transpose(0, 2, 1))

        # check if we can recover the Cholesky vector from Amat
        k_minus_q_idx = dfk_inst.k_transfer_map[qidx, kidx]
        test_chol = dfk_inst.build_chol_part_from_A_B(kidx, qidx, Amats, Bmats)
        assert np.allclose(test_chol, dfk_inst.chol[kidx, k_minus_q_idx])

        # check if factorized is working numerically exact case
        assert np.allclose(dfk_inst.amat_n_mats[kidx, qidx], Amats)
        assert np.allclose(dfk_inst.bmat_n_mats[kidx, qidx], Bmats)

        for nn in range(Amats.shape[0]):
            w, v = np.linalg.eigh(Amats[nn, :, :])
            non_zero_idx = np.where(w > 1.0e-4)[0]
            w = w[non_zero_idx]
            v = v[:, non_zero_idx]
            assert len(w) <= 2 * nmo

    for qidx in range(nkpts):
        for nn in range(naux):
            for kidx in range(nkpts):
                eigs_a_fixed_n_q = dfk_inst.amat_lambda_vecs[kidx, qidx, nn]
                eigs_b_fixed_n_q = dfk_inst.bmat_lambda_vecs[kidx, qidx, nn]
                assert len(eigs_a_fixed_n_q) <= 2 * nmo
                assert len(eigs_b_fixed_n_q) <= 2 * nmo

    for kidx in range(nkpts):
        for kpidx in range(nkpts):
            for qidx in range(nkpts):
                kmq_idx = dfk_inst.k_transfer_map[qidx, kidx]
                kpmq_idx = dfk_inst.k_transfer_map[qidx, kpidx]
                exact_eri_block = dfk_inst.get_eri_exact(
                    [kidx, kmq_idx, kpmq_idx, kpidx]
                )
                test_eri_block = dfk_inst.get_eri([kidx, kmq_idx, kpmq_idx, kpidx])
                assert np.allclose(exact_eri_block, test_eri_block)


def test_supercell_df_amat_bmat():
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
    cell.build(parse_arg=False)

    kmesh = [1, 1, 1]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.with_df.mesh = cell.mesh
    mf.kernel()

    mymp = mp.KMP2(mf)
    Luv = cholesky_from_df_ints(mymp)
    helper = DFABKpointIntegrals(cholesky_factor=Luv, kmf=mf)
    helper.double_factorize(thresh=1.0e-13)

    from pyscf.pbc.tools.k2gamma import k2gamma
    from kpoint_eri.resource_estimates.utils.k2gamma import k2gamma

    supercell_mf = k2gamma(mf, make_real=False)
    # supercell_mf.kernel()
    supercell_mf.e_tot = supercell_mf.energy_tot()
    assert np.isclose(mf.e_tot, supercell_mf.e_tot / np.prod(kmesh))

    supercell_mymp = mp.KMP2(supercell_mf)
    supercell_Luv = cholesky_from_df_ints(supercell_mymp)
    supercell_helper = DFABKpointIntegrals(
        cholesky_factor=supercell_Luv, kmf=supercell_mf
    )
    supercell_helper.double_factorize(thresh=1.0e-13)
    sc_help = supercell_helper

    sc_nk = supercell_helper.nk
    for kidx in range(sc_nk):
        for kpidx in range(sc_nk):
            for qidx in range(sc_nk):
                kmq_idx = supercell_helper.k_transfer_map[qidx, kidx]
                kpmq_idx = supercell_helper.k_transfer_map[qidx, kpidx]
                exact_eri_block = supercell_helper.get_eri_exact(
                    [kidx, kmq_idx, kpmq_idx, kpidx]
                )
                test_eri_block = supercell_helper.get_eri(
                    [kidx, kmq_idx, kpmq_idx, kpidx]
                )
                assert np.allclose(exact_eri_block, test_eri_block)
                print(np.allclose(exact_eri_block, test_eri_block))

    from kpoint_eri.resource_estimates.df.integral_helper_df import get_df_factor

    for qidx, kidx in itertools.product(range(sc_help.nk), repeat=2):
        Amats, Bmats = sc_help.build_A_B_n_q_k_from_chol(qidx, kidx)
        for nc in range(sc_help.naux):
            amat_n_eigs, amat_n_eigv = get_df_factor(Amats[nc], 1.0e-13)
            bmat_n_eigs, bmat_n_eigv = get_df_factor(Bmats[nc], 1.0e-13)
            print(sc_help.amat_lambda_vecs[kidx, qidx, nc])
            print(sc_help.bmat_lambda_vecs[kidx, qidx, nc])


if __name__ == "__main__":
    test_supercell_df_amat_bmat()
    test_df_amat_bmat()
