from functools import reduce
import numpy as np
from pyscf.pbc import gto, scf, mp

from kpoint_eri.resource_estimates.sparse.integral_helper_sparse import (
    SparseFactorizationHelper,
)
from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints


def test_sparse_int_obj():
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
    for thresh in [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]:
        abs_sum_coeffs = 0
        helper = SparseFactorizationHelper(
            cholesky_factor=Luv, kmf=mf, threshold=thresh
        )
        nkpts = len(kpts)
        # recall (k, k-q|k'-q, k')
        for kidx in range(nkpts):
            for kpidx in range(nkpts):
                for qidx in range(nkpts):
                    kmq_idx = helper.k_transfer_map[qidx, kidx]
                    kpmq_idx = helper.k_transfer_map[qidx, kpidx]
                    test_eri_block = helper.get_eri([kidx, kmq_idx, kpmq_idx, kpidx])
                    abs_sum_coeffs += np.sum(np.abs(test_eri_block.real)) + np.sum(
                        np.abs(test_eri_block.imag)
                    )
        print(thresh, abs_sum_coeffs)  # this should always be increasing


def test_get_num_unique():
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

    kmesh = [1, 2, 3]
    kpts = cell.make_kpts(kmesh)
    nk = len(kpts)
    nmo = cell.nao

    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.with_df.mesh = cell.mesh
    mf.kernel()

    mymp = mp.KMP2(mf)
    Luv = cholesky_from_df_ints(mymp)
    helper = SparseFactorizationHelper(cholesky_factor=Luv, kmf=mf)

    import itertools
    from pyscf.pbc.lib.kpts_helper import KptsHelper, loop_kkk

    # import the iteration routines
    from kpoint_eri.resource_estimates.sparse.integral_helper_sparse import (
        unique_iter,
        unique_iter_pr_qs,
        unique_iter_ps_qr,
        unique_iter_pq_rs,
    )

    tally4 = np.zeros((nmo, nmo, nmo, nmo), dtype=int)
    for ft in unique_iter(nmo):
        p, q, r, s = ft
        if p == q == r == s:
            tally4[p, q, r, s] += 1
        elif p == r and q == s:
            tally4[p, q, r, s] += 1
            tally4[q, p, s, r] += 1
        elif p == s and q == r:
            tally4[p, q, r, s] += 1
            tally4[q, p, s, r] += 1
        elif p == q and r == s:
            tally4[p, q, r, s] += 1
            tally4[r, s, p, q] += 1
        else:
            tally4[p, q, r, s] += 1
            tally4[q, p, s, r] += 1
            tally4[s, r, q, p] += 1
            tally4[r, s, p, q] += 1
    assert np.allclose(tally4, 1)

    kpts_helper = KptsHelper(cell, kpts)
    nkpts = len(kpts)
    completed = np.zeros((nkpts, nkpts, nkpts), dtype=bool)
    tally = np.zeros((nkpts, nkpts, nkpts), dtype=int)
    fulltally = np.zeros((nkpts, nkpts, nkpts, nmo, nmo, nmo, nmo), dtype=int)
    nk_count = 0
    for kvals in loop_kkk(nk):
        kp, kq, kr = kvals
        ks = kpts_helper.kconserv[kp, kq, kr]
        if not completed[kp, kq, kr]:
            eri_block = helper.get_eri([kp, kq, kr, ks])
            nk_count += 1
            if kp == kq == kr == ks:
                completed[kp, kq, kr] = True
                tally[kp, kq, kr] += 1
                for ftuple in unique_iter(
                    nmo
                ):  # iterate over unique whenever all momentum indices are the same
                    p, q, r, s = ftuple
                    if p == q == r == s:
                        fulltally[kp, kq, kr, p, q, r, s] += 1
                    elif p == r and q == s:
                        fulltally[kp, kq, kr, p, q, r, s] += 1
                        fulltally[kp, kq, kr, q, p, s, r] += 1
                    elif p == s and q == r:
                        fulltally[kp, kq, kr, p, q, r, s] += 1
                        fulltally[kp, kq, kr, q, p, s, r] += 1
                    elif p == q and r == s:
                        fulltally[kp, kq, kr, p, q, r, s] += 1
                        fulltally[kp, kq, kr, r, s, p, q] += 1
                    else:
                        fulltally[kp, kq, kr, p, q, r, s] += 1
                        fulltally[kp, kq, kr, q, p, s, r] += 1
                        fulltally[kp, kq, kr, s, r, q, p] += 1
                        fulltally[kp, kq, kr, r, s, p, q] += 1

            elif kp == kq and kr == ks:
                completed[kp, kq, kr] = True
                completed[kr, ks, kp] = True
                tally[kp, kq, kr] += 1
                tally[kr, ks, kp] += 1

                # the full (kp kq | kr ks) gets mapped to (kr, ks | kp  kq)
                # fulltally[kp, kq, kr] += 1
                # fulltally[kr, ks, kp] += 1

                # we only need to count the single eri block because (kp, kq kr ks) -> (kr, ks, kp kq) 1:1.
                # but for (p q | r s) -> (qp|sr) so we are overcounting. by a little.
                # (q p | s r)
                # since we have a complex conjugation symmetry in both we really have (npair, npair) here.
                # we can use pyscf.ao2mo to do this.

                test_block = np.zeros_like(eri_block, dtype=int)
                num_terms_in_block = 0
                for p, q, r, s in unique_iter_pq_rs(nmo):
                    num_terms_in_block += 1
                    test_block[p, q, r, s] += 1
                    fulltally[kp, kq, kr, p, q, r, s] += 1
                    fulltally[kr, ks, kp, r, s, p, q] += 1
                    if p == q and r == s:
                        continue
                    else:
                        test_block[q, p, s, r] += 1
                        fulltally[kp, kq, kr, q, p, s, r] += 1
                        fulltally[kr, ks, kp, s, r, q, p] += 1

                for p, q, r, s in itertools.product(range(helper.nao), repeat=4):
                    if not np.isclose(test_block[p, q, r, s], 1):
                        print(p, q, r, s, test_block[p, q, r, s])
                assert np.allclose(test_block, 1)

                assert num_terms_in_block <= helper.nao**4

            elif kp == ks and kq == kr:
                completed[kp, kq, kr] = True
                completed[kr, ks, kp] = True
                tally[kp, kq, kr] += 1
                tally[kr, ks, kp] += 1

                # fulltally[kp, kq, kr] += 1
                # fulltally[kr, ks, kp] += 1

                test_block = np.zeros_like(eri_block, dtype=int)
                num_terms_in_block = 0
                for p, q, r, s in unique_iter_ps_qr(nmo):
                    num_terms_in_block += 1
                    test_block[p, q, r, s] += 1
                    fulltally[kp, kq, kr, p, q, r, s] += 1
                    fulltally[kr, ks, kp, r, s, p, q] += 1
                    if p == s and q == r:
                        continue
                    else:
                        test_block[s, r, q, p] += 1
                        fulltally[kp, kq, kr, s, r, q, p] += 1
                        fulltally[kr, ks, kp, q, p, s, r] += 1

                for p, q, r, s in itertools.product(range(helper.nao), repeat=4):
                    if not np.isclose(test_block[p, q, r, s], 1):
                        print(p, q, r, s, test_block[p, q, r, s])
                assert np.allclose(test_block, 1)

                assert num_terms_in_block <= helper.nao**4

            elif kp == kr and kq == ks:
                completed[kp, kq, kr] = True
                completed[kq, kp, ks] = True
                tally[kp, kq, kr] += 1
                tally[kq, kp, ks] += 1
                # symmetry takes account of [kq, kp, ks] only need to do one of the blocks
                # fulltally[kp, kq, kr] += 1
                # fulltally[kq, kp, ks] += 1

                test_block = np.zeros_like(eri_block, dtype=int)
                num_terms_in_block = 0
                for p, q, r, s in unique_iter_pr_qs(nmo):
                    num_terms_in_block += 1
                    test_block[p, q, r, s] += 1
                    fulltally[kp, kq, kr, p, q, r, s] += 1
                    fulltally[kq, kp, ks, q, p, s, r] += 1
                    if p == r and q == s:
                        continue
                    else:
                        test_block[r, s, p, q] += 1
                        fulltally[kp, kq, kr, r, s, p, q] += 1
                        fulltally[kq, kp, ks, s, r, q, p] += 1

                for p, q, r, s in itertools.product(range(helper.nao), repeat=4):
                    if not np.isclose(test_block[p, q, r, s], 1):
                        print(p, q, r, s, test_block[p, q, r, s])
                assert np.allclose(test_block, 1)

                assert num_terms_in_block <= helper.nao**4

            else:
                completed[kp, kq, kr] = True
                completed[kr, ks, kp] = True
                completed[kq, kp, ks] = True
                completed[ks, kr, kq] = True

                tally[kp, kq, kr] += 1
                tally[kr, ks, kp] += 1
                tally[kq, kp, ks] += 1
                tally[ks, kr, kq] += 1

                # just assign entire 4-tensor +1 value because each pqrs is unique because
                # kp, kq, kr, ks is unique for this case we would only need to grab one of
                # these blocks of 4.
                fulltally[kp, kq, kr] += 1
                fulltally[kq, kp, ks] += 1
                fulltally[ks, kr, kq] += 1
                fulltally[kr, ks, kp] += 1

    assert np.allclose(completed, True)
    assert np.allclose(tally, 1)

    for kvals in loop_kkk(nk):
        kp, kq, kr = kvals
        ks = kpts_helper.kconserv[kp, kq, kr]
        if len(set([kp, kq, kr, ks])) == 4:
            # print(kp, kq, kr, np.allclose(fulltally[kp, kq, kr], 1))
            assert np.allclose(fulltally[kp, kq, kr], 1)
        elif kp == kr and kq == ks:
            assert np.allclose(fulltally[kp, kq, kr], 1)
            assert np.allclose(fulltally[kq, kp, ks], 1)
        elif kp == ks and kq == kr:
            assert np.allclose(fulltally[kp, kq, kr], 1)
            assert np.allclose(fulltally[kr, ks, kp], 1)
        elif kp == kq and kr == ks:
            assert np.allclose(fulltally[kp, kq, kr], 1)
            assert np.allclose(fulltally[kr, ks, kp], 1)

        assert np.allclose(fulltally[kp, kp, kp], 1)

    print("PASSED TEST")


if __name__ == "__main__":
    test_get_num_unique()
    test_sparse_int_obj()
