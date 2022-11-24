from functools import reduce
import numpy as np
from pyscf.pbc import gto, scf, mp, cc

from kpoint_eri.resource_estimates import sparse
from kpoint_eri.resource_estimates import utils
from kpoint_eri.resource_estimates.sparse.ncr_sparse_integral_helper import NCRSSparseFactorizationHelper
from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints

from kpoint_eri.resource_estimates.sparse.compute_lambda_sparse import compute_lambda_ncr


def test_ncr_sparse_int_obj():
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
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    mf.chkfile = 'ncr_test_C_density_fitints.chk'
    mf.with_df._cderi_to_save = 'ncr_test_C_density_fitints_gdf.h5'
    mf.init_guess = 'chkfile'
    mf.kernel()

    mymp = mp.KMP2(mf)
    Luv = cholesky_from_df_ints(mymp)
    for thresh in [1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6]:
        abs_sum_coeffs = 0
        helper = NCRSSparseFactorizationHelper(cholesky_factor=Luv, kmf=mf, threshold=thresh)
        nkpts = len(kpts)
        # recall (k, k-q|k'-q, k')
        for kidx in range(nkpts):
            for kpidx in range(nkpts):
                for qidx in range(nkpts):                 
                    kmq_idx = helper.k_transfer_map[qidx, kidx]
                    kpmq_idx = helper.k_transfer_map[qidx, kpidx]
                    test_eri_block = helper.get_eri([kidx, kmq_idx, kpmq_idx, kpidx])
                    abs_sum_coeffs += np.sum(np.abs(test_eri_block.real)) + np.sum(np.abs(test_eri_block.imag))
        print(thresh, abs_sum_coeffs) # this should always be increasing

def get_num_unique():
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

    kmesh = [1, 3, 4]
    kpts = cell.make_kpts(kmesh)
    nk = len(kpts)
    nmo = cell.nao

    import itertools
    from pyscf.pbc.lib.kpts_helper import KptsHelper, loop_kkk, get_kconserv

    def _symmetric_two_body_terms(quad, complex_valued):
        p, q, r, s = quad
        yield p, q, r, s
        yield q, p, s, r
        yield s, r, q, p
        yield r, s, p, q
        if not complex_valued:
            yield p, s, r, q
            yield q, r, s, p
            yield s, p, q, r
            yield r, q, p, s 

    def unique_iter(nmo):
        seen = set()
        for quad in itertools.product(range(nmo), repeat=4):
            if quad not in seen:
                seen |= set(_symmetric_two_body_terms(quad, True))
                yield tuple(quad)

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
    completed = np.zeros((nkpts,nkpts,nkpts), dtype=bool)
    tally = np.zeros((nkpts,nkpts,nkpts), dtype=int)
    fulltally = np.zeros((nkpts,nkpts,nkpts, nmo, nmo, nmo, nmo), dtype=int)
    for kvals in loop_kkk(nk):
        kp, kq, kr = kvals
        ks = kpts_helper.kconserv[kp, kq, kr]
        if not completed[kp,kq,kr]:
            if kp == kq == kr == ks:
                completed[kp,kq,kr] = True
                tally[kp,kq,kr] += 1 
                for ftuple in unique_iter(nmo):
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
                completed[kp,kq,kr] = True
                completed[kr,ks,kp] = True
                tally[kp,kq,kr] += 1 
                tally[kr,ks,kp] += 1 

                fulltally[kp, kq, kr] += 1
                fulltally[kr, ks, kp] += 1

            elif kp == ks and kq == kr:
                completed[kp,kq,kr] = True
                completed[kr,ks,kp] = True
                tally[kp,kq,kr] += 1 
                tally[kr,ks,kp] += 1 

                fulltally[kp, kq, kr] += 1
                fulltally[kr, ks, kp] += 1


            elif kp == kr and kq == ks:
                completed[kp,kq,kr] = True
                completed[kq,kp,ks] = True
                tally[kp,kq,kr] += 1 
                tally[kq,kp,ks] += 1 
                # symmetry takes account of [kq, kp, ks] only need to do one of the blocks
                fulltally[kp, kq, kr] += 1
                fulltally[kq, kp, ks] += 1

            else:
                completed[kp,kq,kr] = True
                completed[kr,ks,kp] = True
                completed[kq,kp,ks] = True
                completed[ks,kr,kq] = True

                tally[kp,kq,kr] += 1 
                tally[kr,ks,kp] += 1 
                tally[kq,kp,ks] += 1 
                tally[ks,kr,kq] += 1 

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
        if len(set([kp, kq, kr ,ks])) == 4:
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


if __name__ == "__main__":
    get_num_unique()
    # test_ncr_sparse_int_obj()