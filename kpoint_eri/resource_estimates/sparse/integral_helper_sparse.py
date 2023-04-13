import itertools
import numpy as np
from typing import Union, Tuple

from pyscf.pbc import scf
from pyscf.pbc.lib.kpts_helper import KptsHelper, loop_kkk

from kpoint_eri.resource_estimates.utils.misc_utils import (
    build_momentum_transfer_mapping,
)


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


def _pq_rs_two_body_terms(quad):
    """kp = kq and kr = ks
    thus a subset of the four-fold symmetry can be applied
    (pkp, qkp|rkr, skr) = (qkp,pkp|skr,rkr) by complex conjucation
    """
    p, q, r, s = quad
    yield p, q, r, s
    yield q, p, s, r


def unique_iter_pq_rs(nmo):
    seen = set()
    for quad in itertools.product(range(nmo), repeat=4):
        if quad not in seen:
            seen |= set(_pq_rs_two_body_terms(quad))
            yield tuple(quad)


def _ps_qr_two_body_terms(quad):
    """kp = ks and kq = kr
    Thus a subset of the four-fold symmetry can be applied
    (pkp,qkq|rkq,skp) -> (skp,rkq|qkq,pkp) by complex conj and dummy index exchange"""
    p, q, r, s = quad
    yield p, q, r, s
    yield s, r, q, p


def unique_iter_ps_qr(nmo):
    seen = set()
    for quad in itertools.product(range(nmo), repeat=4):
        if quad not in seen:
            seen |= set(_ps_qr_two_body_terms(quad))
            yield tuple(quad)


def _pr_qs_two_body_terms(quad):
    """kp = kr and kq = ks
    Thus a subset of the four-fold symmetry can be applied
    (pkp,qkq|rkp,skq) -> (rkp,skq|pkp,rkq) by dummy index exchange"""
    p, q, r, s = quad
    yield p, q, r, s
    yield r, s, p, q


def unique_iter_pr_qs(nmo):
    seen = set()
    for quad in itertools.product(range(nmo), repeat=4):
        if quad not in seen:
            seen |= set(_pr_qs_two_body_terms(quad))
            yield tuple(quad)


class SparseFactorizationHelper:
    def __init__(self, cholesky_factor: np.ndarray, kmf: scf.HF, threshold=1.0e-14):
        """
        Initialize a ERI object for CCSD from Cholesky factors and a
        pyscf mean-field object

        :param cholesky_factor: Cholesky factor tensor that is [nkpts, nkpts, naux, nao, nao].
                                To see how to generate this go to
                                kpoint_eri.factorizations.pyscf_chol_form_df.py
        :param kmf: pyscf k-object.  Currently only used to obtain the number of k-points.
                    must have an attribute kpts which len(self.kmf.kpts) returns number of
                    kpts.
        :param threshold: Default 1.0E-8 is the value at which to ignore the integral
        """
        self.chol = cholesky_factor
        self.kmf = kmf
        self.nk = len(self.kmf.kpts)
        self.nao = cholesky_factor[0, 0].shape[-1]
        k_transfer_map = build_momentum_transfer_mapping(self.kmf.cell, self.kmf.kpts)
        self.k_transfer_map = k_transfer_map
        self.threshold = threshold

    def get_total_unique_terms_above_thresh(self, return_nk_counter=False) -> Union[int, Tuple[int, int]]:
        """
        Determine all unique (pkp, qkq|rkr, sks) given momentum conservation and four fold symmetry

        :returns: set of tuples (kp, kq, kr, p, q, r, s). To regenerate the last momentum you can
                  use kts_helper = KptsHelper(self.kmf.cell, self.kmf.kpts); ks = kpts_helper.kconserv[kp, kq, kr]
        """
        kpts_helper = KptsHelper(self.kmf.cell, self.kmf.kpts)
        nkpts = len(self.kmf.kpts)
        completed = np.zeros((nkpts, nkpts, nkpts), dtype=bool)
        counter = 0
        nk_counter = 0
        for kvals in loop_kkk(nkpts):
            kp, kq, kr = kvals
            ks = kpts_helper.kconserv[kp, kq, kr]
            if not completed[kp, kq, kr]:
                nk_counter += 1
                eri_block = self.get_eri([kp, kq, kr, ks])
                if kp == kq == kr == ks:
                    completed[kp, kq, kr] = True
                    n = self.nao
                    assert all(nx == n for nx in eri_block.shape)
                    Dd = np.zeros((n,), dtype=eri_block.dtype)
                    for p in range(n):
                        Dd[p] = eri_block[p, p, p, p]
                        eri_block[p, p, p, p] = 0.0
                    Dp = np.zeros((n, n), dtype=eri_block.dtype)
                    for p, q in itertools.product(range(n), repeat=2):
                        Dp[p, q] = eri_block[p, q, p, q]
                        eri_block[p, q, p, q] = 0.0
                    Dc = np.zeros((n, n), dtype=eri_block.dtype)
                    for p, r in itertools.product(range(n), repeat=2):
                        Dc[p, r] = eri_block[p, p, r, r]
                        eri_block[p, p, r, r] = 0.0
                    Dpc = np.zeros((n, n), dtype=eri_block.dtype)
                    for p, q in itertools.product(range(n), repeat=2):
                        Dpc[p, q] = eri_block[p, q, q, p]
                        eri_block[p, q, q, p] = 0.0
                    counter += np.count_nonzero(Dd)
                    counter += np.count_nonzero(Dp) // 2
                    counter += np.count_nonzero(Dc) // 2
                    counter += np.count_nonzero(Dpc) // 2
                    counter += np.count_nonzero(eri_block) // 4
                    # for ftuple in unique_iter(self.nao):
                    #    p, q, r, s = ftuple
                    #    counter += np.count_nonzero(eri_block[p, q, r, s])
                elif kp == kq and kr == ks:
                    completed[kp, kq, kr] = True
                    completed[kr, ks, kp] = True
                    n = self.nao
                    assert all(nx == n for nx in eri_block.shape)
                    Dc = np.zeros((n, n), dtype=eri_block.dtype)
                    for p, r in itertools.product(range(n), repeat=2):
                        Dc[p, r] = eri_block[p, p, r, r]
                        eri_block[p, p, r, r] = 0.0
                    counter += np.count_nonzero(Dc)
                    counter += np.count_nonzero(eri_block) // 2
                    # for ftuple in unique_iter_ps_qr(self.nao):
                    # for ftuple in unique_iter_pq_rs(self.nao):
                    #    p, q, r, s = ftuple
                    #    counter += np.count_nonzero(eri_block[p, q, r, s])
                elif kp == ks and kq == kr:
                    completed[kp, kq, kr] = True
                    completed[kr, ks, kp] = True
                    n = self.nao
                    assert all(nx == n for nx in eri_block.shape)
                    Dpc = np.zeros((n, n), dtype=eri_block.dtype)
                    for p, q in itertools.product(range(n), repeat=2):
                        Dpc[p, q] = eri_block[p, q, q, p]
                        eri_block[p, q, q, p] = 0.0
                    counter += np.count_nonzero(Dpc)
                    counter += np.count_nonzero(eri_block) // 2
                    # for ftuple in unique_iter_ps_qr(self.nao):
                    #    p, q, r, s = ftuple
                    #    counter += np.count_nonzero(eri_block[p, q, r, s])
                elif kp == kr and kq == ks:
                    completed[kp, kq, kr] = True
                    completed[kq, kp, ks] = True
                    n = self.nao
                    assert all(nx == n for nx in eri_block.shape)
                    Dp = np.zeros((n, n), dtype=eri_block.dtype)
                    for p, q in itertools.product(range(n), repeat=2):
                        Dp[p, q] = eri_block[p, q, p, q]
                        eri_block[p, q, p, q] = 0.0
                    counter += np.count_nonzero(Dp)
                    counter += np.count_nonzero(eri_block) // 2
                    # for ftuple in unique_iter_pr_qs(self.nao):
                    #    p, q, r, s = ftuple
                    #    counter += np.count_nonzero(eri_block[p, q, r, s])
                else:
                    counter += np.count_nonzero(eri_block)
                    completed[kp, kq, kr] = True
                    completed[kr, ks, kp] = True
                    completed[kq, kp, ks] = True
                    completed[ks, kr, kq] = True
        if return_nk_counter:
            return counter, nk_counter
        return counter

    def get_eri(self, ikpts, check_eq=False):
        """
        Construct (pkp qkq| rkr sks) via \\sum_{n}L_{pkp,qkq,n}L_{sks, rkr, n}^{*}

        Note: 3-tensor L_{sks, rkr} = L_{rkr, sks}^{*}

        :param ikpts: list of four integers representing the index of the kpoint in self.kmf.kpts
        :param check_eq: optional value to confirm a symmetry in the Cholesky vectors.
        """
        ikp, ikq, ikr, iks = ikpts
        if check_eq:
            assert np.allclose(
                np.einsum(
                    "npq,nsr->pqrs",
                    self.chol[ikp, ikq],
                    self.chol[iks, ikr].conj(),
                    optimize=True,
                ),
                np.einsum(
                    "npq,nrs->pqrs",
                    self.chol[ikp, ikq],
                    self.chol[ikr, iks],
                    optimize=True,
                ),
            )
        kpoint_eri_tensor = np.einsum(
            "npq,nsr->pqrs",
            self.chol[ikp, ikq],
            self.chol[iks, ikr].conj(),
            optimize=True,
        )
        zero_mask = np.abs(kpoint_eri_tensor) < self.threshold
        kpoint_eri_tensor[zero_mask] = 0
        return kpoint_eri_tensor

    def get_eri_exact(self, kpts):
        """
        Construct (pkp qkq| rkr sks) exactly from mean-field object.  This is for constructing the J and K like terms
        needed for the one-body component lambda

        :param kpts: list of four integers representing the index of the kpoint in self.kmf.kpts
        """
        # kp, kq, kr, ks = kpts
        # eri_kpt = self.kmf.with_df.ao2mo([self.kmf.mo_coeff[i] for i in (kp,kq,kr,ks)],
        #                                 [self.kmf.kpts[i] for i in
        #                                  (kp,kq,kr,ks)],
        #                                  compact=False)
        # shape_pqrs = [self.kmf.mo_coeff[i].shape[-1] for i in (kp,kq,kr,ks)]
        # eri_kpt = eri_kpt.reshape(shape_pqrs)
        # return eri_kpt

        ikp, ikq, ikr, iks = kpts
        return np.einsum(
            "npq,nsr->pqrs",
            self.chol[ikp, ikq],
            self.chol[iks, ikr].conj(),
            optimize=True,
        )
