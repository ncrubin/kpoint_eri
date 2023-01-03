import itertools
import numpy as np

from pyscf.pbc import scf
from pyscf.pbc.lib.kpts_helper import KptsHelper, loop_kkk, get_kconserv

from kpoint_eri.resource_estimates.utils.misc_utils import build_momentum_transfer_mapping

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



class NCRSSparseFactorizationHelper:
    def __init__(self, cholesky_factor: np.ndarray, kmf: scf.HF, threshold=1.0E-14):
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

    def get_total_unique_terms_above_thresh(self,):
        """
        Determine all unique (pkp, qkq|rkr, sks) given momentum conservation and four fold symmetry

        :returns: set of tuples (kp, kq, kr, p, q, r, s). To regenerate the last momentum you can
                  use kts_helper = KptsHelper(self.kmf.cell, self.kmf.kpts); ks = kpts_helper.kconserv[kp, kq, kr]
        """
        kpts_helper = KptsHelper(self.kmf.cell, self.kmf.kpts)
        nkpts = len(self.kmf.kpts)
        completed = np.zeros((nkpts,nkpts,nkpts), dtype=bool)
        counter = 0
        for kvals in loop_kkk(nkpts):
            kp, kq, kr = kvals
            ks = kpts_helper.kconserv[kp, kq, kr]
            if not completed[kp,kq,kr]:
                eri_block = self.get_eri([kp, kq, kr, ks])
                # zero_mask = np.abs(eri_block) < self.threshold
                # eri_block[zero_mask] = 0
                if kp == kq == kr == ks:
                    completed[kp,kq,kr] = True
                    for ftuple in unique_iter(self.nao):
                        p, q, r, s = ftuple
                        if p == q == r == s:
                            counter += np.count_nonzero(eri_block[p, q, r, s])
                        elif p == r and q == s:
                            counter += np.count_nonzero(eri_block[p, q, r, s])
                        elif p == s and q == r:
                            counter += np.count_nonzero(eri_block[p, q, r, s])
                        elif p == q and r == s:
                            counter += np.count_nonzero(eri_block[p, q, r, s])
                        else:
                            counter += np.count_nonzero(eri_block[p, q, r, s])

                elif kp == kq and kr == ks:
                    completed[kp,kq,kr] = True
                    completed[kr,ks,kp] = True
                    counter += np.count_nonzero(eri_block)

                elif kp == ks and kq == kr:
                    completed[kp,kq,kr] = True
                    completed[kr,ks,kp] = True
                    counter += np.count_nonzero(eri_block)

                elif kp == kr and kq == ks:
                    completed[kp,kq,kr] = True
                    completed[kq,kp,ks] = True
                    counter += np.count_nonzero(eri_block)

            else:
                counter += np.count_nonzero(eri_block)
                completed[kp,kq,kr] = True
                completed[kr,ks,kp] = True
                completed[kq,kp,ks] = True
                completed[ks,kr,kq] = True
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
            assert np.allclose(np.einsum('npq,nsr->pqrs', self.chol[ikp, ikq], self.chol[iks, ikr].conj(), optimize=True),
                               np.einsum('npq,nrs->pqrs', self.chol[ikp, ikq], self.chol[ikr, iks], optimize=True))
        kpoint_eri_tensor = np.einsum('npq,nsr->pqrs', self.chol[ikp, ikq], self.chol[iks, ikr].conj(), optimize=True)
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
        return np.einsum('npq,nsr->pqrs', self.chol[ikp, ikq], self.chol[iks, ikr].conj(), optimize=True)

