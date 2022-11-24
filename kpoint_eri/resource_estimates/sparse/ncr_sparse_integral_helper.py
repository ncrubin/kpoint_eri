import itertools
import numpy as np

from pyscf.pbc import scf
from pyscf.pbc.lib.kpts_helper import KptsHelper, loop_kkk, get_kconserv

from kpoint_eri.resource_estimates.utils.misc_utils import build_momentum_transfer_mapping



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

    def get_total_symm_unique_termsh(self,):
        """
        Determine all unique (pkp, qkq|rkr, sks) given momentum conservation and four fold symmetry

        :returns: set of tuples (kp, kq, kr, p, q, r, s). To regenerate the last momentum you can
                  use kts_helper = KptsHelper(self.kmf.cell, self.kmf.kpts); ks = kpts_helper.kconserv[kp, kq, kr]
        """
        kpts_helper = KptsHelper(self.kmf.cell, self.kmf.kpts)
        nkpts = len(self.kmf.kpts)
        fulltally = np.zeros((nkpts,nkpts,nkpts, self.nao, self.nao, self.nao, self.nao), dtype=int)
        seven_tuple_set = set() 
        for kvals in loop_kkk(self.nk):
            kp, kq, kr = kvals
            ks = kpts_helper.kconserv[kp, kq, kr]
            for p, q, r, s in itertools.product(range(self.nao), repeat=4):
                seven_tuple_set |= {(kp, kq, kr, p, q, r, s),
                                     (kq, kp, ks, q, p, s, r),
                                     (kr, ks, kp, r, s, p, q),
                                     (ks, kr, kq, s, r, q, p)}
        return seven_tuple_set


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
        zero_mask = kpoint_eri_tensor < self.threshold
        kpoint_eri_tensor[zero_mask] = 0
        return kpoint_eri_tensor
