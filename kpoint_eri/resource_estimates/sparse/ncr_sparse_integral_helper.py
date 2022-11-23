import itertools
import numpy as np

from pyscf.pbc import scf

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

    def get_total_terms_above_thresh():
        """
        compute the symm
        """
        pass

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
