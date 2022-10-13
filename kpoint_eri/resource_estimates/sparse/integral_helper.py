import numpy as np

from pyscf.pbc import scf

from kpoint_eri.resource_estimates.sf.ncr_integral_helper import (
    NCRSingleFactorizationHelper,
)


class SparseFactorizationHelper(NCRSingleFactorizationHelper):
    def __init__(
        self,
        cholesky_factor: np.ndarray,
        kmf: scf.HF,
        naux: int = None,
        sparse_threshold: float = 0.0,
    ):
        self.sparse_threshold = sparse_threshold
        super().__init__(cholesky_factor, kmf, naux)

    def get_eri(self, ikpts, check_eq=False):
        """
        Construct (pkp qkq| rkr sks) via \\sum_{n}L_{pkp,qkq,n}L_{sks, rkr, n}^{*}

        Note: 3-tensor L_{sks, rkr} = L_{rkr, sks}^{*}

        :param ikpts: list of four integers representing the index of the kpoint in self.kmf.kpts
        :param check_eq: optional value to confirm a symmetry in the Cholesky vectors.
        """
        ikp, ikq, ikr, iks = ikpts
        n = self.naux
        naux_pq = self.chol[ikp, ikq].shape[0]
        if n > naux_pq:
            print("WARNING: specified naux ({}) is too large!".format(n))
            n = naux_pq
        if check_eq:
            assert np.allclose(
                np.einsum(
                    "npq,nsr->pqrs",
                    self.chol[ikp, ikq][:n],
                    self.chol[iks, ikr][:n].conj(),
                    optimize=True,
                ),
                np.einsum(
                    "npq,nrs->pqrs",
                    self.chol[ikp, ikq][:n],
                    self.chol[ikr, iks][:n],
                    optimize=True,
                ),
            )
        eri_exact = np.einsum(
            "npq,nsr->pqrs",
            self.chol[ikp, ikq][:n],
            self.chol[iks, ikr][:n].conj(),
            optimize=True,
        )
        eri_exact[np.abs(eri_exact) < self.sparse_threshold] = 0.0
        return eri_exact
