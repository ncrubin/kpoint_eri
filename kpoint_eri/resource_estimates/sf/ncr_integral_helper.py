import numpy as np

from pyscf.pbc import scf

# Single-Factorization
class NCRSingleFactorizationHelper:
    def __init__(self, cholesky_factor: np.ndarray, kmf: scf.HF, naux: int = None):
        """
        Initialize a ERI object for CCSD from Cholesky factors and a
        pyscf mean-field object

        :param cholesky_factor: Cholesky factor tensor that is [nkpts, nkpts, naux, nao, nao].
                                To see how to generate this go to
                                kpoint_eri.factorizations.pyscf_chol_form_df.py
        :param kmf: pyscf k-object.  Currently only used to obtain the number of k-points.
                    must have an attribute kpts which len(self.kmf.kpts) returns number of
                    kpts.
        """
        self.chol = cholesky_factor
        self.kmf = kmf
        self.nk = len(self.kmf.kpts)
        self.naux = naux

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
            assert np.allclose(np.einsum('npq,nsr->pqrs', self.chol[ikp, ikq][:n], self.chol[iks, ikr][:n].conj(), optimize=True),
                               np.einsum('npq,nrs->pqrs', self.chol[ikp, ikq][:n], self.chol[ikr, iks][:n], optimize=True))
        return np.einsum('npq,nsr->pqrs', self.chol[ikp, ikq][:n], self.chol[iks, ikr][:n].conj(), optimize=True)

