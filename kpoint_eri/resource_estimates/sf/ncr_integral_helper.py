import itertools
import numpy as np

from pyscf.pbc import scf

from kpoint_eri.resource_estimates.utils.misc_utils import build_momentum_transfer_mapping

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
        if naux is None:
            naux = cholesky_factor[0, 0].shape[0]
        self.naux = naux
        self.nao = cholesky_factor[0, 0].shape[-1]
        k_transfer_map = build_momentum_transfer_mapping(self.kmf.cell, self.kmf.kpts)
        self.k_transfer_map = k_transfer_map


    def build_AB_from_chol(self, qidx):
        """
        Construct A and B matrices given Q-kpt idx.  This constructs
        all matrices association with n-chol

        :param qidx: index for momentum mode Q.  
        :returns: Tuple(np.ndarray, np.ndarray) where each array is 
                  [naux, nmo * kpts, nmk * kpts] set of matrices.
        """
        nmo = self.nao # number of gaussians used
        naux = self.naux

        # now form A and B
        # First set up matrices to store. We will loop over Q index and construct
        # entire set of matrices index by n-aux.
        rho = np.zeros((naux, nmo * self.nk, nmo * self.nk), dtype=np.complex128)

        for kidx in range(self.nk):
            k_minus_q_idx = self.k_transfer_map[qidx, kidx]
            for p, q in itertools.product(range(nmo), repeat=2):
                P = int(kidx * nmo + p)  #a^_{pK}
                Q = int(k_minus_q_idx * nmo + q)  #a_{q(K-Q)}
                rho[:,P,Q] += self.chol[kidx, k_minus_q_idx][:, p, q]  # L_{pK, q(K-Q)}a^_{pK}a_{q(K-Q)}
            
        A = 0.5  * (rho + rho.transpose((0,2,1)).conj())
        B = 0.5j * (rho - rho.transpose((0,2,1)).conj())

        assert np.allclose(rho, A + -1j * B)  # This can be removed later
        return A, B

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
        print(n, naux_pq)
        if n > naux_pq:
            print("WARNING: specified naux ({}) is too large!".format(n))
            n = naux_pq
        if check_eq:
            assert np.allclose(np.einsum('npq,nsr->pqrs', self.chol[ikp, ikq][:n], self.chol[iks, ikr][:n].conj(), optimize=True),
                               np.einsum('npq,nrs->pqrs', self.chol[ikp, ikq][:n], self.chol[ikr, iks][:n], optimize=True))
        return np.einsum('npq,nsr->pqrs', self.chol[ikp, ikq][:n], self.chol[iks, ikr][:n].conj(), optimize=True)

