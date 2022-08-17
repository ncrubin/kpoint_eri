from typing import Tuple 
import itertools 
import numpy as np
from pyscf.pbc import gto, scf, cc, df, mp, tools
import h5py

from kpoint_eri.resource_estimates.utils.misc_utils import build_momentum_transfer_mapping

def get_df_factor(mat: np.ndarray, thresh: float, verify_adjoint=False) -> Tuple:
    """
    Represent a matrix via non-zero eigenvalue vector pairs.
    anything above thresh is considered non-zero

    :params np.ndarray mat: matrix to diagonalize should be hermitian
    :params float thresh: threshold that indicates a non-zero eigenvalue
    :returns: Tuple eigen values and eigen vectors (lambda, V)
    """
    if verify_adjoint:
        assert np.allclose(mat, mat.conj().T)
    eigs, eigv = np.linalg.eigh(mat)
    normSC = np.sum(np.abs(eigs))
    ix = np.argsort(np.abs(eigs))[::-1]
    eigs = eigs[ix]
    eigv = eigv[:,ix]
    truncation = normSC * np.abs(eigs)
    to_zero = truncation < thresh
    eigs[to_zero] = 0.0
    eigv[:,to_zero] = 0.0
    return eigs, eigv


class DFABKpointIntegrals:
    def __init__(self, cholesky_factor: np.ndarray, kmf: scf.HF):
        """
        Initialize a ERI object for CCSD from Cholesky factors and a
        pyscf mean-field object

        We need to form the A and B objects which are indexed by Cholesky index n and 
        momentum mode Q. This is accomplished by constructing rho[Q, n, kpt, nao, nao] by 
        reshaping the cholesky object.  We don't form the matrix  

        :param cholesky_factor: Cholesky factor tensor that is [nkpts, nkpts, naux, nao, nao]
        :param kmf: pyscf k-object.  Currently only used to obtain the number of k-points.
                    must have an attribute kpts which len(self.kmf.kpts) returns number of 
                    kpts.
        """
        self.chol = cholesky_factor
        self.kmf = kmf 
        self.nk = len(self.kmf.kpts)
        self.nao = self.kmf.cell.nao
        self.naux = self.chol[0, 0].shape[0]
        kpts = self.kmf.kpts
        cell = self.kmf.cell
        k_transfer_map = build_momentum_transfer_mapping(self.kmf.cell, self.kmf.kpts)
        self.k_transfer_map = k_transfer_map

        # set up for later when we construct DF
        self.df_factors = None
        self.a_mats = None
        self.b_mats = None
        
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
    
    def build_chol_from_AB(self, a_by_kq: np.ndarray, b_by_kq: np.ndarray) -> np.ndarray:
        """
        Construct rho_{n, Q} which is equal to the cholesky factor by summing
        together A and -iB.  

        A_{n}(Q) = 0.5 (rho_{n}(Q) + rho_{n}(Q)^)
        B_{n}(Q) = 0.5j(rho_{n}(Q) - rho_{n}(Q)^)
        rho_{n}(Q) = A_{n}(Q) + -1j * B_{n}(Q)

        :param a_by_kq: [kq_idx, naux, nmo * k, nmo * k]
        :param b_by_kq: [kq_idx, naux, nmo * k , nmo * k]
        :returns: cholesky factor [ki, kj, naux, nao, nao]
        """
        rho = np.zeros((self.nk, self.naux, self.nk, self.nao, self.nao), dtype=self.chol[-1, -1].dtype)
        nmo = self.nao
        for qidx in range(self.nk):
            for kidx in range(self.nk):
                k_minus_q_idx = self.k_transfer_map[qidx][kidx]
                for p, q in itertools.product(range(self.nao), repeat=2):
                    P = int(kidx * nmo + p)  #a^_{pK}
                    Q = int(k_minus_q_idx * nmo + q)  #a_{q(K-Q)}
                    rho[kidx, :, k_minus_q_idx, p, q] += a_by_kq[qidx][:, P, Q] + -1.j * b_by_kq[qidx][:, P, Q]
                # I should make this a test.  Checks if rho is equal to chol
                # only would pass assert in the case where no truncation is performed
                # assert np.allclose(rho[kidx, :, k_minus_q_idx, :, :], self.chol[kidx, k_minus_q_idx])
        return rho.transpose(0, 2, 1, 3, 4)


    def double_factorize(self, thresh):
        """
        construct a double factorization of the Hamiltonian check's object if 
        we have already constructed the DF then returns

        :returns dict: Dict with keys U, lambda_U, V, lambda_V
                       where U and V are matrices size nk * nmo x Z
                       lambda_U and lambda_v are size Z vectors.  
        """
        if self.df_factors is not None:
            return self.df_factors

        a_mats = np.empty((self.nk,), dtype=object)
        b_mats = np.empty((self.nk,), dtype=object)
        u_basis = np.empty((self.nk, self.naux), dtype=object)
        u_lambda = np.empty((self.nk, self.naux), dtype=object)
        v_basis = np.empty((self.nk, self.naux), dtype=object)
        v_lambda = np.empty((self.nk, self.naux), dtype=object)

        for iq in range(self.nk):
            A, B = self.build_AB_from_chol(iq)
            a_mats[iq] = np.zeros((self.naux, self.nao * self.nk, self.nao * self.nk), dtype=np.complex128)
            b_mats[iq] = np.zeros((self.naux, self.nao * self.nk, self.nao * self.nk), dtype=np.complex128)
            for nc in range(self.naux):
                # diagonalize A-mat at thresh 
                A_eigs, A_eigv = get_df_factor(A[nc], thresh)
                u_basis[iq, nc] = A_eigv[:, :]
                u_lambda[iq, nc] = A_eigs[:]
                a_mats[iq][nc] = A_eigv @ np.diag(A_eigs) @ A_eigv.conj().T

                # diagonalize B-mat at thresh
                B_eigs, B_eigv = get_df_factor(B[nc], thresh)
                v_basis[iq, nc] = B_eigv[:, :]
                v_lambda[iq, nc] = B_eigs[:]
                b_mats[iq][nc] = B_eigv @ np.diag(B_eigs) @ B_eigv.conj().T

        self.df_factors  = {'U': u_basis, 'lambda_U': u_lambda,
                            'V': v_basis, 'lambda_V': v_lambda}
        self.a_mats = a_mats
        self.b_mats = b_mats
        return self.df_factors

    def get_eri(self, ikpts, a_mats=None, b_mats=None):
        """
        Construct (pkp qkq| rkr sks) via A and B tensors that have already been constructed

        :param ikpts: list of four integers representing the index of the kpoint in self.kmf.kpts
        """
        if (self.a_mats is None or self.b_mats is None) and (a_mats is None and b_mats is None):
            raise ValueError("No DF factorization has occured yet. Rerun by calling inst.double_factorize()")
        
        if a_mats is None:
            a_mats = self.a_mats
        if b_mats is None:
            b_mats = self.b_mats 

        ikp, ikq, ikr, iks = ikpts

        # build Cholesky vector from truncated A and B
        Luv = self.build_chol_from_AB(a_mats, b_mats)  
        return np.einsum('npq,nsr->pqrs', Luv[ikp, ikq], Luv[iks, ikr].conj(), optimize=True)
