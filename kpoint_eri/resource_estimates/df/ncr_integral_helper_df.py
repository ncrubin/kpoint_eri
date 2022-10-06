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
    idx_not_zero = np.where(~to_zero==True)[0]
    eigs = eigs[idx_not_zero]
    eigv = eigv[:, idx_not_zero]
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
        self.naux = self.chol[0, 0].shape[0]
        self.nao = cholesky_factor[0, 0].shape[-1]
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
        rho = np.zeros((self.nk, self.naux, self.nk, self.nao, self.nao), dtype=a_by_kq[0].dtype)
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


    def double_factorize(self, thresh=None):
        """
        construct a double factorization of the Hamiltonian check's object if 
        we have already constructed the DF then returns

        :returns dict: Dict with keys U, lambda_U, V, lambda_V
                       where U and V are matrices size nk * nmo x Z
                       lambda_U and lambda_v are size Z vectors.  
        """
        if thresh is None:
            thresh = 1.0E-13
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
    
    def get_eri_exact(self, ikpts):
        """
        Construct (pkp qkq| rkr sks) exactly from Cholesky vector.  This is for constructing the J and K like terms
        needed for the one-body component lambda

        :param ikpts: list of four integers representing the index of the kpoint in self.kmf.kpts
        """
        ikp, ikq, ikr, iks = ikpts
        return np.einsum('npq,nsr->pqrs', self.chol[ikp, ikq], self.chol[iks, ikr].conj(), optimize=True)

class DFAlphaBetaKpointIntegrals:
    def __init__(self, cholesky_factor: np.ndarray, kmf: scf.HF):
        """
        Initialize a ERI object for CCSD from Cholesky factors and a
        pyscf mean-field object

        We need to form the alpha and beta objects which are indexed by Cholesky index n, 
        momentum modes, k, k', Q, and an even/odd index. This is accomplished by constructing 
        rho[Q, k, n, nao, nao] by 
        reshaping the cholesky object.  We don't form the matrix  

        :param cholesky_factor: Cholesky factor tensor that is [nkpts, nkpts, naux, nao, nao]
        :param kmf: pyscf k-object.  Currently only used to obtain the number of k-points.
                    must have an attribute kpts which len(self.kmf.kpts) returns number of 
                    kpts.
        """
        self.chol = cholesky_factor
        self.kmf = kmf 
        self.nk = len(self.kmf.kpts)
        self.nao = cholesky_factor[0, 0].shape[-1]
        self.naux = self.chol[0, 0].shape[0]
        kpts = self.kmf.kpts
        cell = self.kmf.cell
        k_transfer_map = build_momentum_transfer_mapping(self.kmf.cell, self.kmf.kpts)
        self.k_transfer_map = k_transfer_map # [qidx, kidx] = (k - q)idx
        self.reverse_k_transfer_map = np.zeros_like(self.k_transfer_map)  # [kidx, kmq_idx] = qidx
        for kidx in range(self.nk):
            for qidx in range(self.nk):
                kmq_idx = self.k_transfer_map[qidx, kidx]
                self.reverse_k_transfer_map[kidx, kmq_idx] = qidx

        # set up for later when we construct DF
        self.df_factors = None
        self.a_mats = None
        self.b_mats = None

        # slice setup
        nmo =self.nao
        self.k_slice = slice(0, nmo, 1)
        self.kmq_slice = slice(1 * nmo, (1 + 1) * nmo, 1)
        self.kp_slice = slice(2 * nmo, (2 + 1) * nmo, 1)
        self.kpmq_slice = slice(3 * nmo, (3 + 1) * nmo, 1)


        
    def build_alpha_beta_from_chol(self, kidx, kpidx, qidx):
        """
        Construct alpha and beta matrices given three momentum mode indices

        Every alpha and beta are defined over a one-particle Hilbert space of size 4 * nmo
        
        the 4 is from the fixed momentum modes (k, k-Q, k', k'-Q) below is a diagram of how we 
        store the matrix

              k  | k-Q | k' | k' - Q
            ------------------------
        k   |    |     |    |      |
        ----------------------------
        k-Q |    |     |    |      | 
        ----------------------------
        k'  |    |     |    |      |
        ----------------------------
        k'-Q|    |     |    |      |
        ----------------------------

        a particular slice can be accessed by 
        k-slice = [0 * nmo: (0 + 1) * nmo:1]
        k-Q slice = [1 * nmo:(1 + 1) * nmo:1]
        k'-slice =  [2 * nmo:(2 + 1) * nmo:1]
        k'-Q-slice =  [3 * nmo:(3 + 1) * nmo:1]

        :param kidx: k-momentum index
        :param kpidx: k'-momentum index
        :param qidx: index for momentum mode Q.  
        :returns: Tuple(np.ndarray, np.ndarray) where each array is 
                  [naux, 4 * nmo, 4 * nmo] set of matrices.
        """
        nmo = self.nao # number of gaussians used
        naux = self.naux

        # now form A and B
        # First set up matrices to store. We will loop over Q index and construct
        # entire set of matrices index by n-aux.
        rho_k = np.zeros((naux, 4 * nmo, 4 * nmo), dtype=np.complex128)
        rho_kp = np.zeros((naux, 4 * nmo, 4 * nmo), dtype=np.complex128)

        k_minus_q_idx = self.k_transfer_map[qidx, kidx]
        kp_minus_q_idx = self.k_transfer_map[qidx, kpidx]

        rho_k[:, self.k_slice, self.kmq_slice] = self.chol[kidx, k_minus_q_idx][:, :, :]
        rho_kp[:, self.kp_slice, self.kpmq_slice] = self.chol[kpidx, kp_minus_q_idx][:, :, :]
            
        A = 0.5  * (rho_k + rho_kp.transpose((0,2,1)).conj())
        Adag = 0.5 * (rho_k.transpose((0,2,1)).conj() + rho_kp)
        B = 0.5j * (rho_k - rho_kp.transpose((0,2,1)).conj())
        Bdag = 0.5j * (rho_kp - rho_k.transpose((0,2,1)).conj())

        assert np.allclose(rho_k, A + -1j * B)          
        assert np.allclose(rho_kp, Adag + -1j * Bdag)

        alpha_p = 0.5 * (A + Adag)
        alpha_m = 0.5 * (A - Adag)
        beta_p = 0.5 * (B + Bdag)
        beta_m = 0.5 * (B - Bdag)
        return alpha_p, alpha_m, beta_p, beta_m
    
    def build_chol_part_from_alpha_beta(self, 
                                        kidx: int, 
                                        kpidx: int, 
                                        qidx: int,
                                        alpha_p: np.ndarray, 
                                        alpha_m: np.ndarray,
                                        beta_p: np.ndarray,
                                        beta_m: np.ndarray) -> np.ndarray:
        """
        Construct rho_{n, k, Q} which is equal to the cholesky factor by summing
        together via the following relationships

        A_{n, k, k', Q} = alpha_p + alpha_m
        A_{n, k, k', Q}^dag = alpha_p - alpha_m
        B_{n, k, k', Q} = beta_p + beta_m
        B_{n, k, k', Q}^dag = beta_p - beta_m

        and 

        rho_k = A + 1j * B
        rho_kp = Adag + -1j * Bdag

        :param kidx: k-momentum index
        :param kpidx: k'-momentum index
        :param qidx: Q-momentum index
        :param alpha_p: [naux, 4 * nmo, 4 * nmo]
        :param alpha_m: [naux, 4 * nmo, 4 * nmo]
        :param beta_p: [naux, 4 * nmo, 4 * nmo]
        :param beta_m: [naux, 4 * nmo, 4 * nmo]
        :returns: cholesky factors 3-tensors (k, k-Q)[naux, nao, nao], (kp, kp-Q)[naux, nao, nao]
        """
        A = alpha_p + alpha_m
        Adag = alpha_p - alpha_m
        B = beta_p + beta_m
        Bdag = beta_p - beta_m
        rho_k = A + -1j * B
        rho_kp = Adag + -1j * Bdag
        return rho_k[:, self.k_slice, self.kmq_slice], rho_kp[:, self.kp_slice, self.kpmq_slice]


    def double_factorize(self, thresh=None) -> None:
        """
        construct a double factorization of the Hamiltonian

        We store each matrix that is 2N x 2N as objects in the following objects

        alpha_p_mats = np.empty((self.nk, self.nk, self.nk, self.naux), dtype=object)
        alpha_m_mats = np.empty((self.nk, self.nk, self.nk, self.naux), dtype=object)
        beta_p_mats = np.empty((self.nk, self.nk, self.nk, self.naux), dtype=object)
        beta_m_mats = np.empty((self.nk, self.nk, self.nk, self.naux), dtype=object)

        where the input is [k-idx, k'-idx, q-idx, naux] and output is the 2N x 2N matrix

        :returns: None. we mutate the object and store listed objects above 
        """
        if thresh is None:
            thresh = 1.0E-13
        if self.df_factors is not None:
            return self.df_factors

        alpha_p_mats = np.zeros((self.nk, self.nk, self.nk, self.naux, 4 * self.nao, 4 * self.nao), dtype=np.complex128)
        alpha_m_mats = np.zeros((self.nk, self.nk, self.nk, self.naux, 4 * self.nao, 4 * self.nao), dtype=np.complex128)
        beta_p_mats = np.zeros((self.nk, self.nk, self.nk, self.naux, 4 * self.nao, 4 * self.nao), dtype=np.complex128)
        beta_m_mats = np.zeros((self.nk, self.nk, self.nk, self.naux, 4 * self.nao, 4 * self.nao), dtype=np.complex128)

        kconserv = tools.get_kconserv(self.kmf.cell, self.kmf.kpts)
        nkpts = self.nk
        # recall (k, k-q|k'-q, k')
        for kidx in range(nkpts):
            for kpidx in range(nkpts):
                for qidx in range(nkpts):                 
                    kmq_idx = self.k_transfer_map[qidx, kidx]
                    kpmq_idx = self.k_transfer_map[qidx, kpidx]
                    alpha_p, alpha_m, beta_p, beta_m = \
                       self.build_alpha_beta_from_chol(kidx, kpidx, qidx)
                    for nc in range(self.naux):
                        alphap_eigs, alphap_eigv = get_df_factor(alpha_p[nc], thresh)
                        alpha_p_mats[kidx, kpidx, qidx][nc, :, :] = alphap_eigv @ np.diag(alphap_eigs) @ alphap_eigv.conj().T
                        assert np.allclose(alpha_m[nc].conj().T, -alpha_m[nc])
                        alpham_eigs, alpham_eigv = get_df_factor(1j * alpha_m[nc], thresh)
                        alpha_m_mats[kidx, kpidx, qidx][nc, :, :] = alpham_eigv @ np.diag(-1j * alpham_eigs) @ alpham_eigv.conj().T

                        betap_eigs, betap_eigv = get_df_factor(beta_p[nc], thresh)
                        beta_p_mats[kidx, kpidx, qidx][nc, :, :] = betap_eigv @ np.diag(betap_eigs) @ betap_eigv.conj().T
                        assert np.allclose(beta_m[nc].conj().T, -beta_m[nc])
                        betam_eigs, betam_eigv = get_df_factor(1j * beta_m[nc], thresh)
                        beta_m_mats[kidx, kpidx, qidx][nc, :, :] = betam_eigv @ np.diag(-1j * betam_eigs) @ betam_eigv.conj().T

        self.alpha_p_mats = alpha_p_mats 
        self.alpha_m_mats = alpha_m_mats        
        self.beta_p_mats = beta_p_mats         
        self.beta_m_mats = beta_m_mats
        return

    def get_eri(self, ikpts):
        """
        Construct (pkp qkq| rkr sks) via A and B tensors that have already been constructed

        :param ikpts: list of four integers representing the index of the kpoint in self.kmf.kpts
        """
        ikp, ikq, ikr, iks = ikpts # (k, k-q, k'-q, k')
        qidx = self.reverse_k_transfer_map[ikp, ikq]
        test_qidx = self.reverse_k_transfer_map[iks, ikr]
        assert test_qidx == qidx

        # build Cholesky vector from truncated A and B
        chol_val_k_kmq, chol_val_kp_kpmq = self.build_chol_part_from_alpha_beta(ikp, 
                                                             iks,  
                                                             qidx, 
                                                             self.alpha_p_mats[ikp, iks, qidx],
                                                             self.alpha_m_mats[ikp, iks, qidx],
                                                             self.beta_p_mats[ikp, iks, qidx],
                                                             self.beta_m_mats[ikp, iks, qidx]
                                                             )

        # return np.einsum('npq,nsr->pqrs', Luv[ikp, ikq], Luv[iks, ikr].conj(), optimize=True)
        return np.einsum('npq,nsr->pqrs', chol_val_k_kmq, chol_val_kp_kpmq.conj(), optimize=True)

    
    def get_eri_exact(self, ikpts):
        """
        Construct (pkp qkq| rkr sks) exactly from Cholesky vector.  This is for constructing the J and K like terms
        needed for the one-body component lambda

        :param ikpts: list of four integers representing the index of the kpoint in self.kmf.kpts
        """
        ikp, ikq, ikr, iks = ikpts
        return np.einsum('npq,nsr->pqrs', self.chol[ikp, ikq], self.chol[iks, ikr].conj(), optimize=True)
