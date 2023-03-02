import itertools
import numpy as np
from typing import Tuple

from pyscf.pbc import scf

from kpoint_eri.resource_estimates.utils.misc_utils import (
    build_momentum_transfer_mapping,
)


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
    eigv = eigv[:, ix]
    truncation = normSC * np.abs(eigs)
    to_zero = truncation < thresh
    eigs[to_zero] = 0.0
    eigv[:, to_zero] = 0.0
    idx_not_zero = np.where(~to_zero == True)[0]
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
        naux = 0
        for i, j in itertools.product(range(self.nk), repeat=2):
            naux = max(self.chol[i, j].shape[0], naux)
        self.naux = naux
        self.nao = cholesky_factor[0, 0].shape[-1]
        k_transfer_map = build_momentum_transfer_mapping(self.kmf.cell, self.kmf.kpts)
        self.k_transfer_map = k_transfer_map
        self.reverse_k_transfer_map = np.zeros_like(
            self.k_transfer_map
        )  # [kidx, kmq_idx] = qidx
        for kidx in range(self.nk):
            for qidx in range(self.nk):
                kmq_idx = self.k_transfer_map[qidx, kidx]
                self.reverse_k_transfer_map[kidx, kmq_idx] = qidx

        # set up for later when we construct DF
        self.df_factors = None
        self.a_mats = None
        self.b_mats = None

    def build_A_B_n_q_k_from_chol(self, qidx, kidx):
        """
        Builds matrices that are block in two momentum indices

              k  | k-Q |
            ------------
        k   |    |     |
        ----------------
        k-Q |    |     |
        ----------------

        where the off diagonal blocks are the ones that are populated.  All
        matrices for every Cholesky vector is constructed.

        :param qidx: index for momentum mode Q.
        :param kidx: index for momentum mode K.
        """
        k_minus_q_idx = self.k_transfer_map[qidx, kidx]
        naux = self.chol[kidx, k_minus_q_idx].shape[0]
        nmo = self.nao
        Amat = np.zeros((naux, 2 * nmo, 2 * nmo), dtype=np.complex128)
        Bmat = np.zeros((naux, 2 * nmo, 2 * nmo), dtype=np.complex128)
        if k_minus_q_idx == kidx:
            Amat[:, :nmo, :nmo] = self.chol[
                kidx, k_minus_q_idx
            ]  # beacuse L_{pK, qK,n}= L_{qK,pK,n}^{*}
            Bmat[:, :nmo, :nmo] = 0.5j * (
                self.chol[kidx, k_minus_q_idx]
                - self.chol[kidx, k_minus_q_idx].conj().transpose(0, 2, 1)
            )
        else:
            Amat[:, :nmo, nmo:] = (
                0.5 * self.chol[kidx, k_minus_q_idx]
            )  # [naux, nmo, nmo]
            Amat[:, nmo:, :nmo] = 0.5 * self.chol[kidx, k_minus_q_idx].conj().transpose(
                0, 2, 1
            )

            Bmat[:, :nmo, nmo:] = (
                0.5j * self.chol[kidx, k_minus_q_idx]
            )  # [naux, nmo, nmo]
            Bmat[:, nmo:, :nmo] = -0.5j * self.chol[
                kidx, k_minus_q_idx
            ].conj().transpose(0, 2, 1)

        return Amat, Bmat

    def build_chol_part_from_A_B(
        self,
        kidx: int,
        qidx: int,
        Amats: np.ndarray,
        Bmats: np.ndarray,
    ) -> np.ndarray:
        """
        Construct rho_{n, k, Q} which is equal to the cholesky factor by summing
        together via the following relationships

        :param kidx: k-momentum index
        :param qidx: Q-momentum index
        :param Amats: [naux, 2 * nmo, 2 * nmo]
        :param Bmats: [naux, 2 * nmo, 2 * nmo]
        :returns: cholesky factors 3-tensors (k, k-Q)[naux, nao, nao], (kp, kp-Q)[naux, nao, nao]
        """
        k_minus_q_idx = self.k_transfer_map[qidx, kidx]
        nmo = self.nao
        if k_minus_q_idx == kidx:
            return Amats[:, :nmo, :nmo]
        else:
            return Amats[:, :nmo, nmo:] + -1j * Bmats[:, :nmo, nmo:]

    def double_factorize(self, thresh=None) -> None:
        """
        construct a double factorization of the Hamiltonian.

        Iterate through qidx, kidx and get factorized Amat and Bmat for each Cholesky rank

        :returns: None. we mutate the object and store listed objects above
        """
        if thresh is None:
            thresh = 1.0e-13
        if self.df_factors is not None:
            return self.df_factors

        nkpts = self.nk
        nmo = self.nao
        naux = self.naux
        self.amat_n_mats = np.zeros((nkpts, nkpts, naux, 2 * nmo, 2 * nmo), dtype=np.complex128)
        self.bmat_n_mats = np.zeros((nkpts, nkpts, naux, 2 * nmo, 2 * nmo), dtype=np.complex128)
        self.amat_lambda_vecs = np.empty((nkpts, nkpts, naux), dtype=object)
        self.bmat_lambda_vecs = np.empty((nkpts, nkpts, naux), dtype=object)
        for qidx, kidx in itertools.product(range(nkpts), repeat=2):
            Amats, Bmats = self.build_A_B_n_q_k_from_chol(qidx, kidx)
            naux_qk = Amats.shape[0]
            assert naux_qk <= naux
            for nc in range(naux_qk):
                amat_n_eigs, amat_n_eigv = get_df_factor(Amats[nc], thresh)
                self.amat_n_mats[kidx, qidx][nc, :, :] = amat_n_eigv @ np.diag(amat_n_eigs) @ amat_n_eigv.conj().T
                self.amat_lambda_vecs[kidx, qidx, nc] = amat_n_eigs

                bmat_n_eigs, bmat_n_eigv = get_df_factor(Bmats[nc], thresh)
                self.bmat_n_mats[kidx, qidx][nc, :, :] = bmat_n_eigv @ np.diag(bmat_n_eigs) @ bmat_n_eigv.conj().T
                self.bmat_lambda_vecs[kidx, qidx, nc] = bmat_n_eigs

        return

    def get_eri(self, ikpts):
        """
        Construct (pkp qkq| rkr sks) via A and B tensors that have already been constructed

        :param ikpts: list of four integers representing the index of the kpoint in self.kmf.kpts
        """
        ikp, ikq, ikr, iks = ikpts  # (k, k-q, k'-q, k')
        qidx = self.reverse_k_transfer_map[ikp, ikq]
        test_qidx = self.reverse_k_transfer_map[iks, ikr]
        assert test_qidx == qidx

        # build Cholesky vector from truncated A and B
        chol_val_k_kmq = self.build_chol_part_from_A_B(
            ikp, qidx, self.amat_n_mats[ikp, qidx], self.bmat_n_mats[ikp, qidx]
        )
        chol_val_kp_kpmq = self.build_chol_part_from_A_B(
            iks, qidx, self.amat_n_mats[iks, qidx], self.bmat_n_mats[iks, qidx]
        )

        # return np.einsum('npq,nsr->pqrs', Luv[ikp, ikq], Luv[iks, ikr].conj(), optimize=True)
        return np.einsum(
            "npq,nsr->pqrs", chol_val_k_kmq, chol_val_kp_kpmq.conj(), optimize=True
        )

    def get_eri_exact(self, ikpts):
        """
        Construct (pkp qkq| rkr sks) exactly from Cholesky vector.  This is for constructing the J and K like terms
        needed for the one-body component lambda

        :param ikpts: list of four integers representing the index of the kpoint in self.kmf.kpts
        """
        ikp, ikq, ikr, iks = ikpts
        return np.einsum(
            "npq,nsr->pqrs",
            self.chol[ikp, ikq],
            self.chol[iks, ikr].conj(),
            optimize=True,
        )