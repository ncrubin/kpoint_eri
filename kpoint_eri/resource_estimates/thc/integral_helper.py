import numpy as np

from pyscf.pbc import scf
from pyscf.pbc.lib.kpts_helper import unique, member

from kpoint_eri.resource_estimates.utils.misc_utils import (
    build_momentum_transfer_mapping,
)

from kpoint_eri.factorizations.isdf import (
    build_G_vector_mappings_double_translation,
    build_G_vector_mappings_single_translation,
    build_eri_isdf_double_translation,
    build_eri_isdf_single_translation,
)


class KPTHCHelperDoubleTranslation(object):
    def __init__(
        self,
        chi: np.ndarray,
        zeta: np.ndarray,
        kmf: scf.HF,
        cholesky_factor: np.ndarray,
        nthc: int = None,
    ):
        """
        Initialize a ERI object for CCSD from KP-THC factors and a
        pyscf mean-field object

        :param chi: array of interpolating orbitals of shape [num_kpts, num_mo, num_interp_points]
        :param zeta: central tensor of dimension [num_kpts, num_G, num_G, num_interp_points, num_interp_points].
        :param kmf: pyscf k-object.  Currently only used to obtain the number of k-points.
                    must have an attribute kpts which len(self.kmf.kpts) returns number of
                    kpts.
        :param cholesky_factor: Cholesky object for computing exact integrals

        """
        self.chol = cholesky_factor
        self.naux = self.chol[0, 0].shape[0]
        self.chi = chi
        self.zeta = zeta
        self.kmf = kmf
        self.nk = len(self.kmf.kpts)
        if nthc is None:
            nthc = self.chi.shape[-1]
            assert nthc == self.zeta[0].shape[-1]
        self.nthc = nthc
        self.kpts = self.kmf.kpts
        self.k_transfer_map = build_momentum_transfer_mapping(
            self.kmf.cell, self.kmf.kpts
        )
        self.reverse_k_transfer_map = np.zeros_like(
            self.k_transfer_map
        )  # [kidx, kmq_idx] = qidx
        for kidx in range(self.nk):
            for qidx in range(self.nk):
                kmq_idx = self.k_transfer_map[qidx, kidx]
                self.reverse_k_transfer_map[kidx, kmq_idx] = qidx
        # Two-translation ISDF zeta[iq, dG, dG']
        _, _, G_map_unique, delta_Gs = build_G_vector_mappings_double_translation(
            self.kmf.cell, self.kmf.kpts, self.k_transfer_map
        )
        self.G_mapping = G_map_unique

    def get_eri(self, ikpts):
        """
        Construct (pkp qkq| rkr sks) via \\sum_{mu nu} zeta[iq, dG, dG', mu, nu]
            chi[kp,p,mu]* chi[kq,q,mu] chi[kp,p,nu]* chi[ks,s,nu]

        :param ikpts: list of four integers representing the index of the kpoint in self.kmf.kpts
        :param check_eq: optional value to confirm a symmetry in the Cholesky vectors.
        """
        ikp, ikq, ikr, iks = ikpts
        q_indx = self.reverse_k_transfer_map[ikp, ikq]
        return build_eri_isdf_double_translation(
            self.chi, self.zeta, q_indx, ikpts, self.G_mapping
        )

    def get_eri_exact(self, ikpts):
        """
        Construct (pkp qkq| rkr sks) exactly from Cholesky vector.  This is for constructing the J and K like terms
        needed for the one-body component lambda

        :param ikpts: list of four integers representing the index of the kpoint in self.kmf.kpts
        """
        ikp, ikq, ikr, iks = ikpts
        return np.einsum('npq,nsr->pqrs', self.chol[ikp, ikq], self.chol[iks, ikr].conj(), optimize=True)


class KPTHCHelperSingleTranslation(KPTHCHelperDoubleTranslation):
    def __init__(
        self,
        chi: np.ndarray,
        zeta: np.ndarray,
        kmf: scf.HF,
        cholesky_factor: np.ndarray,
        nthc: int = None,
    ):
        """
        Initialize a ERI object for CCSD from KP-THC factors and a
        pyscf mean-field object

        :param chi: array of interpolating orbitals of shape [num_kpts, num_mo, num_interp_points]
        :param zeta: central tensor of dimension [num_kpts, num_G, num_G, num_interp_points, num_interp_points].
        :param kmf: pyscf k-object.  Currently only used to obtain the number of k-points.
                    must have an attribute kpts which len(self.kmf.kpts) returns number of
                    kpts.
        :param cholesky factor: for computing exact integrals
        """
        super().__init__(chi, zeta, kmf, cholesky_factor, nthc)
        # one-translation ISDF zeta[iq, dG]
        num_kpts = len(self.kmf.kpts)
        kpts = self.kmf.kpts
        kpts_pq = np.array(
            [(kp, kpts[ikq]) for ikp, kp in enumerate(kpts) for ikq in range(num_kpts)]
        )
        kpts_pq_indx = np.array(
            [(ikp, ikq) for ikp, kp in enumerate(kpts) for ikq in range(num_kpts)]
        )
        transfers = kpts_pq[:, 0] - kpts_pq[:, 1]
        unique_q, unique_indx, unique_inverse = unique(transfers)
        _, _, G_map_unique, delta_Gs = build_G_vector_mappings_single_translation(
            kmf.cell, kpts, kpts_pq_indx[unique_indx]
        )
        self.G_mapping = G_map_unique
        self.momentum_transfers = transfers[unique_indx]

    def get_eri(self, ikpts):
        """
        Construct (pkp qkq| rkr sks) via \\sum_{mu nu} zeta[iq, dG, mu, nu]
            chi[kp,p,mu]* chi[kq,q,mu] chi[kp,p,nu]* chi[ks,s,nu]

        :param ikpts: list of four integers representing the index of the kpoint in self.kmf.kpts
        :param check_eq: optional value to confirm a symmetry in the Cholesky vectors.
        """
        ikp, ikq, ikr, iks = ikpts
        mom_transfer = self.kpts[ikp] - self.kpts[ikq]
        q_indx = member(mom_transfer, self.momentum_transfers)[0]
        return build_eri_isdf_single_translation(
            self.chi, self.zeta, q_indx, ikpts, self.G_mapping
        )
