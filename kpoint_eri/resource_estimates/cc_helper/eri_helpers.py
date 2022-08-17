import numpy as np

from pyscf.pbc import scf
from kpoint_eri.resource_estimates import sparse, sf, df, thc

# Sparse
class SparseHelper:

    def __init__(
            self,
            df,
            mo_coeffs,
            kpoints,
            threshold=0.0
            ):
        self.mo_coeffs = mo_coeffs
        self.kpoints = kpoints
        self.df = df
        self.thresh = threshold

    def get_eri(self, ikpts):
        kpt_pqrs = [self.kpoints[ik] for ik in ikpts]
        mos_pqrs = [self.mo_coeffs[ik] for ik in ikpts]
        mos_shape = [C.shape[1] for C in mos_pqrs]
        nk = len(self.kpoints)
        eri_pqrs = nk * sparse.build_eris_kpt(
                self.df,
                mos_pqrs,
                kpt_pqrs,
                compact=False).reshape(mos_shape)
        eri_pqrs[np.abs(eri_pqrs) < self.thresh] = 0.0
        return eri_pqrs

# Single-Factorization
class SingleFactorizationHelper:

    def __init__(
            self,
            chol,
            mom_map,
            kpoints,
            ):
        self.chol = chol
        nchol_pk = [L.shape[-1] for L in chol]
        self.nchol_pk = nchol_pk
        self.kpoints = kpoints
        k1k2_q = np.zeros_like(mom_map)
        nk = len(kpoints)
        for iq in range(nk):
            for ik1 in range(nk):
                for ik2 in range(nk):
                    if mom_map[iq,ik1] == ik2:
                        k1k2_q[ik1,ik2] = iq
        self.mom_map_12 = k1k2_q

    def get_eri(self, ikpts):
        ikp, ikq, ikr, iks = ikpts
        iq = self.mom_map_12[ikp, ikq]
        iq_ = self.mom_map_12[iks, ikr]
        nk = len(self.kpoints)
        return nk * sf.build_eris_kpt(self.chol[iq], ikp, iks)


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


class DoubleFactorizationHelper:

    def __init__(
            self,
            chol,
            mom_map,
            kpoints,
            nmo_pk,
            df_thresh=0.0):
        self.chol = chol
        nchol_pk = [L.shape[-1] for L in chol]
        self.offsets = np.cumsum(nmo_pk) - nmo_pk[0]
        self.nchol_pk = nchol_pk
        self.kpoints = kpoints
        self.mom_map = mom_map
        self.df_thresh = df_thresh
        self.nmo_pk = nmo_pk
        k1k2_q = np.zeros_like(mom_map)
        nk = len(kpoints)
        nmo_tot = sum(nmo_pk)
        for iq in range(nk):
            for ik1 in range(nk):
                for ik2 in range(nk):
                    if mom_map[iq,ik1] == ik2:
                        k1k2_q[ik1,ik2] = iq
        self.mom_map_12 = k1k2_q
        self.df_factors = df.double_factorize(
                chol,
                self.mom_map,
                self.nmo_pk,
                df_thresh=self.df_thresh)

    def get_eri(self, ikpts):
        ikp, ikq, ikr, iks = ikpts
        iq = self.mom_map_12[ikp, ikq]
        iq_ = self.mom_map_12[iks, ikr]
        assert iq == iq_
        Uiq = self.df_factors['U'][iq]
        lambda_Uiq = self.df_factors['lambda_U'][iq]
        Viq = self.df_factors['V'][iq]
        lambda_Viq = self.df_factors['lambda_V'][iq]
        offsets = self.offsets
        nmo_pk = self.nmo_pk
        P = slice(offsets[ikp], offsets[ikp] + nmo_pk[ikp])
        Q = slice(offsets[ikq], offsets[ikq] + nmo_pk[ikq])
        R = slice(offsets[ikr], offsets[ikr] + nmo_pk[ikr])
        S = slice(offsets[iks], offsets[iks] + nmo_pk[iks])
        eri_pqrs = df.build_eris_kpt(
                Uiq,
                lambda_Uiq,
                Viq,
                lambda_Viq,
                (P,Q,R,S),
                )
        nk = len(self.kpoints)
        return nk * eri_pqrs

class THCHelper:

    def __init__(
            self,
            etapP,
            MPQ,
            ao=False,
            mo_coeffs=None):
        self.etapP = etapP
        self.MPQ = MPQ
        self.ao = ao
        self.mo_coeffs = mo_coeffs

    def get_eri(self, ikpts):
        if self.ao:
            etapP = np.einsum('pi,pP->iP', self.mo_coeffs, self.etapP)
        else:
            etapP = self.etapP
        eris = np.einsum(
                'pP,qP,PQ,rQ,sQ->pqrs',
                etapP.conj(),
                etapP,
                self.MPQ,
                etapP.conj(),
                etapP,
                optimize=True)
        return eris
