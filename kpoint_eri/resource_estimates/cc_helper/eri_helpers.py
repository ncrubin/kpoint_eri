import numpy as np

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
