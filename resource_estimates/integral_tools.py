# Tools for building kpoint / supercell integrals.
from itertools import product
import numpy as np

from pyscf.pbc.df import fft, fft_ao2mo

# 1. Supercell integrals
def supercell_eris(
        supercell,
        mo_coeff=None,
        threshold=0.0,
        ):
    df = fft.FFTDF(supercell, kpts=np.zeros((4,3)))
    nao = supercell.nao_nr()
    if mo_coeff is not None:
        nmo = mo_coeff.shape[-1]
        eris = df.ao2mo(mo_coeff, compact=False).reshape((nmo,)*4)
    else:
        eris = df.get_eri(compact=False).reshape((nao,)*4)

    eris[np.abs(eris) < threshold] = 0.0
    return eris

# 2. kpoint-integrals (pkp qkq | rkr sks)
def kpoint_eris(
        cell,
        mo_coeffs,
        kpoints,
        momentum_map,
        threshold=0.0
        ):
    df = fft.FFTDF(cell, kpts=kpoints)
    nmo = sum(C.shape[1] for C in mo_coeffs)
    eris = np.zeros((nmo,)*4, dtype=np.complex128)
    num_kpoints = momentum_map.shape[0]
    nmo_pk = [C.shape[1] for C in mo_coeffs]
    offsets = np.cumsum(nmo_pk, dtype=np.int32) - nmo_pk[0]
    for iq in range(num_kpoints):
        for ikp, iks in product(range(num_kpoints), repeat=2):
            ikq = momentum_map[iq, ikp]
            ikr = momentum_map[iq, iks]
            kpt_pqrs = [kpoints[ik] for ik in [ikp,ikq,ikr,iks]]
            mos_pqrs = [mo_coeffs[ik] for ik in [ikp,ikq,ikr,iks]]
            mos_shape = [C.shape[1] for C in mos_pqrs]
            eri_pqrs = df.ao2mo(
                    mos_pqrs,
                    kpts=kpt_pqrs,
                    compact=False).reshape(mos_shape) / num_kpoints
            P = slice(ikp*offsets[ikp], ikp*offsets[ikp] + nmo_pk[ikp])
            Q = slice(ikq*offsets[ikq], ikq*offsets[ikq] + nmo_pk[ikq])
            R = slice(ikr*offsets[ikr], ikr*offsets[ikr] + nmo_pk[ikr])
            S = slice(iks*offsets[iks], iks*offsets[iks] + nmo_pk[iks])
            eris[P,Q,R,S] = eri_pqrs

    eris[np.abs(eris) < threshold] = 0.0

    return eris

class ERIHelper:

    def __init__(
            self,
            df,
            mo_coeffs,
            kpoints
            ):
        self.mo_coeffs = mo_coeffs
        self.kpoints = kpoints
        self.df = df

    def get_eri(self, ikpts):
        kpt_pqrs = [self.kpoints[ik] for ik in ikpts]
        mos_pqrs = [self.mo_coeffs[ik] for ik in ikpts]
        mos_shape = [C.shape[1] for C in mos_pqrs]
        eri_pqrs = self.df.ao2mo(
                mos_pqrs,
                kpts=kpt_pqrs,
                compact=False).reshape(mos_shape)
        return eri_pqrs

# 3. cholesky kpoint integrals
def kpoint_cholesky_eris(
        chol,
        kpoints,
        momentum_map,
        nmo_pk,
        chol_thresh=None,
        ):
    nmo = sum(nmo_pk)
    eris = np.zeros((nmo,)*4, dtype=np.complex128)
    num_kpoints = momentum_map.shape[0]
    offsets = np.cumsum(nmo_pk) - nmo_pk[0]
    nchol_pk = [L.shape[-1] for L in chol]
    if chol_thresh is not None:
        nchol_pk = [chol_thresh] * num_kpoints
    for iq in range(num_kpoints):
        for ikp, iks in product(range(num_kpoints), repeat=2):
            Lkp_minu_q = chol[iq][ikp,:,:,:nchol_pk[ikq]]
            Lks_minu_q = chol[iq][iks,:,:,:nchol_pk[ikq]]
            ikq = momentum_map[iq, ikp]
            ikr = momentum_map[iq, iks]
            eri_pqrs = np.einsum(
                    'pqn,srn->pqrs',
                    Lkp_minu_q,
                    Lks_minu_q.conj(),
                    optimize=True)
            P = slice(ikp*offsets[ikp], ikp*offsets[ikp] + nmo_pk[ikp])
            Q = slice(ikq*offsets[ikq], ikq*offsets[ikq] + nmo_pk[ikq])
            R = slice(ikr*offsets[ikr], ikr*offsets[ikr] + nmo_pk[ikr])
            S = slice(iks*offsets[iks], iks*offsets[iks] + nmo_pk[iks])
            eris[P,Q,R,S] = eri_pqrs

    return eris

class CholeskyHelper:

    def __init__(
            self,
            chol,
            mom_map,
            kpoints,
            chol_thresh=None):
        self.chol = chol
        nchol_pk = [L.shape[-1] for L in chol]
        if chol_thresh is not None:
            nchol_pk = [chol_thresh] * len(kpoints)
        self.nchol_pk = nchol_pk
        self.kpoints = kpoints
        k1k2_q = np.zeros_like(mom_map)
        nk = len(kpoints)
        # print(nk)
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
        assert iq == iq_
        Lkp_minu_q = self.chol[iq][ikp,:,:,:self.nchol_pk[ikp]]
        Lks_minu_q = self.chol[iq][iks,:,:,:self.nchol_pk[ikp]]
        eri_pqrs = np.einsum(
                'pqn,srn->pqrs',
                Lkp_minu_q,
                Lks_minu_q.conj(),
                optimize=True)
        # pyscf expects to divide this by nk again
        return len(self.kpoints) * eri_pqrs

def cholesky_eri_helper(
        df,
        ikpts,
        kpts,
        mo_coeffs):
    kpt_pqrs = [kpoints[ik] for ik in [ikp,ikq,ikr,iks]]
    mos_pqrs = [mo_coeffs[ik] for ik in [ikp,ikq,ikr,iks]]
    mos_shape = [C.shape[1] for C in mos_pqrs]
    eri_pqrs = df.ao2mo(
            mos_pqrs,
            kpts=kpt_pqrs,
            compact=False).reshape(mos_shape) / num_kpoints
    return eri_pqrs

# 4. DF kpoint integrals
class DFHelper:

    def __init__(
            self,
            chol,
            mom_map,
            kpoints,
            nmo_pk,
            chol_thresh=None,
            df_thresh=0.0):
        self.chol = chol
        nchol_pk = [L.shape[-1] for L in chol]
        self.offsets = np.cumsum(nmo_pk) - nmo_pk[0]
        if chol_thresh is not None:
            nchol_pk = [chol_thresh] * len(kpoints)
        nchol = chol_thresh
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
        # store contiguously for ease of einsum
        self.Us = np.zeros((nk, nchol, nmo_tot, nmo_tot), dtype=np.complex128)
        self.Vs = np.zeros((nk, nchol, nmo_tot, nmo_tot), dtype=np.complex128)
        self.lambda_A = np.zeros((nk, nchol, nmo_tot), dtype=np.complex128)
        self.lambda_B = np.zeros((nk, nchol, nmo_tot), dtype=np.complex128)
        # Build DF factors
        for iq in range(nk):
            # print(iq)
            for nc in range(nchol):
                # print(nc)
                A, B = self.build_AB(iq, nc, self.mom_map)
                eigs, eigv = self.get_df_factor(A, self.df_thresh)
                self.lambda_A[iq, nc] = eigs
                self.Us[iq, nc] = eigv
                eigs, eigv = self.get_df_factor(B, self.df_thresh)
                self.lambda_B[iq, nc] = eigs
                self.Vs[iq, nc] = eigv

    def get_eri(self, ikpts):
        ikp, ikq, ikr, iks = ikpts
        iq = self.mom_map_12[ikp, ikq]
        iq_ = self.mom_map_12[iks, ikr]
        assert iq == iq_
        # (p kp q q kp - Q | r ks - Q s ks)
        # (pq|rs) = A^2 + B^2
        #         = U[n,pq,t]* e^2 U[n,pq, t]
        P = slice(ikp*self.offsets[ikp], ikp*self.offsets[ikp] + self.nmo_pk[ikp])
        Q = slice(ikq*self.offsets[ikq], ikq*self.offsets[ikq] + self.nmo_pk[ikq])
        R = slice(ikr*self.offsets[ikr], ikr*self.offsets[ikr] + self.nmo_pk[ikr])
        S = slice(iks*self.offsets[iks], iks*self.offsets[iks] + self.nmo_pk[iks])
        # print(iq)
        # U = self.Us[iq]
        # eri_A = np.einsum(
                # 'nPt,nt,nQt,nRs,ns,nSs->PQRS',
                # U,
                # self.lambda_A[iq],
                # U.conj(),
                # U,
                # self.lambda_A[iq],
                # U.conj(),
                # optimize=True)
        # V = self.Vs[iq]
        # eri_B = np.einsum(
                # 'nPt,nt,nQt,nRs,ns,nSs->PQRS',
                # V,
                # self.lambda_B[iq],
                # V.conj(),
                # V,
                # self.lambda_B[iq],
                # V.conj(),
                # optimize=True)
        # eris = eri_A + eri_B
        # return eris[P,Q,R,S] * len(self.kpoints)
        UP = self.Us[iq,:,P,:].copy()
        UQ = self.Us[iq,:,Q,:].copy()
        UR = self.Us[iq,:,R,:].copy()
        US = self.Us[iq,:,S,:].copy()
        eri_A = np.einsum(
                'nPt,nt,nQt,nRr,nr,nSr->PQRS',
                UP,
                self.lambda_A[iq],
                UQ.conj(),
                UR,
                self.lambda_A[iq],
                US.conj(),
                optimize=True)
        VP = self.Vs[iq,:,P,:].copy()
        VQ = self.Vs[iq,:,Q,:].copy()
        VR = self.Vs[iq,:,R,:].copy()
        VS = self.Vs[iq,:,S,:].copy()
        eri_B = np.einsum(
                'nPt,nt,nQt,nRr,nr,nSr->PQRS',
                VP,
                self.lambda_B[iq],
                VQ.conj(),
                VR,
                self.lambda_B[iq],
                VS.conj(),
                optimize=True)

        eri = eri_A + eri_B
        return eri * len(self.kpoints)

    @staticmethod
    def get_df_factor(mat, thresh):
        eigs, eigv = np.linalg.eigh(mat)
        # normSC = np.sum(np.abs(eigs))
        # ix = np.argsort(np.abs(eigs))[::-1]
        # eigs = eigs[ix]
        # eigv = eigv[:,ix]
        # truncation = normSC * np.abs(eigs)
        # to_zero = truncation < thresh
        # eigs[to_zero] = 0.0
        # eigv[:,to_zero] = 0.0
        return eigs, eigv

    def build_AB(self, Q_index, chol_indx, momentum_map):
        # print(Q_index, chol_indx)
        LQn = self.chol[Q_index][:, :, :, chol_indx]
        nmo = LQn.shape[1]
        assert len(LQn.shape) == 3
        num_kpoints = LQn.shape[0]
        M = np.zeros((nmo*num_kpoints, nmo*num_kpoints), dtype=np.complex128)
        A = np.zeros((nmo*num_kpoints, nmo*num_kpoints), dtype=np.complex128)
        B = np.zeros((nmo*num_kpoints, nmo*num_kpoints), dtype=np.complex128)
        for kp in range(num_kpoints):
            kq = momentum_map[Q_index, kp]
            for p, q in product(range(nmo), repeat=2):
                P = int(kp * nmo + p)
                Q = int(kq * nmo + q)
                M[P,Q] += LQn[kp, p, q]
                # M[Q,P] += LQn[kp, p, q].conj()
        A = 0.5 * (M + M.conj().T)
        B = 0.5*1j * (M - M.conj().T)
        assert np.linalg.norm(A-A.conj().T) < 1e-12
        assert np.linalg.norm(B-B.conj().T) < 1e-12
        return A, B

# 5. THC supercell integrals
def thc_eris(
        orbs,
        muv):
    eris = np.einsum(
            'pP,qP,PQ,rQ,sQ->pqrs',
            orbs,
            orbs,
            muv,
            orbs,
            orbs,
            optimize=True)
    return eris

class THCHelper:

    def __init__(
            self,
            orbs,
            muv):
        self.orbs = orbs
        self.muv = muv

    def get_eri(self, ikpts):
        eris = np.einsum(
                'pP,qP,PQ,rQ,sQ->pqrs',
                self.orbs,
                self.orbs,
                self.muv,
                self.orbs,
                self.orbs,
                optimize=True)
        return eris
