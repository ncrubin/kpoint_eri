import numpy as np
from itertools import product

def build_eris_kpt(
        Uq,
        lambda_Uq,
        Vq,
        lambda_Vq,
        PQRS):
    """
    Compute (momentum conserving) kpoint-integrals (pkp qkq | rkr sks) block
    (pkp qkp-Q | rks-Q sks) =
    """
    # (p kp q q kp - Q | r ks - Q s ks)
    # (pq|rs) = A^2 + B^2
    #         = U[n,pq,t]* e^2 U[n,pq, t]
    P, Q, R, S = PQRS
    UP = Uq[:,P,:]
    UQ = Uq[:,Q,:]
    UR = Uq[:,R,:]
    US = Uq[:,S,:]
    LPQ = np.einsum('nPt,nt,nQt->nPQ', UP, lambda_Uq, UQ.conj(), optimize=True)
    LRS = np.einsum('nPt,nt,nQt->nPQ', UR, lambda_Uq, US.conj(), optimize=True)
    eri_A = np.einsum('nPQ,nRS->PQRS', LPQ, LRS, optimize=True)
    VP = Vq[:,P,:]
    VQ = Vq[:,Q,:]
    VR = Vq[:,R,:]
    VS = Vq[:,S,:]
    LPQ = np.einsum('nPt,nt,nQt->nPQ', VP, lambda_Vq, VQ.conj(), optimize=True)
    LRS = np.einsum('nPt,nt,nQt->nPQ', VR, lambda_Vq, VS.conj(), optimize=True)
    eri_B = np.einsum('nPQ,nRS->PQRS', LPQ, LRS, optimize=True)

    eri = eri_A + eri_B
    return eri

def kpoint_df_eris(
        df_factors,
        kpoints,
        momentum_map,
        nmo_pk,
        ):
    nmo = sum(nmo_pk)
    eris = np.zeros((nmo,)*4, dtype=np.complex128)
    num_kpoints = momentum_map.shape[0]
    offsets = np.cumsum(nmo_pk) - nmo_pk[0]
    for iq in range(num_kpoints):
        for ikp, iks in product(range(num_kpoints), repeat=2):
            ikq = momentum_map[iq, ikp]
            ikr = momentum_map[iq, iks]
            Uiq = df_factors['U'][iq]
            lambda_Uiq = df_factors['lambda_U'][iq]
            Viq = df_factors['V'][iq]
            lambda_Viq = df_factors['lambda_V'][iq]
            P = slice(offsets[ikp], offsets[ikp] + nmo_pk[ikp])
            Q = slice(offsets[ikq], offsets[ikq] + nmo_pk[ikq])
            R = slice(offsets[ikr], offsets[ikr] + nmo_pk[ikr])
            S = slice(offsets[iks], offsets[iks] + nmo_pk[iks])
            eri_pqrs = build_eris_kpt(
                    Uiq,
                    lambda_Uiq,
                    Viq,
                    lambda_Viq,
                    (P,Q,R,S),
                    )
            eris[P,Q,R,S] = eri_pqrs

    return eris
