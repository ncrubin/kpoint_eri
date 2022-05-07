import numpy as np

from kpoint_eri.resource_estimates import df

def compute_lambda_df(
        hcore,
        df_factors,
        kpoints,
        momentum_map,
        nmo_pk,
        ):
    nmo = sum(nmo_pk)
    eris = np.zeros((nmo,)*4, dtype=np.complex128)
    num_kpoints = momentum_map.shape[0]
    offsets = np.cumsum(nmo_pk) - nmo_pk[0]
    eigs_A = df_factors['lambda_U'] # Q, nchol, neig
    eigs_B = df_factors['lambda_V'] # Q, nchol, neig
    lambda_F  = np.sum(np.abs(eigs_A)**2)
    lambda_F += np.sum(np.abs(eigs_B)**2)
    lambda_F *= 0.125

    # one-body contribution only contains contributions from Q = 0
    Uiq = df_factors['U'][0]
    lambda_Uiq = df_factors['lambda_U'][0]
    Viq = df_factors['V'][0]
    lambda_Viq = df_factors['lambda_V'][0]
    lambda_T = 0.0
    for ik in range(num_kpoints):
        h1b = hcore[ik]
        h2b = np.zeros_like(h1b)
        for ik_prime in range(num_kpoints):
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
            h2b += 0.5 * np.einsum('pqrr->pq', er_pqrs, optimize=True)
        T = h1b + h2b
        lambda_T = np.sum(np.abs(T))

    lambda_tot = lambda_T + lambda_F
    return lambda_tot, lambda_T, lambda_F
