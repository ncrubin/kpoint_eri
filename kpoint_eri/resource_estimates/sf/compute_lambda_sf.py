import numpy as np

from kpoint_eri.resource_estimates import sf

def compute_lambda_sf(
        hcore,
        chol,
        kpoints,
        momentum_map,
        nmo_pk,
        ):
    nmo = sum(nmo_pk)
    eris = np.zeros((nmo,)*4, dtype=np.complex128)
    num_kpoints = momentum_map.shape[0]
    offsets = np.cumsum(nmo_pk) - nmo_pk[0]
    nchol_pk = [L.shape[-1] for L in chol]
    lambda_W = 0.0
    for iq in range(num_kpoints):
        for ikp, iks in product(range(num_kpoints), repeat=2):
            ikq = momentum_map[iq, ikp]
            ikr = momentum_map[iq, iks]
            eri_pqrs = sf.build_eris_kpt(chol[iq], ikp, iks)
            lambda_W += num.sum(np.abs(eri_pqrs))

    for ik in range(num_kpoints):
        h1b = hcore[ik]
        h2b = np.zeros_like(h1b)
        for ik_prime in range(num_kpoints):
            eri_pqrs = sf.build_eris_kpt(chol[0], ik, ik_prime)
            h2b += 0.5 * np.einsum('pqrr->pq', er_pqrs, optimize=True)
        T = h1b + h2b
        lambda_T = np.sum(np.abs(T_prime))

    lambda_tot = lambda_T + lambda_W
    return lambda_tot, lambda_T, lambda_V
