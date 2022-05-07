import numpy as np
from itertools import product

from kpoint_eri.resource_estimates import sf

def compute_lambda(
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
        An, Bn = sf.build_ABn(chol[iq,:,:,], iq, momentum_map)
        lambda_W += np.sum(np.einsum('npq->n', np.abs(An))**2.0)
        lambda_W += np.sum(np.einsum('npq->n', np.abs(Bn))**2.0)

    # TODO check prefactor.
    lambda_W *= 0.5

    for ik in range(num_kpoints):
        h1b = hcore[ik]
        h2b = np.zeros_like(h1b)
        for ik_prime in range(num_kpoints):
            eri_pqrs = sf.build_eris_kpt(chol[0], ik, ik_prime)
            h2b += np.einsum('pqrr->pq', eri_pqrs, optimize=True)
        T = h1b - 0.5 * h2b
        lambda_T = np.sum(np.abs(T))

    lambda_tot = lambda_T + lambda_W
    return lambda_tot, lambda_T, lambda_W
