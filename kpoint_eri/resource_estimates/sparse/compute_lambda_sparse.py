import numpy as np
from itertools import product

from kpoint_eri.resource_estimates import utils, sparse

def compute_lambda(
        pyscf_mf,
        integrals=None,
        localization='ibo',
        threshold=1e-5):
    kpoints = pyscf_mf.kpts
    mo_coeffs = pyscf_mf.mo_coeff
    momentum_map = utils.build_momentum_transfer_mapping(pyscf_mf.cell,
                                                   pyscf_mf.kpts)
    num_kpoints = len(kpoints)
    nmo_pk = [C.shape[1] for C in mo_coeffs]
    lambda_V = 0.0
    num_nnz = 0
    for iq in range(num_kpoints):
        for ikp, iks in product(range(num_kpoints), repeat=2):
            ikq = momentum_map[iq, ikp]
            ikr = momentum_map[iq, iks]
            kpt_pqrs = [kpoints[ik] for ik in [ikp,ikq,ikr,iks]]
            mos_pqrs = [mo_coeffs[ik] for ik in [ikp,ikq,ikr,iks]]
            mos_shape = [C.shape[1] for C in mos_pqrs]
            if integrals is None:
                eri_pqrs = sparse.build_eris_kpt(
                        pyscf_mf.with_df,
                        mos_pqrs,
                        kpt_pqrs,
                        compact=False)
            else:
                eri_pqrs = integrals[(ikq,ikp,iks)]
            eri_pqrs[abs(eri_pqrs) < threshold] = 0.0
            num_nnz += np.sum(eri_pqrs > threshold)
            lambda_V += np.sum(np.abs(eri_pqrs))

    # factor of 1/4 in hamiltonian gets multiplied by factor of 4 for spin
    # summation. so no further prefactor of num_nnz or lambda

    tot_size = num_kpoints**3 * max(nmo_pk)**4
    hcore = pyscf_mf.get_hcore(pyscf_mf.cell, kpts=pyscf_mf.kpts)
    lambda_T = 0.0
    for ik in range(num_kpoints):
        h1b = mo_coeffs[ik].conj().T @ hcore[ik] @ mo_coeffs[ik]
        h2b = np.zeros_like(h1b)
        for ik_prime in range(num_kpoints):
            kpt_pqrs = [
                    kpoints[ik],
                    kpoints[ik],
                    kpoints[ik_prime],
                    kpoints[ik_prime]
                    ]
            mos_pqrs = [
                    mo_coeffs[ik],
                    mo_coeffs[ik],
                    mo_coeffs[ik_prime],
                    mo_coeffs[ik_prime]
                    ]
            mos_shape = [C.shape[1] for C in mos_pqrs]
            if integrals is None:
                eri_pqrs = sparse.build_eris_kpt(
                        pyscf_mf.with_df,
                        mos_pqrs,
                        kpt_pqrs,
                        compact=False).reshape(mos_shape)
            else:
                eri_pqrs = integrals[(0,ik,ik_prime)].reshape(mos_shape)
            eri_pqrs[abs(eri_pqrs) < threshold] = 0.0
            h2b += np.einsum('pqrr->pq', eri_pqrs, optimize=True)
        T = h1b - 0.5 * h2b
        lambda_T = np.sum(np.abs(T))

    lambda_tot = lambda_T + lambda_V
    return lambda_tot, lambda_T, lambda_V, num_nnz, 1-num_nnz/tot_size