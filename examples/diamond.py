#!/usr/bin/env python

# adapted from pyscf example

import numpy as np
from pyscf.pbc import cc as pbccc
from pyscf.pbc import scf as pbchf
from pyscf.pbc import gto
from pyscf.pbc.tools.pbc import super_cell

nmp = [1, 1, 2]
cell = gto.M(
    unit='B',
    a=[[0., 3.37013733, 3.37013733],
       [3.37013733, 0., 3.37013733],
       [3.37013733, 3.37013733, 0.]],
    mesh=[24,]*3,
    atom='''C 0 0 0
              C 1.68506866 1.68506866 1.68506866''',
    basis='gth-szv',
    pseudo='gth-pade',
    verbose=4
)

# We build a supercell composed of 'nmp' replicated units and run
# our usual molecular Hartree-Fock program, but using integrals
# between periodic gaussians.
#cell = build_cell(ase_atom, ke=50., basis=basis)

kpts = cell.make_kpts(nmp)
kmf = pbchf.KRHF(cell, kpts)
kmf.chkfile = 'scf_kpoint.chk'
kpoint_energy = kmf.kernel()

from kpoint_eri.resource_estimates import sparse
# 0. Sparse
# sparsity = 1 - num_nnz/num_elements
lambda_tot, lambda_T, lambda_V, num_nnz, sparsity = sparse.compute_lambda(kmf)
print("sparse lambda : ", lambda_tot)

chol_file = 'kpoint_chol.h5'
# swap for queiter output 
# verbose = ''
verbose = '-vvv'
import os
os.system(f"""
mpirun -np 2 python -u ../bin/gen_chol.py -i scf_kpoint.chk -o {chol_file} \
        -c 1e-6 -b mo {verbose}
""")

# 1. Single-Factorization
from kpoint_eri.resource_estimates import sf
from kpoint_eri.resource_estimates import utils

ham = utils.read_cholesky_contiguous(chol_file, frac_chol_to_keep=1.0)
cell, kmf = utils.init_from_chkfile(f'{kmf.chkfile}')
mo_coeffs = kmf.mo_coeff
num_kpoints = len(mo_coeffs)
kpoints = ham['kpoints']
momentum_map = ham['qk_k2']
chol = ham['chol']
lambda_tot, lambda_T, lambda_W, nchol = sf.compute_lambda(
        ham['hcore'],
        chol,
        kpoints,
        momentum_map,
        ham['nmo_pk']
        )
print("sf lambda : ", lambda_tot)

# 2. DF
from kpoint_eri.resource_estimates import df
df_factors = df.double_factorize_batched(
        ham['chol'],
        ham['qk_k2'],
        ham['nmo_pk'],
        df_thresh=1e-5)
lambda_tot_df, lambda_T, lambda_F, num_eig = df.compute_lambda(
        ham['hcore'],
        df_factors,
        kpoints,
        momentum_map,
        ham['nmo_pk']
        )

print("df lambda : ", lambda_tot_df)

# Run CCSD
mycc = pbccc.KRCCSD(kmf)
print("exact cc : ", mycc.kernel()[0])

# 0 sparse
from kpoint_eri.resource_estimates.cc_helper import (
        build_krcc_sparse_eris,
        build_krcc_sf_eris,
        build_krcc_df_eris)

cc = build_krcc_sparse_eris(kmf, threshold=1e-1)
print("sparse cc: ", cc.kernel()[0])
# Truncate cholesky too (keep 10 % )
ham = utils.read_cholesky_contiguous(chol_file, frac_chol_to_keep=0.1)
cc = build_krcc_sf_eris(kmf, ham['chol'], ham['qk_k2'], ham['kpoints'])
print("sf cc 0.1: ", cc.kernel()[0])
ham = utils.read_cholesky_contiguous(chol_file, frac_chol_to_keep=0.8)
cc = build_krcc_sf_eris(kmf, ham['chol'], ham['qk_k2'], ham['kpoints'])
print("sf cc 0.8: ", cc.kernel()[0])
cc = build_krcc_df_eris(
        kmf, ham['chol'], ham['qk_k2'], ham['kpoints'],
        ham['nmo_pk'], df_thresh=1e-1
        )
print("df cc (sf 0.8, df 1e-1): ", cc.kernel()[0])

# Supercell comparison
# from openfermion molecular
def compute_lambda(h1, eri_full, sf_factors):
    """ Compute lambda for Hamiltonian using SF method of Berry, et al.

    Args:
        pyscf_mf - PySCF mean field object
        sf_factors (ndarray) - (N x N x rank) array of SF factors from rank
            reduction of ERI

    Returns:
        lambda_tot (float) - lambda value for the single factorized Hamiltonian
    """

    # Effective one electron operator contribution
    T = h1 - 0.5 * np.einsum("pqqs->ps", eri_full, optimize=True) +\
        np.einsum("pqrr->pq", eri_full, optimize = True)

    lambda_T = np.sum(np.abs(T))

    # Two electron operator contributions
    lambda_W = 0.25 * np.einsum(
        "ijP,klP->", np.abs(sf_factors), np.abs(sf_factors), optimize=True)
    lambda_tot = lambda_T + lambda_W

    return lambda_tot

import time
def modified_cholesky(M, tol=1e-6, verbose=True, cmax=20):
    """Modified cholesky decomposition of matrix.
    See, e.g. [Motta17]_
    Parameters
    ----------
    M : :class:`numpy.ndarray`
        Positive semi-definite, symmetric matrix.
    tol : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors.
    """
    # matrix of residuals.
    assert len(M.shape) == 2
    delta = np.copy(M.diagonal())
    nchol_max = int(cmax*M.shape[0]**0.5)
    # index of largest diagonal element of residual matrix.
    nu = np.argmax(np.abs(delta))
    delta_max = delta[nu]
    if verbose:
        print ("# max number of cholesky vectors = %d"%nchol_max)
        print ("# iteration %d: delta_max = %f"%(0, delta_max.real))
    # Store for current approximation to input matrix.
    Mapprox = np.zeros(M.shape[0], dtype=M.dtype)
    chol_vecs = np.zeros((nchol_max, M.shape[0]), dtype=M.dtype)
    nchol = 0
    chol_vecs[0] = np.copy(M[:,nu])/delta_max**0.5
    while abs(delta_max) > tol:
        # Update cholesky vector
        start = time.time()
        Mapprox += chol_vecs[nchol]*chol_vecs[nchol].conj()
        delta = M.diagonal() - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        nchol += 1
        Munu0 = np.dot(chol_vecs[:nchol,nu].conj(), chol_vecs[:nchol,:])
        chol_vecs[nchol] = (M[:,nu] - Munu0) / (delta_max)**0.5
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print ("# iteration %d: delta_max = %13.8e: time = %13.8e"%info)

    return np.array(chol_vecs[:nchol])


supcell = super_cell(cell, nmp)
mf = pbchf.RHF(supcell)
mf.chkfile = 'scf_supercell.chk'
mf.kernel()
supcell_energy = mf.energy_tot() / np.prod(nmp)

# 8-fold symmetry at gamma.
C = mf.mo_coeff
h1 = C.T @ mf.get_hcore() @ C
eri = mf.with_df.ao2mo(C)
from pyscf import ao2mo

eri_full = ao2mo.restore(1, eri, C.shape[1])
nmo = C.shape[1]
L = modified_cholesky(eri_full.reshape((nmo*nmo, nmo*nmo)))
# produces naux x nmo x nmo, openfermion expects naux on last index
L = L.reshape((-1, nmo, nmo)).transpose((1, 2, 0))
print("cholesky error: ", np.linalg.norm((np.einsum('pqn,rsn->pqrs', L, L)-eri_full)))
print("lambda sf mol: ", compute_lambda(h1, eri_full, L))
