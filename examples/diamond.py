#!/usr/bin/env python

# adapted from pyscf

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
    verbose=0
)

# We build a supercell composed of 'nmp' replicated units and run
# our usual molecular Hartree-Fock program, but using integrals
# between periodic gaussians.
#cell = build_cell(ase_atom, ke=50., basis=basis)
# supcell = super_cell(cell, nmp)
# mf = pbchf.RHF(supcell)
# mf.chkfile = 'scf_supercell.chk'
# mf.kernel()
# supcell_energy = mf.energy_tot() / np.prod(nmp)

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
verbose = ''
# verbose = '- vvv'
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
