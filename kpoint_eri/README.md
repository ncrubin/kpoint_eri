# Resource Estimation for Periodic Systems 

Module `openfermion.resource_estimates.pbc` to facilitate fault-tolerant (FT) resource estimates for second-quantized symmetry-adapted Hamiltonians with periodic boundary conditions (PBC).

The module provides symmetry adapated sparse, single, double and tensor hypercontraction representations of the Hamiltonians. 

For the methods listed above, there are sub-routines which:
* factorize the two-electron integrals if appropriate
* compute the lambda values, `compute_lambda()`
* estimate the number of logical qubits and Toffoli gates required to simulate with this factorization, `compute_cost()`

### Details

Given a pyscf scf calculation of a periodic system with k-points:

```python
from pyscf.pbc import gto, scf

cell = gto.Cell()
cell.atom = '''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-hf-rev'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 0
cell.build()

kmesh = [1, 1, 3]
kpts = cell.make_kpts(kmesh)
nkpts = len(kpts)
mf = scf.KRHF(cell, kpts).rs_density_fit()
mf.with_df.mesh = mf.cell.mesh
mf.kernel()
```

There are helper functions for the SF, DF, and THC factorization schemes that will make a nice table given some parameters. For example:

```python
from openfermion.resource_estimates.pbc import sf

table = sf.generate_costing_table(mf, name='water', rank_range=[20,25,30,35,40,45,50])
```
which outputs to a file called `single_factorization_water.json`, and contains:

```
```

Note that the automated costing relies on error in MP2 which may be a poor model chemistry depending on the system. 

The philosophy is that all costing methods are captured in the namespace related to the type of factorization (e.g., . So if one wanted to repeat the costing for DF or THC factorizations, one could 

```python
from openfermion.resource_estimates import df, thc

# make pretty DF costing table
df.generate_costing_table(mf, name='carbon-diamond', cutoffs=[1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5]) 

# make pretty THC costing table
# if you want to save each THC result to a file, you can set 'save_thc' to True
thc.generate_costing_table(mf, name='carbon-diamond', thc_rank_params=[2, 4, 6]) 
```

More fine-grained control is given by subroutines that compute the factorization, the lambda values, and the cost estimates.

Again, since we do not wish to burden all OpenFermion users with these dependencies, testing with GitHub workflows is disabled, but if you install the dependencies, running `pytest` should pass.

## Requirements

```
pyscf
jax
jaxlib
ase
pandas
```