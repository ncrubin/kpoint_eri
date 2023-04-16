"""
Example resource estimates for solid Hydrogen
"""
import numpy as np

from ase.build import bulk
from ase.lattice.cubic import Diamond

from pyscf.pbc.tools import pyscf_ase
from pyscf.pbc import gto, scf, mp, cc

from kpoint_eri.resource_estimates.sparse.ncr_sparse_integral_helper import NCRSSparseFactorizationHelper
from kpoint_eri.resource_estimates.sparse.compute_sparse_resources import cost_sparse
from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
from kpoint_eri.resource_estimates.sparse.compute_lambda_sparse import compute_lambda_ncr as compute_lambda_sparse

from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_approximate_eris


def hydrogen_in_diamond_config():
    # ase_atom=Diamond(symbol='H', latticeconstant=1.2) 
    ase_atom = bulk('H', 'bcc', a=2, cubic=True)
    cell = gto.Cell()
    cell.verbose = 4
    cell.atom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a=ase_atom.cell
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.build()

    return cell
    
def run_scf(cell:gto.Cell, kmesh):
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts=kpts).rs_density_fit()
    mf.kernel()
    return mf

def get_cholesky(mf):
    mymp = mp.KMP2(mf)
    Luv = cholesky_from_df_ints(mymp)
    return Luv

def generate_sparse_cost_table(mf, Luv):
    helper = NCRSSparseFactorizationHelper(cholesky_factor=Luv, kmf=mf)
    approx_cc = cc.KRCCSD(mf)
    approx_cc.verbose = 0
    approx_cc = build_approximate_eris(approx_cc, helper)
    eris = approx_cc.ao2mo(lambda x: x)
    exact_emp2, _, _ = approx_cc.init_amps(eris)
    exact_nnz_unique = helper.get_total_unique_terms_above_thresh()

    print("EMP2(Exact): ", exact_emp2)
    print("NNZ(Exact): ", exact_nnz_unique)

    hcore_mo = [
        C.conj().T @ hcore @ C for (C, hcore) in zip(mf.mo_coeff, mf.get_hcore())
    ]
    

    print("{}    {}   {}   {}  {}   {}       {}   {}".format("index", "Threshold", "RelativeErr", "AbsErr", "NNZ", "lamTot", "lamone", "lamtwo")) 
    for i, thresh in enumerate(np.linspace(4.0E-2, 1.0E-3, 10)): # [1.0E-1, 1.0e-2]):
        abs_sum_coeffs = 0
        helper = NCRSSparseFactorizationHelper(cholesky_factor=Luv, kmf=mf, threshold=thresh)
        approx_cc = build_approximate_eris(approx_cc, helper)
        eris = approx_cc.ao2mo(lambda x: x)
        emp2, _, _ = approx_cc.init_amps(eris)
        delta = abs((emp2 - exact_emp2) / exact_emp2) * 100
        test_nnz_unique = helper.get_total_unique_terms_above_thresh()
        lambda_tot, lambda_one_body, lambda_two_body, nnz_unique = compute_lambda_sparse(hcore_mo, helper)
        assert test_nnz_unique == nnz_unique

        n = helper.nao * 2
        nk = helper.nk
        dE = 0.001
        chi = 10
        stps = cost_sparse(n, nk, lambda_tot, nnz_unique, dE, chi, 20_000)
        iter_tof_cost, total_tof_cost, total_qubit_cost = cost_sparse(n, nk, lambda_tot, nnz_unique, dE, chi, stps[0])

        print(" {:4d}  {: 10.6e} {:10.4e} {:10.4e} {:4d} {:10.3e} {:10.3e} {:10.3e} {:4.5e} {:4.5e} {:4.5e}".format(i, thresh, delta, abs(emp2 - exact_emp2),
        nnz_unique, lambda_tot, lambda_one_body, lambda_two_body,
        iter_tof_cost, total_tof_cost, total_qubit_cost))


if __name__ == "__main__":
    cell = hydrogen_in_diamond_config()
    mf = run_scf(cell, [1, 1, 2])
    Luv = get_cholesky(mf)
    generate_sparse_cost_table(mf, Luv)
