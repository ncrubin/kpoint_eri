from dataclasses import dataclass, field
from functools import reduce

import numpy as np

from pyscf.pbc import scf, cc
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

from kpoint_eri.resource_estimates.utils.misc_utils import PBCResources
from kpoint_eri.resource_estimates.sparse.integral_helper_sparse import (
    SparseFactorizationHelper,
)
from kpoint_eri.factorizations.hamiltonian_utils import build_hamiltonian
from kpoint_eri.resource_estimates.cc_helper.cc_helper import (
    build_approximate_eris,
    build_cc_inst,
    build_approximate_eris_rohf,
)
from kpoint_eri.resource_estimates.sparse.compute_lambda_sparse import compute_lambda
from kpoint_eri.resource_estimates.sparse.compute_sparse_resources import cost_sparse



def generate_costing_table(
    pyscf_mf: scf.HF,
    name="pbc",
    chi: int = 10,
    thresholds: np.ndarray = np.logspace(-1, -5, 6),
    dE_for_qpe=0.0016,
    energy_method="MP2",
) -> PBCResources:
    """Generate resource estimate costing table given a set of cutoffs for
        sparse Hamiltonian.

    Arguments:
        pyscf_mf: k-point pyscf mean-field object
        name: Optional descriptive name for simulation.
        chi: the number of bits for the representation of the coefficients
        thresholds: Array of sparse thresholds to generate the table for.
        dE_for_qpe: Phase estimation epsilon.

    Returns
        resources: Table of resource estimates.
    """
    kmesh = kpts_to_kmesh(pyscf_mf.cell, pyscf_mf.kpts)
    cc_inst = build_cc_inst(pyscf_mf)
    exact_eris = cc_inst.ao2mo()
    if energy_method == "MP2":
        energy_function = lambda x: cc_inst.init_amps(x)
        reference_energy, _, _ = energy_function(exact_eris) 
    elif energy_method == "CCSD":
        energy_function = lambda x: cc_inst.kernel(eris=x)
        reference_energy, _, _ = energy_function(exact_eris)
    else:
        raise ValueError(f"Unknown value for energy_method: {energy_method}")

    hcore, chol = build_hamiltonian(pyscf_mf)
    num_spin_orbs = 2 * hcore[0].shape[-1]

    num_kpts = np.prod(kmesh)

    sparse_resource_obj = PBCResources(
        system_name=name,
        num_spin_orbitals=num_spin_orbs,
        num_kpts=num_kpts,
        dE=dE_for_qpe,
        chi=chi,
        exact_energy=np.real(reference_energy),
    )
    approx_eris = exact_eris
    for thresh in thresholds:
        sparse_helper = SparseFactorizationHelper(
            cholesky_factor=chol, kmf=pyscf_mf, threshold=thresh
        )
        if pyscf_mf.cell.spin == 0:
            approx_eris = build_approximate_eris(cc_inst, sparse_helper, eris=approx_eris)
        else:
            approx_eris = build_approximate_eris_rohf(cc_inst, sparse_helper, eris=approx_eris)
        approx_energy, _, _ = energy_function(approx_eris)

        sparse_data = compute_lambda(hcore, sparse_helper)

        sparse_res_cost = cost_sparse(
            num_spin_orbs,
            sparse_data.lambda_total,
            sparse_data.num_sym_unique,
            list(kmesh),
            dE_for_qpe=dE_for_qpe,
            chi=chi,
        )
        sparse_resource_obj.add_resources(
            ham_properties=sparse_data,
            resource_estimates=sparse_res_cost,
            cutoff=thresh,
            approx_energy=np.real(approx_energy),
        )

    return sparse_resource_obj
