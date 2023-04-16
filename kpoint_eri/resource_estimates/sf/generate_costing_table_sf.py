import numpy as np
import numpy.typing as npt
import pandas as pd

from pyscf.pbc import cc, scf
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

from kpoint_eri.factorizations.hamiltonian_utils import build_hamiltonian
from kpoint_eri.resource_estimates.cc_helper.cc_helper import (
    build_approximate_eris,
    build_cc_inst,
    build_approximate_eris_rohf,
)
from kpoint_eri.resource_estimates.sf.compute_lambda_sf import compute_lambda
from kpoint_eri.resource_estimates.sf.compute_sf_resources import (
    cost_single_factorization,
)
from kpoint_eri.resource_estimates.sf.integral_helper_sf import (
    SingleFactorizationHelper,
)
from kpoint_eri.resource_estimates.utils.misc_utils import PBCResources


def generate_costing_table(
    pyscf_mf: scf.HF,
    naux_cutoffs: npt.NDArray[np.int32],
    name: str = "pbc",
    chi: int = 10,
    dE_for_qpe=0.0016,
    energy_method="MP2",
) -> PBCResources:
    """Generate resource estimate costing table given a set of cutoffs for
        single-factorized Hamiltonian.

    Arguments:
        pyscf_mf: k-point pyscf mean-field object
        naux_cutoffs: Array of (integer) auxiliary index cutoff values
        name: Optional descriptive name for simulation.
        chi: the number of bits for the representation of the coefficients
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

    sf_resource_obj = PBCResources(
        system_name=name,
        num_spin_orbitals=num_spin_orbs,
        num_kpts=num_kpts,
        dE=dE_for_qpe,
        chi=chi,
        exact_energy=np.real(reference_energy),
    )
    approx_eris = exact_eris
    for cutoff in naux_cutoffs:
        sf_helper = SingleFactorizationHelper(
            cholesky_factor=chol, kmf=pyscf_mf, naux=cutoff
        )
        if pyscf_mf.cell.spin == 0:
            approx_eris = build_approximate_eris(cc_inst, sf_helper, eris=approx_eris)
        else:
            approx_eris = build_approximate_eris_rohf(cc_inst, sf_helper, eris=approx_eris)
        approx_energy, _, _ = energy_function(approx_eris)

        sf_lambda = compute_lambda(hcore, sf_helper)
        sparse_res_cost = cost_single_factorization(
            num_spin_orbs,
            sf_lambda.lambda_total,
            sf_lambda.num_aux,
            list(kmesh),
            chi=chi,
            dE_for_qpe=dE_for_qpe,
        )

        sf_resource_obj.add_resources(
            ham_properties=sf_lambda,
            resource_estimates=sparse_res_cost,
            cutoff=sf_lambda.num_aux,
            approx_energy=np.real(approx_energy),
        )

    return sf_resource_obj
