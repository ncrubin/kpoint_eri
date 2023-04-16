from dataclasses import dataclass, field
from functools import reduce
from typing import Union

import numpy as np

from pyscf.pbc import scf
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

from kpoint_eri.resource_estimates.utils.misc_utils import PBCResources
from kpoint_eri.resource_estimates.df.integral_helper_df import DFABKpointIntegrals
from kpoint_eri.factorizations.hamiltonian_utils import build_hamiltonian
from kpoint_eri.resource_estimates.cc_helper.cc_helper import (
    build_approximate_eris,
    build_cc_inst,
    build_approximate_eris_rohf,
)
from kpoint_eri.resource_estimates.df.compute_lambda_df import compute_lambda
from kpoint_eri.resource_estimates.df.compute_df_resources import (
    cost_double_factorization,
)
from kpoint_eri.resource_estimates.utils.misc_utils import compute_beta_for_resources


@dataclass
class DFResources(PBCResources):
    num_aux: int = -1
    beta: int = 20


def generate_costing_table(
    pyscf_mf: scf.HF,
    cutoffs: np.ndarray,
    name: str = "pbc",
    chi: int = 10,
    beta: int = 20,
    dE_for_qpe: float = 0.0016,
    energy_method: str = "MP2",
) -> DFResources:
    """Generate resource estimate costing table given a set of cutoffs for
        double-factorized Hamiltonian.

    Arguments:
        pyscf_mf: k-point pyscf mean-field object
        cutoffs: Array of (integer) auxiliary index cutoff values
        name: Optional descriptive name for simulation.
        chi: the number of bits for the representation of the coefficients
        beta: the number of bits for controlled rotation. If None we estimate
            the value given N and dE_for_qpe. See compute_beta_for_resources for
            details.
        dE_for_qpe: Phase estimation epsilon.
        energy_method: Which model chemistry to use to estimate convergence of
            factorization with respect to threshold. values are MP2 or CCSD.

    Returns
        resources: Table of resource estimates.

    Raises:
        ValueError if energy_method is unknown.
    """
    kmesh = kpts_to_kmesh(pyscf_mf.cell, pyscf_mf.kpts)
    cc_inst = build_cc_inst(pyscf_mf)
    exact_eris = cc_inst.ao2mo()
    if energy_method.lower() == "mp2":
        energy_function = lambda x: cc_inst.init_amps(x)
        reference_energy, _, _ = energy_function(exact_eris)
    elif energy_method.lower() == "ccsd":
        energy_function = lambda x: cc_inst.kernel(eris=x)
        reference_energy, _, _ = energy_function(exact_eris)
    else:
        raise ValueError(f"Unknown value for energy_method: {energy_method}")

    hcore, chol = build_hamiltonian(pyscf_mf)
    num_spin_orbs = 2 * hcore[0].shape[-1]
    num_kpts = np.prod(kmesh)

    num_aux = chol[0, 0].shape[0]
    # This is constant as we don't truncate first factorization for DF
    num_aux_df = 2 * num_aux * num_kpts
    df_resource_obj = DFResources(
        system_name=name,
        num_spin_orbitals=num_spin_orbs,
        num_kpts=num_kpts,
        dE=dE_for_qpe,
        chi=chi,
        beta=beta,
        exact_energy=np.real(reference_energy),
        num_aux=num_aux_df,
    )
    # Save some space and overwrite eris object from exact CC
    approx_eris = exact_eris
    if beta is None:
        num_kpts = np.prod(kmesh)
        beta = compute_beta_for_resources(num_spin_orbs, num_kpts, dE_for_qpe)
    for cutoff in cutoffs:
        df_helper = DFABKpointIntegrals(cholesky_factor=chol, kmf=pyscf_mf)
        df_helper.double_factorize(thresh=cutoff)
        df_lambda = compute_lambda(hcore, df_helper)
        if pyscf_mf.cell.spin == 0:
            approx_eris = build_approximate_eris(cc_inst, df_helper, eris=approx_eris)
        else:
            approx_eris = build_approximate_eris_rohf(cc_inst, df_helper, eris=approx_eris)
        approx_energy, _, _ = energy_function(approx_eris)
        df_res_cost = cost_double_factorization(
            num_spin_orbs,
            df_lambda.lambda_total,
            num_aux_df,
            df_lambda.num_eig,
            list(kmesh),
            chi=chi,
            beta=beta,
            dE_for_qpe=dE_for_qpe,
        )
        df_resource_obj.add_resources(
            ham_properties=df_lambda,
            resource_estimates=df_res_cost,
            cutoff=cutoff,
            approx_energy=np.real(approx_energy),
        )

    return df_resource_obj
