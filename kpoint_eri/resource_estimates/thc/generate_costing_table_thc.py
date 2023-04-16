from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as npt

from pyscf.pbc import scf
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

from kpoint_eri.factorizations.thc_jax import kpoint_thc_via_isdf
from kpoint_eri.resource_estimates.utils.misc_utils import PBCResources
from kpoint_eri.resource_estimates.thc.integral_helper_thc import (
    KPTHCHelperDoubleTranslation,
)
from kpoint_eri.resource_estimates.cc_helper.cc_helper import (
    build_approximate_eris,
    build_cc_inst,
    build_approximate_eris_rohf,
)
from kpoint_eri.factorizations.hamiltonian_utils import build_hamiltonian
from kpoint_eri.resource_estimates.thc.compute_lambda_thc import compute_lambda
from kpoint_eri.resource_estimates.thc.compute_thc_resources import (
    cost_thc,
)


@dataclass
class THCResources(PBCResources):
    beta: int = 20


def generate_costing_table(
    pyscf_mf: scf.HF,
    thc_rank_params: Union[list, npt.NDArray],
    name="pbc",
    chi: int = 10,
    beta: int = 20,
    dE_for_qpe: float = 0.0016,
    reoptimize: bool = True,
    bfgs_maxiter: int = 3000,
    adagrad_maxiter: int = 3000,
    fft_df_mesh: Union[None, list] = None,
    energy_method: str = "MP2",
) -> THCResources:
    """Generate resource estimate costing table given a set of cutoffs for
        THC Hamiltonian.

    Arguments:
        pyscf_mf: k-point pyscf mean-field object
        cutoffs: Array of (integer) auxiliary index cutoff values
        name: Optional descriptive name for simulation.
        chi: the number of bits for the representation of the coefficients
        beta: the number of bits for rotations.
        dE_for_qpe: Phase estimation epsilon.
        reoptimize: Whether or not to perform regularized reoptimization of THC
            factors.
        bfgs_maxiter: Max number of BFGS steps.
        adagrad_maxiter: Max number of AdaGrad steps.
        fft_df_mesh: FFTDF mesh for ISDF.
    Returns
        resources: Table of resource estimates.
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
    thc_resource_obj = THCResources(
        system_name=name,
        num_spin_orbitals=num_spin_orbs,
        num_kpts=num_kpts,
        dE=dE_for_qpe,
        chi=chi,
        beta=beta,
        exact_energy=np.real(reference_energy),
    )

    # For the ISDF guess we need an FFTDF MF object (really just need the grids so a bit of a hack)
    # Have not checked carefully the sensitivity of ISDF to real space grid size
    # and just use that determined by pyscf, which is usually not too big. If
    # you find a ridiculously large FFT grid it might be necessary to set
    # mf_fftdf.cell.mesh = [40, 40, 40], or simimlar (I've tested up to ~ 50^3).
    # Since we're fitting to RSGDF it isn't very important what the value is and
    # there is some tradeoff (grid density vs comp time) which afaik has not
    # been carefully studied (or at least not published)
    # Subsequent optimization attempts to fit to the RSGDF integrals which
    # should hopefully be somewhat close to the FFTDF ones for ISDF to be a good starting point.
    mf_fftdf = scf.KRHF(pyscf_mf.cell, pyscf_mf.kpts)
    mf_fftdf.max_memory = 180000
    mf_fftdf.kpts = pyscf_mf.kpts
    mf_fftdf.e_tot = pyscf_mf.e_tot
    mf_fftdf.mo_coeff = pyscf_mf.mo_coeff
    mf_fftdf.mo_energy = pyscf_mf.mo_energy
    mf_fftdf.mo_occ = pyscf_mf.mo_occ
    if fft_df_mesh is not None:
        mf_fftdf.with_df.mesh = fft_df_mesh
    approx_eris = exact_eris
    for thc_rank in thc_rank_params:
        num_thc = thc_rank * num_spin_orbs // 2
        kpt_thc, _ = kpoint_thc_via_isdf(
            mf_fftdf,
            chol,
            num_thc,
            perform_adagrad_opt=reoptimize,
            perform_bfgs_opt=reoptimize,
            bfgs_maxiter=bfgs_maxiter,
            adagrad_maxiter=adagrad_maxiter,
        )
        thc_helper = KPTHCHelperDoubleTranslation(
            kpt_thc.chi, kpt_thc.zeta, pyscf_mf, chol=chol
        )
        thc_lambda = compute_lambda(hcore, thc_helper)
        kmesh = kpts_to_kmesh(pyscf_mf.cell, pyscf_mf.kpts)
        if pyscf_mf.cell.spin == 0:
            approx_eris = build_approximate_eris(cc_inst, thc_helper, eris=approx_eris)
        else:
            approx_eris = build_approximate_eris_rohf(cc_inst, thc_helper, eris=approx_eris)
        approx_energy, _, _ = energy_function(approx_eris)
        thc_res_cost = cost_thc(
            num_spin_orbs,
            thc_lambda.lambda_total,
            thc_rank * num_spin_orbs // 2,
            list(kmesh),
            chi=chi,
            beta=beta,
            dE_for_qpe=dE_for_qpe,
        )
        thc_resource_obj.add_resources(
            ham_properties=thc_lambda,
            resource_estimates=thc_res_cost,
            cutoff=thc_rank,
            approx_energy=np.real(approx_energy),
        )

    return thc_resource_obj
