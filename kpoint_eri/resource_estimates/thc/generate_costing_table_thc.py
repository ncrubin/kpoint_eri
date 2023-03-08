from dataclasses import dataclass, field
from functools import reduce
from typing import Union

import pandas as pd
import numpy as np

from pyscf.pbc import scf, mp, cc
from pyscf.pbc.mp.kmp2 import _add_padding
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

from kpoint_eri.factorizations.thc_jax import kpoint_thc_via_isdf
from kpoint_eri.resource_estimates.utils.misc_utils import PBCResources
from kpoint_eri.resource_estimates.thc.integral_helper import (
    KPTHCHelperDoubleTranslation,
)
from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_cc
from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
from kpoint_eri.resource_estimates.thc.compute_lambda_thc import compute_lambda
from kpoint_eri.resource_estimates.thc.compute_thc_resources import (
    compute_cost,
)


@dataclass
class THCResources(PBCResources):
    beta: int = 20
    num_thc: list = field(default_factory=list)

    def add_resources(
        self,
        lambda_tot: float,
        lambda_one_body: float,
        lambda_two_body: float,
        toffolis_per_step: float,
        total_toffolis: float,
        logical_qubits: float,
        cutoff: float,
        mp2_energy: float,
        num_thc: int,
    ) -> None:
        super().add_resources(
            lambda_tot,
            lambda_one_body,
            lambda_two_body,
            toffolis_per_step,
            total_toffolis,
            logical_qubits,
            cutoff,
            mp2_energy,
        )
        self.num_thc.append(num_thc)


def generate_costing_table(
    pyscf_mf: scf.HF,
    thc_rank_params: np.ndarray,
    name="pbc",
    chi: int = 10,
    beta: int = 20,
    dE_for_qpe: float = 0.0016,
    reoptimize: bool = True,
    bfgs_maxiter: int = 3000,
    adagrad_maxiter: int = 3000,
    fft_df_mesh: Union[None, list] = None,
):
    kmesh = kpts_to_kmesh(pyscf_mf.cell, pyscf_mf.kpts)

    exact_cc = cc.KRCCSD(pyscf_mf)
    exact_cc.verbose = 0
    eris = exact_cc.ao2mo()
    exact_emp2, _, _ = exact_cc.init_amps(eris)

    mp2_inst = mp.KMP2(pyscf_mf)
    Luv = cholesky_from_df_ints(mp2_inst)  # [kpt, kpt, naux, nmo_padded, nmo_padded]
    mo_coeff_padded = _add_padding(mp2_inst, mp2_inst.mo_coeff, mp2_inst.mo_energy)[0]

    # get hcore mo
    hcore_ao = pyscf_mf.get_hcore()
    hcore_mo = np.asarray(
        [
            reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo))
            for k, mo in enumerate(mo_coeff_padded)
        ]
    )
    num_spin_orbs = 2 * hcore_mo[0].shape[-1]

    ### SPARSE RESOURCE ESTIMATE ###
    num_kpts = np.prod(kmesh)

    thc_resource_obj = THCResources(
        system_name=name,
        num_spin_orbitals=num_spin_orbs,
        num_kpts=num_kpts,
        dE=dE_for_qpe,
        chi=chi,
        beta=beta,
        exact_emp2=exact_emp2,
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
    naux = Luv[0, 0].shape[0]
    for thc_rank in thc_rank_params:
        num_thc = thc_rank * num_spin_orbs // 2
        kpt_thc, loss = kpoint_thc_via_isdf(
            mf_fftdf,
            Luv,
            num_thc,
            perform_adagrad_opt=reoptimize,
            perform_bfgs_opt=reoptimize,
            bfgs_maxiter=bfgs_maxiter,
            adagrad_maxiter=adagrad_maxiter,
        )
        thc_helper = KPTHCHelperDoubleTranslation(
            kpt_thc.chi, kpt_thc.zeta, pyscf_mf, chol=Luv
        )
        thc_lambda_tot, thc_lambda_one_body, thc_lambda_two_body = compute_lambda(
            hcore_mo, thc_helper
        )
        kmesh = kpts_to_kmesh(pyscf_mf.cell, pyscf_mf.kpts)
        thc_res_cost = compute_cost(
            n=num_spin_orbs,
            lam=thc_lambda_tot,
            dE=dE_for_qpe,
            chi=chi,
            beta=beta,
            M=kpt_thc.chi.shape[-1],
            Nkx=kmesh[0],
            Nky=kmesh[1],
            Nkz=kmesh[2],
            stps=20_000,
        )
        approx_cc = cc.KRCCSD(pyscf_mf)
        approx_cc.verbose = 0
        approx_cc = build_cc(approx_cc, thc_helper)
        eris = approx_cc.ao2mo(lambda x: x)
        approx_emp2, _, _ = approx_cc.init_amps(eris)
        thc_resource_obj.add_resources(
            lambda_tot=thc_lambda_tot,
            lambda_one_body=thc_lambda_one_body,
            lambda_two_body=thc_lambda_two_body,
            cutoff=thc_rank,
            toffolis_per_step=thc_res_cost[0],
            total_toffolis=thc_res_cost[1],
            logical_qubits=thc_res_cost[2],
            mp2_energy=approx_emp2,
            num_thc=num_thc,
        )

    return pd.DataFrame(thc_resource_obj.dict())
