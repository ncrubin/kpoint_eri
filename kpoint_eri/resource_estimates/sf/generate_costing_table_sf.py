from dataclasses import dataclass, field
import json
from functools import reduce

import pandas as pd
import numpy as np

from pyscf.pbc import scf, mp, cc
from pyscf.pbc.mp.kmp2 import _add_padding
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

from kpoint_eri.resource_estimates.utils.misc_utils import PBCResources
from kpoint_eri.resource_estimates.sf.integral_helper_sf import (
    SingleFactorizationHelper,
)
from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_cc
from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
from kpoint_eri.resource_estimates.sf.compute_lambda_sf import compute_lambda
from kpoint_eri.resource_estimates.sf.compute_sf_resources import (
    cost_single_factorization,
)


@dataclass
class SFResources(PBCResources):
    num_aux: list = field(default_factory=list)

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
        num_aux: int,
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
        self.num_aux.append(num_aux)


def generate_costing_table(
    pyscf_mf: scf.HF,
    cutoffs: np.ndarray,
    name="pbc",
    chi: int = 10,
    dE_for_qpe=0.0016,
    write_to_file=True,
) -> pd.DataFrame:
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

    sf_resource_obj = SFResources(
        system_name=name,
        num_spin_orbitals=num_spin_orbs,
        num_kpts=num_kpts,
        dE=dE_for_qpe,
        chi=chi,
        exact_emp2=exact_emp2,
    )
    naux = Luv[0, 0].shape[0]
    for cutoff in cutoffs:
        naux_cutoff = max(int(cutoff * naux), 1)
        sf_helper = SingleFactorizationHelper(
            cholesky_factor=Luv, kmf=pyscf_mf, naux=naux_cutoff
        )

        sf_lambda_tot, sf_lambda_one_body, sf_lambda_two_body = compute_lambda(
            hcore_mo, sf_helper
        )

        L = (
            sf_helper.naux
        )  # No factor of 2 accounting for A and B because they use the same data
        sf_res_cost = cost_single_factorization(
            n=num_spin_orbs,
            lam=sf_lambda_tot,
            M=L,
            dE=dE_for_qpe,
            chi=chi,
            stps=20000,
            Nkx=kmesh[0],
            Nky=kmesh[0],
            Nkz=kmesh[0],
        )
        sf_res_cost = cost_single_factorization(
            n=num_spin_orbs,
            lam=sf_lambda_tot,
            M=L,
            dE=dE_for_qpe,
            chi=chi,
            stps=sf_res_cost[0],
            Nkx=kmesh[0],
            Nky=kmesh[0],
            Nkz=kmesh[0],
        )
        approx_cc = cc.KRCCSD(pyscf_mf)
        approx_cc.verbose = 0
        approx_cc = build_cc(approx_cc, sf_helper)
        eris = approx_cc.ao2mo(lambda x: x)
        approx_emp2, _, _ = approx_cc.init_amps(eris)

        sf_resource_obj.add_resources(
            lambda_tot=sf_lambda_tot,
            lambda_one_body=sf_lambda_one_body,
            lambda_two_body=sf_lambda_two_body,
            cutoff=cutoff,
            num_aux=naux_cutoff,
            toffolis_per_step=sf_res_cost[0],
            total_toffolis=sf_res_cost[1],
            logical_qubits=sf_res_cost[2],
            mp2_energy=approx_emp2,
        )

    df = pd.DataFrame(sf_resource_obj.dict())
    if write_to_file:
        df.to_csv(f"{name}_sf_num_kpts_{num_kpts}.csv")

    return df 
