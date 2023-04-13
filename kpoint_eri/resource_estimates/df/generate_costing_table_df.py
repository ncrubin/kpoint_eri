from dataclasses import dataclass, field
from functools import reduce

import pandas as pd
import numpy as np

from pyscf.pbc import scf, mp, cc
from pyscf.pbc.mp.kmp2 import _add_padding
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

from kpoint_eri.resource_estimates.utils.misc_utils import PBCResources
from kpoint_eri.resource_estimates.df.integral_helper_df import DFABKpointIntegrals
from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_approximate_eris
from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
from kpoint_eri.resource_estimates.df.compute_lambda_df import compute_lambda
from kpoint_eri.resource_estimates.df.compute_df_resources import (
    compute_cost,
)


@dataclass
class DFResources(PBCResources):
    beta: int = 20
    num_aux: list = field(default_factory=list)
    num_eig: list = field(default_factory=list)

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
        num_eig: int,
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
        self.num_eig.append(num_eig)


def generate_costing_table(
    pyscf_mf: scf.HF,
    cutoffs: np.ndarray,
    name="pbc",
    chi: int = 10,
    beta: int = 20,
    dE_for_qpe: float=0.0016,
) -> pd.DataFrame:
    kmesh = kpts_to_kmesh(pyscf_mf.cell, pyscf_mf.kpts)

    cc_inst = cc.KRCCSD(pyscf_mf)
    cc_inst.verbose = 0
    exact_eris = cc_inst.ao2mo()
    exact_emp2, _, _ = cc_inst.init_amps(exact_eris)

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

    num_kpts = np.prod(kmesh)

    df_resource_obj = DFResources(
        system_name=name,
        num_spin_orbitals=num_spin_orbs,
        num_kpts=num_kpts,
        dE=dE_for_qpe,
        chi=chi,
        beta=beta,
        exact_emp2=exact_emp2,
    )
    naux = Luv[0, 0].shape[0]
    # Save some space and overwrite eris object from exact CC 
    approx_eris = exact_eris
    for cutoff in cutoffs:
        df_helper = DFABKpointIntegrals(cholesky_factor=Luv, kmf=pyscf_mf)
        df_helper.double_factorize(thresh=cutoff)
        (
            df_lambda_tot,
            df_lambda_one_body,
            df_lambda_two_body,
            num_eigs,
        ) = compute_lambda(hcore_mo, df_helper)
        L = df_helper.naux * 2 * df_helper.nk  # factor of 2 accounts for A and B terms
        df_res_cost = compute_cost(
            n=num_spin_orbs,
            lam=df_lambda_tot,
            dE=dE_for_qpe,
            L=L,
            Lxi=num_eigs,
            chi=chi,
            beta=beta,
            Nkx=kmesh[0],
            Nky=kmesh[1],
            Nkz=kmesh[2],
            stps=20_000,
        )
        df_res_cost = compute_cost(
            n=num_spin_orbs,
            lam=df_lambda_tot,
            dE=dE_for_qpe,
            L=L,
            Lxi=num_eigs,
            chi=chi,
            beta=beta,
            Nkx=kmesh[0],
            Nky=kmesh[1],
            Nkz=kmesh[2],
            stps=df_res_cost[0],
        )
        approx_eris = build_approximate_eris(cc_inst, df_helper, eris=approx_eris)
        approx_emp2, _, _ = cc_inst.init_amps(approx_eris)
        df_resource_obj.add_resources(
            lambda_tot=df_lambda_tot,
            lambda_one_body=df_lambda_one_body,
            lambda_two_body=df_lambda_two_body,
            cutoff=cutoff,
            num_aux=naux,
            num_eig=num_eigs,
            toffolis_per_step=df_res_cost[0],
            total_toffolis=df_res_cost[1],
            logical_qubits=df_res_cost[2],
            mp2_energy=approx_emp2,
        )

    df = pd.DataFrame(df_resource_obj.dict())

    return df 
