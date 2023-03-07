from dataclasses import dataclass
from functools import reduce
from typing import Union

import pandas as pd
import numpy as np

from pyscf.pbc import scf, mp, cc
from pyscf.pbc.mp.kmp2 import _add_padding
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

from kpoint_eri.resource_estimates.utils.misc_utils import ResourcesHelper
from kpoint_eri.resource_estimates.sparse.integral_helper_sparse import (
    SparseFactorizationHelper,
)
from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_cc
from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
from kpoint_eri.resource_estimates.sparse.compute_lambda_sparse import compute_lambda
from kpoint_eri.resource_estimates.sparse.compute_sparse_resources import cost_sparse


@dataclass
class SparseResources(ResourcesHelper):
    chi: int = 10


def generate_costing_table(
    pyscf_mf: scf.HF,
    name="pbc",
    chi: int = 10,
    thresholds: np.ndarray = np.logspace(-1, -5, 6),
    dE_for_qpe=0.0016,
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

    sparse_resource_obj = SparseResources(
        system_name=name,
        num_spin_orbitals=num_spin_orbs,
        num_kpts=num_kpts,
        dE=dE_for_qpe,
        chi=chi,
        exact_emp2=exact_emp2
    )
    for thresh in thresholds:
        sparse_helper = SparseFactorizationHelper(
            cholesky_factor=Luv, kmf=pyscf_mf, threshold=thresh
        )
        approx_cc = cc.KRCCSD(pyscf_mf)
        approx_cc.verbose = 0
        approx_cc = build_cc(approx_cc, sparse_helper)
        eris = approx_cc.ao2mo(lambda x: x)
        approx_emp2, _, _ = approx_cc.init_amps(eris)

        (
            sparse_lambda_tot,
            sparse_lambda_one_body,
            sparse_lambda_two_body,
            num_nnz,
        ) = compute_lambda(hcore_mo, sparse_helper)

        sparse_res_cost = cost_sparse(
            n=num_spin_orbs,
            lam=sparse_lambda_tot,
            d=num_nnz,
            dE=dE_for_qpe,
            chi=chi,
            stps=20_000,
            Nkx=kmesh[0],
            Nky=kmesh[1],
            Nkz=kmesh[2],
        )
        sparse_res_cost = cost_sparse(
            n=num_spin_orbs,
            lam=sparse_lambda_tot,
            d=num_nnz,
            dE=dE_for_qpe,
            chi=chi,
            stps=sparse_res_cost[0],
            Nkx=kmesh[0],
            Nky=kmesh[1],
            Nkz=kmesh[2],
        )
        sparse_resource_obj.add_resources(
            lambda_tot=sparse_lambda_tot,
            lambda_one_body=sparse_lambda_one_body,
            lambda_two_body=sparse_lambda_two_body,
            number_of_sym_unique_terms=num_nnz,
            toffolis_per_step=sparse_res_cost[0],
            total_toffolis=sparse_res_cost[1],
            logical_qubits=sparse_res_cost[2],
            threshold=thresh,
            mp2_energy=approx_emp2,
        )

    return pd.DataFrame(sparse_resource_obj.dict())