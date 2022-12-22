"""
In this example we load a kpoint chkfile and calculate the 
resources required for sparse, single, and double factorizations

Sparse thresholds for eri integral value cutoff
Mean cutoff values for k-point: 1.085e-03 +- 9.165e-04
Mean cutoff values for supercell: 6.321e-04 +- 1.311e-03
so safe value is 1.0E-4

SF thresholds for c where  naux is c * nmo * 2 (2 because of spin orbitals)
Mean c values for k-point: 1.944e+00 +- 6.990e-01 # these are L / num_spin_orbital
Mean c values for supercell: 2.081e+00 +- 7.939e-01 #htes are L / num_spin_orbtial
so safe value is c = 2.5. We use c * nmo = naux so c = 5 in this case.

DF Threshold for second factorization
Mean cutoff values for k-point: 8.000e-03 +- 1.735e-18
Mean cutoff values for supercell: 8.000e-03 +- 0.000e+00
this is the cutoff for the second factorization.  
so safe value is 1.0E-3.

"""
from dataclasses import dataclass, asdict
from typing import List
import sys
import os
import time
from itertools import product
from functools import reduce
import h5py
import numpy as np
import pandas as pd
from pyscf.pbc import gto, cc, scf, mp
from pyscf.pbc.mp.kmp2 import _add_padding
from pyscf.pbc.scf.chkfile import load_scf
from pyscf.pbc.tools.k2gamma import k2gamma
from pyscf import ao2mo

from kpoint_eri.resource_estimates.sparse.ncr_sparse_integral_helper import (
    NCRSSparseFactorizationHelper,
)
from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_cc
from kpoint_eri.resource_estimates.sparse.compute_lambda_sparse import (
    compute_lambda_ncr as compute_lambda_sparse
)
from kpoint_eri.resource_estimates.sparse.compute_sparse_resources import (
    cost_sparse,
)

from kpoint_eri.resource_estimates.sf.ncr_integral_helper import (
    NCRSingleFactorizationHelper,
)
from kpoint_eri.resource_estimates.sf.compute_lambda_sf import (
    compute_lambda_ncr2 as compute_lambda_sf
)
from kpoint_eri.resource_estimates.sf.compute_sf_resources import kpoint_single_factorization_costs

from kpoint_eri.resource_estimates.df.ncr_integral_helper_df import (
    DFABV2KpointIntegrals,
)
from kpoint_eri.resource_estimates.df.compute_lambda_df import (
    compute_lambda_ncr_v2 as compute_lambda_df
)
from kpoint_eri.resource_estimates.df.compute_df_resources import (
    compute_cost as kpoint_df_costs
)

def initialize_scf():
    kmesh = [1, 1, 3]
    cell = gto.M(
        unit='B',
        a=[[0., 3.37013733, 3.37013733],
           [3.37013733, 0., 3.37013733],
           [3.37013733, 3.37013733, 0.]],
        atom='''C 0 0 0
                  C 1.68506866 1.68506866 1.68506866''',
        basis='gth-szv',
        pseudo='gth-hf-rev',
        verbose=4
    )
    cell.build()
    
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts = kpts, exxdiv=None).rs_density_fit()
    mf.chkfile = 'ncr_c2.chk'
    mf.init_guess = 'chkfile'
    mf.kernel()

@dataclass
class SparseResources:
    system_name: str
    num_spin_orbitals: int
    nkpts: int
    lambda_tot: float
    lambda_one_body: float
    lambda_two_body: float
    number_of_sym_unique_terms: int
    toffolis_per_step: int
    total_toffolis: int
    logical_qubits: int
    dE: float
    chi: float
    threshold: float
    mp2_energy: None

    dict = asdict

@dataclass
class SingleFactorizationResources:
    system_name: str
    num_spin_orbitals: int
    nkpts: int
    lambda_tot: float
    lambda_one_body: float
    lambda_two_body: float
    toffolis_per_step: int
    total_toffolis: int
    logical_qubits: int
    dE: float
    chi: float
    naux: float
    mp2_energy: None

    dict = asdict

@dataclass
class DoubleFactorizationResources:
    system_name: str
    num_spin_orbitals: int
    nkpts: int
    lambda_tot: float
    lambda_one_body: float
    lambda_two_body: float
    toffolis_per_step: int
    total_toffolis: int
    logical_qubits: int
    dE: float
    chi: float
    second_factor_threshold: float
    L: float
    Lxi: float
    mp2_energy: None

    dict = asdict


def get_resources(chkfile, sys_name, rs_gdf_h5file=None, sparse_threshold=1.0E-4, sf_nmo_multiplier=5.0, df_thresh=1.0E-3):
    """
    Calculate k-point qubitization resource estimates using sparse, single factorization, and double factorization LCUs

    :param chkfile: pyscf chkfile for kpoint RHF or ROHF.
    :param sys_name: system name. Recommended that this include kpoint mesh used.
    :param rs_gdf_h5file: Optional h5 file for range separated file. This gets loaded into kmf.with_df._cderi 
    :param sparse_threshold: Optional 1.0E-4 sets the value for sparse threshold
    :param sf_nmo_multiplier: Optional 5.0 sets the value for the naux value for single factorization as a function
                              of the number of bandss.
    :param df_thresh: Optional 1.0E-3 value for second factorization for DF LCU.
    :returns: three Dataclasses containing resource estimates. sparse, sf, df (in that order). 
    """
    cell, scf_dict = load_scf(chkfile)
    kpts = scf_dict['kpts']
    kmf = scf.KRHF(cell, kpts=kpts).rs_density_fit()  # I assume this dispatches to ROHF appropriately
    if rs_gdf_h5file is not None:
        if os.path.isfile(rs_gdf_h5file):
            kmf.with_df._cderi = rs_gdf_h5file
    kmf.max_memory = 118000
    kmf.e_tot = scf_dict["e_tot"]
    kmf.mo_coeff = scf_dict["mo_coeff"]
    kmf.mo_energy = scf_dict["mo_energy"]
    kmf.mo_occ = scf_dict["mo_occ"]

    mymp = mp.KMP2(kmf)
    Luv = cholesky_from_df_ints(
        mymp
    )  # [kpt, kpt, naux, nmo_padded, nmo_padded]
    naux = Luv[0, 0].shape[0]
    mo_coeff_padded = _add_padding(mymp, mymp.mo_coeff, mymp.mo_energy)[0]

    # get hcore mo
    hcore_ao = kmf.get_hcore()
    hcore_mo = np.asarray(
        [
            reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo))
            for k, mo in enumerate(mo_coeff_padded)
        ]
    )
    num_spin_orbs = 2 * hcore_mo[0].shape[-1]
    nmo = hcore_mo[0].shape[-1]
    dE_for_qpe = 0.0016 # 1 kcal/mol[]
    chi = 10

    ### SPARSE RESOURCE ESTIMATE ###
    start_time = time.time()
    sparse_helper = NCRSSparseFactorizationHelper(cholesky_factor=Luv, kmf=kmf, threshold=sparse_threshold)
    end_time = time.time()
    print("Time to initialize SparseFactorizationHelper ", end_time - start_time)

    start_time = time.time()
    sparse_lambda_tot, sparse_lambda_one_body, sparse_lambda_two_body, num_nnz = compute_lambda_sparse(
        hcore_mo, sparse_helper
    )
    end_time = time.time()
    print("Time to calculate sparse lambda ", end_time - start_time)

    sparse_res_cost = cost_sparse(
            n=num_spin_orbs, Nk=len(kmf.kpts), lam=sparse_lambda_tot, d=num_nnz, dE=dE_for_qpe, chi=chi, stps=20_000
    )
    sparse_res_cost = cost_sparse(
            n=num_spin_orbs, Nk=len(kmf.kpts), lam=sparse_lambda_tot, d=num_nnz, dE=dE_for_qpe, chi=chi, stps=sparse_res_cost[0]
    )
    sparse_resource_obj = SparseResources(system_name='{}_sparse'.format(sys_name),
                                          num_spin_orbitals=num_spin_orbs,
                                          nkpts=len(kmf.kpts),
                                          lambda_tot=sparse_lambda_tot,
                                          lambda_one_body=sparse_lambda_one_body,
                                          lambda_two_body=sparse_lambda_two_body,
                                          number_of_sym_unique_terms=num_nnz,
                                          toffolis_per_step=sparse_res_cost[0],
                                          total_toffolis=sparse_res_cost[1],
                                          logical_qubits=sparse_res_cost[2],
                                          dE=dE_for_qpe,
                                          chi=chi,
                                          threshold=sparse_threshold,
                                          mp2_energy=None
                                        )

    ### SF Costs ###
    start_time = time.time()
    sf_helper = NCRSingleFactorizationHelper(cholesky_factor=Luv, 
                                             kmf=kmf, 
                                             naux=int(np.ceil(sf_nmo_multiplier * nmo))
                                             )
    end_time = time.time()
    print("Time to initialize the SingleFactorizationHelper ", end_time - start_time)

    start_time = time.time()
    sf_lambda_tot, sf_lambda_one_body, sf_lambda_two_body = compute_lambda_sf(hcore_mo, sf_helper)
    end_time = time.time()
    print("Time to compute lambda for SF ", end_time - start_time)

    L = sf_helper.naux # No factor of 2 accounting for A and B because they use the same data
    sf_res_cost = kpoint_single_factorization_costs(
        n=num_spin_orbs, lam=sf_lambda_tot, M=L, Nk=(len(kmf.kpts)), dE=dE_for_qpe, chi=chi, stps=20000
    )
    sf_res_cost = kpoint_single_factorization_costs(
        n=num_spin_orbs, lam=sf_lambda_tot, M=L, Nk=(len(kmf.kpts)), dE=dE_for_qpe, chi=chi, stps=sf_res_cost[0]
    )
    sf_resource_obj = SingleFactorizationResources(system_name='{}_sf'.format(sys_name),
                                                   num_spin_orbitals=num_spin_orbs,
                                                   nkpts=len(kmf.kpts),
                                                   lambda_tot=sf_lambda_tot,
                                                   lambda_one_body=sf_lambda_one_body,
                                                   lambda_two_body=sf_lambda_two_body,
                                                   toffolis_per_step=sf_res_cost[0],
                                                   total_toffolis=sf_res_cost[1],
                                                   logical_qubits=sf_res_cost[2],
                                                   dE=dE_for_qpe,
                                                   chi=chi,
                                                   naux=L,
                                                   mp2_energy=None
                                                   )


    ### DF Costs ###
    start_time = time.time()
    df_helper = DFABV2KpointIntegrals(cholesky_factor=Luv, kmf=kmf)
    df_helper.double_factorize(thresh=df_thresh)
    end_time = time.time()
    print("Time to initialize and factorize the DoubleFactorizationHelper ", end_time - start_time)

    beta = 20
    num_kpts = df_helper.nk
    nk = max(1, np.ceil(np.log2(num_kpts)))

    start_time = time.time()
    df_lambda_tot, df_lambda_one_body, df_lambda_two_body, num_eigs = compute_lambda_df(
        hcore_mo, df_helper
    )
    end_time = time.time()
    print("Time to compute lambda for DF ", end_time - start_time)

    L = df_helper.naux * 2 * df_helper.nk # factor of 2 accounts for A and B terms
    Lxi = num_eigs
    df_res_cost = kpoint_df_costs(
        n=num_spin_orbs, lam=df_lambda_tot, dE=dE_for_qpe, L=L,
        Lxi=num_eigs, chi=chi, beta=beta, Nk=df_helper.nk, nk=nk, stps=20000
    )
    df_res_cost = kpoint_df_costs(
        n=num_spin_orbs, lam=df_lambda_tot, dE=dE_for_qpe, L=L,
        Lxi=num_eigs, chi=chi, beta=beta, Nk=df_helper.nk, nk=nk, stps=df_res_cost[0]
    )
    df_resource_obj = DoubleFactorizationResources(system_name='{}_df'.format(sys_name),
                                                   num_spin_orbitals=num_spin_orbs,
                                                   nkpts=len(kmf.kpts),
                                                   lambda_tot=sf_lambda_tot,
                                                   lambda_one_body=sf_lambda_one_body,
                                                   lambda_two_body=sf_lambda_two_body,
                                                   toffolis_per_step=df_res_cost[0],
                                                   total_toffolis=df_res_cost[1],
                                                   logical_qubits=df_res_cost[2],
                                                   dE=dE_for_qpe,
                                                   chi=chi,
                                                   second_factor_threshold=df_thresh,
                                                   L=L,
                                                   Lxi=num_eigs,
                                                   mp2_energy=None
                                                   )
    return sparse_resource_obj, sf_resource_obj, df_resource_obj



if __name__ == "__main__":
    TEST_RUN = True
    ### Setting up run. You won't need this if for your resource estimates ###
    if TEST_RUN:
        if not os.path.isfile('ncr_c2.chk'):
            initialize_scf() # this sets up a sample scf 

    ### START HERE IF RUNNING IN PRODUCTION AND SET TEST_RUN = False ###
    scf_chkfile_name = 'ncr_c2.chk' # name of LNO system goes here
    sys_name = 'C2'
    sparse_resource_obj, sf_resource_obj, df_resource_obj = get_resources(scf_chkfile_name, sys_name=sys_name)
    ### SAVE THE OBJECTS using dict() to JSON dump ###
    import json
    with open('{}_sparse.json'.format(sys_name), 'w') as fid:
        json.dump(sparse_resource_obj.dict(), fp=fid)
    with open('{}_sf.json'.format(sys_name), 'w') as fid:
        json.dump(sf_resource_obj.dict(), fp=fid)
    with open('{}_df.json'.format(sys_name), 'w') as fid:
        json.dump(df_resource_obj.dict(), fp=fid)

    if TEST_RUN:
        with open('{}_sparse.json'.format(sys_name), 'r') as fid:
            test_sparse_resource_dict = json.load(fid)
        with open('{}_sf.json'.format(sys_name), 'r') as fid:
            test_sf_resource_dict = json.load(fid)
        with open('{}_df.json'.format(sys_name), 'r') as fid:
            test_df_resource_dict = json.load(fid)

        assert test_sparse_resource_dict == sparse_resource_obj.dict()
        assert test_sf_resource_dict == sf_resource_obj.dict()
        assert test_df_resource_dict == df_resource_obj.dict()
