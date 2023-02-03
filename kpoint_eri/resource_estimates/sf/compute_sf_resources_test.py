import numpy as np
from kpoint_eri.resource_estimates.sf.compute_sf_resources import kpoint_single_factorization_costs, QR2, QI2

def test_qr2():
    L = 728
    npp = 182
    bpp = 21
    test_val = QR2(L + 1, npp, bpp)
    assert np.isclose(test_val, 3416)

    L = 56
    npp = 28
    bpp = 91
    test_val = QR2(L + 1, npp, bpp)
    assert np.isclose(test_val, 679)

def test_qi2():
    L1 = 728
    npp = 182
    test_val = QI2(L1 + 1, npp)
    assert np.isclose(test_val, 785)

    L1 = 56
    npp = 28
    test_val = QI2(L1 + 1, npp)
    assert np.isclose(test_val, 88)

def test_estimate():
    n = 152
    lam = 3071.8
    L = 275
    dE = 0.001
    chi = 10
    
    res = kpoint_single_factorization_costs(n, lam, L, dE, chi, 20_000, 3, 3, 3)
    # 1663687, 8027577592851, 438447}
    assert np.isclose(res[0], 1663687)
    assert np.isclose(res[1], 8027577592851)
    assert np.isclose(res[2], 438447)

    res = kpoint_single_factorization_costs(n, lam, L, dE, chi, 20_000, 3, 5, 1)
    # 907828, 4380427154244, 219526
    assert np.isclose(res[0], 907828)
    assert np.isclose(res[1], 4380427154244)
    assert np.isclose(res[2], 219526)

def test_carbon_multikpoint():
    from functools import reduce
    from pyscf.pbc import gto, scf, mp, cc
    import h5py
    from tqdm import tqdm
    
    from kpoint_eri.resource_estimates.sf.ncr_integral_helper import NCRSingleFactorizationHelper
    from kpoint_eri.factorizations.pyscf_chol_from_df import cholesky_from_df_ints
    from kpoint_eri.resource_estimates.cc_helper.cc_helper import build_cc
    from kpoint_eri.resource_estimates.sf.compute_sf_resources import kpoint_single_factorization_costs
    from kpoint_eri.resource_estimates.sf.compute_lambda_sf import compute_lambda_ncr2

    cell = gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-dzv'
    cell.pseudo = 'gth-hf-rev'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()

    kmesh = [1, 1, 2]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    # mf.chkfile = 'ncr_test_C_density_fitints.chk'
    # mf.with_df._cderi_to_save = 'ncr_test_C_density_fitints_gdf.h5'
    # mf.init_guess = 'chkfile'
    mf.kernel()

    exact_cc = cc.KRCCSD(mf)
    eris = exact_cc.ao2mo()
    exact_emp2, _, _ = exact_cc.init_amps(eris)
    mf.e_tot = mf.energy_tot()
    exact_emp2 += mf.e_tot
    import copy
    # get mo-one-body Hamiltonian
    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray(
        [
            reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo))
            for k, mo in enumerate(mf.mo_coeff)
        ]
    )
    mymp = mp.KMP2(mf)

    Luv = cholesky_from_df_ints(mymp)  # [kpt, kpt, naux, nao, nao]
    naux = Luv[0, 0].shape[0]
    nmo = mf.mo_coeff[0].shape[-1]

    print("nmo: ", nmo, naux)
    print(" naux  error (Eh)")
    naux_values = []
    thresh_values = []
    emp2_values = []
    delta_values = []
    lambda_tot_values = []
    lambda_one_body_values = []
    lambda_two_body_values = []
    num_spin_orbs = nmo * 2# * len(mf.kpts)
    num_kpts = len(mf.kpts)
    tof_per_iter = []
    total_tof = []
    total_logical_qubits = []

    dE = 0.0016
    chi = 10
    print(" naux  error (Eh)")
    thresholds = np.linspace(max(1, int(0.1*naux)), naux, 8, dtype=int)
    print(thresholds)
    for thresh in tqdm(thresholds):
        approx_cc = cc.KRCCSD(mf)
        approx_cc.max_memory = 1e7
        approx_cc.verbose = 0
        helper = NCRSingleFactorizationHelper(cholesky_factor=Luv,
                kmf=mf, naux=thresh)
        approx_cc = build_cc(approx_cc, helper)
        eris = approx_cc.ao2mo(lambda x: x)
        emp2, _, _ = approx_cc.init_amps(eris)  # correlation energy
        emp2 += mf.e_tot  # total mp2 energy
        delta = abs(emp2 - exact_emp2)

        thresh_values.append(thresh)
        emp2_values.append(emp2)
        delta_values.append(delta)

        lambda_tot, lambda_one_body, lambda_two_body = compute_lambda_ncr2(hcore_mo, helper)
        print(thresh, delta, lambda_tot, lambda_one_body, lambda_two_body)

        resource = kpoint_single_factorization_costs(2 * nmo, lambda_tot, len(kpts) * helper.naux * 2, Nk=len(kpts), chi=10, dE=dE, stps=20_000)
        resource = kpoint_single_factorization_costs(2 * nmo, lambda_tot, len(kpts) * helper.naux * 2, Nk=len(kpts), chi=10, dE=dE, stps=resource[0])
        tof_per_iter.append(resource[0])
        total_tof.append(resource[1])
        total_logical_qubits.append(resource[2])
        with h5py.File("mp2_chol_energy.h5", "w") as fid:
            fid.create_dataset(name="thresh_values", data=np.array(thresh_values))
            fid.create_dataset(name="emp2", data=np.array(emp2_values))
            fid.create_dataset(name="delta", data=np.array(delta_values))
            fid.create_dataset(name='toffoli_per_step', data=np.array(tof_per_iter))
            fid.create_dataset(name='total_toffoli', data=np.array(total_tof))
            fid.create_dataset(name='total_qubits', data=np.array(total_logical_qubits))


if __name__ == "__main__":
    test_qr2()
    test_qi2()
    test_estimate()
#     # 4 1.3687733767446255 2 1 0.0016 10 (n, lambda, M, nkpts, de, chi)
#     res = kpoint_single_factorization_costs(n=4, lam=1.3687733767446255, M=2, Nk=1, dE=0.0016, chi=10, stps=20_000)
# 
#     # test_carbon_multikpoint()
# 
#     import h5py
#     with h5py.File("mp2_chol_energy.h5", 'r') as fid:
#         print(fid.keys())
#         print(fid['delta'][...])
#         print(fid['emp2'][...])
#         print(fid['thresh_values'][...])
#         print(fid['toffoli_per_step'][...])