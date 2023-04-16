import numpy as np

from kpoint_eri.resource_estimates.thc.compute_thc_resources import _cost_thc, cost_thc


def test_thc_resources():
    lam = 307.68
    dE = 0.001
    n = 108
    chi = 10
    beta = 16
    M = 350

    res = _cost_thc(n, lam, dE, chi, beta, M, 1, 1, 1, 20_000)
    # print(res) # 26205, 12664955115, 2069
    print(res)  # (80098, 38711603694, 17630)
    assert np.isclose(res[0], 80098)
    assert np.isclose(res[1], 38711603694)
    assert np.isclose(res[2], 17630)

    res = _cost_thc(n, lam, dE, chi, beta, M, 3, 3, 3, 20_000)
    # print(res)  # {205788, 99457957764, 78813
    print(res)  # (270394, 130682231382, 78815)
    assert np.isclose(res[0], 270394)
    assert np.isclose(res[1], 130682231382)
    assert np.isclose(res[2], 78815)

    res = _cost_thc(n, lam, dE, chi, beta, M, 3, 5, 1, 20_000)
    # print(res)  # 151622, 73279367466, 39628
    print(res)  #  (202209, 97728216327, 77517)
    assert np.isclose(res[0], 202209)
    assert np.isclose(res[1], 97728216327)
    assert np.isclose(res[2], 77517)


def test_thc_resources_helper():
    lam = 307.68
    dE = 0.001
    n = 108
    chi = 10
    beta = 16
    M = 350

    res = cost_thc(
        num_spin_orbs=n,
        lambda_tot=lam,
        thc_dim=M,
        kmesh=[1, 1, 1],
        dE_for_qpe=dE,
        chi=chi,
        beta=beta,
    )
    assert np.isclose(res.toffolis_per_step, 80098)
    assert np.isclose(res.total_toffolis, 38711603694)
    assert np.isclose(res.logical_qubits, 17630)

    res = cost_thc(
        num_spin_orbs=n,
        lambda_tot=lam,
        thc_dim=M,
        kmesh=[3, 3, 3],
        dE_for_qpe=dE,
        chi=chi,
        beta=beta,
    )
    assert np.isclose(res.toffolis_per_step, 270394)
    assert np.isclose(res.total_toffolis, 130682231382)
    assert np.isclose(res.logical_qubits, 78815)

    res = cost_thc(
        num_spin_orbs=n,
        lambda_tot=lam,
        thc_dim=M,
        kmesh=[3, 5, 1],
        dE_for_qpe=dE,
        chi=chi,
        beta=beta,
    )
    assert np.isclose(res.toffolis_per_step, 202209)
    assert np.isclose(res.total_toffolis, 97728216327)
    assert np.isclose(res.logical_qubits, 77517)
