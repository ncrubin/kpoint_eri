import numpy as np

from kpoint_eri.resource_estimates.thc.compute_thc_resources import compute_cost

def test_thc_resources():
    lam = 307.68
    dE = 0.001
    n = 108
    chi = 10
    beta = 16
    M = 350

    res = compute_cost(n, lam, dE, chi, beta, M, 1, 1, 1, 20_000)
    # print(res) # 26205, 12664955115, 2069
    print(res)  # (80098, 38711603694, 17630)
    assert np.isclose(res[0], 80098)
    assert np.isclose(res[1], 38711603694)
    assert np.isclose(res[2], 17630)

    res = compute_cost(n, lam, dE, chi, beta, M, 3, 3, 3, 20_000)
    # print(res)  # {205788, 99457957764, 78813
    print(res)  # (270394, 130682231382, 78815)
    assert np.isclose(res[0], 270394)
    assert np.isclose(res[1], 130682231382)
    assert np.isclose(res[2], 78815)

    res = compute_cost(n, lam, dE, chi, beta, M, 3, 5, 1, 20_000)
    # print(res)  # 151622, 73279367466, 39628
    print(res)  #  (202209, 97728216327, 77517)
    assert np.isclose(res[0], 202209)
    assert np.isclose(res[1], 97728216327)
    assert np.isclose(res[2], 77517)