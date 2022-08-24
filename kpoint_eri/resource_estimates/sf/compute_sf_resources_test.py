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

    stps2 = kpoint_single_factorization_costs(n, lam, L, dE, chi, 20000)
    assert np.isclose(stps2[0], 33721)
    assert np.isclose(stps2[1], 162709658733)
    assert np.isclose(stps2[2], 7206)
