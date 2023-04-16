import numpy as np
from kpoint_eri.resource_estimates.sf.compute_sf_resources import (
    _cost_single_factorization,
    cost_single_factorization,
    QR2,
    QI2,
)


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

    res = _cost_single_factorization(n, lam, L, dE, chi, 20_000, 3, 3, 3)
    # 1663687, 8027577592851, 438447}
    assert np.isclose(res[0], 1663687)
    assert np.isclose(res[1], 8027577592851)
    assert np.isclose(res[2], 438447)
    res = _cost_single_factorization(n, lam, L, dE, chi, res[0], 3, 3, 3)
    assert np.isclose(res[0], 1663707)
    assert np.isclose(res[1], 8027674096311)
    assert np.isclose(res[2], 438452)

    res = _cost_single_factorization(n, lam, L, dE, chi, 20_000, 3, 5, 1)
    # 907828, 4380427154244, 219526
    assert np.isclose(res[0], 907828)
    assert np.isclose(res[1], 4380427154244)
    assert np.isclose(res[2], 219526)
    res = _cost_single_factorization(n, lam, L, dE, chi, res[0], 3, 5, 1)
    assert np.isclose(res[0], 907828)
    assert np.isclose(res[1], 4380427154244)
    assert np.isclose(res[2], 219526)


def test_estimate_helper():
    n = 152
    lam = 3071.8
    L = 275
    dE = 0.001
    chi = 10

    res = cost_single_factorization(
        num_spin_orbs=n,
        lambda_tot=lam,
        num_aux=L,
        kmesh=[3, 3, 3],
        dE_for_qpe=dE,
        chi=chi,
    )
    assert np.isclose(res.toffolis_per_step, 1663707)
    assert np.isclose(res.total_toffolis, 8027674096311)
    assert np.isclose(res.logical_qubits, 438452)

    res = cost_single_factorization(
        num_spin_orbs=n,
        lambda_tot=lam,
        num_aux=L,
        kmesh=[3, 5, 1],
        dE_for_qpe=dE,
        chi=chi,
    )
    # 1663687, 8027577592851, 438447}
    assert np.isclose(res.toffolis_per_step, 907828)
    assert np.isclose(res.total_toffolis, 4380427154244)
    assert np.isclose(res.logical_qubits, 219526)


if __name__ == "__main__":
    test_qr2()
    test_qi2()
    test_estimate()
