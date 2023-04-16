import numpy as np

from kpoint_eri.resource_estimates.df.compute_df_resources import (
    _cost_double_factorization,
    cost_double_factorization,
)


def test_costing():
    nRe = 108
    lamRe = 294.8
    dE = 0.001
    LRe = 360
    LxiRe = 13031
    chi = 10
    betaRe = 16

    # (*The Li et al orbitals.*)
    nLi = 152
    lamLi = 1171.2
    LLi = 394
    LxiLi = 20115
    betaLi = 20

    res = _cost_double_factorization(
        nRe, lamRe, dE, LRe, LxiRe, chi, betaRe, 2, 2, 2, 20_000
    )
    res = _cost_double_factorization(
        nRe, lamRe, dE, LRe, LxiRe, chi, betaRe, 2, 2, 2, res[0]
    )
    # 48250, 22343175750, 8174
    assert np.isclose(res[0], 48250)
    assert np.isclose(res[1], 22343175750)
    assert np.isclose(res[2], 8174)

    res = _cost_double_factorization(
        nRe, lamRe, dE, LRe, LxiRe, chi, betaRe, 3, 5, 1, 20_000
    )
    res = _cost_double_factorization(
        nRe, lamRe, dE, LRe, LxiRe, chi, betaRe, 3, 5, 1, res[0]
    )
    # 53146, 24610371366, 8945
    assert np.isclose(res[0], 53146)
    assert np.isclose(res[1], 24610371366)
    assert np.isclose(res[2], 8945)

    res = _cost_double_factorization(
        nLi, lamLi, dE, LLi, LxiLi, chi, betaLi, 2, 2, 2, 20_000
    )
    res = _cost_double_factorization(
        nLi, lamLi, dE, LLi, LxiLi, chi, betaLi, 2, 2, 2, res[0]
    )
    # print(res) # 79212, 145727663004, 13873
    assert np.isclose(res[0], 79212)
    assert np.isclose(res[1], 145727663004)
    assert np.isclose(res[2], 13873)
    res = _cost_double_factorization(
        nLi, lamLi, dE, LLi, LxiLi, chi, betaLi, 3, 5, 1, res[0]
    )
    # print(res) # 86042, 158292930114, 14952
    assert np.isclose(res[0], 86042)
    assert np.isclose(res[1], 158292930114)
    assert np.isclose(res[2], 14952)


def test_costing_helper():
    nRe = 108
    lamRe = 294.8
    dE = 0.001
    LRe = 360
    LxiRe = 13031
    chi = 10
    betaRe = 16

    # (*The Li et al orbitals.*)
    nLi = 152
    lamLi = 1171.2
    LLi = 394
    LxiLi = 20115
    betaLi = 20

    res = cost_double_factorization(
        num_spin_orbs=nRe,
        lambda_tot=lamRe,
        num_aux=LRe,
        num_eig=LxiRe,
        kmesh=[2, 2, 2],
        dE_for_qpe=dE,
        chi=chi,
        beta=betaRe,
    )
    # 48250, 22343175750, 8174
    assert np.isclose(res.toffolis_per_step, 48250)
    assert np.isclose(res.total_toffolis, 22343175750)
    assert np.isclose(res.logical_qubits, 8174)

    res = cost_double_factorization(
        num_spin_orbs=nRe,
        lambda_tot=lamRe,
        num_aux=LRe,
        num_eig=LxiRe,
        kmesh=[3, 5, 1],
        dE_for_qpe=dE,
        chi=chi,
        beta=betaRe,
    )
    # 53146, 24610371366, 8945
    assert np.isclose(res.toffolis_per_step, 53146)
    assert np.isclose(res.total_toffolis, 24610371366)
    assert np.isclose(res.logical_qubits, 8945)

    res = cost_double_factorization(
        num_spin_orbs=nLi,
        lambda_tot=lamLi,
        num_aux=LLi,
        num_eig=LxiLi,
        kmesh=[3, 5, 1],
        dE_for_qpe=dE,
        chi=chi,
        beta=betaLi,
    )
    # print(res) # 79212, 145727663004, 13873
    assert np.isclose(res.toffolis_per_step, 86042)
    assert np.isclose(res.total_toffolis, 158292930114)
    assert np.isclose(res.logical_qubits, 14952)


if __name__ == "__main__":
    test_costing()
