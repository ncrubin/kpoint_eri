import numpy as np

from kpoint_eri.resource_estimates.df.compute_df_resources import compute_cost


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

    res = compute_cost(nRe, lamRe, dE, LRe, LxiRe, chi, betaRe, 2, 2, 2, 20_000)
    res = compute_cost(nRe, lamRe, dE, LRe, LxiRe, chi, betaRe, 2, 2, 2, res[0])
    # 48250, 22343175750, 8174
    assert np.isclose(res[0], 48250)
    assert np.isclose(res[1], 22343175750)
    assert np.isclose(res[2], 8174)

    res = compute_cost(nRe, lamRe, dE, LRe, LxiRe, chi, betaRe, 3, 5, 1, 20_000)
    res = compute_cost(nRe, lamRe, dE, LRe, LxiRe, chi, betaRe, 3, 5, 1, res[0])
    # 53146, 24610371366, 8945
    assert np.isclose(res[0], 53146)
    assert np.isclose(res[1], 24610371366)
    assert np.isclose(res[2], 8945)

    res = compute_cost(nLi, lamLi, dE, LLi, LxiLi, chi, betaLi, 2, 2, 2, 20_000)
    res = compute_cost(nLi, lamLi, dE, LLi, LxiLi, chi, betaLi, 2, 2, 2, res[0])
    # print(res) # 79212, 145727663004, 13873
    assert np.isclose(res[0], 79212)
    assert np.isclose(res[1], 145727663004)
    assert np.isclose(res[2], 13873)
    res = compute_cost(nLi, lamLi, dE, LLi, LxiLi, chi, betaLi, 3, 5, 1, res[0])
    # print(res) # 86042, 158292930114, 14952
    assert np.isclose(res[0], 86042)
    assert np.isclose(res[1], 158292930114)
    assert np.isclose(res[2], 14952)


if __name__ == "__main__":
    test_costing()
