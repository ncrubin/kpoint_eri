import numpy as np

from kpoint_eri.resource_estimates.df.compute_df_resources import compute_cost


def test_estimate():
    n = 52
    lam = 92.8
    L = 336
    Lxi = 616
    dE = 0.001
    chi = 10
    beta = 20

    stps2 = compute_cost(
        n=n,
        lam=lam,
        dE=dE,
        L=2 * L, # 2 for A and B terms
        Lxi=Lxi,
        chi=chi,
        beta=beta,
        stps=20000,
    )
    assert np.isclose(stps2[0], 6745)
    assert np.isclose(stps2[1], 983218650)
    assert np.isclose(stps2[2], 730)

def test_costing():
    nRe = 108
    lamRe = 294.8
    dE = 0.001
    LRe = 360
    LxiRe = 13031
    chi = 10
    betaRe = 16

    #(*The Li et al orbitals.*)
    nLi = 152
    lamLi = 1171.2
    LLi = 394
    LxiLi = 20115
    betaLi = 20

    res = compute_cost(nRe, lamRe, dE, LRe, LxiRe, chi, betaRe, 8, 3, 20_000) 
    res = compute_cost(nRe, lamRe, dE, LRe, LxiRe, chi, betaRe, 8, 3, res[0]) 
    assert np.isclose(res[0], 48151)
    assert np.isclose(res[1], 22297331721)
    assert np.isclose(res[2], 8171)
    print(res) #{48151, 22297331721, 8171}

    res = compute_cost(nLi, lamLi, dE, LLi, LxiLi, chi, betaLi, 8, 3, 20_000) 
    # res = compute_cost(nLi, lamLi, dE, LLi, LxiLi, chi, betaLi, 8, 3, res[0]) 
    print(res) # 79014, 145363399038, 13867
    assert np.isclose(res[0], 79014)
    assert np.isclose(res[1], 145363399038)
    assert np.isclose(res[2], 13867)
    res = compute_cost(nLi, lamLi, dE, LLi, LxiLi, chi, betaLi, 8, 3, res[0])
    assert np.isclose(res[0], 79014)
    assert np.isclose(res[1], 145363399038)
    assert np.isclose(res[2], 13867)

if __name__ == "__main__":
    test_costing()