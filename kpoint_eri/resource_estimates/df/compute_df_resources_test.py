import numpy as np

from kpoint_eri.resource_estimates.df.compute_df_resources import compute_cost


def test_estimate():
    n = 302
    lam = 3071.8
    L = 275
    Lxi = 11
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
    assert np.isclose(stps2[0], 33721)
    assert np.isclose(stps2[1], 162709658733)
    assert np.isclose(stps2[2], 7206)
