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
