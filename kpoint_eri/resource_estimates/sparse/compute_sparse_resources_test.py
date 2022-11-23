from kpoint_eri.resource_estimates.sparse.compute_sparse_resources import cost_sparse

def test_cost_sparse():
    nRe = 108
    lam_re = 2135.3
    dRe = 705831
    nk = 7

    nLi = 152
    lam_Li = 1547.3
    dLi = 440501
    dE = 0.001
    chi = 10

    res = cost_sparse(nRe, nk, lam_re, dRe, dE, chi, 20_000)
    res = cost_sparse(nRe, nk, lam_re, dRe, dE, chi, res[0])
    assert res[0] == 28488
    assert res[1] == 95552227536
    assert res[2] == 3452

    res = cost_sparse(nLi, nk, lam_Li, dLi, dE, chi, 20_000)
    res = cost_sparse(nLi, nk, lam_Li, dLi, dE, chi, res[0])
    assert res[0] == 20896
    assert res[1] == 50787602624
    assert res[2] == 4014
