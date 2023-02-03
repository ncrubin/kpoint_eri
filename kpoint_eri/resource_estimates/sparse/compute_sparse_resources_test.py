from kpoint_eri.resource_estimates.sparse.compute_sparse_resources import cost_sparse

def test_cost_sparse():
    nRe = 108
    lam_re = 2135.3
    dRe = 705831
    dE = 0.001
    chi = 10

    nLi = 152
    lam_Li = 1547.3
    dLi = 440501
    dE = 0.001
    chi = 10

    Nkx = 2
    Nky = 2
    Nkz = 2

    res = cost_sparse(nRe, lam_re, dRe, dE, chi, 20_000, Nkx, Nky, Nkz)
    # res = cost_sparse(nRe, lam_re, dRe, dE, chi, res[0], Nkx, Nky, Nkz)
    assert res[0] == 22962
    assert res[1] == 77017349364
    assert res[2] == 11335

    res = cost_sparse(nRe, lam_re, dRe, dE, chi, 20_000, 3, 5, 1)
    assert res[0] == 29004
    assert res[1] == 97282954488
    assert res[2] == 8060

    res = cost_sparse(nLi, lam_Li, dLi, dE, chi, 20_000, Nkx, Nky, Nkz)
    res = cost_sparse(nLi, lam_Li, dLi, dE, chi, res[0], Nkx, Nky, Nkz)
    assert res[0] == 21426
    assert res[1] == 52075764444
    assert res[2] == 7015

    res = cost_sparse(nLi, lam_Li, dLi, dE, chi, 20_000, 3, 5, 1)
    assert res[0] == 28986
    assert res[1] == 70450299084
    assert res[2] == 9231

if __name__ == "__main__":
    test_cost_sparse()
