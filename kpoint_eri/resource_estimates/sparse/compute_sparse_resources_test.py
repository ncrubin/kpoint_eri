from kpoint_eri.resource_estimates.sparse.compute_sparse_resources import cost_sparse

def test_cost_sparse():
    nRe = 108
    lam_re = 2135.3
    dRe = 705831
    nk = 8

    nLi = 152
    lam_Li = 1547.3
    dLi = 440501
    dE = 0.001
    chi = 10

    res = cost_sparse(nRe, nk, lam_re, dRe, dE, chi, 20_000)
    res = cost_sparse(nRe, nk, lam_re, dRe, dE, chi, res[0])
    print(res)
    assert res[0] == 108487
    assert res[1] == 363878633414
    assert res[2] == 1540

    res = cost_sparse(nLi, nk, lam_Li, dLi, dE, chi, 20_000)
    res = cost_sparse(nLi, nk, lam_Li, dLi, dE, chi, res[0])
    assert res[0] == 82798
    assert res[1] == 201240042212
    assert res[2] == 1857

if __name__ == "__main__":
    test_cost_sparse()
