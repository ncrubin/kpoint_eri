from kpoint_eri.resource_estimates.sparse.compute_sparse_resources import (
    _cost_sparse,
    cost_sparse,
)


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

    res = _cost_sparse(nRe, lam_re, dRe, dE, chi, 20_000, Nkx, Nky, Nkz)
    # res = _cost_sparse(nRe, lam_re, dRe, dE, chi, res[0], Nkx, Nky, Nkz)
    assert res[0] == 22962
    assert res[1] == 77017349364
    assert res[2] == 11335

    res = _cost_sparse(nRe, lam_re, dRe, dE, chi, 20_000, 3, 5, 1)
    assert res[0] == 29004
    assert res[1] == 97282954488
    assert res[2] == 8060

    res = _cost_sparse(nLi, lam_Li, dLi, dE, chi, 20_000, Nkx, Nky, Nkz)
    res = _cost_sparse(nLi, lam_Li, dLi, dE, chi, res[0], Nkx, Nky, Nkz)
    assert res[0] == 21426
    assert res[1] == 52075764444
    assert res[2] == 7015

    res = _cost_sparse(nLi, lam_Li, dLi, dE, chi, 20_000, 3, 5, 1)
    assert res[0] == 28986
    assert res[1] == 70450299084
    assert res[2] == 9231


def test_cost_sparse_helper():
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

    res = cost_sparse(
        num_spin_orbs=nLi,
        lambda_tot=lam_Li,
        num_sym_unique=dLi,
        kmesh=[Nkx, Nky, Nkz],
        dE_for_qpe=dE,
        chi=chi,
    )
    assert res.toffolis_per_step == 21426
    assert res.total_toffolis == 52075764444
    assert res.logical_qubits == 7015


if __name__ == "__main__":
    test_cost_sparse()
