from kpoint_eri.resource_estimates.sparse.compute_sparse_resources import cost_sparse

def test_cost_sparse():
    nLi = 152
    lam_Li = 1547.3
    dLi = 440501
    dE = 0.001
    chi = 10
    nK = 8

    # n: int, Nk: int, lam: float, d: int, dE: float, chi: int, stps: int) -> Tuple[int, int, int]
    stps2 = cost_sparse(nLi, nK, lam_Li, dLi, dE, chi, 20_000)
    toff_per_step, total_toff, total_ancilla_cost = cost_sparse(nLi, nK, lam_Li, dLi, dE, chi, stps2[0])
    assert toff_per_step == 18914
    assert total_toff == 45_970_363_516
    assert total_ancilla_cost == 4294

test_cost_sparse()