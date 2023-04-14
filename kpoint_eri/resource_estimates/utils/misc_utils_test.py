import numpy as np

from kpoint_eri.resource_estimates.utils.misc_utils import (
    build_momentum_transfer_mapping,
)


def test_momentum_transfer_map():
    from pyscf.pbc import gto

    cell = gto.Cell()
    cell.atom = """
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.a = """
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000"""
    cell.unit = "B"
    cell.verbose = 0
    cell.build(parse_arg=False)
    kpts = cell.make_kpts([2, 2, 1], scaled_center=[0.1, 0.2, 0.3])
    mom_map = build_momentum_transfer_mapping(cell, kpts)
    for i, Q in enumerate(kpts):
        for j, k1 in enumerate(kpts):
            k2 = kpts[mom_map[i, j]]
            test = Q - k1 + k2
            assert np.amin(np.abs(test[None, :] - cell.Gv - kpts[0][None, :])) < 1e-15
