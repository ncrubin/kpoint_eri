import pandas as pd
import numpy as np

from pyscf.pbc import gto, scf
import pytest

from kpoint_eri.resource_estimates.sparse.generate_costing_table_sparse import (
    generate_costing_table,
)


@pytest.mark.slow
def test_generate_costing_table_sparse():
    kmesh = [1, 1, 3]
    cell = gto.M(
        unit="B",
        a=[
            [0.0, 3.37013733, 3.37013733],
            [3.37013733, 0.0, 3.37013733],
            [3.37013733, 3.37013733, 0.0],
        ],
        atom="""C 0 0 0
                  C 1.68506866 1.68506866 1.68506866""",
        basis="gth-szv",
        pseudo="gth-hf-rev",
        verbose=0,
    )
    cell.build()

    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts=kpts, exxdiv=None).rs_density_fit()
    mf.kernel()
    thresh = np.array([1e-1, 1e-2, 1e-12])
    table = generate_costing_table(mf, thresholds=thresh, chi=17, dE_for_qpe=1e-3)
    assert np.allclose(table.dE, 1e-3)
    assert np.allclose(table.chi, 17)
    assert np.allclose(table.cutoff, thresh)
    assert np.isclose(table.approx_emp2.values[2], table.exact_emp2.values[0])
    assert not np.isclose(table.approx_emp2.values[0], table.exact_emp2.values[0])
