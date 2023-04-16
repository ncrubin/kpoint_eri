import numpy as np
import pandas as pd

from pyscf.pbc import gto, scf
import pytest

from kpoint_eri.resource_estimates.sf.generate_costing_table_sf import (
    generate_costing_table,
)


@pytest.mark.slow
def test_generate_costing_table_sf():
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
    thresh = [10, 54, 108]
    thresh = np.array([0.1, 0.5, 1.0])  # Fraction of naux
    table = generate_costing_table(mf, naux_cutoffs=thresh, chi=17, dE_for_qpe=1e-3)
    assert np.allclose(table.dE, 1e-3)
    assert np.allclose(table.chi, 17)
    assert np.allclose(table.num_aux, [10, 54, 108])
    assert np.isclose(table.approx_emp2.values[2], table.exact_emp2.values[0])
