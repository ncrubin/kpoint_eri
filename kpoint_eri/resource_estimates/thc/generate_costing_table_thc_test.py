import numpy as np
import os
import pandas as pd

from pyscf.pbc import gto, scf

from kpoint_eri.resource_estimates.thc.generate_costing_table_thc import (
    generate_costing_table,
)


def test_generate_costing_table_df():
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
    print(mf.with_df)
    thc_rank_params = np.array([2, 4, 6])
    table = generate_costing_table(
        mf,
        thc_rank_params=thc_rank_params,
        chi=10,
        beta=22,
        dE_for_qpe=1e-3,
        bfgs_maxiter=10,
        adagrad_maxiter=10,
        fft_df_mesh=[11] * 3,
    )
    num_kpts = np.prod(kmesh)
    assert np.allclose(table.dE, 1e-3)
    assert np.allclose(table.chi, 10)
    assert np.allclose(table.beta, 22)
    assert np.allclose(table.cutoff, thc_rank_params)
    filename = f"pbc_thc_num_kpts_{num_kpts}.csv"
    df_from_file = pd.read_csv(filename, index_col=0)
    assert np.allclose(df_from_file.total_toffolis, table.total_toffolis) 
    assert np.allclose(df_from_file.approx_emp2, table.approx_emp2) 
