import numpy as np

from pyscf.pbc import gto, scf

from kpoint_eri.resource_estimates.df.generate_costing_table_df import (
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
        parse_arg=False,
    )
    cell.build(parse_arg=False)

    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts=kpts, exxdiv=None).rs_density_fit()
    mf.kernel()
    thresh = np.array([0.1, 1e-2, 1e-14])  # Eigenvalue threshold for second factorization. 
    table = generate_costing_table(mf, cutoffs=thresh, chi=10, beta=22, dE_for_qpe=1e-3)
    df_table = table.to_dataframe()
    assert np.allclose(df_table.dE, 1e-3)
    assert np.allclose(df_table.chi, 10)
    assert np.allclose(df_table.beta, 22)
    assert np.allclose(df_table.cutoff, thresh)
    assert np.allclose(df_table.num_aux, [648]*3)
    assert np.isclose(df_table.approx_energy.values[2], df_table.exact_energy.values[0])
