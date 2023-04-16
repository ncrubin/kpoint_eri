import numpy as np
from functools import reduce

from pyscf.pbc import gto, scf
from kpoint_eri.resource_estimates.utils.k2gamma import k2gamma
from pyscf.pbc.lib.kpts_helper import conj_mapping


def test_make_real():
    cell = gto.Cell()
    cell.atom = """
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf-rev"
    cell.a = """
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000"""
    cell.unit = "B"
    cell.verbose = 0
    cell.build(parse_arg=False)

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    nkpts = np.prod(kmesh)
    mf.kernel()
    kminus = conj_mapping(cell, kpts)

    hcore_ao = mf.get_hcore()
    hcore_mo = np.asarray(
        [
            reduce(np.dot, (mo.T.conj(), hcore_ao[k], mo))
            for k, mo in enumerate(mf.mo_coeff)
        ]
    )

    supercell_mf = k2gamma(mf, make_real=False)
    supercell_hcore_ao = supercell_mf.get_hcore()
    supercell_hcore_mo = np.asarray(
        [
            reduce(np.dot, (mo.T.conj(), supercell_hcore_ao[k], mo))
            for k, mo in enumerate(supercell_mf.mo_coeff)
        ]
    )
    norm_uc_T = sum(np.sum(np.abs(hk.real) + np.abs(hk.imag)) for hk in hcore_mo)
    norm_sc_T = sum(
        np.sum(np.abs(hk.real) + np.abs(hk.imag)) for hk in supercell_hcore_mo
    )
    assert np.isclose(norm_uc_T, norm_sc_T)
    supercell_mf = k2gamma(mf, make_real=True)
    supercell_hcore_ao = supercell_mf.get_hcore()
    supercell_hcore_mo = np.asarray(
        [
            reduce(np.dot, (mo.T.conj(), supercell_hcore_ao[k], mo))
            for k, mo in enumerate(supercell_mf.mo_coeff)
        ]
    )
    norm_uc_T = sum(np.sum(np.abs(hk.real) + np.abs(hk.imag)) for hk in hcore_mo)
    norm_sc_T = sum(
        np.sum(np.abs(hk.real) + np.abs(hk.imag)) for hk in supercell_hcore_mo
    )
    assert not np.isclose(norm_uc_T, norm_sc_T)


if __name__ == "__main__":
    test_make_real()
