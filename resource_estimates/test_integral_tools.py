import numpy as np

from integral_tools import (
        supercell_eris,
        kpoint_eris,
        kpoint_cholesky_eris,
        thc_eris)
from utils import (
        init_from_chkfile,
        read_qmcpack_cholesky_kpoint,
        read_qmcpack_thc
        )
from k2gamma import k2gamma

def test_supercell_integrals():
    chkfile = 'data/scf_nk2.chk'
    cell, kmf = init_from_chkfile(chkfile)
    supercell_hf = k2gamma(kmf)
    supercell = supercell_hf.cell
    eris = supercell_integrals(supercell_hf.cell)
    assert eris.shape == (supercell.nao_nr(),)*4
    eris = supercell_eris(supercell_hf.cell, mo_coeff=supercell_hf.mo_coeff)
    nmo = supercell_hf.mo_coeff.shape[1]
    assert eris.shape == (nmo,)*4

def test_kpoint_integrals():
    chkfile = 'data/scf_nk2.chk'
    cell, kmf = init_from_chkfile(chkfile)
    hamil_chol = read_qmcpack_cholesky_kpoint('data/chol_nk2.h5')
    eris = kpoint_eris(
            cell,
            kmf.mo_coeff,
            kmf.kpts,
            hamil_chol['qk_k2'])

def test_kpoint_cholesky_integrals():
    chkfile = 'data/scf_nk2.chk'
    cell, kmf = init_from_chkfile(chkfile)
    hamil_chol = read_qmcpack_cholesky_kpoint('data/chol_nk2.h5')
    nmo_pk = hamil_chol['nmo_pk']
    nchol_pk = hamil_chol['nchol_pk']
    nk = len(nmo_pk)
    nmo_k1k2 = []
    max_nmo = np.max(nmo_pk)
    LQ = []
    chol = hamil_chol['chol']
    nchol_per_kp = hamil_chol['nchol_pk']
    qk_to_k2 = hamil_chol['qk_k2']
    for i, c in enumerate(chol):
        LQ.append(c.reshape((nk, max_nmo, max_nmo, nchol_per_kp[i])))
    eris = kpoint_cholesky_eris(
            LQ,
            kmf.kpts,
            qk_to_k2,
            nmo_pk,
            )

def test_thc_integrals():
    hamil_thc = read_qmcpack_thc('data/thc_nk2_cthc12.h5')
    eris = thc_eris(
            hamil_thc['orbs_pu'],
            hamil_thc['Muv']
            )
