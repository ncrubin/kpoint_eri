import numpy as np
from pyscf.pbc.df import fft, fft_ao2mo

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
    eris = kpoint_cholesky_eris(
            chol,
            kmf.kpts,
            qk_to_k2,
            nmo_pk,
            )

def test_df_integrals():
    chkfile = 'data/scf_nk2.chk'
    cell, kmf = init_from_chkfile(chkfile)
    hamil_chol = read_qmcpack_cholesky_kpoint('data/chol_nk2.h5')
    nmo_per_kp = hamil_chol['nmo_pk']
    nchol_per_kp = hamil_chol['nchol_pk']
    nk = len(nmo_per_kp)
    nmo_k1k2 = []
    max_nmo = np.max(nmo_per_kp)
    chol = hamil_chol['chol']
    kpoints = hamil_chol['kpoints']
    nchol_per_kp = hamil_chol['nchol_pk']
    qk_to_k2 = hamil_chol['qk_k2']
    from integral_tools import (
            DFHelper,
            CholeskyHelper,
            ERIHelper)
    df = fft.FFTDF(cell, kpts=kpoints)
    eh = ERIHelper(
            df,
            kmf.mo_coeff,
            kpoints)
    ikpts = [0, 1, 1 ,0]
    eri_ref = eh.get_eri(ikpts)
    ch = CholeskyHelper(
            chol,
            qk_to_k2,
            kpoints,
            chol_thresh=264,
            )
    eri_chol = ch.get_eri(ikpts)
    print(np.max(np.abs(eri_ref-eri_chol)))
    dfh = DFHelper(
            chol,
            qk_to_k2,
            kpoints,
            nmo_per_kp,
            chol_thresh=264,
            df_thresh=0.0)
    eri_df = dfh.get_eri(ikpts)
    # print(eri_ref[0,0,0,0], eri_df[0,0,0,0])
    print(np.max(np.abs(eri_ref-eri_df)))

def test_thc_integrals():
    hamil_thc = read_qmcpack_thc('data/thc_nk2_cthc12.h5')
    eris = thc_eris(
            hamil_thc['orbs_pu'],
            hamil_thc['Muv']
            )

# def test_random_df():
    # no = 4
    # eris = np.random.normal(scale=0.01, size=(no,)*4)
    # eris = eris + eris.transpose((2,3,0,1))
    # eris = eris + eris.transpose((3,2,1,0))
    # eris = eris + eris.transpose((1,0,2,3))
    # eris = eris + eris.transpose((0,1,3,2))
    # eris = eris.reshape((no*no,no*no))
    # eris = np.dot(eris,eris.T)
    # chol = modified_cholesky(eris, tol=1e-12)
    # chol = chol.reshape((-1, no, no))
    # print(np.linalg.norm(
        # eris.reshape((no,)*4)- np.einsum('npq,nrs->pqrs', chol, chol, optimize=True)
        # ))
    # nchol = chol.shape[0]
    # Us = np.zeros((nchol, no, no))
    # ls = np.zeros((nchol, no))
    # for i in range(nchol):
        # print(np.linalg.norm(chol[i] - chol[i].T))
        # eigs, eigv = np.linalg.eigh(chol[i])
        # Us[i] = eigv
        # ls[i] = eigs

    # eri_df = np.einsum(
            # 'nPt,nt,nQt,nRs,ns,nSs->PQRS',
            # Us, ls, Us.conj(), Us, ls, Us.conj(),
            # optimize=True)
    # print(np.linalg.norm(
        # eris.reshape((no,)*4) - eri_df)
        # )

# def test_random_df_cmplx():
    # no = 4
    # from ipie.utils.testing import generate_hamiltonian
    # h1e, chol, enuc, eris = generate_hamiltonian(no, (2,2), cplx=4, sym=4, tol=1e-12)
    # chol = chol.reshape((-1, no, no))
    # print("delta: ", np.linalg.norm(
        # eris.reshape((no,)*4)- np.einsum('npq,nsr->pqrs', chol, chol.conj(), optimize=True)
        # ))
    # nchol = chol.shape[0]
    # Us = np.zeros((nchol, no, no), dtype=np.complex128)
    # ls = np.zeros((nchol, no), dtype=np.complex128)
    # Vs = np.zeros((nchol, no, no), dtype=np.complex128)
    # ts = np.zeros((nchol, no), dtype=np.complex128)
    # for i in range(nchol):
        # # print(np.linalg.norm(chol[i] - chol[i].conj().T))
        # eigs, eigv = np.linalg.eigh(0.5*(chol[i]+chol[i].conj().T))
        # Us[i] = eigv
        # ls[i] = eigs
        # eigs, eigv = np.linalg.eigh(0.5*1j*(chol[i]-chol[i].conj().T))
        # Vs[i] = eigv
        # ts[i] = eigs

    # eri_df_A = np.einsum(
            # 'nPt,nt,nQt,nRs,ns,nSs->PQRS',
            # Us, ls, Us.conj(), Us, ls, Us.conj(),
            # optimize=True)
    # eri_df_B = np.einsum(
            # 'nPt,nt,nQt,nRs,ns,nSs->PQRS',
            # Vs, ts, Vs.conj(), Vs, ts, Vs.conj(),
            # optimize=True)
    # print("delta 2: ", np.linalg.norm(
        # eris - (eri_df_A+eri_df_B))
        # )
