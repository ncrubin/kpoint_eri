import numpy as np
import pytest

from pyscf.pbc import scf, gto

from kpoint_cholesky_factorization import (
        build_momentum_transfer_mapping,
        generate_orbital_products,
        build_eri_diagonal,
        locate_max_residual,
        generate_kpoint_cholesky_factorization,
        generate_eri_column
        )

def test_generate_momentum_transfer_mapping():
    cell = gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()
    kpts = cell.make_kpts([3, 3, 3])
    a = cell.lattice_vectors() / (2*np.pi)
    # k1_minus_k2 = kpoints[:,None,:] - kpoints[:,:,None]
    kpoints = kpts
    num_kpoints = len(kpoints)
    momentum_transfer_map = np.zeros((num_kpoints,)*2, dtype=np.int32)
    for iq, Q in enumerate(kpoints):
        for ik1, k1 in enumerate(kpoints):
            found = False
            for ik2, k2 in enumerate(kpoints):
                q = k1 - k2 - Q
                # We want q.a = 2 n pi
                test = np.einsum('wx,x->w', a, q)
                int_test = np.rint(test)
                if sum(abs(test-int_test)) < 1e-10:
                    assert found == False
                    momentum_transfer_map[iq, ik1] = ik2
                    found = True
    qmap = build_momentum_transfer_mapping(cell, kpts)
    assert np.sum(np.abs(qmap-momentum_transfer_map)) == 0

def test_generate_orbital_products():
    cell = gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()
    kpts = cell.make_kpts([2, 1, 1])
    a = cell.lattice_vectors() / (2*np.pi)
    kpoints = kpts
    num_kpoints = len(kpoints)
    mom_map = build_momentum_transfer_mapping(cell, kpts)
    kmf = scf.KRHF(cell, kpts, exxdiv=None)
    nao = cell.nao_nr()
    kmf.mo_coeff = [np.random.random((nao,nao))]*len(kpoints)
    rho_pq = generate_orbital_products(
            1,
            kmf,
            mom_map,
            kpoints)
    assert rho_pq.shape[0] == 2
    assert rho_pq.shape[1] == 24389
    assert rho_pq.shape[2] == 64

def test_generate_orbital_products():
    cell = gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()
    kpts = cell.make_kpts([2, 1, 1])
    a = cell.lattice_vectors() / (2*np.pi)
    kpoints = kpts
    num_kpoints = len(kpoints)
    mom_map = build_momentum_transfer_mapping(cell, kpts)
    kmf = scf.KRHF(cell, kpts, exxdiv=None)
    nao = cell.nao_nr()
    kmf.mo_coeff = [np.random.random((nao,nao))]*len(kpoints)
    rho_pq = generate_orbital_products(
            1,
            kmf,
            mom_map,
            kpoints)
    assert rho_pq.shape[0] == 2
    assert rho_pq.shape[1] == 24389
    assert rho_pq.shape[2] == 64

def test_build_eri_diagonal():
    cell = gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()
    kpts = cell.make_kpts([3, 1, 1])
    a = cell.lattice_vectors() / (2*np.pi)
    kpoints = kpts
    num_kpoints = len(kpoints)
    mom_map = build_momentum_transfer_mapping(cell, kpts)
    kmf = scf.KRHF(cell, kpts, exxdiv=None)
    nao = cell.nao_nr()
    kmf.mo_coeff = [np.eye(nao)] * len(kpoints)
    num_mo_per_kpoint = [nao] * len(kpoints)
    num_kpoints = len(kpoints)
    num_pq = nao*nao
    from pyscf.pbc.df.fft_ao2mo import get_eri
    for mom_trans_indx, mom_trans in enumerate(kpoints):
        rho_pq = generate_orbital_products(
                mom_trans_indx,
                kmf,
                mom_map,
                kpoints)
        eri_diag = build_eri_diagonal(
                      mom_trans_indx,
                      rho_pq,
                      num_mo_per_kpoint,
                      mom_map)
        diag = np.zeros((num_kpoints,num_pq), dtype=np.complex128)
        for kp1, kpt1 in enumerate(kpoints):
            kp2 = mom_map[mom_trans_indx, kp1]
            kpt_pqrs = [
                        kpt1,
                        kpoints[kp2],
                        kpoints[kp2],
                        kpt1,
                        ]
            eri_pqrs = get_eri(
                    kmf.with_df,
                    kpts=kpt_pqrs,
                    compact=False)
            eri_pqrs = eri_pqrs.reshape((nao,)*4).transpose((0,1,3,2)).reshape((num_pq,)*2)
            diag[kp1] = eri_pqrs.diagonal().ravel()
        assert np.max(diag) - np.max(eri_diag) < 1e-10
        assert np.linalg.norm(diag-eri_diag) < 1e-10
        max_res, max_k1k2, max_pq = locate_max_residual(eri_diag, mom_trans_indx, mom_map, num_mo_per_kpoint)
        print(np.max(eri_diag), np.argmax(eri_diag), eri_diag[0,0])
        print(np.max(diag), diag[1,0], diag[0,0], np.argmax(diag))
        print(np.unravel_index(np.argmax(diag), diag.shape))
        print()
    assert eri_diag.shape == (num_kpoints, num_pq)

def test_generate_cholesky_factorization():
    cell = gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()
    kpts = cell.make_kpts([1, 1, 1])
    a = cell.lattice_vectors() / (2*np.pi)
    kpoints = kpts
    num_kpoints = len(kpoints)
    mom_map = build_momentum_transfer_mapping(cell, kpts)
    kmf = scf.KRHF(cell, kpts, exxdiv=None)
    # kmf.chkfile = 'diamond_222.chk'
    from pyscf.scf.chkfile import load_scf
    _, kmf_dict = load_scf('diamond_222.chk')
    kmf.mo_coeff = kmf_dict['mo_coeff']
    nao = cell.nao_nr()
    # AO basis.
    LQ = generate_kpoint_cholesky_factorization(kmf, kpoints)
    kpts_pqrs = [kpoints[0],kpoints[0],kpoints[0],kpoints[0]]
    from pyscf.pbc.df.fft_ao2mo import get_eri
    eri_pqrs = get_eri(
            kmf.with_df,
            kpts=kpts_pqrs,
            compact=False).reshape((nao,)*4).transpose((0,1,3,2)).reshape((nao*nao,)*2)
    eri_chol = np.einsum('ij,kl->ijkl', LQ[0][0], LQ[0][0].conj())
    print(np.linalg.norm(eri_pqrs-eri_chol))

def test_generate_eri_column():
    cell = gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()
    kpts = cell.make_kpts([2, 2, 1])
    a = cell.lattice_vectors() / (2*np.pi)
    kpoints = kpts
    num_kpoints = len(kpoints)
    mom_map = build_momentum_transfer_mapping(cell, kpts)
    kmf = scf.KRHF(cell, kpts, exxdiv=None)
    # kmf.chkfile = 'diamond_222.chk'
    from pyscf.scf.chkfile import load_scf
    nao = cell.nao_nr()
    # AO basis.
    kmf.mo_coeff = [np.eye(nao)] * len(kpoints)
    num_mo_per_kpoint = [nao] * len(kpoints)
    num_kpoints = len(kpoints)
    num_pq = nao*nao
    from pyscf.pbc.df.fft_ao2mo import get_eri
    iQ = 1
    # print(iQ)
    ik4 = 2
    ik3 = mom_map[iQ,ik4]
    r = 6
    s = 2
    # max_rs = r*nao + s
    # M[pq,rs] = (pq|sr)
    max_rs = r*nao + s
    # for mom_trans_indx, mom_trans in enumerate(kpoints):
    rho_pq = generate_orbital_products(
            iQ,
            kmf,
            mom_map,
            kpoints)
    eri_col = generate_eri_column(
                    kmf,
                    rho_pq,
                    [ik3,ik4],
                    max_rs,
                    iQ,
                    mom_map,
                    kpoints)
    diag = np.zeros_like(eri_col)
    from pyscf.pbc.df.fft_ao2mo import get_ao_pairs_G, _iskconserv
    from pyscf.pbc import tools
    Q = kpoints[iQ]
    eri_col_ref = np.zeros_like(eri_col)
    for kp1, kpt1 in enumerate(kpoints):
        kp2 = mom_map[iQ, kp1]
        kpt_pqrs = [
                    kpt1,
                    kpoints[kp2],
                    kpoints[ik3],
                    kpoints[ik4],
                    ]
        eri_pqrs = get_eri(
                kmf.with_df,
                kpts=kpt_pqrs,
                compact=False)
        coulG = tools.get_coulG(
                    kmf.cell,
                    kpoints[kp2]-kpt1,
                    mesh=kmf.cell.mesh)
        eri_pqrs = eri_pqrs.reshape((nao,)*4).transpose((0,1,3,2)).reshape((num_pq,)*2)
        eri_col_ref[kp1] = eri_pqrs[:,max_rs]
        assert np.linalg.norm(eri_col[kp1]-eri_col_ref[kp1]) < 1e-12
