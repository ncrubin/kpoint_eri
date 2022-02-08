import numpy as np
import pytest

from pyscf.pbc import scf, gto

from kpoint_cholesky_factorization import (
        build_momentum_transfer_mapping,
        generate_orbital_products,
        build_eri_diagonal
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
    kpts = cell.make_kpts([2, 1, 1])
    a = cell.lattice_vectors() / (2*np.pi)
    kpoints = kpts
    num_kpoints = len(kpoints)
    mom_map = build_momentum_transfer_mapping(cell, kpts)
    kmf = scf.KRHF(cell, kpts, exxdiv=None)
    nao = cell.nao_nr()
    kmf.mo_coeff = [np.random.random((nao,nao))] * len(kpoints)
    mom_trans_indx = 1
    num_mo_per_kpoint = [nao] * len(kpoints)
    rho_pq = generate_orbital_products(
            mom_trans_indx,
            kmf,
            mom_map,
            kpoints)
    num_kpoints = len(kpoints)
    num_pq = nao*nao
    residual, max_k1k2, max_pq = build_eri_diagonal(
                                    mom_trans_indx,
                                    rho_pq,
                                    num_mo_per_kpoint,
                                    mom_map)
    assert residual.shape == (num_kpoints, num_pq)
    assert len(max_k1k2) == 2
    assert len(max_pq) == 2
