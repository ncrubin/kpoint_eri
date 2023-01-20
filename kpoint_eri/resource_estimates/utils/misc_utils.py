import h5py
import numpy as np
from pyscf import lib
from pyscf.lib.chkfile import load, load_mol
from pyscf.pbc.lib.chkfile import load_cell
from pyscf.pbc import scf as pb_scf
# from pyscf import gto, scf, ao2mo, cc

def build_momentum_transfer_mapping(
        cell,
        kpoints
        ):
    # Define mapping momentum_transfer_map[Q][k1] = k2 that satisfies
    # k1 - k2 + G = Q.
    a = cell.lattice_vectors() / (2*np.pi)
    delta_k1_k2_Q = kpoints[:,None,None,:] - kpoints[None,:,None,:] - kpoints[None,None,:,:]
    delta_k1_k2_Q += kpoints[0][None, None, None, :]  # shift to center
    delta_dot_a = np.einsum('wx,kpQx->kpQw', a, delta_k1_k2_Q)
    int_delta_dot_a = np.rint(delta_dot_a)
    # Should be zero if transfer is statisfied (2*pi*n)
    mapping = np.where(np.sum(np.abs(delta_dot_a-int_delta_dot_a), axis=3) < 1e-10)
    num_kpoints = len(kpoints)
    momentum_transfer_map = np.zeros((num_kpoints,)*2, dtype=np.int32)
    # Note index flip due to Q being first index in map but broadcasted last..
    momentum_transfer_map[mapping[1], mapping[0]] = mapping[2]

    return momentum_transfer_map


def init_from_chkfile(chkfile):
    cell = load_cell(chkfile)
    cell.build()
    nao = cell.nao_nr()
    energy = np.asarray(lib.chkfile.load(chkfile, 'scf/e_tot'))
    kpts = np.asarray(lib.chkfile.load(chkfile, 'scf/kpts'))
    nkpts = len(kpts)
    mo_coeff = np.asarray(lib.chkfile.load(chkfile, 'scf/mo_coeff'))
    # print(mo_coeff.shape)
    if len(mo_coeff.shape) == 4:
        kmf = pb_scf.KUHF(cell, kpts)
    else:
        kmf = pb_scf.KRHF(cell, kpts)
    kmf.mo_occ = np.asarray(lib.chkfile.load(chkfile, 'scf/mo_occ'))
    kmf.mo_coeff = mo_coeff
    kmf.mo_energy = np.asarray(lib.chkfile.load(chkfile, 'scf/mo_energy'))
    kmf.e_tot = energy
    return cell, kmf

def build_cc_object(
        hcore,
        eris,
        ovlp,
        nelec,
        mo_coeff,
        mo_occ,
        mo_energy,
        mo_basis=True):
    mol = gto.M()
    mol.nelectron = nelec
    mol.verbose = 4
    mf = scf.RHF(mol)
    nmo = mo_coeff.shape[1]
    if not mo_basis:
        mf.get_hcore = lambda *args: hcore.copy()
        mf.get_ovlp = lambda *args: ovlp.copy()
        mf.mo_coeff = mo_coeff
        mf.mo_occ = mo_occ
        mf.mo_energy = mo_energy
    else:
        mf.mo_coeff = np.eye(nmo)
        mf.get_hcore = lambda *args : hcore
        mf.get_ovlp = lambda *args : np.eye(nmo)
        mf.mo_occ = mo_occ
        mf.mo_energy = mo_energy
    if eris.dtype == np.complex128:
        mf._eri = eris
        # mf._eri = ao2mo.restore(
                # 4,
                # eris,
                # nmo)
    else:
        mf._eri = ao2mo.restore(
                8,
                eris,
                nmo)
    return cc.RCCSD(mf)


def energy_eri(hcore, eris, nocc, enuc):
    e1b = 2*hcore[:nocc, :nocc].trace()
    ecoul = 2*np.einsum('iijj->', eris[:nocc,:nocc,:nocc,:nocc])
    exx = -np.einsum('ijji->', eris[:nocc,:nocc,:nocc,:nocc])
    return e1b + ecoul + exx + enuc, e1b + enuc, ecoul + exx

def eri_thc(orbs, Muv):
    eri_thc = np.einsum(
            'pP,qP,PQ,rQ,sQ->pqrs',
            orbs.conj(), orbs,
            Muv,
            orbs.conj(), orbs,
            optimize=True)
    return eri_thc

def build_test_system_diamond(basis):
    from pyscf.pbc import scf as pb_scf
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = basis
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
    kmf = pb_scf.KRHF(cell, kpts, exxdiv=None)
    kmf.chkfile = 'diamond_221.chk'
    kmf.kernel()
    return kmf


def test_momentum_transfer_map():
    from pyscf.pbc import gto
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
    kpts = cell.make_kpts([2, 2, 1], scaled_center=[0.1, 0.2, 0.3])
    mom_map = build_momentum_transfer_mapping(cell, kpts)
    for i, Q in enumerate(kpts):
        for j, k1 in enumerate(kpts):
            k2 = kpts[mom_map[i, j]]
            test = Q - k1 + k2
            assert np.amin(
                np.abs(test[None, :] - cell.Gv - kpts[0][None, :])) < 1e-15
