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


# taken from qmcpack afqmctools.hamiltonian.converter!
def from_qmcpack_complex(data, shape=None):
    if shape is not None:
        return data.view(np.complex128).ravel().reshape(shape)
    else:
        shape = tuple((s for s in data.shape[:-1]))
        return data.view(np.complex128).ravel().reshape(shape)

def get_dset_simple(fh5, name):
    try:
        dset = fh5[name][:]
        return dset
    except KeyError:
        # print("Error reading {:} dataset".format(name))
        return None

def read_common_input(filename, get_hcore=True):
    with h5py.File(filename, 'r') as fh5:
        try:
            enuc = fh5['Hamiltonian/Energies'][:][0]
        except:
            print(" Error reading Hamiltonian/Energies dataset.")
            enuc = None
        try:
            dims = fh5['Hamiltonian/dims'][:]
        except:
            dims = None
        assert dims is not None, "Error reading Hamiltonian/dims data set."
        assert len(dims) == 8, "Hamiltonian dims data set has incorrect length."
        nmo = dims[3]
        real_ints = False
        if get_hcore:
            try:
                hcore = from_qmcpack_complex(fh5['Hamiltonian/hcore'][:], (nmo,nmo))
            except ValueError:
                hcore = fh5['Hamiltonian/hcore'][:]
                real_ints = True
            except KeyError:
                hcore = None
                pass
            if hcore is None:
                try:
                    # old sparse format only for complex.
                    hcore = fh5['Hamiltonian/H1'][:].view(np.complex128).ravel()
                    idx = fh5['Hamiltonian/H1_indx'][:]
                    row_ix = idx[::2]
                    col_ix = idx[1::2]
                    hcore = scipy.sparse.csr_matrix((hcore, (row_ix, col_ix))).toarray()
                    hcore = np.tril(hcore, -1) + np.tril(hcore, 0).conj().T
                except:
                    hcore = None
                    print("Error reading Hamiltonian/hcore data set.")
            try:
                complex_ints = bool(fh5['Hamiltonian/ComplexIntegrals'][0])
            except KeyError:
                complex_ints = None

            if complex_ints is not None:
                if hcore is not None:
                    hc_type = hcore.dtype
                else:
                    hc_type = type(hcore)
                msg = ("ComplexIntegrals flag conflicts with integral data type. "
                       "dtype = {:} ComplexIntegrals = {:}.".format(hc_type, complex_ints))
                assert real_ints ^ complex_ints, msg
        else:
            hcore = None
    return enuc, dims, hcore, real_ints

def read_qmcpack_thc(filename):
    with h5py.File(filename, 'r') as fh5:
        hcore = from_qmcpack_complex(fh5['Hamiltonian/hcore'][:])
        naux = fh5['Hamiltonian/THC/dims'][:][1]
        nmo = fh5['Hamiltonian/THC/dims'][:][0]
        Luv = from_qmcpack_complex(
                fh5['Hamiltonian/THC/Luv'][:],
                shape=(naux,naux))
        Muv = np.dot(Luv, Luv.conj().T)
        orbs_pu = from_qmcpack_complex(
                fh5['Hamiltonian/THC/Orbitals'][:],
                shape=(nmo, naux))
        e0 = fh5['Hamiltonian/Energies'][0]

    hamil = {
            'hcore': hcore,
            'orbs_pu': orbs_pu,
            'Muv': Muv,
            'enuc': e0
            }
    return hamil

def read_qmcpack_cholesky_kpoint(filename, get_chol=True):
    """Read in integrals from qmcpack hdf5 format. kpoint dependent case.

    Parameters
    ----------
    filename : string
        File containing integrals in qmcpack format.

    Returns
    -------
    hcore : :class:`np.ndarray`
        One-body part of the Hamiltonian.
    chol_vecs : :class:`scipy.sparse.csr_matrix`
        Two-electron integrals. Shape: [nmo*nmo, nchol]
    ecore : float
        Core contribution to the total energy.
    nmo : int
        Number of orbitals.
    nelec : tuple
        Number of electrons.
    nmo_pk : :class:`np.ndarray`
        Number of orbitals per kpoint.
    qk_k2 : :class:`np.ndarray`
        Array mapping (q,k) pair to kpoint: Q = k_i - k_k + G.
        qk_k2[iQ,ik_i] = i_kk.
    """
    enuc, dims, hcore, real_ints = read_common_input(filename, get_hcore=False)
    with h5py.File(filename, 'r') as fh5:
        nmo_pk = get_dset_simple(fh5, 'Hamiltonian/NMOPerKP')
        nchol_pk = get_dset_simple(fh5, 'Hamiltonian/NCholPerKP')
        qk_k2 = get_dset_simple(fh5, 'Hamiltonian/QKTok2')
        minus_k = get_dset_simple(fh5, 'Hamiltonian/MinusK')
        kpoints = get_dset_simple(fh5, 'Hamiltonian/KPoints')
        hcore = []
        nkp = dims[2]
        nmo_tot = dims[3]
        nalpha = dims[4]
        nbeta = dims[5]
        if nmo_pk is None:
            raise KeyError("Could not read NMOPerKP dataset.")
        for i in range(0, nkp):
            hk = get_dset_simple(fh5, 'Hamiltonian/H1_kp{}'.format(i))
            if hk is None:
                raise KeyError("Could not read one-body hamiltonian.")
            nmo = nmo_pk[i]
            hcore.append(hk.view(np.complex128).reshape(nmo,nmo))
        chol_vecs = []
        if nmo_pk is None:
            raise KeyError("Error nmo_pk dataset does not exist.")
        nmo_max = max(nmo_pk)
    if minus_k is None:
        raise KeyError("Error MinusK dataset does not exist.")
    if nchol_pk is None:
        raise KeyError("Error NCholPerKP dataset does not exist.")
    if get_chol:
        for i in range(0, nkp):
            LQi = get_kpoint_chol(filename, nchol_pk, minus_k, i)
            chol_vecs.append(
                    LQi.reshape((nkp, nmo_max, nmo_max, nchol_pk[i]))
                    )
    else:
        chol_vecs = None

    hamil = {
        'hcore': hcore,
        'chol': chol_vecs,
        'enuc': enuc,
        'nelec': (nalpha, nbeta),
        'nmo': int(nmo),
        'nmo_pk': nmo_pk,
        'nchol_pk': nchol_pk,
        'minus_k': minus_k,
        'qk_k2': qk_k2,
        'kpoints': kpoints
        }
    return hamil

def read_cholesky_contiguous(filename, frac_chol_to_keep=1):
    hamil = read_qmcpack_cholesky_kpoint(filename)
    chol = hamil['chol']
    min_nchol = min(cv.shape[-1] for cv in chol)
    min_nchol = int(min_nchol * frac_chol_to_keep)
    shape = list(chol[0].shape)
    shape[-1] = min_nchol
    nk = len(chol)
    chol_contiguous = np.zeros((nk,)+tuple(shape), dtype=np.complex128)
    for ic, cv in enumerate(chol):
        chol_contiguous[ic] = cv[:,:,:,:min_nchol]

    hamil['chol'] = chol_contiguous
    return hamil


def get_kpoint_chol(filename, nchol_pk, minus_k, i):
    with h5py.File(filename, 'r') as fh5:
        try:
            Lk = get_dset_simple(fh5, 'Hamiltonian/KPFactorized/L{}'.format(i))
            if Lk is None:
                raise KeyError
            nchol = nchol_pk[i]
            Lk = Lk.view(np.complex128)[:,:,0]
        except KeyError:
            Lk = get_dset_simple(fh5, 'Hamiltonian/KPFactorized/L{}'.format(minus_k[i]))
            if Lk is None:
                raise TypeError("Could not read Cholesky integrals from file.")
            nchol = nchol_pk[minus_k[i]]
            nchol_pk[i] = nchol
            Lk = Lk.view(np.complex128).conj()[:,:,0]
    return Lk

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
    from pyscf.pbc import scf as pb_scf
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
    kpts = cell.make_kpts([2, 2, 1])
    a = cell.lattice_vectors() / (2*np.pi)
    mom_map = build_momentum_transfer_mapping(cell, kpts)
    for i, Q in enumerate(kpts):
        for j, k1 in enumerate(kpts):
            k2 = kpts[mom_map[i, j]]
            test = Q - k1 + k2
            assert test in cell.Gv
