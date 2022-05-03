import h5py
import numpy
from pyscf import lib
from pyscf.lib.chkfile import load, load_mol
from pyscf.pbc.lib.chkfile import load_cell

def load_from_pyscf_chk(chkfile,hcore=None,orthoAO=False):

    cell = load_cell(chkfile)
    assert(cell is not None)

    kpts=None
    singleK = False
    if lib.chkfile.load(chkfile, 'scf/kpt') is not None :
        kpts = numpy.asarray(lib.chkfile.load(chkfile, 'scf/kpt'))
        singleK = True
    else:
        kpts = numpy.asarray(lib.chkfile.load(chkfile, 'scf/kpts'))
        assert(kpts is not None)
    kpts = numpy.reshape(kpts,(-1,3))
    nkpts = len(kpts)
    nao = cell.nao_nr()
    nao_tot = nao*nkpts

    Xocc = lib.chkfile.load(chkfile, 'scf/mo_occ')
    mo_energy = lib.chkfile.load(chkfile, 'scf/mo_energy')
    mo_coeff = lib.chkfile.load(chkfile, 'scf/mo_coeff')
    fock = numpy.asarray(lib.chkfile.load(chkfile, 'scf/fock'))
    assert(fock is not None)
    if isinstance(Xocc,list):
        # 3 choices:
        if isinstance(Xocc[0],list):
            # KUHF
            isUHF = True
            assert(len(Xocc[0])==nkpts)
        elif singleK:
            # UHF
            isUHF = True
            assert(len(Xocc) == 2)
            Xocc = numpy.asarray(Xocc)
        else:
            # KRHF
            isUHF = False
            assert(len(Xocc) == nkpts)
    else:
        assert(singleK)
        if len(Xocc) == 2:
            isUHF = True
        else:
            # single kpoint RHF
            isUHF = False
            Xocc = ([Xocc])
            assert(len(Xocc)==nkpts)

    if hcore is None:
        hcore = numpy.asarray(lib.chkfile.load(chkfile, 'scf/hcore'))
        assert(hcore is not None)
    hcore = numpy.reshape(hcore,(-1,nao,nao))
    assert(hcore.shape[0]==nkpts)

    if(cell.spin!=0 and isUHF==False):
        print(" cell.spin!=0 only allowed with UHF calculation \n")
        comm.abort()

    if(orthoAO==False and isUHF==True):
        print(" orthoAO=True required with UHF calculation \n")
        quit()

    if orthoAO:
        X_ = numpy.asarray(lib.chkfile.load(chkfile, 'scf/orthoAORot')).reshape(nkpts,nao,-1)
        assert(X_ is not None)
        nmo_pk = numpy.asarray(lib.chkfile.load(chkfile, 'scf/nmo_per_kpt'))
        # do this properly!!!
        if len(nmo_pk.shape) == 0:
            nmo_pk = numpy.asarray([nmo_pk])
        X = []
        for k in range(len(nmo_pk)):
            X.append(X_[k][:,0:nmo_pk[k]])
        assert(nmo_pk is not None)
    else:
        # can safely assume isUHF == False
        X = lib.chkfile.load(chkfile, 'scf/mo_coeff')
        if singleK:
            assert(len(X.shape) == 2)
            assert(X.shape[0] == nao)
            X = ([X])
        assert(len(X) == nkpts)
        nmo_pk = numpy.zeros(nkpts,dtype=numpy.int32)
        for ki in range(nkpts):
            nmo_pk[ki]=X[ki].shape[1]
            assert(nmo_pk[ki] == Xocc[ki].shape[0])

    if singleK:
        assert(nkpts==1)
        if isUHF:
            assert len(fock.shape) == 3
            assert fock.shape[0] == 2
            assert fock.shape[1] == nao
            fock = fock.reshape((2,1,fock.shape[1],fock.shape[2]))
            assert len(Xocc.shape) == 2
            Xocc = Xocc.reshape((2,1,Xocc.shape[1]))
            assert len(mo_energy.shape) == 2
            mo_energy = mo_energy.reshape((2,1,mo_energy.shape[1]))
        else:
            assert len(fock.shape) == 2
            assert fock.shape[0] == nao
            fock = fock.reshape((1,1)+fock.shape)
            mo_energy = mo_energy.reshape((1,-1))
    if len(fock.shape) == 3:
        fock = fock.reshape((1,)+fock.shape)
    scf_data = {'cell': cell, 'kpts': kpts,
                'Xocc': Xocc, 'isUHF': isUHF,
                'hcore': hcore, 'X': X, 'nmo_pk': nmo_pk,
                'mo_coeff': mo_coeff,
                'nao': nao, 'fock': fock,
                'mo_energy': mo_energy}
    return scf_data

# taken from qmcpack afqmctools.hamiltonian.converter!
def from_qmcpack_complex(data, shape=None):
    if shape is not None:
        return data.view(numpy.complex128).ravel().reshape(shape)
    else:
        return data.view(numpy.complex128).ravel()

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
                    hcore = fh5['Hamiltonian/H1'][:].view(numpy.complex128).ravel()
                    idx = fh5['Hamiltonian/H1_indx'][:]
                    row_ix = idx[::2]
                    col_ix = idx[1::2]
                    hcore = scipy.sparse.csr_matrix((hcore, (row_ix, col_ix))).toarray()
                    hcore = numpy.tril(hcore, -1) + numpy.tril(hcore, 0).conj().T
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

def read_qmcpack_hamiltonian(filename, get_chol=True):
    """Read Hamiltonian from QMCPACK format.

    Parameters
    ----------
    filename : string
        QMPACK Hamiltonian file.

    Returns
    -------
    hamil : dict
        Data read from file.
    """
    hc, chol, enuc, nmo, nelec, nmok, qkk2, nchol_pk, minus_k, kpoints = (
            read_qmcpack_cholesky_kpoint(filename, get_chol=get_chol)
            )
    hamil = {
        'hcore': hc,
        'chol': chol,
        'enuc': enuc,
        'nelec': nelec,
        'nmo': nmo,
        'nmo_pk': nmok,
        'nchol_pk': nchol_pk,
        'minus_k': minus_k,
        'qk_k2': qkk2,
        'kpoints': kpoints
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
    hcore : :class:`numpy.ndarray`
        One-body part of the Hamiltonian.
    chol_vecs : :class:`scipy.sparse.csr_matrix`
        Two-electron integrals. Shape: [nmo*nmo, nchol]
    ecore : float
        Core contribution to the total energy.
    nmo : int
        Number of orbitals.
    nelec : tuple
        Number of electrons.
    nmo_pk : :class:`numpy.ndarray`
        Number of orbitals per kpoint.
    qk_k2 : :class:`numpy.ndarray`
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
            hcore.append(hk.view(numpy.complex128).reshape(nmo,nmo))
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
            chol_vecs.append(get_kpoint_chol(filename, nchol_pk, minus_k, i))
    else:
        chol_vecs = None

    return (hcore, chol_vecs, enuc, int(nmo_tot), (int(nalpha), int(nbeta)),
            nmo_pk, qk_k2, nchol_pk, minus_k, kpoints)

def get_kpoint_chol(filename, nchol_pk, minus_k, i):
    with h5py.File(filename, 'r') as fh5:
        try:
            Lk = get_dset_simple(fh5, 'Hamiltonian/KPFactorized/L{}'.format(i))
            if Lk is None:
                raise KeyError
            nchol = nchol_pk[i]
            Lk = Lk.view(numpy.complex128)[:,:,0]
        except KeyError:
            Lk = get_dset_simple(fh5, 'Hamiltonian/KPFactorized/L{}'.format(minus_k[i]))
            if Lk is None:
                raise TypeError("Could not read Cholesky integrals from file.")
            nchol = nchol_pk[minus_k[i]]
            nchol_pk[i] = nchol
            Lk = Lk.view(numpy.complex128).conj()[:,:,0]
    return Lk
