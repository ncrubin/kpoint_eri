import h5py
import numpy
from pyscf import lib
from pyscf.lib.chkfile import load, load_mol
from pyscf.pbc.lib.chkfile import load_cell

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

def read_qmcpack_thc(filename):
    with open(filename, 'r') as fh5:
        hcore = fh5['Hamiltonian/hcore'][:]
        Luv = fh5['Hamiltonian/Luv'][:]
        Muv = np.dot(Luv, Luv.conj().T)
        orbs_pu = fh5['Hamiltonian/Oribtals'][:]
        naux = fh5['Hamiltonian/THC/dims'][:][1]
        e0 = fh5['Hamiltonian/Energies'][0]
        assert Muv.shape == (naux, naux)

    hamil = {
            'hcore': hcore,
            'orbs_pu'; orbs_pu,
            'Muv': Muv
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

    hamil = {
        'hcore': hcore,
        'chol': chol_vecs,
        'enuc': enuc,
        'nelec': nelec,
        'nmo': int(nmo),
        'nmo_pk': nmo_pk,
        'nchol_pk': nchol_pk,
        'minus_k': minus_k,
        'qk_k2': qk_k2,
        'kpoints': kpoints
        }
    return hamil

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
