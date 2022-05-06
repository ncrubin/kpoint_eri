import sys
import numpy as np
from mpi4py import MPI

from pyscf.pbc.lib.chkfile import load_cell
from pyscf.pbc import scf
from pyscf import lib
import argparse

from kpoint_eri.resource_estimates.cholesky import write_hamil_kpoints

def parse_args(args, comm):
    """Parse command-line arguments.

    Parameters
    ----------
    args : list of strings
        command-line arguments.

    Returns
    -------
    options : :class:`argparse.ArgumentParser`
        Command line arguments.
    """

    if comm.rank == 0:
        parser = argparse.ArgumentParser(description = __doc__)
        parser.add_argument('-i', '--input', dest='chkfile', type=str,
                            default=None, help='Input pyscf .chk file.')
        parser.add_argument('-o', '--output', dest='output',
                            type=str, default='chol.h5',
                            help='Output file name for QMCPACK hamiltonian.')
        parser.add_argument('-t', '--lindep-thresh', dest='threshold',
                            type=float, default=1e-16,
                            help='Linear dependency threshold.')
        parser.add_argument('-c', '--chol-thresh', dest='chol_threshold',
                            type=float, default=1e-6,
                            help='Linear dependency threshold.')
        parser.add_argument('-b', '--basis', dest='basis',
                            type=str, default='mo',
                            help='Linear dependency threshold.')
        parser.add_argument('-v', '--verbose', action='count', default=0,
                            help='Verbose output.')

        options = parser.parse_args(args)
    else:
        options = None
    options = comm.bcast(options, root=0)

    if not options.chkfile:
        if comm.rank == 0:
            parser.print_help()
        sys.exit()

    return options


def init_from_chkfile(chkfile):
    cell = load_cell(chkfile)
    cell.build()
    nao = cell.nao_nr()
    energy = np.asarray(lib.chkfile.load(chkfile, 'scf/e_tot'))
    kpts = np.asarray(lib.chkfile.load(chkfile, 'scf/kpts'))
    nkpts = len(kpts)
    kmf = scf.KRHF(cell, kpts)
    kmf.mo_occ = np.asarray(lib.chkfile.load(chkfile, 'scf/mo_occ'))
    kmf.mo_coeff = np.asarray(lib.chkfile.load(chkfile, 'scf/mo_coeff'))
    kmf.mo_energy = np.asarray(lib.chkfile.load(chkfile, 'scf/mo_energy'))
    return cell, kmf

# from afqmctools
def get_ortho_ao(cell, kpts, LINDEP_CUTOFF=0):
    """Generate canonical orthogonalization transformation matrix.

    Parameters
    ----------
    cell : :class:`pyscf.pbc.cell' object.
        PBC cell object.
    kpts : :class:`numpy.array'
        List of kpoints.
    LINDEP_CUTOFF : float
        Linear dependency cutoff. Basis functions whose eigenvalues lie below
        this value are removed from the basis set. Should be set in accordance
        with value in pyscf (pyscf.scf.addons.remove_linear_dep_).

    Returns
    -------
    X : :class:`numpy.array`
        Transformation matrix.
    nmo_per_kpt : :class:`numpy.array`
        Number of OAOs orbitals per kpoint.
    """
    kpts = np.reshape(kpts,(-1,3))
    nkpts = len(kpts)
    nao = cell.nao_nr()
    s1e = lib.asarray(cell.pbc_intor('cint1e_ovlp_sph',hermi=1,kpts=kpts))
    X = np.zeros((nkpts,nao,nao),dtype=np.complex128)
    nmo_per_kpt = np.zeros(nkpts,dtype=np.int32)
    for k in range(nkpts):
        sdiag, Us = np.linalg.eigh(s1e[k])
        nmo_per_kpt[k] = sdiag[sdiag>LINDEP_CUTOFF].size
        norm = np.sqrt(sdiag[sdiag>LINDEP_CUTOFF])
        X[k,:,0:nmo_per_kpt[k]] = Us[:,sdiag>LINDEP_CUTOFF] / norm
    return X, nmo_per_kpt

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    options = parse_args(sys.argv[1:], comm)
    if comm.rank == 0:
        print(" Options")
        print("---------")
        for k, v in vars(options).items():
            print(" {:14s} : {:14s}".format(k, str(v)))
    assert options.basis in ['mo', 'ao', 'oao']

    cell, kmf = init_from_chkfile(options.chkfile)

    nmo_pk = np.array([C.shape[1] for C in kmf.mo_coeff])
    if options.basis == 'mo':
        X = kmf.mo_coeff
    elif options.basis == 'oao':
        if comm.rank == 0:
            # only on root for unique eigen decomposition.
            _X, _nmo_pk = get_ortho_ao(cell, kmf.kpts, LINDEP_CUTOFF=options.threshold)
            assert (_nmo_pk == nmo_pk).all(), "Number of discarded basis functions not consistent."
        else:
            _X = None
        X = comm.bcast(_X)
    else:
        # AO basis
        X = [np.eye(kmf.mo_coeff[k].shape[0])[:,:nmo_pk[k]].copy() for k in range(len(kmf.kpts))]

    scf_data = {
            'cell': cell,
            'hcore': kmf.get_hcore(),
            'X': X,
            'kpts': kmf.kpts,
            'Xocc': kmf.mo_occ,
            'nmo_pk': nmo_pk,
            }

    nk = len(kmf.mo_coeff)
    write_hamil_kpoints(
            comm,
            scf_data,
            options.output,
            options.chol_threshold,
            verbose=options.verbose)
