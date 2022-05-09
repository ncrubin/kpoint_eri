import sys
import numpy as np
from mpi4py import MPI

from pyscf.pbc.lib.chkfile import load_cell
from pyscf.pbc import scf
from pyscf import lib
import argparse

from kpoint_eri.resource_estimates import sparse

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
        parser.add_argument('-c', '--threshold', dest='threshold',
                            type=float, default=0.0,
                            help='Sparse Threshold.')
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

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    options = parse_args(sys.argv[1:], comm)
    if comm.rank == 0:
        print(" Options")
        print("---------")
        for k, v in vars(options).items():
            print(" {:14s} : {:14s}".format(k, str(v)))

    cell, kmf = init_from_chkfile(options.chkfile)

    nk = len(kmf.mo_coeff)
    sparse.write_hamil_sparse(
            comm,
            kmf,
            filename=options.output,
            threshold=options.threshold
            )
