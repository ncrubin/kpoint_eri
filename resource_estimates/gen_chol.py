import sys
import numpy as np
from mpi4py import MPI

from pyscf.pbc.lib.chkfile import load_cell
from pyscf.pbc import tools
from pyscf.pbc import scf
from pyscf import lib

from cholesky import write_hamil_kpoints

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

filename = sys.argv[1]
cell, kmf = init_from_chkfile(filename)

comm = MPI.COMM_WORLD

scf_data = {
        'cell': cell,
        'hcore': kmf.get_hcore(),
        'X': kmf.mo_coeff,
        'kpts': kmf.kpts,
        'Xocc': kmf.mo_occ,
        'nmo_pk': np.array([C.shape[1] for C in kmf.mo_coeff])
        }

nk = len(kmf.mo_coeff)

write_hamil_kpoints(comm, scf_data, chol_path+'ham_{:d}.h5'.format(nk), 1e-5)
