import sys
import os
from pyscf.pbc import scf, gto, tools
from pyscf.pbc.tools.k2gamma import k2gamma
from pyscf import lib
import numpy
import time
import h5py

cell = gto.Cell()
alat0 = 3.6
nk = 2
cell.a = (numpy.ones((3,3))-numpy.eye(3))*alat0/2.0
cell.atom = (('C',0,0,0),('C',numpy.array([0.25,0.25,0.25])*alat0))
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.mesh = [29,29,29]
cell.verbose = 4
cell.build()
supercell = tools.pbc.super_cell(cell, [1, 1, 2])
nk = numpy.array([1,1,2])
kpt = cell.make_kpts(nk)
mf = scf.KRHF(cell,kpt)
mf.exxdiv = 'None'
chkfile = "scf_{:d}.chk".format(numpy.prod(nk))
# ehf = mf.kernel()
from thcpy.transform_basis import (
        init_from_chkfile,
        write_thc_data
        )
cell, kscf = init_from_chkfile(chkfile)

scmf = k2gamma(kscf)
print(scmf.cell.atom)
C = scmf.mo_coeff
import numpy as np
with h5py.File('orbitals_ao.h5') as fh5:
    orbs_ao = fh5['aoR'][:]
with h5py.File('orbitals_mo.h5') as fh5:
    orbs_mo = fh5['aoR'][:]
print(orbs_ao.shape)
print(np.linalg.norm(
    np.einsum('Gp,pi->Gi', orbs_ao, C)
    -
    orbs_mo
    )
    )
print(np.linalg.norm(
    np.dot(orbs_ao, C)
    -
    orbs_mo
    )
    )


from pyscf.pbc import cc
# mycc = cc.CCSD(scmf)
# mycc.kernel()
from pyscf import cc
from pyscf import gto, scf, ao2mo

from pyscf.pbc.df import fft, fft_ao2mo
import numpy as np
df = fft.FFTDF(scmf.cell, kpts=np.zeros((4,3)))
nmo = scmf.mo_coeff.shape[-1]
C = scmf.mo_coeff
mol = gto.M()
mol.nelectron = 8
mol.verbose=0
mf = scf.RHF(mol)
hcore = scmf.get_hcore()
ovlp = scmf.get_ovlp()
mf.get_hcore = lambda *args: hcore.copy()
mf.get_ovlp = lambda *args: ovlp.copy()
mf._eri = ao2mo.restore(
        8,
        df.get_eri(compact=False).reshape((nmo,)*4),
        nmo)
eris_mo = df.ao2mo(C, compact=False).reshape((nmo,)*4)
eris_ao = df.get_eri(compact=False).reshape((nmo,)*4)
# print(mf._eri.e_tot)
mf.mo_coeff = C
mf.mo_occ = scmf.mo_occ
mf.mo_energy = scmf.mo_energy
import h5py
with h5py.File(chkfile, 'r') as fh5:
    e_tot = fh5['/scf/e_tot'][()]
mf.e_tot = 2*e_tot
mycc = cc.RCCSD(mf)
mycc.e_hf = None
eref, t1, t2 = mycc.kernel()
et_ref = mycc.ccsd_t()
hcore = C.T @ scmf.get_hcore() @ C
nocc = int(sum(scmf.mo_occ)//2)
def energy_eri(hcore, eris, nocc):
    e1b = 2*hcore[:nocc, :nocc].trace()
    ecoul = 2*np.einsum('iijj->', eris[:nocc,:nocc,:nocc,:nocc])
    exx = -np.einsum('ijji->', eris[:nocc,:nocc,:nocc,:nocc])
    # print(e1b, ecoul, exx)
    return e1b + ecoul + exx, e1b, ecoul + exx
for i in range(2,34,2):
    filename = '1x1x2_mo/thc_{:d}.h5'.format(i)
    # try:
    with h5py.File(filename, 'r') as fh5:
        orbs = fh5['Hamiltonian/THC/Orbitals'][:,:,0]
        Luv = fh5['Hamiltonian/THC/Luv'][:,:,0]
        e0 = fh5['Hamiltonian/Energies'][0] / nk
        orbs_ov_occ = fh5['Hamiltonian/THC/HalfTransformedOccOrbitals'][:,:,0]
        Muv_ov = fh5['Hamiltonian/THC/HalfTransformedMuv'][:,:,0]
        # Luv = Luv + Luv.conj().T
        Muv = np.dot(Luv, Luv.conj().T).real
        # orbs = np.einsum("mp,mP->pP", C, orbs)
    eri_thc = np.einsum('pP,qP,PQ,rQ,sQ->pqrs', orbs, orbs, Muv, orbs, orbs, optimize=True)
    from pyscf.pbc.cc import CCSD
    from pyscf import ao2mo
    # eri_thc_mo = np.einsum('Pp,Qq,PQRS,Rr,Ss->pqrs', C, C, eri_thc, C, C,
            # optimize=True)
    mf.mo_coeff = np.eye(nmo)
    mf.get_hcore = lambda *args : hcore
    mf.get_ovlp = lambda *args : np.eye(nmo)
    mf._eri = ao2mo.restore(
            8,
            eri_thc,
            # eris_mo,
            # eri_thc_mo,
            nmo)
    mycc = cc.RCCSD(mf)
    mycc.e_hf = None
    ecc, t1, t2 = mycc.kernel()
    et = mycc.ccsd_t()
    print(i,
            # np.linalg.norm(eris_ao-eri_thc),
            np.linalg.norm(eris_mo.reshape((nmo*nmo,)*2)
                           -
                           eri_thc.reshape((nmo*nmo,)*2),
                           ord=1),
            # np.linalg.norm(eris_mo-eri_thc_mo),
            # energy_eri(hcore, eri_thc, nocc)[2]-energy_eri(hcore, eris_mo,
                # nocc)[2],
            ecc - eref,
            et - et_ref,
            )
    # except:
        # pass
