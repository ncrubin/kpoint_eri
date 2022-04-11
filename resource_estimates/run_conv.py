from utils import (
        build_cc_object,
        init_from_chkfile,
        )

from integral_tools import supercell_eris

import numpy as np
from k2gamma import k2gamma
from pyscf.pbc import cc

chkfile = 'data/scf_nk2.chk'
cell, kmf = init_from_chkfile(chkfile)

# kmf.exxdiv = None
# ref: Ec = -0.247905392728507
# kcc = cc.KCCSD(kmf)
# kcc.kernel()

# supercell_hf = k2gamma(kmf, make_real=True)
# supercell = supercell_hf.cell
# mo_coeff = supercell_hf.mo_coeff
# mo_occ = supercell_hf.mo_occ
# mo_energy = supercell_hf.mo_energy
# eris = supercell_eris(
        # supercell_hf.cell,
        # mo_coeff=mo_coeff
        # )
# hcore = mo_coeff.conj().T @ supercell_hf.get_hcore() @ mo_coeff

# cc = build_cc_object(
        # hcore,
        # eris,
        # np.eye(hcore.shape[0]),
        # int(sum(np.concatenate(kmf.mo_occ))),
        # mo_coeff,
        # mo_occ,
        # mo_energy)
# ecc, t1, t2 = cc.kernel()
# print(ecc/2)

supercell_hf = k2gamma(kmf, make_real=False)
supercell = supercell_hf.cell
mo_coeff = supercell_hf.mo_coeff
mo_occ = supercell_hf.mo_occ
mo_energy = supercell_hf.mo_energy
eris = supercell_eris(
        supercell_hf.cell,
        mo_coeff=mo_coeff
        )
hcore = mo_coeff.conj().T @ supercell_hf.get_hcore() @ mo_coeff

cc = build_cc_object(
        hcore,
        eris,
        np.eye(hcore.shape[0]),
        int(sum(np.concatenate(kmf.mo_occ))),
        mo_coeff,
        mo_occ,
        mo_energy)
ecc, t1, t2 = cc.kernel()
print(ecc/2)
