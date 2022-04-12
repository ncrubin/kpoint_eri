from utils import (
        build_cc_object,
        init_from_chkfile,
        )

from integral_tools import supercell_eris

import numpy as np
from k2gamma import k2gamma
from pyscf.pbc import cc

chkfile = 'data/scf_nk8.chk'
cell, kmf = init_from_chkfile(chkfile)

mycc = cc.KRCCSD(kmf)
ecc_ref, t1, t2 = mycc.kernel()

# Sparse ERIS?
from custom_ao2mo import _ERIS
from integral_tools import ERIHelper
# helper = ERIHelper(kmf.with_df, kmf.mo_coeff, kmf.kpts)
# mycc = cc.KRCCSD(kmf)
# eris = _ERIS(mycc, kmf.mo_coeff,  eri_helper=helper, method='incore')
# def ao2mo(self, mo_coeff=None):
    # return eris
# mycc.ao2mo = ao2mo
# mycc.kernel()

# Cholesky
from integral_tools import CholeskyHelper
from utils import read_qmcpack_cholesky_kpoint
chkfile = 'data/scf_nk8.chk'
cell, kmf = init_from_chkfile(chkfile)
hamil_chol = read_qmcpack_cholesky_kpoint('data/chol_nk8.h5')
shape = [L.shape[-1] for L in hamil_chol['chol']]
cell.verbose = 0
mycc = cc.KRCCSD(kmf)
for fac in range(26, np.max(shape), 20):
    helper = CholeskyHelper(
                hamil_chol['chol'],
                hamil_chol['qk_k2'],
                hamil_chol['kpoints'],
                chol_thresh=fac)
    eris = _ERIS(mycc, kmf.mo_coeff,  eri_helper=helper, method='incore')
    def ao2mo(self, mo_coeff=None):
        return eris
    mycc.ao2mo = ao2mo
    ecc, t1, t2 = mycc.kernel()
    print(fac, ecc, ecc_ref)
from integral_tools import THCHelper
from utils import read_qmcpack_thc
chkfile = 'data/scf_nk4.chk'
# cell, kmf = init_from_chkfile(chkfile)
# hamil_thc = read_qmcpack_thc('data/thc_nk2_cthc12.h5')
# print(list(hamil_thc.keys()))
# helper = THCHelper(
        # hamil_thc['orbs_pu'],
        # hamil_thc['Muv'],
        # )
# scmf = k2gamma(kmf, make_real=True)
# from pyscf.pbc import scf
# _scmf = scf.KRHF(scmf.cell)
# mo_coeff = scmf.mo_coeff
# nmo = mo_coeff.shape[1]
# hcore = mo_coeff.conj().T @ scmf.get_hcore() @ mo_coeff
# _scmf.mo_coeff = [np.eye(nmo)]
# _scmf.get_hcore = lambda *args : [hcore]
# _scmf.get_ovlp = lambda *args : [np.eye(nmo)]
# _scmf.mo_occ = [scmf.mo_occ]
# _scmf.mo_energy = [scmf.mo_energy]
# mycc = cc.KRCCSD(_scmf)
# eris = _ERIS(mycc, [scmf.mo_coeff],  eri_helper=helper, method='incore')
# def ao2mo(self, mo_coeff=None):
    # return eris
# mycc.ao2mo = ao2mo
# mycc.kernel()
