from utils import (
        build_cc_object,
        init_from_chkfile,
        )

from integral_tools import supercell_eris
import sys

import numpy as np
from k2gamma import k2gamma
from pyscf.pbc import cc

nk = sys.argv[1]
chkfile = f'data/scf_nk{nk}.chk'
cell, kmf = init_from_chkfile(chkfile)

mycc = cc.KRCCSD(kmf)
ecc_ref, t1, t2 = mycc.kernel()
ecct_ref = mycc.ccsd_t(t1=t1, t2=t2, eris=mycc.eris)

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
# from integral_tools import CholeskyHelper
# from utils import read_qmcpack_cholesky_kpoint
# chkfile = 'data/scf_nk8.chk'
# cell, kmf = init_from_chkfile(chkfile)
# hamil_chol = read_qmcpack_cholesky_kpoint('data/chol_nk8.h5')
# shape = [L.shape[-1] for L in hamil_chol['chol']]
# cell.verbose = 0
# mycc = cc.KRCCSD(kmf)
# for fac in range(26, np.max(shape), 20):
    # helper = CholeskyHelper(
                # hamil_chol['chol'],
                # hamil_chol['qk_k2'],
                # hamil_chol['kpoints'],
                # chol_thresh=fac)
    # eris = _ERIS(mycc, kmf.mo_coeff,  eri_helper=helper, method='incore')
    # def ao2mo(self, mo_coeff=None):
        # return eris
    # mycc.ao2mo = ao2mo
    # ecc, t1, t2 = mycc.kernel()
    # print(fac, ecc, ecc_ref)
from integral_tools import DFHelper
from utils import read_qmcpack_cholesky_kpoint
# chkfile = 'data/scf_nk2.chk'
cell, kmf = init_from_chkfile(chkfile)
# hamil_chol = read_qmcpack_cholesky_kpoint(f'data/chol_nk{nk}.h5')
# shape = [L.shape[-1] for L in hamil_chol['chol']]
# cell.verbose = 0
# mycc = cc.KRCCSD(kmf)
# nmo = 26
# df_thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
# for thresh in df_thresholds:
    # helper = DFHelper(
                # hamil_chol['chol'],
                # hamil_chol['qk_k2'],
                # hamil_chol['kpoints'],
                # hamil_chol['nmo_pk'],
                # chol_thresh=4*nmo,
                # df_thresh=thresh)
    # eris = _ERIS(mycc, kmf.mo_coeff,  eri_helper=helper, method='incore')
    # def ao2mo(self, mo_coeff=None):
        # return eris
    # mycc.ao2mo = ao2mo
    # ecc, t1, t2 = mycc.kernel()
    # ecct = mycc.ccsd_t(t1=t1, t2=t2, eris=mycc.eris)
    # print(thresh, ecc, ecc_ref, ecct, ecct_ref)
from integral_tools import THCHelper
from utils import read_qmcpack_thc
chkfile = 'data/scf_nk2.chk'
cell, kmf = init_from_chkfile(chkfile)
hamil_thc = read_qmcpack_thc('data/thc_nk2_cthc12.h5')
print(list(hamil_thc.keys()))
helper = THCHelper(
        hamil_thc['orbs_pu'],
        hamil_thc['Muv'],
        )
scmf = k2gamma(kmf, make_real=True)
from pyscf.pbc import scf
_scmf = scf.KRHF(scmf.cell)
mo_coeff = scmf.mo_coeff
nmo = mo_coeff.shape[1]
_scmf.mo_coeff = [mo_coeff]
_scmf.get_hcore = lambda *args : [scmf.get_hcore()]
_scmf.get_ovlp = lambda *args : [scf.get_ovlp()]
_scmf.mo_occ = [scmf.mo_occ]
_scmf.mo_energy = [scmf.mo_energy]
# scmf.dump_flags()
# _scmf.dump_flags()
# print(scmf.energy_tot())
# print(_scmf.energy_tot())
eris = helper.get_eri([])
from utils import energy_eri
hcore = mo_coeff.conj().T @ scmf.get_hcore() @ mo_coeff
energy_thc = energy_eri(
                hcore,
                eris,
                4,
                hamil_thc['enuc'])
# print(energy_thc)
mycc = cc.KRCCSD(_scmf)
eris = _ERIS(mycc, [scmf.mo_coeff],  eri_helper=helper, method='incore')
# print(cc.KRCCSD(scmf))
def ao2mo(self, mo_coeff=None):
    return eris
mycc.ao2mo = ao2mo
# mycc.kernel()
ecc_ref, t1, t2 = mycc.kernel()
print(ecc_ref/int(nk))
