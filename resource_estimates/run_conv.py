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

# mycc = cc.KRCCSD(kmf)
# mycc.kernel()

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
chkfile = 'data/scf_nk2.chk'
cell, kmf = init_from_chkfile(chkfile)
hamil_chol = read_qmcpack_cholesky_kpoint('data/chol_nk2.h5')
helper = CholeskyHelper(
        hamil_chol['chol'],
        hamil_chol['qk_k2'],
        hamil_chol['kpoints'])
mycc = cc.KRCCSD(kmf)
eris = _ERIS(mycc, kmf.mo_coeff,  eri_helper=helper, method='incore')
def ao2mo(self, mo_coeff=None):
    return eris
mycc.ao2mo = ao2mo
mycc.kernel()
