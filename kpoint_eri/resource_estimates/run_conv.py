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

chkfile = f"scf_{nk}.chk"
cell, kmf = init_from_chkfile(chkfile)
nk = len(kmf.kpts)

mycc = cc.KRCCSD(kmf)
ecc_ref, t1, t2 = mycc.kernel()
ecct_ref = mycc.ccsd_t(t1=t1, t2=t2, eris=mycc.eris)

# # Sparse ERIS?
from custom_ao2mo import _ERIS
# from integral_tools import ERIHelper

# # Cholesky
print("Running DF")
from integral_tools import CholeskyHelper
from utils import read_qmcpack_cholesky_kpoint
hamil_chol = read_qmcpack_cholesky_kpoint(f'chol/ham_{nk}.h5')
shape = [L.shape[-1] for L in hamil_chol['chol']]
cell.verbose = 0
mycc = cc.KRCCSD(kmf)
with open(f'chol_cc_conv_{nk}.dat', 'w') as f:
    for fac in range(kmf.cell.nao_nr(), np.max(shape), 20):
        print(f"Chol {fac}")
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
        ecct = mycc.ccsd_t(t1=t1, t2=t2, eris=mycc.eris)
        f.write("{} {} {} {} {}\n".format(fac, ecc, ecc_ref, ecct, ecct_ref))
        f.flush()
# # DF
print("Running DF")
from integral_tools import DFHelper
shape = [L.shape[-1] for L in hamil_chol['chol']]
cell.verbose = 0
mycc = cc.KRCCSD(kmf)
nao = cell.nao_nr()
df_thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
with open(f'df_cc_conv_{nk}.dat', 'w') as f:
    for thresh in df_thresholds:
        print(f"DF {thresh}")
        helper = DFHelper(
                    hamil_chol['chol'],
                    hamil_chol['qk_k2'],
                    hamil_chol['kpoints'],
                    hamil_chol['nmo_pk'],
                    chol_thresh=6*nao,
                    df_thresh=thresh)
        eris = _ERIS(mycc, kmf.mo_coeff,  eri_helper=helper, method='incore')
        def ao2mo(self, mo_coeff=None):
            return eris
        mycc.ao2mo = ao2mo
        ecc, t1, t2 = mycc.kernel()
        ecct = mycc.ccsd_t(t1=t1, t2=t2, eris=mycc.eris)
        f.write("{} {} {} {} {}\n".format(thresh, ecc, ecc_ref, ecct, ecct_ref))
        f.flush()
from integral_tools import THCHelper
from utils import read_qmcpack_thc
# thc_base = sys.argv[3]
print("Running THC")
try:
    with open(f'thc_cc_conv_{nk}.dat', 'w') as f:
        scmf = k2gamma(kmf, make_real=False)
        for nthc in range(2, 14, 2):
            print(f"THC: {nthc}")
            thc_file = f'thc/thc_{nk}_{nthc}.h5'
            hamil_thc = read_qmcpack_thc(thc_file)
            helper = THCHelper(
                    hamil_thc['orbs_pu'],
                    hamil_thc['Muv'],
                    )
            from pyscf.pbc import scf
            _scmf = scf.KRHF(scmf.cell)
            mo_coeff = scmf.mo_coeff
            nmo = mo_coeff.shape[1]
            _scmf.mo_coeff = [mo_coeff]
            _scmf.get_hcore = lambda *args : [scmf.get_hcore()]
            _scmf.get_ovlp = lambda *args : [scf.get_ovlp()]
            _scmf.mo_occ = [scmf.mo_occ]
            _scmf.mo_energy = [scmf.mo_energy]
            mycc = cc.KRCCSD(_scmf)
            eris = _ERIS(mycc, [scmf.mo_coeff],  eri_helper=helper, method='incore')
            # print(cc.KRCCSD(scmf))
            def ao2mo(self, mo_coeff=None):
                return eris
            mycc.ao2mo = ao2mo
            # mycc.kernel()
            # ecc_ref, t1, t2 = mycc.kernel()
            ecc, t1, t2 = mycc.kernel()
            ecct = mycc.ccsd_t(t1=t1, t2=t2, eris=mycc.eris)
            f.write("{} {} {} {} {}\n".format(nthc, ecc/nk, ecc_ref, ecct/nk, ecct_ref))
            f.flush()
except:
    with open(f'thc_cc_conv_{nk}.dat', 'w') as f:
        f.write(f"# THC not run for this system {nk}.")
