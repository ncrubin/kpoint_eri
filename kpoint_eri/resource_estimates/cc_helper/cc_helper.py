from pyscf.pbc import cc, scf

from kpoint_eri.resource_estimates import cc_helper
from kpoint_eri.resource_estimates.k2gamma import k2gamma

def build_cc(approx_cc, helper):
    eris = cc_helper._ERIS(
            approx_cc,
            approx_cc.mo_coeff,
            eri_helper=helper,
            method='incore')
    def ao2mo(self, mo_coeff=None):
        return eris
    approx_cc.ao2mo = ao2mo
    return approx_cc

def build_krcc_sparse_eris(pyscf_mf, threshold=1e-5):
    approx_cc = cc.KRCCSD(pyscf_mf)
    helper = cc_helper.SparseHelper(
            pyscf_mf.with_df,
            pyscf_mf.mo_coeff,
            pyscf_mf.kpts,
            threshold=threshold
            )
    return build_cc(approx_cc, helper)

def build_krcc_sf_eris(
        pyscf_mf,
        chol,
        mom_map,
        kpoints
        ):
    approx_cc = cc.KRCCSD(pyscf_mf)
    helper = cc_helper.SingleFactorizationHelper(
            chol,
            mom_map,
            kpoints
            )
    return build_cc(approx_cc, helper)

def build_krcc_df_eris(
        pyscf_mf,
        chol,
        mom_map,
        kpoints,
        nmo_pk,
        df_thresh=1e-5
        ):
    approx_cc = cc.KRCCSD(pyscf_mf)
    helper = cc_helper.DoubleFactorizationHelper(
            chol,
            mom_map,
            kpoints,
            nmo_pk,
            df_thresh=df_thresh
            )
    return build_cc(approx_cc, helper)

def build_krcc_thc_eris(
        pyscf_mf,
        etapP,
        MPQ
        ):
    scmf = k2gamma(pyscf_mf)
    kscmf = scf.KRHF(scmf.cell)
    kscmf.mo_coeff = [scmf.mo_coeff]
    kscmf.mo_occ = [scmf.mo_occ]
    kscmf.get_hcore = lambda *args: [scmf.get_hcore()]
    kscmf.get_ovlp = lambda *args: [scmf.get_ovlp()]
    kscmf.mo_energy = [scmf.mo_energy]
    approx_cc = cc.KRCCSD(kscmf)
    helper = cc_helper.THCHelper(
            etapP,
            MPQ
            )
    return build_cc(approx_cc, helper)
