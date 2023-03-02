"""Utilities for overwriting CCSD pbc eris with integral factorizations."""
from pyscf.pbc import cc
from kpoint_eri.resource_estimates.cc_helper.custom_ao2mo import _ERIS

def build_cc(approx_cc, integral_helper):
    """Build modified coupled cluster object which uses integral_helper to
    construct ERIs.

    Args:
        approx_cc (cc.CCSD): pyscf pbc CCSD object we will overwrite the integrals computation of.
        integral_helper (IntegralHelper object): Integral helper that builds _ERIS object. 

    Returns:
        approx_cc (cc.CCSD): Updated pyscf pbc CCSD object.
    """
    eris = _ERIS(
            approx_cc,
            approx_cc.mo_coeff,
            eri_helper=integral_helper,
            method='incore')
    def ao2mo(self, mo_coeff=None):
        return eris
    approx_cc.ao2mo = ao2mo
    return approx_cc

def compute_emp2_approx(mf, helper):
    approx_cc = cc.KRCCSD(mf)
    approx_cc = build_cc(approx_cc, helper)
    eris = approx_cc.ao2mo(lambda x: x)
    emp2, _, _ = approx_cc.init_amps(eris)
    emp2 += mf.e_tot
    return emp2