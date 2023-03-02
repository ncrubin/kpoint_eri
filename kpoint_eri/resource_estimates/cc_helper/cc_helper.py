"""Utilities for overwriting CCSD pbc eris with integral factorizations."""
from kpoint_eri.resource_estimates import cc_helper

def build_cc(approx_cc, integral_helper):
    """Build modified coupled cluster object which uses integral_helper to
    construct ERIs.

    Args:
        approx_cc (cc.CCSD): pyscf pbc CCSD object we will overwrite the integrals computation of.
        integral_helper (IntegralHelper object): Integral helper that builds _ERIS object. 

    Returns:
        approx_cc (cc.CCSD): Updated pyscf pbc CCSD object.
    """
    eris = cc_helper._ERIS(
            approx_cc,
            approx_cc.mo_coeff,
            eri_helper=helper,
            method='incore')
    def ao2mo(self, mo_coeff=None):
        return eris
    approx_cc.ao2mo = ao2mo
    return approx_cc