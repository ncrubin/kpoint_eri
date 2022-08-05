import argparse
import sys
import numpy as np

from kpoint_eri.resource_estimates import cc_helper, utils

def parse_args(args):
    """Parse command-line arguments.

    Parameters
    ----------
    args : list of strings
        command-line arguments.

    Returns
    -------
    options : :class:`argparse.ArgumentParser`
        Command line arguments.
    """

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--scf_chk', dest='chkfile', type=str,
                        default=None, help='Input pyscf .chk file.')
    parser.add_argument('--output', dest='output', type=str,
                        default=None, help='Output file for results.')
    parser.add_argument('--integrals', dest='integrals',
                        type=str,
                        help='Integral file')
    parser.add_argument('--factorization', dest='factorization',
                        type=str,
                        help='sparse, thc, sf, df')
    parser.add_argument('--sf-thresh', dest='sf_thresh',
                        type=float, default=1e-16,
                        help='ERI Threshold (sparse, chol_frac, '
                        'df_thresh, thc rank.')
    parser.add_argument('--sparse-thresh', dest='sparse_thresh',
                        type=float, default=1e-16,
                        help='ERI Threshold (sparse, chol_frac, '
                        'df_thresh, thc rank.')
    parser.add_argument('--thc-thresh', dest='thc_thresh',
                        type=float, default=1e-16,
                        help='ERI Threshold (sparse, chol_frac, '
                        'df_thresh, thc rank.')
    parser.add_argument('--df-thresh', dest='df_thresh',
                        type=float, default=1e-16,
                        help='ERI Threshold (sparse, chol_frac, '
                        'df_thresh, thc rank.')
    options = parser.parse_args(args)

    if not options.chkfile:
        parser.print_help()
        sys.exit()

    return options

def run_approx_cc(options):
    cell, kmf = utils.init_from_chkfile(options.chkfile)
    nk = len(kmf.kpts)
    factorization = options.factorization.lower()
    if factorization == 'sparse':
        print("Sparse Factoriation")
        kcc = cc_helper.build_krcc_sparse_eris(kmf,
                threshold=options.sparse_thresh)
        threshold = options.sparse_thresh
    elif factorization == 'sf':
        print("Single-Factorization")
        ham = utils.read_cholesky_contiguous(
                options.integrals,
                frac_chol_to_keep=options.sf_thresh)
        kcc = cc_helper.build_krcc_sf_eris(
                kmf,
                ham['chol'],
                ham['qk_k2'],
                ham['kpoints']
                )
        threshold = options.sf_thresh
    elif factorization == 'df':
        print("Double-Factorization")
        ham = utils.read_cholesky_contiguous(
                options.integrals,
                frac_chol_to_keep=options.sf_thresh)
        kcc = cc_helper.build_krcc_df_eris(
                kmf,
                ham['chol'],
                ham['qk_k2'],
                ham['kpoints'],
                ham['nmo_pk'],
                df_thresh=options.df_thresh,
                )
        threshold = f"{options.sf_thresh}_{options.df_thresh}"
    elif factorization == 'thc':
        ham = utils.read_qmcpack_thc(options.integrals)
        kcc = cc_helper.build_krcc_thc_eris(
                kmf,
                ham['orbs_pu'],
                ham['Muv']
                )
        threshold = options.thc_thresh
    else:
        print("unknown factorization type.")
        sys.exit(1)

    ecc, t1, t2 = kcc.kernel()
    ecct = kcc.ccsd_t(t1=t1, t2=t2, eris=kcc.eris)
    np.savez(
            options.output+'.npz',
            e_cc=ecc,
            e_cct=ecct,
            sf_thresh=options.sf_thresh,
            df_thresh=options.df_thresh,
            thc_thresh=options.thc_thresh,
            sparse_thresh=options.sparse_thresh,
            )

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    run_approx_cc(args)
