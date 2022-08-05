import argparse
import sys
import numpy as np

from kpoint_eri.resource_estimates import sparse, sf, df, thc
from kpoint_eri.resource_estimates import utils

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
                        default=None, help='Input pyscf .chk file.',
                        required=True)
    parser.add_argument('--output', dest='output', type=str,
                        default=None, help='Output file for results.',
                        required=True
                        )
    parser.add_argument('--integrals', dest='integrals',
                        type=str, default=None,
                        help='Integral file')
    parser.add_argument('--factorization', dest='factorization',
                        type=str,
                        help='sparse, thc, sf, df',
                        required=True)
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

def compute_lambda(options):
    cell, kmf = utils.init_from_chkfile(options.chkfile)
    nk = len(kmf.kpts)
    factorization = options.factorization.lower()
    if factorization == 'sparse':
        print("Sparse Factoriation")
        if options.integrals is not None:
            integrals = sparse.read_hamil_sparse(
                    options.integrals,
                    nk)
        else:
            integrals = None
        lambda_tot, lambda_T, lambda_V, measure, sparsity = sparse.compute_lambda(
                kmf,
                integrals=integrals,
                threshold=options.sparse_thresh
                )
        print(lambda_tot, lambda_T, lambda_V, measure, sparsity)
    elif factorization == 'sf':
        print("Single-Factorization")
        ham = utils.read_cholesky_contiguous(
                options.integrals,
                frac_chol_to_keep=options.sf_thresh)
        lambda_tot, lambda_T, lambda_V, measure = sf.compute_lambda(
                ham['hcore'],
                ham['chol'],
                ham['kpoints'],
                ham['qk_k2'],
                ham['nmo_pk'],
                )
        threshold = options.sf_thresh
    elif factorization == 'df':
        print("Double-Factorization")
        ham = utils.read_cholesky_contiguous(
                options.integrals,
                frac_chol_to_keep=options.sf_thresh)
        df_factors = df.double_factorize_batched(
                ham['chol'],
                ham['qk_k2'],
                ham['nmo_pk'],
                df_thresh=options.df_thresh)
        lambda_tot, lambda_T, lambda_V, measure = df.compute_lambda(
                ham['hcore'],
                df_factors,
                ham['kpoints'],
                ham['qk_k2'],
                ham['nmo_pk'],
                )
        threshold = f"{options.sf_thresh}_{options.df_thresh}"
    elif factorization == 'thc':
        ham = utils.read_qmcpack_thc(options.integrals)
        lambda_tot, lambda_T, lambda_V, measure = thc.compute_lambda(
                ham['hcore'],
                ham['orbs_pu'],
                ham['Muv']
                )
        threshold = options.thc_thresh
    else:
        print("unknown factorization type.")
        sys.exit(1)

    np.savez(
            'lambda_'+options.output+'.npz',
            lambda_tot=lambda_tot,
            lambda_T=lambda_T,
            lambda_V=lambda_V,
            measure=measure,
            sf_thresh=options.sf_thresh,
            df_thresh=options.df_thresh,
            thc_thresh=options.thc_thresh,
            sparse_thresh=options.sparse_thresh,
            )

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    compute_lambda(args)
