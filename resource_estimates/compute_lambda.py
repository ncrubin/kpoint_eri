import numpy as np

# import openfermion as of
from utils import (
        # build_cc_object,
        init_from_chkfile,
        )


def get_lambda_unregularized(
        etaPp,
        MPQ,
        h1,
        eri_full,
        use_eri_thc_for_t=False):
    nthc = etaPp.shape[0]

    # computing Least-squares THC residual
    CprP = np.einsum("Pp,Pr->prP", etaPp,
                     etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    BprQ = np.tensordot(CprP, MPQ, axes=([2], [0]))
    Iapprox = np.tensordot(CprP, np.transpose(BprQ), axes=([2], [0]))
    deri = eri_full - Iapprox
    res = 0.5 * np.sum((deri)**2)

    # NOTE: remove in future once we resolve why it was used in the first place.
    # NOTE: see T construction for details.
    eri_thc = np.einsum("Pp,Pr,Qq,Qs,PQ->prqs",
                        etaPp,
                        etaPp,
                        etaPp,
                        etaPp,
                        MPQ,
                        optimize=True)

    # projecting into the THC basis requires each THC factor mu to be nrmlzd.
    # we roll the normalization constant into the central tensor zeta
    SPQ = etaPp.dot(
        etaPp.T)  # (nthc x nmo)  x (nmo x nthc) -> (nthc  x nthc) metric
    cP = np.diag(np.diag(
        SPQ))  # grab diagonal elements. equivalent to np.diag(np.diagonal(SPQ))
    # no sqrts because we have two normalized THC vectors (index by mu and nu)
    # on each side.
    MPQ_normalized = cP.dot(MPQ).dot(cP)  # get normalized zeta in Eq. 11 & 12

    lambda_z = np.sum(np.abs(MPQ_normalized)) * 0.5  # Eq. 13
    # NCR: originally Joonho's code add np.einsum('llij->ij', eri_thc)
    # NCR: I don't know how much this matters.
    if use_eri_thc_for_t:
        # use eri_thc for second coulomb contraction.  This was in the original
        # code which is different than what the paper says.
        T = h1 - 0.5 * np.einsum("illj->ij", eri_full) + np.einsum(
            "llij->ij", eri_thc)  # Eq. 3 + Eq. 18
    else:
        print(eri_full.shape, h1.shape)
        T = h1 - 0.5 * np.einsum("illj->ij", eri_full) + np.einsum(
            "llij->ij", eri_full)  # Eq. 3 + Eq. 18
    #e, v = np.linalg.eigh(T)
    e = np.linalg.eigvalsh(T)  # only need eigenvalues
    lambda_T = np.sum(
        np.abs(e))  # Eq. 19. NOTE: sum over spin orbitals removes 1/2 factor

    lambda_tot = lambda_z + lambda_T  # Eq. 20

    return lambda_tot, nthc, np.sqrt(res), res, lambda_T, lambda_z

if __name__ == '__main__':
    from integral_tools import (
            THCHelper,
            supercell_eris
            )
    from utils import read_qmcpack_thc
    import sys
    chkfile = sys.argv[1]
    thc_file = sys.argv[2]
    max_iter = int(sys.argv[3])
    cell, kmf = init_from_chkfile(chkfile)
    hamil_thc = read_qmcpack_thc(thc_file)
    nmo = hamil_thc
    from utils import read_qmcpack_cholesky_kpoint
    from k2gamma import k2gamma
    cell, kmf = init_from_chkfile(chkfile)
    scmf = k2gamma(kmf, make_real=True)
    eris = supercell_eris(scmf.cell, scmf.mo_coeff)
    nmo = hamil_thc['hcore'].shape[0]
    hcore = hamil_thc['hcore'].real
    muv   = hamil_thc['Muv'].real
    etaPp = hamil_thc['orbs_pu'].real.T.copy()
    nthc = muv.shape[0]
    # print(get_lambda_unregularized(etaPp, muv, hcore, eris))
    # import openfermion as of
    # print(np.max(np.abs(hamil_thc['orbs_pu'].imag)))
    # print(np.max(np.abs(hamil_thc['Muv'].imag)))
    etaPp = hamil_thc['orbs_pu'].real.T.copy()
    muv = hamil_thc['Muv'].real.copy()
    from openfermion.resource_estimates.thc.utils import (
            lbfgsb_opt_thc_l2reg,
            adagrad_opt_thc
            )
    import h5py
    # for max_iter in range(10, 100, 100):
    init = np.hstack((etaPp.ravel(), muv.ravel()))
    params = lbfgsb_opt_thc_l2reg(
            eris,
            nthc,
            initial_guess=init,
            maxiter=max_iter,
            chkfile_name=f'thc/bfgs_reopt_{max_iter}_{nthc}.h5')
    orbs = params[:nthc*nmo].reshape((nthc, nmo))
    muv = params[nthc*nmo:].reshape((nthc, nthc))
    lamb, nthc, de, de2, x, y = get_lambda_unregularized(orbs, muv, hcore, eris)
    # print(max_iter, de2, lamb)
    init = np.hstack((orbs.ravel(), muv.ravel()))
    params = adagrad_opt_thc(
            eris,
            nthc,
            initial_guess=init,
            maxiter=50000,
            # gtol
            chkfile_name=f'thc/adagrad_reopt_{max_iter}_{nthc}.h5')
    orbs = params[:nthc*nmo].reshape((nthc, nmo))
    muv = params[nthc*nmo:].reshape((nthc, nthc))
    lamb, nthc, de, de2, x, y = get_lambda_unregularized(orbs, muv, hcore, eris)
    # print(max_iter, de2, lamb)
