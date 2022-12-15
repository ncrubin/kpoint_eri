import h5py
import numpy as np
from scipy.optimize import minimize
from uuid import uuid4
import math

from jax.config import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp


from openfermion.resource_estimates.thc.utils import adagrad
from openfermion.resource_estimates.thc.utils.thc_factorization import CallBackStore

from kpoint_eri.factorizations.isdf import solve_kmeans_kpisdf
from kpoint_eri.resource_estimates.utils.misc_utils import (
    build_momentum_transfer_mapping,
)


def load_thc_factors(chkfile_name):
    with h5py.File(chkfile_name, "r") as fh5:
        chi = fh5["chi"][:]
        G_mapping = fh5["G_mapping"][:]
        num_kpts = G_mapping.shape[0]
        zeta = np.zeros((num_kpts,), dtype=object)
        for iq in range(G_mapping.shape[0]):
            zeta[iq] = fh5[f"zeta_{iq}"][:]
    return chi, zeta, G_mapping


def get_zeta_size(zeta):
    return sum([z.size for z in zeta])


def unpack_thc_factors(xcur, num_thc, num_orb, num_kpts, num_G_per_Q):
    # leaf tensor (num_kpts, num_orb, num_thc)
    chi_size = num_kpts * num_orb * num_thc
    chi_real = xcur[:chi_size].reshape(num_kpts, num_orb, num_thc)
    chi_imag = xcur[chi_size : 2 * chi_size].reshape(num_kpts, num_orb, num_thc)
    chi = chi_real + 1j * chi_imag
    zeta_packed = xcur[2 * chi_size :]
    zeta = []
    start = 0
    for iQ in range(num_kpts):
        num_G = num_G_per_Q[iQ]
        size = num_G * num_G * num_thc * num_thc
        zeta_real = zeta_packed[start : start + size].reshape(
            (num_G, num_G, num_thc, num_thc)
        )
        zeta_imag = zeta_packed[start + size : start + 2 * size].reshape(
            (num_G, num_G, num_thc, num_thc)
        )
        zeta.append(zeta_real + 1j * zeta_imag)
        start += 2 * size
    return chi, zeta


def pack_thc_factors(chi, zeta, buffer):
    # leaf tensor (num_kpts, num_orb, num_thc)
    assert len(chi.shape) == 3
    buffer[: chi.size] = chi.real.ravel()
    buffer[chi.size : 2 * chi.size] = chi.imag.ravel()
    start = 2 * chi.size
    num_kpts = len(zeta)
    for iQ in range(num_kpts):
        size = zeta[iQ].size
        buffer[start : start + size] = zeta[iQ].real.ravel()
        buffer[start + size : start + 2 * size] = zeta[iQ].imag.ravel()
        start += 2 * size


@jax.jit
def compute_objective_batched(chis, zetas, chols, norm_factors, penalty_param=0):
    eri_thc = jnp.einsum(
        "Jpm,Jqm,Jmn,Jrn,Jsn->Jpqrs",
        chis[0].conj(),
        chis[1],
        zetas,
        chis[2].conj(),
        chis[3],
        optimize=True,
    )
    eri_ref = jnp.einsum("Jnpq,Jnrs->Jpqrs", chols[0], chols[1], optimize=True)
    deri = eri_thc - eri_ref
    norm_left = norm_factors[0] * norm_factors[1]
    norm_right = norm_factors[2] * norm_factors[3]
    MPQ_normalized = jnp.einsum("JP,JPQ,JQ->JPQ", norm_left, zetas, norm_right,
                               optimize=True)

    lambda_z = jnp.sum(jnp.einsum("JPQ->J", 0.5 * jnp.abs(MPQ_normalized)) ** 2.0)

    res = 0.5 * jnp.sum((jnp.abs(deri)) ** 2) + penalty_param * lambda_z
    return res.conj()


def prepare_batched_data_indx_arrays(
    momentum_map, Gpq_map, num_thc, num_mo, batch_size=None, verbose=False
):
    """Create arrays to batch over."""
    num_kpts = momentum_map.shape[0]
    indx_pqrs = np.zeros((num_kpts, num_kpts**2, 4), dtype=jnp.int32)
    zetas = np.zeros((num_kpts, num_kpts**2, 2), dtype=jnp.int32)
    for iq in range(num_kpts):
        indx = 0
        for ik in range(num_kpts):
            ik_minus_q = momentum_map[iq, ik]
            Gpq = Gpq_map[iq, ik]
            for ik_prime in range(num_kpts):
                ik_prime_minus_q = momentum_map[iq, ik_prime]
                Gsr = Gpq_map[iq, ik_prime]
                indx_pqrs[iq, indx] = [ik, ik_minus_q, ik_prime_minus_q, ik_prime]
                zetas[iq, indx] = [Gpq, Gsr]
                indx += 1
    return indx_pqrs, zetas


@jax.jit
def get_batched_data_1indx(array, indx):
    return array[indx]


@jax.jit
def get_batched_data_2indx(array, indxa, indxb):
    return array[indxa, indxb]


def thc_objective_regularized_batched(
    xcur,
    num_orb,
    num_thc,
    momentum_map,
    Gpq_map,
    chol,
    indx_arrays,
    batch_size,
    penalty_param=0.0,
):
    num_kpts = momentum_map.shape[0]
    num_G_per_Q = [len(np.unique(GQ)) for GQ in Gpq_map]
    chi, zeta = unpack_thc_factors(xcur, num_thc, num_orb, num_kpts, num_G_per_Q)
    nthc = zeta[0].shape[-1]
    # Normalization factor, no factor of sqrt as there are 4 chis in total when
    # building ERI.
    norm_kP = jnp.einsum("kpP,kpP->kP", chi.conj(), chi, optimize=True)
    num_batches = math.ceil(num_kpts**2 / batch_size)

    indx_pqrs, indx_zeta = indx_arrays
    objective = 0.0
    for iq in range(num_kpts):
        for ibatch in range(num_batches):
            start = ibatch * batch_size
            end = (ibatch + 1) * batch_size
            chi_p = get_batched_data_1indx(chi, indx_pqrs[iq, start:end, 0])
            chi_q = get_batched_data_1indx(chi, indx_pqrs[iq, start:end, 1])
            chi_r = get_batched_data_1indx(chi, indx_pqrs[iq, start:end, 2])
            chi_s = get_batched_data_1indx(chi, indx_pqrs[iq, start:end, 3])
            norm_k1 = get_batched_data_1indx(norm_kP, indx_pqrs[iq, start:end, 0])
            norm_k2 = get_batched_data_1indx(norm_kP, indx_pqrs[iq, start:end, 1])
            norm_k3 = get_batched_data_1indx(norm_kP, indx_pqrs[iq, start:end, 2])
            norm_k4 = get_batched_data_1indx(norm_kP, indx_pqrs[iq, start:end, 3])
            zeta_batch = get_batched_data_2indx(
                zeta[iq], indx_zeta[iq, start:end, 0], indx_zeta[iq, start:end, 1]
            )
            chol_batch_pq = get_batched_data_2indx(
                chol, indx_pqrs[iq, start:end, 0], indx_pqrs[iq, start:end, 1]
            )
            chol_batch_rs = get_batched_data_2indx(
                chol, indx_pqrs[iq, start:end, 2], indx_pqrs[iq, start:end, 3]
            )
            objective += compute_objective_batched(
                (chi_p, chi_q, chi_r, chi_s),
                zeta_batch,
                (chol_batch_pq, chol_batch_rs),
                (norm_k1, norm_k2, norm_k3, norm_k4),
                penalty_param=penalty_param,
            )
    return objective / num_kpts


def thc_objective_regularized(
    xcur,
    num_orb,
    num_thc,
    momentum_map,
    Gpq_map,
    chol,
    penalty_param=0.0,
):
    res = 0.0
    num_kpts = momentum_map.shape[0]
    num_G_per_Q = [len(np.unique(GQ)) for GQ in Gpq_map]
    chi, zeta = unpack_thc_factors(xcur, num_thc, num_orb, num_kpts, num_G_per_Q)
    num_kpts = momentum_map.shape[0]
    norm_kP = jnp.einsum("kpP,kpP->kP", chi.conj(), chi, optimize=True)
    for iq in range(num_kpts):
        for ik in range(num_kpts):
            ik_minus_q = momentum_map[iq, ik]
            Gpq = Gpq_map[iq, ik]
            for ik_prime in range(num_kpts):
                ik_prime_minus_q = momentum_map[iq, ik_prime]
                Gsr = Gpq_map[iq, ik_prime]
                eri_thc = jnp.einsum(
                    "pm,qm,mn,rn,sn->pqrs",
                    chi[ik].conj(),
                    chi[ik_minus_q],
                    zeta[iq][Gpq, Gsr],
                    chi[ik_prime_minus_q].conj(),
                    chi[ik_prime],
                )
                eri_ref = jnp.einsum(
                    "npq,nsr->pqrs",
                    chol[ik, ik_minus_q],
                    chol[ik_prime, ik_prime_minus_q].conj(),
                )
                deri = eri_thc - eri_ref
                norm_left = norm_kP[ik] * norm_kP[ik_minus_q]
                norm_right = norm_kP[ik_prime_minus_q] * norm_kP[ik_prime]
                MPQ_normalized = jnp.einsum(
                    "P,PQ,Q->PQ", norm_left, zeta[iq][Gpq, Gsr], norm_right
                )

                lambda_z = 0.5 * jnp.sum(jnp.abs(MPQ_normalized))
                res += 0.5 * jnp.sum((jnp.abs(deri)) ** 2) + penalty_param * (
                    lambda_z**2
                )

    return res / num_kpts


def lbfgsb_opt_kpthc_l2reg(
    chi,
    zeta,
    momentum_map,
    Gpq_map,
    chol,
    chkfile_name=None,
    random_seed=None,
    maxiter=150_000,
    disp_freq=98,
    penalty_param=None,
    disp=False,
):
    """
    Least-squares fit of two-electron integral tensors with  L-BFGS-B with
    l2-regularization of lambda

    disp is ignored.
    disp_freq sets the freqnecy of printing
    """
    if disp_freq > 98 or disp_freq < 1:
        raise ValueError(
            "disp_freq {} is not valid. must be between [1, 98]".format(disp_freq)
        )

    if chkfile_name is None:
        # chkfile_name = str(uuid4()) + '.h5'
        callback_func = None
    else:
        # callback func stores checkpoints
        # callback_func = CallBackStore(chkfile_name)
        callback_func = None

    initial_guess = np.zeros(2 * (chi.size + get_zeta_size(zeta)), dtype=np.float64)
    pack_thc_factors(chi, zeta, initial_guess)
    assert len(chi.shape) == 3
    assert len(zeta[0].shape) == 4
    num_kpts = chi.shape[0]
    num_orb = chi.shape[1]
    num_thc = chi.shape[-1]
    assert zeta[0].shape[-1] == num_thc
    assert zeta[0].shape[-2] == num_thc
    print(initial_guess)
    loss = thc_objective_regularized(
        jnp.array(initial_guess),
        num_orb,
        num_thc,
        momentum_map,
        Gpq_map,
        jnp.array(chol),
        penalty_param=0.0,
    )
    reg_loss = thc_objective_regularized(
        jnp.array(initial_guess),
        num_orb,
        num_thc,
        momentum_map,
        Gpq_map,
        jnp.array(chol),
        penalty_param=1.0,
    )
    # set penalty
    lambda_z = (reg_loss - loss) ** 0.5
    if penalty_param is None:
        # loss + lambda_z^2 - loss
        penalty_param = loss / lambda_z
    print("loss {}".format(loss))
    print("lambda_z {} {} {}".format(lambda_z, penalty_param, loss))
    print("penalty_param {}".format(penalty_param))

    # L-BFGS-B optimization
    thc_grad = jax.grad(thc_objective_regularized, argnums=[0])
    print("Initial Grad")
    print(
        thc_grad(
            jnp.array(initial_guess),
            num_orb,
            num_thc,
            momentum_map,
            Gpq_map,
            jnp.array(chol),
            penalty_param,
        )
    )
    # print()
    res = minimize(
        thc_objective_regularized,
        initial_guess,
        args=(num_orb, num_thc, momentum_map, Gpq_map, jnp.array(chol), penalty_param),
        jac=thc_grad,
        method="L-BFGS-B",
        options={"disp": None, "iprint": disp_freq, "maxiter": maxiter},
        callback=callback_func,
    )

    # print(res)
    params = np.array(res.x)
    if chkfile_name is not None:
        num_G_per_Q = [len(np.unique(GQ)) for GQ in Gpq_map]
        chi, zeta = unpack_thc_factors(params, num_thc, num_orb, num_kpts, num_G_per_Q)
        with h5py.File(chkfile_name, "w") as fh5:
            fh5["chi"] = chi
            fh5["G_mapping"] = Gpq_map
            for iq in range(num_kpts):
                fh5[f"zeta_{iq}"] = zeta[iq]
    return np.array(params)


def lbfgsb_opt_kpthc_l2reg_batched(
    chi,
    zeta,
    momentum_map,
    Gpq_map,
    chol,
    chkfile_name=None,
    random_seed=None,
    maxiter=150_000,
    disp_freq=98,
    penalty_param=None,
    disp=False,
    batch_size=None,
):
    """
    Least-squares fit of two-electron integral tensors with  L-BFGS-B with
    l2-regularization of lambda

    disp is ignored.
    disp_freq sets the freqnecy of printing
    """
    if disp_freq > 98 or disp_freq < 1:
        raise ValueError(
            "disp_freq {} is not valid. must be between [1, 98]".format(disp_freq)
        )

    if chkfile_name is None:
        # chkfile_name = str(uuid4()) + '.h5'
        callback_func = None
    else:
        # callback func stores checkpoints
        # callback_func = CallBackStore(chkfile_name)
        callback_func = None

    initial_guess = np.zeros(2 * (chi.size + get_zeta_size(zeta)), dtype=np.float64)
    pack_thc_factors(chi, zeta, initial_guess)
    assert len(chi.shape) == 3
    assert len(zeta[0].shape) == 4
    num_kpts = chi.shape[0]
    num_orb = chi.shape[1]
    num_thc = chi.shape[-1]
    assert zeta[0].shape[-1] == num_thc
    assert zeta[0].shape[-2] == num_thc
    if batch_size is None:
        batch_size = num_kpts**2
    indx_arrays = prepare_batched_data_indx_arrays(
        momentum_map, Gpq_map, num_thc, num_orb
    )
    data_amount = batch_size * (
        4 * num_orb * num_thc + num_thc * num_thc  # chi[p,m] + zeta[m,n]
    )
    data_size_gb = (data_amount * 16) / (1024**3)
    print(f"Batch size in GB: {data_size_gb}")
    loss = thc_objective_regularized_batched(
        jnp.array(initial_guess),
        num_orb,
        num_thc,
        momentum_map,
        Gpq_map,
        jnp.array(chol),
        indx_arrays,
        batch_size,
        penalty_param=0.0,
    )
    reg_loss = thc_objective_regularized_batched(
        jnp.array(initial_guess),
        num_orb,
        num_thc,
        momentum_map,
        Gpq_map,
        jnp.array(chol),
        indx_arrays,
        batch_size,
        penalty_param=1.0,
    )
    print("loss {}".format(loss))
    # set penalty
    lambda_z = (reg_loss - loss) ** 0.5
    if penalty_param is None:
        # loss + lambda_z^2 - loss
        penalty_param = loss / lambda_z
    print("lambda_z {}".format(lambda_z))
    print("penalty_param {}".format(penalty_param))

    # L-BFGS-B optimization
    thc_grad = jax.grad(thc_objective_regularized_batched, argnums=[0])
    print("Initial Grad")
    print(
        thc_grad(
            jnp.array(initial_guess),
            num_orb,
            num_thc,
            momentum_map,
            Gpq_map,
            jnp.array(chol),
            indx_arrays,
            batch_size,
            penalty_param,
        )
    )
    res = minimize(
        thc_objective_regularized_batched,
        initial_guess,
        args=(
            num_orb,
            num_thc,
            momentum_map,
            Gpq_map,
            jnp.array(chol),
            indx_arrays,
            batch_size,
            penalty_param,
        ),
        jac=thc_grad,
        method="L-BFGS-B",
        options={"disp": None, "iprint": disp_freq, "maxiter": maxiter},
        callback=callback_func,
    )

    params = np.array(res.x)
    if chkfile_name is not None:
        num_G_per_Q = [len(np.unique(GQ)) for GQ in Gpq_map]
        chi, zeta = unpack_thc_factors(params, num_thc, num_orb, num_kpts, num_G_per_Q)
        with h5py.File(chkfile_name, "w") as fh5:
            fh5["chi"] = chi
            fh5["G_mapping"] = Gpq_map
            for iq in range(num_kpts):
                fh5[f"zeta_{iq}"] = zeta[iq]
    return np.array(params)


def adagrad_opt_kpthc_batched(
    chi,
    zeta,
    momentum_map,
    Gpq_map,
    chol,
    batch_size=None,
    chkfile_name=None,
    random_seed=None,
    stepsize=0.01,
    momentum=0.9,
    maxiter=50_000,
    gtol=1.0e-5,
):
    """
    THC opt usually starts with BFGS and then is completed with Adagrad or other
    first order solver.  This  function implements an Adagrad optimization.

    Optimization runs for 50 K iterations.  This is the ONLY stopping cirteria
    used in the FT-THC paper by Lee et al.
    """
    assert len(chi.shape) == 3
    assert len(zeta[0].shape) == 4
    num_kpts = chi.shape[0]
    num_orb = chi.shape[1]
    num_thc = chi.shape[-1]
    assert zeta[0].shape[-1] == num_thc
    assert zeta[0].shape[-2] == num_thc
    if chkfile_name is None:
        chkfile_name = str(uuid4()) + ".h5"

    # callback func stores checkpoints
    callback_func = CallBackStore(chkfile_name)
    # set initial guess
    initial_guess = np.zeros(2 * (chi.size + get_zeta_size(zeta)), dtype=np.float64)
    pack_thc_factors(chi, zeta, initial_guess)
    opt_init, opt_update, get_params = adagrad(step_size=stepsize, momentum=momentum)
    opt_state = opt_init(initial_guess)
    thc_grad = jax.grad(thc_objective_regularized_batched, argnums=[0])

    if batch_size is None:
        batch_size = num_kpts**2
    indx_arrays = prepare_batched_data_indx_arrays(
        momentum_map, Gpq_map, num_thc, num_orb
    )

    def update(i, opt_state):
        params = get_params(opt_state)
        gradient = thc_grad(
            params,
            num_orb,
            num_thc,
            momentum_map,
            Gpq_map,
            chol,
            indx_arrays,
            batch_size,
        )
        grad_norm_l1 = np.linalg.norm(gradient[0], ord=1)
        return opt_update(i, gradient[0], opt_state), grad_norm_l1

    for t in range(maxiter):
        opt_state, grad_l1 = update(t, opt_state)
        params = get_params(opt_state)
        if t % callback_func.freq == 0:
            # callback_func(params)
            fval = thc_objective_regularized_batched(
                params,
                num_orb,
                num_thc,
                momentum_map,
                Gpq_map,
                chol,
                indx_arrays,
                batch_size,
            )
            outline = "Objective val {: 5.15f}".format(fval)
            outline += "\tGrad L1-norm {: 5.15f}".format(grad_l1)
            print(outline)
        if grad_l1 <= gtol:
            # break out of loop
            # which sends to save
            break
    else:
        print("Maximum number of iterations reached")
    # save results before returning
    x = np.array(params)
    if chkfile_name is not None:
        num_G_per_Q = [len(np.unique(GQ)) for GQ in Gpq_map]
        chi, zeta = unpack_thc_factors(x, num_thc, num_orb, num_kpts, num_G_per_Q)
        with h5py.File(chkfile_name, "w") as fh5:
            fh5["chi"] = chi
            fh5["G_mapping"] = Gpq_map
            for iq in range(num_kpts):
                fh5[f"zeta_{iq}"] = zeta[iq]
    return params


def make_contiguous_cholesky(cholesky):
    num_kpts = len(cholesky)
    num_mo = cholesky[0, 0].shape[-1]
    if cholesky.dtype == object:
        # Jax requires contiguous arrays so just truncate naux if it's not
        # uniform hopefully shouldn't affect results dramatically as from
        # experience the naux amount only varies by a few % per k-point
        # Alternatively todo: padd with zeros
        min_naux = min([cholesky[k1, k1].shape[0] for k1 in range(num_kpts)])
        cholesky_contiguous = np.zeros(
            (
                num_kpts,
                num_kpts,
                min_naux,
                num_mo,
                num_mo,
            ),
            dtype=np.complex128,
        )
        for ik1 in range(num_kpts):
            for ik2 in range(num_kpts):
                cholesky_contiguous[ik1, ik2] = cholesky[ik1, ik2][:min_naux]
    else:
        cholesky_contiguous = cholesky

    return cholesky_contiguous


def kpoint_thc_via_isdf(
    kmf,
    cholesky,
    num_thc,
    perform_bfgs_opt=True,
    perform_adagrad_opt=True,
    bfgs_maxiter=3000,
    adagrad_maxiter=3000,
    checkpoint_basename="thc",
    save_checkoints=True,
    use_batched_algos=True,
    penalty_param=None,
    batch_size=None,
    max_kmeans_iteration=500,
    verbose=False,
):
    """
    Solve k-point THC using ISDF as an initial guess.

    :param kmf: instance of pyscf.pbc.KRHF object. Must be using FFTDF density
        fitting for integrals.
    :param cholesky: 3-index cholesky integrals needed for BFGS optimization.
    :param num_thc: THC dimensions (int), usually nthc = c_thc * n, where n is the
        number of spatial orbitals in the unit cell and c_thc is some
        poisiitve integer (typically <= 15).
    :param perform_bfgs_opt: Perform subsequent BFGS optimization of THC
        factors?
    :param perform_adagrad_opt: Perform subsequent adagrad optimization of THC
        factors? This is performed after BFGD if perform_bfgs_opt is True.
    :param bfgs_maxiter: Maximum iteration for adagrad optimization.
    :param adagrad_maxiter: Maximum iteration for adagrad optimization.
    :param save_checkoints: Whether to save checkpoint files or not (which will
        store THC factors. For each step we store checkpoints as
        {checkpoint_basename}_{thc_method}.h5, where thc_method is one of the
        strings (isdf, bfgs or adagrad).
    :param checkpoint_basename: Base name for checkpoint files. string,
        default "thc".
    :param use_batched_algos: Whether to use batched algorithms which may be
        faster but have higher memory cost. Bool. Default True.
    :param penalty_param: Penalty parameter for l2 regularization. Float. Default None.
    :param max_kmeans_iteration: Maximum number of iterations for KMeansCVT
        step. int. Default 500.
    :param verbose: Print information? Bool, default False.
    :returns (chi, zeta, G_map)
    """
    # Perform initial ISDF calculation of THC factors
    chi, zeta, xi, G_mapping = solve_kmeans_kpisdf(
        kmf,
        num_thc,
        single_translation=False,
        verbose=verbose,
        max_kmeans_iteration=max_kmeans_iteration,
    )
    num_mo = kmf.mo_coeff[0].shape[-1]
    num_kpts = len(kmf.kpts)
    if save_checkoints:
        chkfile_name = f"{checkpoint_basename}_isdf.h5"
        with h5py.File(chkfile_name, "w") as fh5:
            fh5["chi"] = chi
            fh5["G_mapping"] = G_mapping
            for iq in range(num_kpts):
                fh5[f"zeta_{iq}"] = zeta[iq]
    momentum_map = build_momentum_transfer_mapping(kmf.cell, kmf.kpts)
    cholesky_contiguous = make_contiguous_cholesky(cholesky)
    if perform_bfgs_opt:
        if save_checkoints:
            chkfile_name = f"{checkpoint_basename}_bfgs.h5"
        else:
            chkfile_name = None
        if use_batched_algos:
            opt_params = lbfgsb_opt_kpthc_l2reg_batched(
                chi,
                zeta,
                momentum_map,
                G_mapping,
                cholesky_contiguous,
                chkfile_name=chkfile_name,
                maxiter=bfgs_maxiter,
                penalty_param=penalty_param,
                batch_size=batch_size,
            )
        else:
            opt_params = lbfgsb_opt_kpthc_l2reg(
                chi,
                zeta,
                momentum_map,
                G_mapping,
                cholesky_contiguous,
                chkfile_name=chkfile_name,
                maxiter=bfgs_maxiter,
                penalty_param=penalty_param,
            )
        num_G_per_Q = [len(np.unique(GQ)) for GQ in G_mapping]
        chi, zeta = unpack_thc_factors(
            opt_params, num_thc, num_mo, num_kpts, num_G_per_Q
        )
    if perform_adagrad_opt:
        if save_checkoints:
            chkfile_name = f"{checkpoint_basename}_adagrad.h5"
        else:
            chkfile_name = None
        if use_batched_algos:
            opt_params = lbfgsb_opt_kpthc_l2reg_batched(
                chi,
                zeta,
                momentum_map,
                G_mapping,
                cholesky_contiguous,
                chkfile_name=chkfile_name,
                maxiter=bfgs_maxiter,
                penalty_param=penalty_param,
                batch_size=batch_size,
            )
        else:
            opt_params = lbfgsb_opt_kpthc_l2reg(
                chi,
                zeta,
                momentum_map,
                G_mapping,
                cholesky_contiguous,
                chkfile_name=chkfile_name,
                maxiter=bfgs_maxiter,
                penalty_param=penalty_param,
            )
        num_G_per_Q = [len(np.unique(GQ)) for GQ in G_mapping]
        chi, zeta = unpack_thc_factors(
            opt_params, num_thc, num_mo, num_kpts, num_G_per_Q
        )
    return chi, zeta, G_mapping
