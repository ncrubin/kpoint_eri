import h5py
import numpy as np
import math
from scipy.optimize import minimize

import jax
import jax.numpy as jnp

def get_zeta_size(zeta):
    return sum([z.size for z in zeta])


def unpack_thc_factors(xcur, num_thc, num_orb, num_kpts, num_G_per_Q):
    # leaf tensor (num_kpts, num_orb, num_thc)
    chi_size = num_kpts * num_orb * num_thc
    chi_real = xcur[:chi_size].reshape(num_kpts, num_orb, num_thc)
    chi_imag = xcur[chi_size : 2 * chi_size].reshape(num_kpts, num_orb, num_thc)
    chi = chi_real + 1j * chi_imag
    zeta_packed = xcur[2 * chi_size :]
    zeta = np.zeros((num_kpts,), dtype=object)
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
        zeta[iQ] = zeta_real + 1j * zeta_imag
        start += 2 * size
    return chi, zeta


def pack_thc_factors(chi, zeta, buffer):
    # leaf tensor (num_kpts, num_orb, num_thc)
    assert len(chi.shape) == 3
    buffer[: chi.size] = chi.real.ravel()
    buffer[chi.size : 2 * chi.size] = chi.imag.ravel()
    start = 2 * chi.size
    num_kpts = zeta.shape[0]
    for iQ in range(num_kpts):
        size = zeta[iQ].size
        buffer[start : start + size] = zeta[iQ].real.ravel()
        buffer[start + size : start + 2 * size] = zeta[iQ].imag.ravel()
        start += 2 * size

def compute_ojective_batched(
        chis,
        zetas,
        chols,
        cP,
        penalty_param=0
        ):
    eri_thc = jnp.einsum(
        "Jpm,Jqm,Jmn,Jrn,Jsn->Jpqrs",
        chis[0].conj(),
        chis[1],
        zetas,
        chis[2].conj(),
        chis[3],
        optimize=True,
    )
    eri_ref = jnp.einsum(
        "Jnpq,Jnsr->Jpqrs",
        chols[0],
        chols[1].conj(),
        optimize=True
    )
    deri = eri_thc - eri_ref
    MPQ_normalized = jnp.einsum(
        "P,JPQ,Q->JPQ", cP, zetas, cP
    )

    lambda_z = jnp.sum(jnp.einsum("JPQ->J", 0.5*jnp.abs(MPQ_normalized))**2.0)

    res = 0.5 * jnp.sum((jnp.abs(deri)) ** 2) + penalty_param * lambda_z
    return res

def prepare_batched_data(
    xcur,
    num_orb,
    num_thc,
    momentum_map,
    Gpq_map,
    chol):
    """Create arrays to batch over."""
    num_kpts = momentum_map.shape[0]
    num_G_per_Q = [len(np.unique(GQ)) for GQ in Gpq_map]
    chi, zeta = unpack_thc_factors(xcur, num_thc, num_orb, num_kpts, num_G_per_Q)
    num_kpts = momentum_map.shape[0]
    chis_p = []
    chis_q = []
    chis_r = []
    chis_s = []
    zetas = []
    chols_pq = []
    chols_rs = []
    for iq in range(num_kpts):
        for ik in range(num_kpts):
            ik_minus_q = momentum_map[iq, ik]
            Gpq = Gpq_map[iq, ik]
            for ik_prime in range(num_kpts):
                ik_prime_minus_q = momentum_map[iq, ik_prime]
                Gsr = Gpq_map[iq, ik_prime]
                chis_p.append(chi[ik])
                chis_q.append(chi[ik_minus_q])
                chis_r.append(chi[ik_prime_minus_q])
                chis_s.append(chi[ik_prime])
                zetas.append(zeta[iq][Gpq, Gsr])
                chols_pq.append(chol[ik, ik_minus_q])
                chols_rs.append(chol[ik_prime, ik_prime_minus_q])
    chis = (jnp.array(chis_p), jnp.array(chis_q), jnp.array(chis_r), jnp.array(chis_s))
    chols = (jnp.array(chols_pq), jnp.array(chols_rs))
    return chis, jnp.array(zetas), chols


def thc_objective_regularized_batched(
    xcur,
    num_orb,
    num_thc,
    momentum_map,
    Gpq_map,
    chol,
    penalty_param=0.0,
    batch_size=100,
):
    num_kpts = momentum_map.shape[0]
    num_G_per_Q = [len(np.unique(GQ)) for GQ in Gpq_map]
    chi, zeta = unpack_thc_factors(xcur, num_thc, num_orb, num_kpts, num_G_per_Q)
    # Normalization factor, no factor of sqrt as their are 4 chis in total when
    # building ERI.
    cP = jnp.einsum("kpP,kpP->P", chi.conj(), chi, optimize=True)

    num_batch = math.ceil(num_kpts ** 3 / batch_size)
    chis, zetas, chols = prepare_batched_data(xcur, num_orb, num_thc,
                                              momentum_map, Gpq_map, chol)
    objective = compute_ojective_batched(chis, zetas, chols, cP, penalty_param=penalty_param)
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
    cP = jnp.einsum("kpP,kpP->P", chi.conj(), chi, optimize=True)
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
                MPQ_normalized = jnp.einsum(
                    "P,PQ,Q->PQ", cP, zeta[iq][Gpq, Gsr], cP
                )

                lambda_z = jnp.sum(jnp.abs(MPQ_normalized)) * 0.5
                # if iq == 0 and ik == 0 and ik_prime == 0:
                    # print(jnp.sum(jnp.abs(MPQ_normalized)))

                    # print(lambda_z**2)
                res += 0.5 * jnp.sum((jnp.abs(deri)) ** 2) + penalty_param * (lambda_z**2)

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

    initial_guess = np.zeros(2*(chi.size + get_zeta_size(zeta)),
                             dtype=np.float64)
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
        jnp.array(initial_guess), num_orb, num_thc, momentum_map, Gpq_map,
        jnp.array(chol), penalty_param=0.0
    )
    reg_loss = thc_objective_regularized(
        jnp.array(initial_guess), num_orb, num_thc, momentum_map, Gpq_map,
        jnp.array(chol), penalty_param=1.0
    )
    print(loss, reg_loss)
    # set penalty
    if penalty_param is None:
        # loss + lambda_z^2 - loss
        lambda_z = reg_loss - loss
        penalty_param = reg_loss / (lambda_z**0.5)
        print("lambda_z {}".format(lambda_z))
        print("penalty_param {}".format(penalty_param))

    # L-BFGS-B optimization
    thc_grad = jax.grad(thc_objective_regularized, argnums=[0])
    print("Initial Grad")
    print(thc_grad(jnp.array(initial_guess), num_orb, num_thc, momentum_map, Gpq_map,
                   jnp.array(chol), penalty_param))
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
            fh5["etaPp"] = chi
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
    num_batch=None,
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

    initial_guess = np.zeros(2*(chi.size + get_zeta_size(zeta)),
                             dtype=np.float64)
    pack_thc_factors(chi, zeta, initial_guess)
    assert len(chi.shape) == 3
    assert len(zeta[0].shape) == 4
    num_kpts = chi.shape[0]
    num_orb = chi.shape[1]
    num_thc = chi.shape[-1]
    assert zeta[0].shape[-1] == num_thc
    assert zeta[0].shape[-2] == num_thc
    print(initial_guess)
    loss = thc_objective_regularized_batched(
        jnp.array(initial_guess), num_orb, num_thc, momentum_map, Gpq_map,
        jnp.array(chol), penalty_param=0.0
    )
    reg_loss = thc_objective_regularized_batched(
        jnp.array(initial_guess), num_orb, num_thc, momentum_map, Gpq_map,
        jnp.array(chol), penalty_param=1.0
    )
    print(loss, reg_loss)
    # set penalty
    if penalty_param is None:
        # loss + lambda_z^2 - loss
        lambda_z = reg_loss - loss
        penalty_param = reg_loss / (lambda_z**0.5)
        print("lambda_z {}".format(lambda_z))
        print("penalty_param {}".format(penalty_param))

    # L-BFGS-B optimization
    thc_grad = jax.grad(thc_objective_regularized_batched, argnums=[0])
    print("Initial Grad")
    print(thc_grad(jnp.array(initial_guess), num_orb, num_thc, momentum_map, Gpq_map,
                   jnp.array(chol), penalty_param))
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
            fh5["etaPp"] = chi
            for iq in range(num_kpts):
                fh5[f"zeta_{iq}"] = zeta[iq]
    return np.array(params)
