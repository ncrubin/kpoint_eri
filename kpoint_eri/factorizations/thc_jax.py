import h5py
import numpy as np
from scipy.optimize import minimize

# from openfermion.resource_estimates.thc.utils.thc_factorization import CallBackStore

# from kpoint_eri.factorizations.isdf import build_eri_isdf_double_translation


def compute_eri_error(chi, zeta, momentum_map, Gpq_map, chol, mf):
    res = 0.0
    num_kpts = momentum_map.shape[0]
    kpts = mf.kpts
    num_mo = chi.shape[1]
    for iq in range(num_kpts):
        for ik in range(num_kpts):
            ik_minus_q = momentum_map[iq, ik]
            Gpq = Gpq_map[iq, ik]
            for ik_prime in range(num_kpts):
                ik_prime_minus_q = momentum_map[iq, ik_prime]
                Gsr = Gpq_map[iq, ik_prime]
                eri_thc = np.einsum(
                    "pm,qm,mn,rn,sn->pqrs",
                    chi[ik].conj(),
                    chi[ik_minus_q],
                    zeta[iq][Gpq, Gsr],
                    chi[ik_prime_minus_q].conj(),
                    chi[ik_prime],
                    optimize=True,
                )
                eri_ref = np.einsum(
                    "npq,nsr->pqrs",
                    chol[ik, ik_minus_q],
                    chol[ik_prime, ik_prime_minus_q].conj(),
                    optimize=True,
                )
                kpt_pqrs = [
                    kpts[ik],
                    kpts[ik_minus_q],
                    kpts[ik_prime_minus_q],
                    kpts[ik_prime],
                ]
                mos_pqrs = [
                    mf.mo_coeff[ik],
                    mf.mo_coeff[ik_minus_q],
                    mf.mo_coeff[ik_prime_minus_q],
                    mf.mo_coeff[ik_prime],
                ]
                eri_pqrs = mf.with_df.ao2mo(mos_pqrs, kpt_pqrs, compact=False).reshape(
                    (num_mo,) * 4
                )
                delta = eri_thc - eri_ref
                # delta_ex = eri_thc - eri_pqrs
                # delta_df = eri_ref - eri_pqrs
                res += 0.5 * np.sum(np.abs(delta) ** 2.0)
    return res / num_kpts


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


def thc_objective_regularized(
    # chi, zeta, momentum_map, Gpq_map, chol, penalty_param=0.0
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
    cP = np.einsum("kpP,kpP->P", chi.conj(), chi, optimize=True)
    for iq in range(num_kpts):
        for ik in range(num_kpts):
            ik_minus_q = momentum_map[iq, ik]
            Gpq = Gpq_map[iq, ik]
            for ik_prime in range(num_kpts):
                ik_prime_minus_q = momentum_map[iq, ik_prime]
                Gsr = Gpq_map[iq, ik_prime]
                eri_thc = np.einsum(
                    "pm,qm,mn,rn,sn->pqrs",
                    chi[ik].conj(),
                    chi[ik_minus_q],
                    zeta[iq][Gpq, Gsr],
                    chi[ik_prime_minus_q].conj(),
                    chi[ik_prime],
                    optimize=True,
                )
                eri_ref = np.einsum(
                    "npq,nsr->pqrs",
                    chol[ik, ik_minus_q],
                    chol[ik_prime, ik_prime_minus_q].conj(),
                    optimize=True,
                )
                deri = eri_thc - eri_ref
                MPQ_normalized = np.einsum(
                    "P,PQ,Q->PQ", cP, zeta[iq][Gpq, Gsr], cP, optimize=True
                )

                lambda_z = np.sum(np.abs(MPQ_normalized)) * 0.5

                res += 0.5 * np.sum((np.abs(deri)) ** 2) + penalty_param * (lambda_z**2)

    print("residual: ", res)
    return res / num_kpts


def lbfgsb_opt_thc_l2reg(
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
    loss = thc_objective_regularized(
        initial_guess, num_orb, num_thc, momentum_map, Gpq_map, chol, penalty_param=0.0
    )
    reg_loss = thc_objective_regularized(
        initial_guess, num_orb, num_thc, momentum_map, Gpq_map, chol, penalty_param=1.0
    )
    # set penalty
    if penalty_param is None:
        # loss + lambda_z^2 - loss
        lambda_z = reg_loss - loss
        penalty_param = reg_loss / (lambda_z**0.5)
        print("lambda_z {}".format(lambda_z))
        print("penalty_param {}".format(penalty_param))

    # L-BFGS-B optimization
    # thc_grad = jax.grad(thc_objective_regularized, argnums=[0])
    # print("Initial Grad")
    # print(thc_grad(jnp.array(x), norb, nthc, jnp.array(eri), penalty_param))
    # print()
    res = minimize(
        thc_objective_regularized,
        initial_guess,
        args=(num_orb, num_thc, momentum_map, Gpq_map, chol, penalty_param),
        method="L-BFGS-B",
        options={"disp": None, "iprint": disp_freq, "maxiter": maxiter},
        callback=callback_func,
    )

    # print(res)
    params = np.array(res.x)
    if chkfile_name is not None:
        num_G_per_Q = [len(np.unique(GQ)) for GQ in Gpq_map]
        chi, zeta = unpack_thc_factors(params, num_thc, num_orb, num_kpts, num_G_per_Q)
        with h5py.File(chkfile_name, "w+") as fh5:
            fh5["etaPp"] = chi
            for iq in range(num_kpts):
                fh5[f"zeta_{iq}"] = zeta[iq]
    return params
