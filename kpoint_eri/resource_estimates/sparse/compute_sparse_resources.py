"""
Compute resource estimates for sparse LCU of k-point Hamiltonian
"""
from typing import Tuple
import numpy as np
from numpy.lib.scimath import arccos, arcsin  # has analytc continuation to cplx
from sympy import factorint
from openfermion.resource_estimates.utils import QI, power_two


def QR_ncr(L, M1):
    """
    QR[Ll_, m_] := Ceiling[MinValue[{Ll/2^k + m*(2^k - 1), k >= 0}, k \[Element] Integers]];
    QRa = ArgMin[{L/2^k + Mm*(2^k - 1), k >= 0},   k \[Element] Integers];(*Gives the optimal k.*)
    """
    k = 0.5 * np.log2(L / M1)
    value = lambda k: L / np.power(2, k) + M1 * (np.power(2, k) - 1)
    try:
        assert k >= 0
    except AssertionError:
        k_opt = 0
        val_opt = np.ceil(value(k_opt))
        assert val_opt.is_integer()
        return int(k_opt), int(val_opt)
    k_int = [np.floor(k), np.ceil(k)]  # restrict optimal k to integers
    k_opt = k_int[np.argmin(value(k_int))]  # obtain optimal k
    val_opt = np.ceil(value(k_opt))  # obtain ceiling of optimal value given k
    assert k_opt.is_integer()
    assert val_opt.is_integer()
    return int(k_opt), int(val_opt)


def cost_sparse(n: int, Nk: int, lam: float, d: int, dE: float, chi: int,
                stps: int) -> Tuple[int, int, int]:
    """ Determine fault-tolerant costs using sparse decomposition in quantum
        chemistry
    Args:
        n (int) - the number of spin-orbitals
        Nk (int) - the number of k-points
        lam (float) - the lambda-value for the Hamiltonian
        d (int) - number of symmetry unique terms kept in the sparse Hamiltonian
        dE (float) - allowable error in phase estimation
        chi (int) - equivalent to aleph_1 and aleph_2 in the document, the
            number of bits for the representation of the coefficients
        stps (int) - an approximate number of steps to choose the precision
            of single qubit rotations in preparation of the equal superposition
            state
    Returns:
        step_cost (int) - Toffolis per step
        total_cost (int) - Total number of Toffolis
        ancilla_cost (int) - Total ancilla cost
    """
    if n % 2 != 0:
        raise ValueError("The number of spin orbitals is always even!")

    # I think there is a bug in the mathematica notebook. It does not check if
    # 2 is a factor first, which it should, cf. the similar function in
    # costingdf.nb Below is correct using the power_two() function, to give
    # power of 2 that is a factor of d.
    factors = factorint(d)
    eta = factors[min(list(sorted(factors.keys())))]
    # eta = power_two(d) # 0 if power_two(d) == 0 else power_two(d)
    if d % 2 == 1:
        eta = 0

    nN = np.ceil(np.log2(n // 2))
    nNk = np.ceil(np.log2(Nk))

    m = chi + 8 * nN + 6 * nNk + 5  # Eq 26

    oh = [0] * 20

    nM = (np.ceil(np.log2(d)) - eta) / 2

    for p in range(2, 22):
        # JJG note: arccos arg may be > 1
        v = np.round(np.power(2,p+1) / (2 * np.pi) * arccos(np.power(2,nM) /\
            np.sqrt(d/2**eta)/2))
        oh[p-2] = np.real(stps * (1 / (np.sin(3 * arcsin(np.cos(v * 2 * np.pi /\
            np.power(2,p+1)) * \
            np.sqrt(d/2**eta) / np.power(2,nM)))**2) - 1) + 4 * (p + 1))

    # Bits of precision for rotation
    br = int(np.argmin(oh) + 1) + 2

    # Hand selecting the k expansion factor
    k1 = QR_ncr(d, m)[0]

    # Equation (A17)
    cost = QR_ncr(d, m)[1] + QI(d)[1] + 6 * n * Nk + 8 * nN + 10 * nNk + 2 * chi + \
        7 * np.ceil(np.log2(d)) - 6 * eta + 4 * br - 8

    # Number of iterations needed for the phase estimation.
    iters = np.ceil(np.pi * lam / (dE * 2))

    # The following are the number of qubits from the listing on page 39.

    # Control registers for phase estimation and iteration on them.
    ac1 = 2 * np.ceil(np.log2(iters)) - 1

    # System qubits
    ac2 = n * Nk # shoulding this be Nk * n

    # The register used for the QROM
    ac3 = np.ceil(np.log2(d))

    # The qubit used for flagging the success of the equal superposition state
    # preparation and the ancilla qubit for rotation
    # ac45 = 2

    # The phase gradient state
    ac6 = br

    # The equal superposition state for coherent alias sampling
    ac7 = chi

    # The ancillas used for QROM
    ac8 = np.ceil(np.log2(d / np.power(2, k1))) + m * np.power(2, k1)

    ac9 = 9

    ancilla_cost = ac1 + ac2 + ac3 + ac6 + ac7 + ac8 + ac9

    # Sanity checks before returning as int
    assert cost.is_integer()
    assert iters.is_integer()
    assert ancilla_cost.is_integer()

    step_cost = int(cost)
    total_cost = int(cost * iters)
    ancilla_cost = int(ancilla_cost)

    return step_cost, total_cost, ancilla_cost
