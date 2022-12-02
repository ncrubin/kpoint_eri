#coverage:ignore
""" Determine costs for THC decomposition in QC """
from typing import Tuple
import numpy as np
from numpy.lib.scimath import arccos, arcsin  # has analytc continuatn to cplx
from sympy import factorint
from openfermion.resource_estimates.utils import QR, QI


def compute_cost(n: int,
                 lam: float,
                 dE: float,
                 chi: int,
                 beta: int,
                 M: int,
                 Nkx: int,
                 Nky: int,
                 Nkz: int,
                 stps: int,
                 verbose: bool = False) -> Tuple[int, int, int]:
    """ Determine fault-tolerant costs using THC decomposition in quantum chem

    Args:
        n (int) - the number of spin-orbitals
        lam (float) - the lambda-value for the Hamiltonian
        dE (float) - allowable error in phase estimation
        chi (int) - equivalent to aleph in the document, the number of bits for
            the representation of the coefficients
        beta (int) - equivalent to beth in the document, the number of bits for
            the rotations
        M (int) - the dimension for the THC decomposition
        Nkx (int) - This is the number of values of k for the k - point sampling,
                     with each component
        Nky (int) - This is the number of values of k for the k - point sampling,
                     with each component
        Nkz (int) - This is the number of values of k for the k - point sampling,
                     with each component
        stps (int) - an approximate number of steps to choose the precision of
            single qubit rotations in preparation of the equal superpositn state

    Returns:
        step_cost (int) - Toffolis per step
        total_cost (int) - Total number of Toffolis
        ancilla_cost (int) - Total ancilla cost
    """
    nk = np.ceil(np.log2(Nkx)) + np.ceil(np.log2(Nky)) + np.ceil(np.log2(Nkz))
    Nk = Nkx * Nky * Nkz

    # The number of steps needed
    iters = np.ceil(np.pi * lam / (2 * dE))


    #This is the number of distinct items of data we need to output,  see Eq. (28).*)
    d = int((2*Nkx - 1)*(2*Nky - 1)*(2*Nkz - 1) * M**2 / 2 + n * Nk / 2)

    # The number of bits used for the contiguous register
    nc = np.ceil(np.log2(d))    
    
    nM = np.ceil(np.log2(M))

    # The output size is 2* Log[M] for the alt values, chi for the keep value, 
    # and 2 for the two sign bits.
    m = 2 * (2 * nM + nk + 5) + chi

    oh = [0] * 20
    for p in range(20):
        # arccos arg may be > 1
        v = np.round(
            np.power(2, p + 1) / (2 * np.pi) *
            arccos(np.power(2, nc) / np.sqrt(d) / 2))
        oh[p] = stps * (1 / (np.sin(3 * arcsin(np.cos(v * 2 * np.pi / \
            np.power(2,p+1)) * \
            np.sqrt(d) / np.power(2,nc)))**2) - 1) + 4 * (p + 1)

    print(oh)
    # Set it to be the number of bits that minimises the cost, usually 7.
    # Python is 0-index, so need to add the one back in vs mathematica nb
    br = np.argmin(oh) + 1

    if d % 2 == 0:
        factors = factorint(d)
        eta = factors[min(list(sorted(factors.keys())))]
    else:
        eta = 0

    cp1 = 2 * (3 * nc - 3 * eta + 2 * br - 9)


    # This is the cost of the QROM for the state preparation in step 3 and its
    cp3 = QR(d, m)[1] + QI(d)[1]

    # The cost for the inequality test in step 4 and its inverse.
    cp4 = 2 * chi

    # the cost of inequality test and controlled swap of mu and nu registers
    cp5 = 2 * (2 * nM + nk + 4)

    CPCP = cp1 + cp3 + cp4 + cp5 # + cp6

    # The cost of preparing the k superposition. The 7 here is the assumed number 
    # of bits for the ancilla rotation which makes the probability of 
    # failure negligible.
    cks = 4*(Nkx + Nky + Nkz + 8*nk + 6*7 - 24)

    # The cost of the arithmetic computing k-q
    cka = 10 * nk - 18

    # This is the cost of swapping based on the spin register
    cs1 = 3*n* Nk/2

    # The cost of controlled swaps into working registers based on the k or 
    # k - Q value.
    cs2 = 4*n*(Nk - 1)

    # The QROM for the rotation angles the first time.
    cs2a = QR(Nk*(M + n/2), n*beta)[1] + QI(Nk*(M + n/2))[1]

    # The QROM for the rotation angles the second time.
    cs2b = QR(Nk*M, n*beta)[1] + QI(Nk*M)[1]

    # The cost of the rotations.
    cs3 = 16*n*(beta - 2)

    # Cost for constructing contiguous register for outputting rotations.
    cs4 = 12*Nk + 4*np.ceil(np.log2(Nk*(M + n/2))) + 4*np.ceil(np.log2(Nk*M))

    # The cost of the controlled selection of the X vs Y.
    cs5 = 2

    # The total select cost.
    CS = cks + cka + cs1 + cs2 + cs2a + cs2b + cs3 + cs4 + cs5

    # The reflection cost.
    costref = nc + 2*nk + chi + 9

    cost = CPCP + CS + costref

    # Qubits for control for phase estimation
    ac1 = 2*np.ceil(np.log2(iters + 1)) - 1

    # system qubits
    ac2 = n*Nk

    # various control qubits
    ac3 = nc + chi + nk + 10

    # phase gradient state
    ac4 = beta

    # T state
    ac5 = 1

    # kp = 2^QRa[d, m]
    kp = np.power(2, QR(d, m))[0]

    # First round of QROM.
    ac12 = m*kp + np.ceil(np.log2(d/kp)) - 1

    # Temporary qubits from QROM.
    ac6 = m

    # Qubit from inequality test for state preparation.
    ac7 = 1

    # Qubit from inequality test for \[Mu],\[Nu]
    ac8 = 1  

    # QROM for constructing contiguous register.
    ac9 = nk + 7 

    # The contiguous register. 
    ac10 = np.ceil(np.log2(Nk*(M + n/2)))

    kr = np.power(2, QR(Nk*(M + n/2), n * beta))[0]

    ac11 = n*beta*kr + np.ceil(np.log2(Nk*(M + n/2))) - 1

    # Common ancilla costs before the first QROM.
    aca = ac1 + ac2 + ac3 + ac4 + ac5

    # These are the temporary ancillas in between erasing the first
    #  QROM ancillas and inverting that QROM.
    acc = ac6 + ac7 + ac8 + ac8 + ac9 + ac10 + ac11 

    step_cost = int(cost)
    total_cost = int(cost * iters)
    ancilla_cost = int(np.max([aca + ac12, aca + acc]))

    # step-cost, Toffoli count, logical qubits
    return step_cost, total_cost, ancilla_cost

if __name__ == "__main__":
    lam = 307.68
    dE = 0.001 
    n = 108
    chi = 10
    beta = 16
    M = 350
    res = compute_cost(n, lam, dE, chi, beta, M, 3, 2, 1, 20_000)
    print(res) # , 23991644223, 8693
    assert np.isclose(res[0], 49641)
    assert np.isclose(res[1], 23991644223)
    assert np.isclose(res[2], 8693)