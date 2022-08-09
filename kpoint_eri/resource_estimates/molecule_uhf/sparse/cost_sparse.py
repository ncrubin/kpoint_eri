"""
Cost out a sparse
(*The arguments are
n - the number of spin-orbitals
\[Lambda] - the lambda-value for the Hamiltonian
d - The number of terms kept in the sparse Hamiltonian
dE - allowable error in phase estimation
\[Chi] - equivalent to aleph in the document, the number of bits for \
the representation of the coefficients
stps - an approximate number of steps to choose the precision of \
single qubit rotations in preparation of the equal superposition \
state*)

CostSparse[n_, \[Lambda]_, d_, dE_, \[Chi]_, stps_] :=
  Module[{\[Eta], nN, m, QR, QI, oh, nM, p, br, v, k1, cost, iters,
    ac1, ac2, ac3, ac45, ac6, ac7, ac8},
   \[Eta] =
    FactorInteger[d][[1,
     2]];(*This gives the power of 2 that is a factor of d.*)

   nN = Ceiling[Log2[n/2]];
   m = \[Chi] + 8*nN + 4;(*Equation (A13)*)
   m = m + 4;
   (*In the case where spin values need to be output,
   the size of the QROM output is increased by 4 bits.*)

   QR[L_, M_] :=
    Ceiling[
     MinValue[{L/2^k + M*(2^k - 1), k >= 0},
      k \[Element]
       Integers]];(*This gives the minimum cost for a QROM over L \
values of size M.*)

   QI[L_] :=
    Ceiling[MinValue[{L/2^k + 2^k, k >= 0},
      k \[Element]
       Integers]];(*This gives the minimum cost for an inverse QROM \
over L values.*)
   oh = Table[0, 20];
   nM = (Ceiling[Log2[d]] - \[Eta])/2;
   For[p = 3, p <= 22, p++,
    v = Round[2^p/(2*\[Pi])*ArcCos[2^nM/Sqrt[d/2^\[Eta]]/2]];
    oh[[p - 2]] =
     stps*(1/N[
           Sin[3*ArcSin[
               Cos[v*2*\[Pi]/2^p]*Sqrt[d/2^\[Eta]]/2^nM]]^2] - 1) +
      4*p];
   br = Ordering[oh, 1][[1]] + 2;
   k1 = 32; (*This is hand selecting the k expansion factor.*)

   cost = Ceiling[d/k1] + m*(k1 - 1) + QI[d] + 4*n + 8*nN +
     2*\[Chi] + 7*Ceiling[Log2[d]] - 6*\[Eta] + 4*br -
     19; (*Equation (A17)*)
   cost = cost + 3;
    (*As well as the increase in the number of output qubits where we \
modified m,
   we need to do a controlled SWAP of the two pairs of spin qubits \
(cost 2) as well as perform a controlled swap of the two spins for \
symmetry.*)

   iters = Ceiling[\[Pi]*\[Lambda]/(dE*2)]; (*The number of \
iterations needed for the phase estimation.*)
   (*The following are \
the number of qubits from the listing on page 40.*)

   ac1 = 2*Ceiling[Log2[iters]] -
     1; (*Control registers for phase estimation and iteration on \
them.*)
   ac2 = n; (*System qubits.*)

   ac3 = Ceiling[Log2[d]];(*The register used for the QROM.*)

   ac45 = 2; (*The qubit used for flagging the success of the equal \
superposition state preparation and the ancilla qubit for rotation.*)

      ac6 = br; (*The phase gradient state.*)

   ac7 = \[Chi]; (*The equal superposition state for coherent alias \
sampling.*)

   ac8 = Ceiling[Log2[d/k1]] +
     m*k1;(*The ancillas used for the QROAM.*)
   ac9 = 5;
   (*There are another 4 qubits for the output of the index and alt \
values of the spin,
   and there is control qubit needed for the swap of the spins.*)
   \
{cost,(*Toffolis per step.*)

    cost*iters,(*The total number of Toffolis.*)

    ac1 + ac2 + ac3 + ac45 + ac6 + ac7 + ac8 +
     ac9 (*The total ancilla cost.*)}];
"""
import numpy as np
from chemftr.utils import QI, QR, power_two

def cost_sparse_uhf(*, num_spinorbs: int,
                    num_nonzero_terms: int,
                    hamiltonian_lambda: float,
                    allowable_error_in_pe: float,
                    num_bits_for_ham_coeffs: int,
                    stps: int):
    """
    Estimate cost of implementing Qubitization QPE for UHF Hamiltonians

    Args:
        num_spinorbs: the number of spin-orbitals
        hamiltonian_lambda: the lambda-value for the Hamiltonian
        num_nonzero_terms: The number of terms kept in the sparse Hamiltonian
        allowable_error_in_pe: allowable error in phase estimation
        num_bits_for_ham_coeffs: the number of bits for the representation
                                 of hamiltonian the coefficients
        stps: an approximate number of steps to choose the precision of single
              qubit rotations in preparation of the equal superposition
    Returns:
    """
    eta = power_two(num_nonzero_terms)  # (*This gives the power of 2 that is a factor of d.*)

    nN = np.ceil(np.log2(num_spinorbs//2))
    m = num_bits_for_ham_coeffs + 8 * nN + 4 # (*Equation(A13) *)
    m = m + 4  # (*In the case where spin values need to be output, the size of the QROM output is increased by 4 bits.*)


    # QR[L_, M_] := Ceiling[MinValue[{L / 2 ^ k + M * (2 ^ k - 1), k >= 0}, k \[Element] Integers]];
    # (*This gives the minimum cost for a QROM over L values of size M.*)

    # QI[L_] := Ceiling[MinValue[{L / 2 ^ k + 2 ^ k, k >= 0}, k \[Element] Integers]]
    # (*This gives the minimum cost for an inverse QROM over  L values.*)





