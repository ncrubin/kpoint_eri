from dataclasses import dataclass, asdict, field
import numpy as np
from pyscf.pbc import gto


@dataclass
class PBCResources:
    system_name: str
    num_spin_orbitals: int
    num_kpts: int
    dE: float
    chi: int
    exact_emp2: float 
    cutoff: list = field(default_factory=list)
    approx_emp2: list = field(default_factory=list)
    lambda_tot: list = field(default_factory=list)
    lambda_one_body: list = field(default_factory=list)
    lambda_two_body: list = field(default_factory=list)
    toffolis_per_step: list = field(default_factory=list)
    total_toffolis: list = field(default_factory=list)
    logical_qubits: list = field(default_factory=list)

    dict = asdict

    def add_resources(
        self,
        lambda_tot: float,
        lambda_one_body: float,
        lambda_two_body: float,
        toffolis_per_step: float,
        total_toffolis: float,
        logical_qubits: float,
        cutoff: float,
        mp2_energy: float,
    ) -> None:
        self.lambda_tot.append(lambda_tot)
        self.lambda_one_body.append(lambda_one_body)
        self.lambda_two_body.append(lambda_two_body)
        self.toffolis_per_step.append(toffolis_per_step)
        self.total_toffolis.append(total_toffolis)
        self.logical_qubits.append(logical_qubits)
        self.cutoff.append(cutoff)
        self.approx_emp2.append(mp2_energy)

def compute_beta_for_resources(num_spin_orbs, num_kpts, dE_for_qpe):
    return np.ceil(5.652 + np.log2(num_spin_orbs * num_kpts / dE_for_qpe))


def build_momentum_transfer_mapping(cell: gto.Cell, kpoints: np.ndarray) -> np.ndarray:
    # Define mapping momentum_transfer_map[Q][k1] = k2 that satisfies
    # k1 - k2 + G = Q.
    a = cell.lattice_vectors() / (2 * np.pi)
    delta_k1_k2_Q = (
        kpoints[:, None, None, :]
        - kpoints[None, :, None, :]
        - kpoints[None, None, :, :]
    )
    delta_k1_k2_Q += kpoints[0][None, None, None, :]  # shift to center
    delta_dot_a = np.einsum("wx,kpQx->kpQw", a, delta_k1_k2_Q)
    int_delta_dot_a = np.rint(delta_dot_a)
    # Should be zero if transfer is statisfied (2*pi*n)
    mapping = np.where(np.sum(np.abs(delta_dot_a - int_delta_dot_a), axis=3) < 1e-10)
    num_kpoints = len(kpoints)
    momentum_transfer_map = np.zeros((num_kpoints,) * 2, dtype=np.int32)
    # Note index flip due to Q being first index in map but broadcasted last..
    momentum_transfer_map[mapping[1], mapping[0]] = mapping[2]

    return momentum_transfer_map
