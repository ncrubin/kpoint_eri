from dataclasses import dataclass, asdict, field
import numpy as np
from pyscf.pbc import gto
import pandas as pd
from pytest import approx

from kpoint_eri.factorizations.hamiltonian_utils import HamiltonianProperties



@dataclass(frozen=True)
class ResourceEstimates:
    toffolis_per_step: int 
    total_toffolis: int 
    logical_qubits: int 

    dict = asdict

@dataclass
class PBCResources:
    system_name: str
    num_spin_orbitals: int
    num_kpts: int
    dE: float
    chi: int
    exact_energy: float 
    cutoff: list = field(default_factory=list)
    approx_energy: list = field(default_factory=list)
    ham_props: list = field(default_factory=list)
    resources: list = field(default_factory=list)

    dict = asdict

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.dict())
        lambdas = pd.json_normalize(df.pop("ham_props"))
        resources = pd.json_normalize(df.pop("resources"))
        df = df.join(pd.DataFrame(lambdas))
        df = df.join(pd.DataFrame(resources))
        return df

    def add_resources(
        self,
        ham_properties: HamiltonianProperties,
        resource_estimates: ResourceEstimates,
        cutoff: float,
        approx_energy: float,
    ) -> None:
        self.ham_props.append(ham_properties)
        self.resources.append(resource_estimates)
        self.cutoff.append(cutoff)
        self.approx_energy.append(approx_energy)

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
