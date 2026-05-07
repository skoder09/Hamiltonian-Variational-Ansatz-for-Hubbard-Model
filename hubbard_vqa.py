import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pennylane as qml
from numpy.linalg import eigh
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


DEFAULT_WIRES = tuple(range(8))
DEFAULT_SHOT_SWEEP = (100, 1000, 10000, 50000, 100000)
DEFAULT_NOISE_SWEEP = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1)


def site_spin2qubit(site, spin):
    """Map a site and spin to a qubit index."""
    return 2 * site + spin


def generate_hubbard_terms(n_sites, pbc=True, onsite_u=2.0):
    """Generate h_h, h_v, and h_U for the 2x2 Hubbard ladder used in this project."""
    h_h = []

    for u, v in [(0, 1), (2, 3), (0, 2), (1, 3)]:
        for spin in (0, 1):
            term = qml.fermi.FermiWord(
                {
                    (0, site_spin2qubit(u, spin)): "+",
                    (1, site_spin2qubit(v, spin)): "-",
                }
            )
            term_conj = qml.fermi.FermiWord(
                {
                    (0, site_spin2qubit(v, spin)): "+",
                    (1, site_spin2qubit(u, spin)): "-",
                }
            )

            if (u, v) in ((0, 1), (2, 3)):
                coeff = -1.0 - float(pbc)
            else:
                coeff = -1.0

            h_h.append(qml.fermi.FermiSentence({term: coeff, term_conj: coeff}))

    h_v = h_h[4:]
    h_h = h_h[:4]

    h_u = []
    for site in range(n_sites):
        term = qml.fermi.FermiWord(
            {
                (0, site_spin2qubit(site, 0)): "+",
                (1, site_spin2qubit(site, 0)): "-",
                (2, site_spin2qubit(site, 1)): "+",
                (3, site_spin2qubit(site, 1)): "-",
            }
        )
        h_u.append(qml.fermi.FermiSentence({term: onsite_u}))

    return h_h, h_v, h_u


def sum_fermionic_terms(terms):
    """Sum a list of FermiSentence terms into one operator."""
    total = terms[0]
    for term in terms[1:]:
        total += term
    return total


def pure_state_fidelity(reference_state, trial_state):
    """Return |<reference|trial>|^2 for two pure states."""
    overlap = np.vdot(reference_state, trial_state)
    return float(np.abs(overlap) ** 2)


def pure_state_density_fidelity(reference_state, density_matrix):
    """Return <reference|rho|reference> for a pure reference state and a density matrix."""
    amplitude = density_matrix @ reference_state
    return float(np.real(np.vdot(reference_state, amplitude)))


@dataclass
class OptimizationConfig:
    s_tot: int = 3
    optim_pts: int = 6
    init_sigma: float = 0.1
    greedy_n_steps: int = 150
    greedy_start_step_scale: float = 0.5
    greedy_reset_step_scale: float = 0.1
    greedy_decay_start: int = 80
    powell_maxiter: int = 1000
    alternate_rounds: int = 10
    alternate_n_steps: int = 150
    alternate_step_scale: float = 0.1
    tol: float = 1e-9
    acceptance_window: int = 10
    acceptance_cutoff: int = 7
    step_increase_factor: float = 1.1
    step_decrease_factor: float = 0.8
    seed: int = 1234
    verbose: bool = True


@dataclass
class OptimizationResult:
    best_energy: float
    best_params: np.ndarray
    energy_trace: list
    final_result: object
    initial_points: np.ndarray
    initial_energies: np.ndarray


class HubbardVQA:
    """Core model, exact references, and evaluation utilities for the 2x2 Hubbard VQA."""

    def __init__(
        self,
        n_sites=4,
        pbc=True,
        onsite_u=2.0,
        noiseless_device_name="lightning.qubit",
    ):
        self.n_sites = n_sites
        self.n_orbitals = 2 * n_sites
        self.n_electrons = 4
        self.wires = tuple(range(self.n_orbitals))
        self.noiseless_device_name = noiseless_device_name

        self.h_h, self.h_v, self.h_u = generate_hubbard_terms(n_sites, pbc=pbc, onsite_u=onsite_u)
        self.h_h_total = sum_fermionic_terms(self.h_h)
        self.h_v_total = sum_fermionic_terms(self.h_v)
        self.h_u_total = sum_fermionic_terms(self.h_u)

        self.non_interacting_hamiltonian = self.h_h_total + self.h_v_total
        self.full_fermionic_hamiltonian = self.h_h_total + self.h_v_total + self.h_u_total

        self.non_interacting_matrix = self._build_non_interacting_matrix()
        eigenvalues, eigenvectors = eigh(self.non_interacting_matrix)
        self.non_interacting_energies = eigenvalues
        self.basis_rotation = eigenvectors.T

        self.occupation = np.zeros(self.n_orbitals, dtype=int)
        self.occupation[np.argsort(self.non_interacting_energies)[: self.n_electrons]] = 1

        self.qubit_hamiltonian = qml.jordan_wigner(self.non_interacting_hamiltonian)
        self.full_qubit_hamiltonian = qml.jordan_wigner(self.full_fermionic_hamiltonian)
        self.jw_u = [qml.jordan_wigner(term) for term in self.h_u]
        self.jw_h = [qml.jordan_wigner(term) for term in self.h_h]
        self.jw_v = [qml.jordan_wigner(term) for term in self.h_v]

        self.full_hamiltonian_matrix = qml.matrix(
            self.full_qubit_hamiltonian, wire_order=self.wires
        )
        full_eigenvalues, full_eigenvectors = eigh(self.full_hamiltonian_matrix)
        self.exact_energy = float(np.real(full_eigenvalues[0]))
        self.exact_state = full_eigenvectors[:, 0]

        self._qnode_cache = {}

    def _build_non_interacting_matrix(self):
        matrix = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)

        for word, coeff in self.non_interacting_hamiltonian.items():
            ops = sorted(word.items(), key=lambda item: item[0][0])
            (_, p), _ = ops[0]
            (_, q), _ = ops[1]
            matrix[p, q] += coeff

        return matrix

    def _apply_ground_state(self):
        qml.BasisState(self.occupation, wires=self.wires)
        qml.BasisRotation(wires=self.wires, unitary_matrix=self.basis_rotation, check=True)

    def _apply_noise(self, wires, noise_model, strength):
        if strength == 0.0:
            return

        if noise_model == "depolarizing":
            for wire in wires:
                qml.DepolarizingChannel(strength, wires=wire)
        elif noise_model == "amplitude_damping":
            for wire in wires:
                qml.AmplitudeDamping(strength, wires=wire)
        else:
            raise ValueError(f"Unsupported noise model: {noise_model}")

    def _apply_ansatz(self, theta, noise_model=None, strength=0.0):
        self._apply_ground_state()

        for step in range(theta.shape[0]):
            for term in self.jw_u:
                qml.TrotterProduct(term, time=theta[step][0] / 2, n=1)
                self._apply_noise(term.wires, noise_model, strength)

            for term in self.jw_h:
                qml.TrotterProduct(term, time=theta[step][1], n=1)
                self._apply_noise(term.wires, noise_model, strength)

            for term in self.jw_v:
                qml.TrotterProduct(term, time=theta[step][2], n=1)
                self._apply_noise(term.wires, noise_model, strength)

            for term in self.jw_u:
                qml.TrotterProduct(term, time=theta[step][0] / 2, n=1)
                self._apply_noise(term.wires, noise_model, strength)

    def _get_qnode(self, measurement, shots=None, noise_model=None, strength=0.0, readout_prob=None):
        key = (measurement, shots, noise_model, float(strength), readout_prob)
        if key in self._qnode_cache:
            return self._qnode_cache[key]

        if noise_model is None and readout_prob is None:
            device_name = self.noiseless_device_name
            device_kwargs = {}
        else:
            device_name = "default.mixed"
            device_kwargs = {}
            if readout_prob is not None:
                device_kwargs["readout_prob"] = readout_prob

        if shots is not None:
            device_kwargs["shots"] = shots

        device = qml.device(device_name, wires=self.wires, **device_kwargs)

        @qml.qnode(device)
        def qnode(theta):
            self._apply_ansatz(theta, noise_model=noise_model, strength=strength)
            if measurement == "energy":
                return qml.expval(self.full_qubit_hamiltonian)
            if measurement == "state":
                return qml.state()
            if measurement == "density_matrix":
                return qml.density_matrix(wires=self.wires)
            raise ValueError(f"Unsupported measurement: {measurement}")

        self._qnode_cache[key] = qnode
        return qnode

    def energy(self, theta, shots=None, noise_model=None, strength=0.0, readout_prob=None):
        theta = np.asarray(theta, dtype=float)
        return float(
            self._get_qnode(
                "energy",
                shots=shots,
                noise_model=noise_model,
                strength=strength,
                readout_prob=readout_prob,
            )(theta)
        )

    def state(self, theta):
        theta = np.asarray(theta, dtype=float)
        return np.asarray(self._get_qnode("state")(theta))

    def density_matrix(self, theta, noise_model=None, strength=0.0):
        theta = np.asarray(theta, dtype=float)
        return np.asarray(
            self._get_qnode(
                "density_matrix",
                noise_model=noise_model,
                strength=strength,
            )(theta)
        )

    def evaluate_batch(self, thetas):
        return np.array([self.energy(theta) for theta in thetas], dtype=float)


def optimize_parameters(model, config):
    """Run the current fixed-setting optimization workflow and return the best result."""
    rng = np.random.default_rng(config.seed)

    init_pts = rng.normal(0.0, config.init_sigma, (config.optim_pts, config.s_tot, 3))
    exp_energy = model.evaluate_batch(init_pts)
    if config.verbose:
        print(exp_energy)

    step_scale = config.greedy_start_step_scale
    for step in range(config.greedy_n_steps + 1):
        if step > config.greedy_decay_start:
            step_scale /= (step // config.greedy_decay_start) + 1

        new_pts = init_pts + rng.normal(0.0, step_scale, (config.optim_pts, config.s_tot, 3))
        new_exp_energy = model.evaluate_batch(new_pts)

        all_energy = np.concatenate([exp_energy, new_exp_energy])
        all_pts = np.concatenate([init_pts, new_pts], axis=0)
        best_idx = np.argsort(all_energy)[: config.optim_pts]

        exp_energy = all_energy[best_idx]
        init_pts = all_pts[best_idx]

        if config.verbose:
            print("Lowest energies: ", exp_energy, step_scale, step)

        step_scale = config.greedy_reset_step_scale

    def cost_function(theta_flat):
        return model.energy(theta_flat.reshape((config.s_tot, 3)))

    final_res = None
    for index in range(config.optim_pts):
        result = minimize(
            fun=cost_function,
            x0=init_pts[index].flatten(),
            method="Powell",
            options={"disp": config.verbose, "maxiter": config.powell_maxiter},
        )

        if final_res is None or result.fun < final_res.fun:
            final_res = result

    best_energy = float(final_res.fun)
    best_params = final_res.x.reshape((config.s_tot, 3))

    if config.verbose:
        print(f"\nOptimization Success: {final_res.success}")
        print(f"Lowest Energy: {best_energy:.10f}")
        print(f"Best parameters: {best_params}")

    prev_best_energy = 1_000_000.0
    best_energy_arr = []
    current_best_params = best_params.copy()

    for round_idx in range(config.alternate_rounds):
        init_theta = current_best_params.copy()
        exp_energy_scalar = model.energy(init_theta)

        current_step_scale = config.alternate_step_scale
        acceptance_count = 0

        for step in range(config.alternate_n_steps + 1):
            new_theta = init_theta + rng.normal(0.0, current_step_scale, init_theta.shape)
            new_energy = model.energy(new_theta)

            if new_energy < exp_energy_scalar:
                exp_energy_scalar = new_energy
                init_theta = new_theta
                acceptance_count += 1

            if (step + 1) % config.acceptance_window == 0:
                if acceptance_count > config.acceptance_cutoff:
                    current_step_scale *= config.step_increase_factor
                else:
                    current_step_scale *= config.step_decrease_factor

                if config.verbose:
                    print(acceptance_count)

                acceptance_count = 0

        result = minimize(
            fun=cost_function,
            x0=init_theta.flatten(),
            method="Powell",
            options={"disp": config.verbose, "maxiter": config.powell_maxiter},
        )

        best_energy = float(result.fun)
        current_best_params = result.x.reshape((config.s_tot, 3))

        if np.abs(best_energy - prev_best_energy) < config.tol:
            break

        prev_best_energy = best_energy
        best_energy_arr.append(best_energy)

        if config.verbose:
            print("--------------")
            print(
                f"Best energy found ({round_idx}): ",
                best_energy,
                " with step size:",
                current_step_scale,
            )
            print("--------------")

        final_res = result

    return OptimizationResult(
        best_energy=best_energy,
        best_params=current_best_params,
        energy_trace=best_energy_arr,
        final_result=final_res,
        initial_points=init_pts,
        initial_energies=exp_energy,
    )


def analyze_shot_noise(model, theta, shots_list=DEFAULT_SHOT_SWEEP, repetitions=20):
    """Compare finite-shot energy estimates against exact and noiseless references."""
    theta = np.asarray(theta, dtype=float)
    noiseless_energy = model.energy(theta)
    noiseless_state = model.state(theta)
    exact_fidelity = pure_state_fidelity(model.exact_state, noiseless_state)

    results = []
    for shots in shots_list:
        sampled_energies = np.array([model.energy(theta, shots=shots) for _ in range(repetitions)])
        mean_energy = float(np.mean(sampled_energies))
        std_energy = float(np.std(sampled_energies))

        results.append(
            {
                "shots": shots,
                "mean_energy": mean_energy,
                "std_energy": std_energy,
                "exact_energy_error": mean_energy - model.exact_energy,
                "noiseless_energy_error": mean_energy - noiseless_energy,
                "fidelity_to_exact_state": exact_fidelity,
                "fidelity_to_noiseless_state": 1.0,
            }
        )

    return results


def analyze_circuit_noise(model, theta, noise_model, strengths=DEFAULT_NOISE_SWEEP):
    """Analyze channel noise relative to exact and noiseless references."""
    theta = np.asarray(theta, dtype=float)
    noiseless_energy = model.energy(theta)
    noiseless_state = model.state(theta)

    results = []
    for strength in strengths:
        density_matrix = model.density_matrix(theta, noise_model=noise_model, strength=strength)
        energy = model.energy(theta, noise_model=noise_model, strength=strength)

        results.append(
            {
                "strength": strength,
                "mean_energy": energy,
                "std_energy": 0.0,
                "exact_energy_error": energy - model.exact_energy,
                "noiseless_energy_error": energy - noiseless_energy,
                "fidelity_to_exact_state": pure_state_density_fidelity(
                    model.exact_state, density_matrix
                ),
                "fidelity_to_noiseless_state": pure_state_density_fidelity(
                    noiseless_state, density_matrix
                ),
            }
        )

    return results


def analyze_readout_error(model, theta, strengths=DEFAULT_NOISE_SWEEP, shots=DEFAULT_SHOT_SWEEP[-1]):
    """Analyze readout error relative to exact and noiseless references."""
    theta = np.asarray(theta, dtype=float)
    noiseless_energy = model.energy(theta)
    noiseless_state = model.state(theta)
    exact_fidelity = pure_state_fidelity(model.exact_state, noiseless_state)

    results = []
    for strength in strengths:
        energy = model.energy(theta, shots=shots, readout_prob=strength)
        results.append(
            {
                "strength": strength,
                "mean_energy": energy,
                "std_energy": 0.0,
                "exact_energy_error": energy - model.exact_energy,
                "noiseless_energy_error": energy - noiseless_energy,
                "fidelity_to_exact_state": exact_fidelity,
                "fidelity_to_noiseless_state": 1.0,
            }
        )

    return results


def write_summary_csv(path, rows):
    """Write analysis rows to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows, x_key, y_key, title, xlabel, ylabel, path):
    """Save a simple line plot for one analysis metric."""
    import matplotlib.pyplot as plt

    x_values = [row[x_key] for row in rows]
    y_values = [row[y_key] for row in rows]

    plt.figure()
    plt.plot(x_values, y_values, marker="o")
    if x_key == "shots":
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def print_summary(rows, x_key):
    """Print a compact summary table to stdout."""
    for row in rows:
        print(
            f"{x_key}={row[x_key]} | "
            f"mean_energy={row['mean_energy']:.10f} | "
            f"exact_energy_error={row['exact_energy_error']:.10f} | "
            f"noiseless_energy_error={row['noiseless_energy_error']:.10f} | "
            f"fidelity_to_exact_state={row['fidelity_to_exact_state']:.10f} | "
            f"fidelity_to_noiseless_state={row['fidelity_to_noiseless_state']:.10f}"
        )
