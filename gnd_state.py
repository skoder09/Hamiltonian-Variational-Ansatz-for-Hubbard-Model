# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from numpy.linalg import eigh
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


def site_spin2qubit(site, spin):
    """
    Maps a site and spin to a qubit index. up=0, down = 1.
    """
    return 2 * site + spin


def generate_hubbard_terms(n_sites, PBC=True):
    """
    Generates hermitian terms for the Hubbard model on a two-leg ladder. In all code, here we assume N=4.
    Args:
        n_sites (int): Total number of sites (must be even for a 2-leg ladder).
    """

    # Generate h_h
    h_h = []

    for u, v in [(0, 1), (2, 3), (0, 2), (1, 3)]:  # the last 2 elements are for h_v
        for s in [0, 1]:
            term = qml.fermi.FermiWord(
                {(0, site_spin2qubit(u, s)): "+", (1, site_spin2qubit(v, s)): "-"}
            )
            term_conj = qml.fermi.FermiWord(
                {(0, site_spin2qubit(v, s)): "+", (1, site_spin2qubit(u, s)): "-"}
            )

            if (u, v) == (0, 1) or (u, v) == (2, 3):  # PBC only in horizontal direction
                hermitian_term = qml.fermi.FermiSentence(
                    {term: -1.0 - float(PBC), term_conj: -1.0 - float(PBC)}
                )  # having PBC same as horizontal coupling with double strength (for N=4 only)
            else:
                hermitian_term = qml.fermi.FermiSentence({term: -1.0, term_conj: -1.0})
            h_h.append(hermitian_term)

    # Generate h_v
    h_v = []
    h_v = h_h[4:]
    h_h = h_h[:4]

    # Generate h_U

    U = 2
    h_U = []
    for i in range(4):
        term = qml.fermi.FermiWord(
            {
                (0, site_spin2qubit(i, 0)): "+",
                (1, site_spin2qubit(i, 0)): "-",
                (2, site_spin2qubit(i, 1)): "+",
                (3, site_spin2qubit(i, 1)): "-",
            }
        )
        h_U.append(qml.fermi.FermiSentence({term: U}))
    return h_h, h_v, h_U


h_h, h_v, h_U = generate_hubbard_terms(4)

h_h_total = h_h[0]
for h_h_i in h_h[1:]:
    h_h_total += h_h_i

h_v_total = h_v[0]
for h_v_i in h_v[1:]:  # adds together all terms of h_v
    h_v_total += h_v_i

h_U_total = h_U[0]
for h_U_i in h_U[1:]:
    h_U_total += h_U_i


non_int_Ham = h_h_total + h_v_total

# %%

n_orbitals = 8
n_electrons = 4

h = np.zeros((n_orbitals, n_orbitals), dtype=complex)

for word, coeff in non_int_Ham.items():
    ops = sorted(word.items(), key=lambda x: x[0][0])
    (_, p), op0 = ops[0]
    (_, q), op1 = ops[1]
    h[p, q] += coeff

eps, V = eigh(h)
U = V.T

# occupation = qml.qchem.hf_state(n_electrons, n_orbitals)   #previous approach
occupation = np.zeros(n_orbitals, dtype=int)
occupation[np.argsort(eps)[:n_electrons]] = 1  # =[1 1 1 1 0 0 0 0]


dev = qml.device("default.qubit", wires=n_orbitals)
qubit_H = qml.jordan_wigner(non_int_Ham)
full_Ham = qml.jordan_wigner(non_int_Ham + h_U_total)


def ground_state():
    qml.BasisState(occupation, wires=range(n_orbitals))
    qml.BasisRotation(wires=range(n_orbitals), unitary_matrix=U, check=True)


# %%

# ---------------
# Ansatz preparation
# ---------------

jw_U = [qml.jordan_wigner(t) for t in h_U]
jw_h = [qml.jordan_wigner(t) for t in h_h]
jw_v = [qml.jordan_wigner(t) for t in h_v]


@qml.qnode(dev)
def circuit(S, theta, ret_val="expval"):
    """
    S: what the paper calls steps in eq. 3
    theta: a vector of form [[theta_U^1, theta_h^1, theta_v^1], ..., [theta_U^S, theta_h^S, theta_v^S]]
    """

    # GS preparation
    ground_state()

    for step in range(S):
        for term in jw_U:
            qml.exp(term, 1j * theta[step][0] / 2)
        # at this point in code, circ = e^{i*theta*h_U}
        for term in jw_h:
            qml.exp(term, 1j * theta[step][1])
        # at this point in code, circ = e^{i*theta*h_U} @ e^{i*theta*h_h}
        for term in jw_v:
            qml.exp(term, 1j * theta[step][2])
        # at this point in code, circ = e^{i*theta*h_U} @ e^{i*theta*h_h} @e^{i*theta*h_v}

        for term in jw_U:
            qml.exp(term, 1j * theta[step][0] / 2)

        # at this point in code, circ = e^{i*theta*h_U/2} @ e^{i*theta*h_h} @e^{i*theta*h_v} @ e^{i*theta*h_U/2}
    if ret_val == "expval":
        return qml.expval(full_Ham)
    elif ret_val == "state":
        return qml.state()


full_Ham_matrix = qml.matrix(full_Ham, wire_order=range(n_orbitals))
full_Ham_matrix = np.array(full_Ham_matrix, dtype=complex)
full_E_vals, full_E_vecs = eigh(full_Ham_matrix)
exact_ground_state_index = np.argmin(full_E_vals)
exact_energy = float(full_E_vals[exact_ground_state_index])
exact_state = full_E_vecs[:, exact_ground_state_index]

print("Exact energy: ", exact_energy)

# %%

# ------------
# Full Optimization
# ------------


def run_full_optimization(
    circuit_function,
    S,
    optim_pts=6,
    init_sigma=0.1,
    greedy_n_steps=5,
    greedy_step_scale=0.1,
    greedy_decay_start=80,
    powell_options=None,
    max_alternate_rounds=10,
    alternate_n_steps=150,
    alternate_step_scale=0.001,
    tol=1e-9,
    acceptance_window=30,
    acceptance_cutoff=15,
    step_increase_factor=1.2,
    step_decrease_factor=0.8,
    verbose=True,
):
    if powell_options is None:
        powell_options = {
            "disp": True,
            "maxiter": 100,
            "maxfev": 100,
            "xtol": 1e-9,
            "ftol": 1e-9,
        }

    init_pts = np.random.normal(0, init_sigma, (optim_pts, S, 3))
    exp_energy = np.array([circuit_function(S, theta) for theta in init_pts])

    # --------------
    # GREEDY SEARCH
    # --------------

    if verbose:
        print("Energy at initial points in parameter space: ", exp_energy)

    step_scale = greedy_step_scale

    for i in range(greedy_n_steps + 1):
        if i > greedy_decay_start:
            step_scale /= (i // greedy_decay_start) + 1  # adapt the step size

        new_pts = init_pts + np.random.normal(0, step_scale, (optim_pts, S, 3))
        new_exp_energy = np.array([circuit_function(S, theta) for theta in new_pts])

        all_energy = np.concatenate([exp_energy, new_exp_energy])
        all_pts = np.concatenate([init_pts, new_pts], axis=0)
        idx = np.argsort(all_energy)[:optim_pts]

        lowest_energies = all_energy[idx]
        lowest_points = all_pts[idx]

        if verbose:
            print("Lowest energies: ", lowest_energies, step_scale, i)

        exp_energy = lowest_energies
        init_pts = lowest_points
        step_scale = greedy_step_scale

    # --------------
    # POWELL MINIMIZATION
    # --------------

    def cost_function(theta_1d):
        theta_reshaped = theta_1d.reshape((S, 3))
        energy = circuit_function(S, theta_reshaped)
        return float(energy)

    final_res = None

    for i in range(optim_pts):
        result = minimize(
            fun=cost_function,
            x0=init_pts[i].flatten(),
            method="Powell",
            options=powell_options,
        )

        if final_res == None:
            final_res = result
        else:
            if result.fun < final_res.fun:
                final_res = result

    best_energy = final_res.fun
    best_params = final_res.x

    if verbose:
        print(f"\nOptimization Success: {final_res.success}")
        print(f"Lowest Energy: {best_energy:.10f}")
        print(f"Best parameters: {best_params}")

    # --------------
    # GREEDY AND POWELL ALTERNATE SEARCH
    # --------------#

    prev_best_energy = 1000000
    best_energy_arr = []

    for _ in range(max_alternate_rounds):
        init_pts = np.reshape(best_params, (S, 3))
        exp_energy = circuit_function(S, init_pts)

        current_step_scale = alternate_step_scale
        acceptance_count = 0

        for i in range(alternate_n_steps + 1):
            new_pts = init_pts + np.random.normal(0, current_step_scale, (S, 3))
            new_exp_energy = circuit_function(S, new_pts)

            if new_exp_energy < exp_energy:
                lowest_energy = new_exp_energy
                lowest_point = new_pts
                acceptance_count += 1
            else:
                lowest_energy = exp_energy
                lowest_point = init_pts

            exp_energy = lowest_energy
            init_pts = lowest_point

            if (i + 1) % acceptance_window == 0:
                if acceptance_count > acceptance_cutoff:
                    current_step_scale *= step_increase_factor
                else:
                    current_step_scale *= step_decrease_factor

                acceptance_count = 0

        result = minimize(
            fun=cost_function,
            x0=init_pts.flatten(),
            method="Powell",
            options=powell_options,
        )

        best_energy = result.fun
        best_params = result.x

        if np.abs(best_energy - prev_best_energy) < tol:
            break
        prev_best_energy = best_energy

        if verbose:
            print("--------------")
            print(
                f"Best energy found ({_}): ",
                best_energy,
                " with step size:",
                current_step_scale,
            )
            print("--------------")
        best_energy_arr.append(float(best_energy))

    return best_energy, best_params, best_energy_arr, final_res


# %%

# ----------
# INITAL PARAM. GENERATION AND CIRCUIT EVAL. FOR THOSE PARAMS.
# ----------
S_tot = 3

powell_options = {
    "disp": True,
    "maxiter": 100,
    "maxfev": 100,
    "xtol": 1e-9,
    "ftol": 1e-9,
}

# %%

# ----------
# OPTIMIZATION
# ----------

best_energy, best_params, best_energy_arr, final_res = run_full_optimization(
    circuit,
    S_tot,
    optim_pts=6,  # from paper
    init_sigma=0.1,  # stddev of inital param. distribution
    greedy_n_steps=150,  # num of greedy optimization steps
    greedy_step_scale=0.1,  # step size for greedy optimization
    greedy_decay_start=80,  # iteration at which to start step size decay for 1st greedy search
    powell_options=powell_options,
    max_alternate_rounds=10,  # max num of alternate optimization rounds we allow (about 30 rounds to converge)
    alternate_n_steps=150,  # num of steps for alternate optimization
    alternate_step_scale=0.001,  # step size for alternate optimization
    tol=1e-9,
    acceptance_window=30,  # window size for step size decrease
    acceptance_cutoff=15,  # num of consecutive iterations with no improvement before decreasing step size
    step_increase_factor=1.2,
    step_decrease_factor=0.8,
    verbose=True,
)


# Plot the best energy found at each iteration
plt.plot(best_energy_arr)
plt.xlabel("Iteration")
plt.ylabel("Best Energy")
plt.show()

print(best_energy_arr)
# %%

# -----------
# SHOT NOISE ANALYSIS
# -----------

best_params_reshaped = np.reshape(best_params, (S_tot, 3))
noiseless_energy = circuit(S_tot, best_params_reshaped)


noiseless_state = circuit(S_tot, best_params_reshaped, ret_val="state")
fidelity_to_exact_state = np.abs(np.vdot(exact_state, noiseless_state)) ** 2
fidelity_to_noiseless_state = np.abs(np.vdot(noiseless_state, noiseless_state)) ** 2


print("Noiseless energy:", noiseless_energy)
print("Fidelity to exact state:", fidelity_to_exact_state)
print("Fidelity to noiseless state:", fidelity_to_noiseless_state)

# %%
shot_list = [100, 1000, 100000]
n_shot_repeats = 1

shot_mean_energies = []
shot_std_energies = []
exact_energy_errors = []
noiseless_energy_errors = []

for shots in shot_list:
    print("Investigating for: ", shots, "shots.....")
    dev_shot = qml.device("default.qubit", wires=n_orbitals, shots=shots)

    @qml.qnode(dev_shot)
    def circuit_shot_noise(S, theta, ret_val="expval"):
        ground_state()

        for step in range(S):
            for term in jw_U:
                qml.exp(term, 1j * theta[step][0] / 2)
            for term in jw_h:
                qml.exp(term, 1j * theta[step][1])
            for term in jw_v:
                qml.exp(term, 1j * theta[step][2])
            for term in jw_U:
                qml.exp(term, 1j * theta[step][0] / 2)

        if ret_val == "expval":
            return qml.expval(full_Ham)
        elif ret_val == "samples":
            return qml.sample(wires=range(n_orbitals))

    shot_best_energy, shot_best_params, shot_best_energy_arr, shot_final_res = (
        run_full_optimization(
            circuit_shot_noise,
            S_tot,
            optim_pts=6,
            init_sigma=0.1,
            greedy_n_steps=150,
            greedy_step_scale=0.1,
            greedy_decay_start=80,
            powell_options=powell_options,
            max_alternate_rounds=10,
            alternate_n_steps=150,
            alternate_step_scale=0.001,
            tol=1e-9,
            acceptance_window=30,
            acceptance_cutoff=15,
            step_increase_factor=1.2,
            step_decrease_factor=0.8,
            verbose=False,
        )
    )

    shot_best_params_reshaped = np.reshape(shot_best_params, (S_tot, 3))

    sampled_energies = np.array(
        [
            circuit_shot_noise(S_tot, shot_best_params_reshaped)
            for _ in range(n_shot_repeats)
        ]
    )

    mean_energy = float(np.mean(sampled_energies))
    std_energy = float(np.std(sampled_energies))

    shot_mean_energies.append(mean_energy)
    shot_std_energies.append(std_energy)
    exact_energy_errors.append(mean_energy - exact_energy)
    noiseless_energy_errors.append(mean_energy - noiseless_energy)


print("\nShot-noise summary")
print("shots | mean_energy | stddev_energy | error_to_exact | error_to_noiseless ")
for i, shots in enumerate(shot_list):
    print(
        f"{shots} | "
        f"{shot_mean_energies[i]:.10f} | "
        f"{shot_std_energies[i]:.10f} | "
        f"{exact_energy_errors[i]:.10f} | "
        f"{noiseless_energy_errors[i]:.10f} | "
    )
# %%

plt.figure()
plt.errorbar(
    shot_list,
    shot_mean_energies,
    yerr=shot_std_energies,
    marker="o",
    capsize=4,
    label="Shot-based energy estimate",
)
plt.axhline(exact_energy, linestyle="--", color="green", label="Exact energy")
plt.axhline(
    noiseless_energy, linestyle=":", color="orange", label="Noiseless circuit energy"
)
plt.xscale("log")
plt.xlabel("Shots")
plt.ylabel("Estimated energy")
plt.title("Shot-noise energy estimates")
plt.legend()
plt.show()


plt.figure()
plt.plot(shot_list, np.abs(exact_energy_errors), marker="o", label="|E_shot - E_exact|")
plt.plot(
    shot_list,
    np.abs(noiseless_energy_errors),
    marker="s",
    label="|E_shot - E_noiseless|",
)
plt.xscale("log")
plt.xlabel("Shots")
plt.ylabel("Absolute energy error")
plt.title("Shot-noise energy error")
plt.legend()
plt.show()


# %%

# -----------
# CIRCUIT NOISE ANALYSIS
# -----------
circuit_noise_list = [0.001, 0.0005, 0.0001]

# %%

# -----------
# Depolarizing Noise
# -----------

depolarizing_energies = []
depolarizing_exact_energy_errors = []
depolarizing_noiseless_energy_errors = []
depolarizing_fidelity_to_exact = []
depolarizing_fidelity_to_noiseless = []

for p in circuit_noise_list:
    print("Investigating depolarizing noise for p =", p, ".....")
    dev_depolarizing = qml.device("default.mixed", wires=n_orbitals)

    @qml.qnode(dev_depolarizing)
    def circuit_depolarizing_noise(S, theta, ret_val="expval"):
        ground_state()

        for step in range(S):
            for term in jw_U:
                qml.exp(term, 1j * theta[step][0] / 2, num_steps=1)
                for wire in term.wires:
                    qml.DepolarizingChannel(p, wires=wire)
            for term in jw_h:
                qml.exp(term, 1j * theta[step][1], num_steps=1)
                for wire in term.wires:
                    qml.DepolarizingChannel(p, wires=wire)
            for term in jw_v:
                qml.exp(term, 1j * theta[step][2], num_steps=1)
                for wire in term.wires:
                    qml.DepolarizingChannel(p, wires=wire)
            for term in jw_U:
                qml.exp(term, 1j * theta[step][0] / 2, num_steps=1)
                for wire in term.wires:
                    qml.DepolarizingChannel(p, wires=wire)

        if ret_val == "expval":
            return qml.expval(full_Ham)
        elif ret_val == "density_matrix":
            return qml.density_matrix(wires=range(n_orbitals))

    (
        depolarizing_best_energy,
        depolarizing_best_params,
        depolarizing_best_energy_arr,
        depolarizing_final_res,
    ) = run_full_optimization(
        circuit_depolarizing_noise,
        S_tot,
        optim_pts=6,
        init_sigma=0.1,
        greedy_n_steps=1,
        greedy_step_scale=0.1,
        greedy_decay_start=80,
        powell_options=powell_options,
        max_alternate_rounds=1,
        alternate_n_steps=1,
        alternate_step_scale=0.001,
        tol=1e-9,
        acceptance_window=30,
        acceptance_cutoff=15,
        step_increase_factor=1.2,
        step_decrease_factor=0.8,
        verbose=False,
    )
    depolarizing_best_params_reshaped = np.reshape(depolarizing_best_params, (S_tot, 3))

    depolarizing_density_matrix = circuit_depolarizing_noise(
        S_tot, depolarizing_best_params_reshaped, ret_val="density_matrix"
    )
    depolarizing_energy = float(depolarizing_best_energy)
    depolarizing_fid_exact = float(
        np.real(np.vdot(exact_state, depolarizing_density_matrix @ exact_state))
    )
    depolarizing_fid_noiseless = float(
        np.real(np.vdot(noiseless_state, depolarizing_density_matrix @ noiseless_state))
    )

    depolarizing_energies.append(depolarizing_energy)
    depolarizing_exact_energy_errors.append(depolarizing_energy - exact_energy)
    depolarizing_noiseless_energy_errors.append(depolarizing_energy - noiseless_energy)
    depolarizing_fidelity_to_exact.append(depolarizing_fid_exact)
    depolarizing_fidelity_to_noiseless.append(depolarizing_fid_noiseless)


print("\nDepolarizing-noise summary")
print(
    "p | energy | error_to_exact | error_to_noiseless | fid_to_exact | fid_to_noiseless"
)
for i, p in enumerate(circuit_noise_list):
    print(
        f"{p} | "
        f"{depolarizing_energies[i]:.10f} | "
        f"{depolarizing_exact_energy_errors[i]:.10f} | "
        f"{depolarizing_noiseless_energy_errors[i]:.10f} | "
        f"{depolarizing_fidelity_to_exact[i]:.10f} | "
        f"{depolarizing_fidelity_to_noiseless[i]:.10f}"
    )

# %%

plt.figure()
plt.plot(
    circuit_noise_list, depolarizing_energies, marker="o", label="Depolarizing energy"
)
plt.axhline(exact_energy, linestyle="--", color="green", label="Exact energy")
plt.axhline(
    noiseless_energy, linestyle=":", color="orange", label="Noiseless circuit energy"
)
plt.xlabel("Depolarizing probability p")
plt.ylabel("Energy")
plt.title("Depolarizing-noise energy estimates")
plt.legend()
plt.show()


plt.figure()
plt.plot(
    circuit_noise_list,
    np.abs(depolarizing_exact_energy_errors),
    marker="o",
    label="|E_dep - E_exact|",
)
plt.plot(
    circuit_noise_list,
    np.abs(depolarizing_noiseless_energy_errors),
    marker="s",
    label="|E_dep - E_noiseless|",
)
plt.xlabel("Depolarizing probability p")
plt.ylabel("Absolute energy error")
plt.title("Depolarizing-noise energy error")
plt.legend()
plt.show()


plt.figure()
plt.plot(
    circuit_noise_list,
    depolarizing_fidelity_to_exact,
    marker="o",
    label="Fidelity to exact state",
)
plt.plot(
    circuit_noise_list,
    depolarizing_fidelity_to_noiseless,
    marker="s",
    label="Fidelity to noiseless state",
)
plt.xlabel("Depolarizing probability p")
plt.ylabel("State fidelity")
plt.title("Depolarizing-noise state fidelities")
plt.legend()
plt.show()


# %%

# -----------
# Amplitude Damping Noise
# -----------

amplitude_damping_energies = []
amplitude_damping_exact_energy_errors = []
amplitude_damping_noiseless_energy_errors = []
amplitude_damping_fidelity_to_exact = []
amplitude_damping_fidelity_to_noiseless = []

for gamma in circuit_noise_list:
    print("Investigating amplitude damping for gamma =", gamma, ".....")
    dev_amplitude_damping = qml.device("default.mixed", wires=n_orbitals)

    @qml.qnode(dev_amplitude_damping)
    def circuit_amplitude_damping_noise(S, theta, ret_val="expval"):
        ground_state()

        for step in range(S):
            for term in jw_U:
                qml.exp(term, 1j * theta[step][0] / 2, num_steps=1)
                for wire in term.wires:
                    qml.AmplitudeDamping(gamma, wires=wire)
            for term in jw_h:
                qml.exp(term, 1j * theta[step][1], num_steps=1)
                for wire in term.wires:
                    qml.AmplitudeDamping(gamma, wires=wire)
            for term in jw_v:
                qml.exp(term, 1j * theta[step][2], num_steps=1)
                for wire in term.wires:
                    qml.AmplitudeDamping(gamma, wires=wire)
            for term in jw_U:
                qml.exp(term, 1j * theta[step][0] / 2, num_steps=1)
                for wire in term.wires:
                    qml.AmplitudeDamping(gamma, wires=wire)
        if ret_val == "expval":
            return qml.expval(full_Ham)
        elif ret_val == "density_matrix":
            return qml.density_matrix(wires=range(n_orbitals))

    (
        amplitude_best_energy,
        amplitude_best_params,
        amplitude_best_energy_arr,
        amplitude_final_res,
    ) = run_full_optimization(
        circuit_amplitude_damping_noise,
        S_tot,
        optim_pts=6,
        init_sigma=0.1,
        greedy_n_steps=1,
        greedy_step_scale=0.1,
        greedy_decay_start=80,
        powell_options=powell_options,
        max_alternate_rounds=1,
        alternate_n_steps=1,
        alternate_step_scale=0.001,
        tol=1e-9,
        acceptance_window=30,
        acceptance_cutoff=15,
        step_increase_factor=1.2,
        step_decrease_factor=0.8,
        verbose=False,
    )
    amplitude_best_params_reshaped = np.reshape(amplitude_best_params, (S_tot, 3))

    amplitude_damping_density_matrix = circuit_amplitude_damping_noise(
        S_tot, amplitude_best_params_reshaped, ret_val="density_matrix"
    )
    amplitude_damping_energy = float(amplitude_best_energy)
    amplitude_damping_fid_exact = float(
        np.real(np.vdot(exact_state, amplitude_damping_density_matrix @ exact_state))
    )
    amplitude_damping_fid_noiseless = float(
        np.real(
            np.vdot(noiseless_state, amplitude_damping_density_matrix @ noiseless_state)
        )
    )

    amplitude_damping_energies.append(amplitude_damping_energy)
    amplitude_damping_exact_energy_errors.append(
        amplitude_damping_energy - exact_energy
    )
    amplitude_damping_noiseless_energy_errors.append(
        amplitude_damping_energy - noiseless_energy
    )
    amplitude_damping_fidelity_to_exact.append(amplitude_damping_fid_exact)
    amplitude_damping_fidelity_to_noiseless.append(amplitude_damping_fid_noiseless)


print("\nAmplitude-damping summary")
print(
    "gamma | energy | error_to_exact | error_to_noiseless | fid_to_exact | fid_to_noiseless"
)
for i, gamma in enumerate(circuit_noise_list):
    print(
        f"{gamma} | "
        f"{amplitude_damping_energies[i]:.10f} | "
        f"{amplitude_damping_exact_energy_errors[i]:.10f} | "
        f"{amplitude_damping_noiseless_energy_errors[i]:.10f} | "
        f"{amplitude_damping_fidelity_to_exact[i]:.10f} | "
        f"{amplitude_damping_fidelity_to_noiseless[i]:.10f}"
    )

# %%

plt.figure()
plt.plot(
    circuit_noise_list,
    amplitude_damping_energies,
    marker="o",
    label="Amplitude-damping energy",
)
plt.axhline(exact_energy, linestyle="--", color="green", label="Exact energy")
plt.axhline(
    noiseless_energy, linestyle=":", color="orange", label="Noiseless circuit energy"
)
plt.xlabel("Amplitude damping gamma")
plt.ylabel("Energy")
plt.title("Amplitude-damping energy estimates")
plt.legend()
plt.show()

print(amplitude_damping_energies)
plt.figure()
plt.plot(
    circuit_noise_list,
    np.abs(amplitude_damping_exact_energy_errors),
    marker="o",
    label="|E_amp - E_exact|",
)
plt.plot(
    circuit_noise_list,
    np.abs(amplitude_damping_noiseless_energy_errors),
    marker="s",
    label="|E_amp - E_noiseless|",
)
plt.xlabel("Amplitude damping gamma")
plt.ylabel("Absolute energy error")
plt.title("Amplitude-damping energy error")
plt.legend()
plt.show()


plt.figure()
plt.plot(
    circuit_noise_list,
    amplitude_damping_fidelity_to_exact,
    marker="o",
    label="Fidelity to exact state",
)
plt.plot(
    circuit_noise_list,
    amplitude_damping_fidelity_to_noiseless,
    marker="s",
    label="Fidelity to noiseless state",
)
plt.xlabel("Amplitude damping gamma")
plt.ylabel("State fidelity")
plt.title("Amplitude-damping state fidelities")
plt.legend()
plt.show()

# %%

# -----------
# ROUTING OVERHEAD ANALYSIS
# -----------

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import DensityMatrix
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error

tuna9_edges = [
    (0, 1),
    (0, 2),
    (1, 3),
    (1, 4),
    (2, 4),
    (2, 5),
    (3, 6),
    (4, 6),
    (4, 7),
    (5, 7),
    (6, 8),
    (7, 8),
]
tuna9_coupling_map = CouplingMap(tuna9_edges)
tuna9_coupling_map.make_symmetric()

tuna9_basis_gates = [
    "rx",
    "ry",
    "rz",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "id",
    "cz",
]
routing_noise_list = [0.01, 0.005, 0.001]
routing_initial_layout = list(range(9))
routing_seed = 1234
routing_optimization_level = 1

# %%


# -----------------------------------------------------------------------------
# QISKIT EQUIVALENT: Build a Qiskit QuantumCircuit that mirrors the PennyLane
# `circuit(S, theta)` qnode above. This function is intentionally monolithic
# and appended at the bottom of the file so it can be used for routing /
# transpilation analysis on the `tuna-9` coupling map.
# -----------------------------------------------------------------------------

from scipy.linalg import expm


def build_qiskit_equivalent_circuit(S, theta):
    # Basic validation and conversion
    theta = np.array(theta, dtype=float)
    if S == 0:
        # allow theta to be (1,3) or (0,3); it will be unused when S==0
        pass
    else:
        if theta.shape != (S, 3):
            raise ValueError("theta must have shape (S, 3) for the given S")

    # Build a 9-qubit circuit so we can map onto tuna-9 (last qubit unused)
    qc = QuantumCircuit(9)

    # 1) Prepare ground state using existing PennyLane ground_state() / qnode
    # Ask the PennyLane qnode for the statevector after GS preparation only.
    # We call the existing `circuit` qnode with S=0 (no ansatz layers) and ask
    # for the statevector. This re-uses the `ground_state()` implementation above.
    ground_sv = circuit(0, np.zeros((1, 3)), ret_val="state")

    # Initialize the first `n_orbitals` qubits with that statevector.
    qc.initialize(ground_sv, list(range(n_orbitals)))

    # 2) For each ansatz layer, append exponentials of JW-mapped terms as dense
    # unitaries. We compute the dense matrix for each jw term on-the-fly using
    # pennylane's qml.matrix(..., wire_order=range(n_orbitals)).
    for step in range(S):
        # U half-layer
        for term in jw_U:
            # fix, why make a qml matrix first, then convert to numpy and then to qiskit?
            M = np.array(qml.matrix(term, wire_order=range(n_orbitals)), dtype=complex)
            Umat = expm(1j * theta[step][0] / 2.0 * M)
            qc.append(UnitaryGate(Umat), list(range(n_orbitals)))

        # h full-layer
        for term in jw_h:
            M = np.array(qml.matrix(term, wire_order=range(n_orbitals)), dtype=complex)
            Umat = expm(1j * theta[step][1] * M)
            qc.append(UnitaryGate(Umat), list(range(n_orbitals)))

        # v full-layer
        for term in jw_v:
            M = np.array(qml.matrix(term, wire_order=range(n_orbitals)), dtype=complex)
            Umat = expm(1j * theta[step][2] * M)
            qc.append(UnitaryGate(Umat), list(range(n_orbitals)))

        # U half-layer (again)
        for term in jw_U:
            M = np.array(qml.matrix(term, wire_order=range(n_orbitals)), dtype=complex)
            Umat = expm(1j * theta[step][0] / 2.0 * M)
            qc.append(UnitaryGate(Umat), list(range(n_orbitals)))

    return qc


# Example usage (commented out so importing this module does not build the big matrices):
# qc = build_qiskit_equivalent_circuit(S_tot, np.reshape(best_params, (S_tot, 3)))
# transpiled = transpile(qc, coupling_map=tuna9_coupling_map, basis_gates=tuna9_basis_gates,
#                        optimization_level=routing_optimization_level,
#                        initial_layout=routing_initial_layout, seed_transpiler=routing_seed)
# print(transpiled)
# ----------------------------------------------------------------------------
