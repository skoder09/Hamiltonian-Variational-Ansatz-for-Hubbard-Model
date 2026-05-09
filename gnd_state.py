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

# ----------
# INITAL PARAM. GENERATION AND CIRCUIT EVAL. FOR THOSE PARAMS.
# ----------
S_tot = 3
optim_pts = 6  # from paper

init_pts = np.random.normal(0, 0.1, (optim_pts, S_tot, 3))

exp_energy = np.array([circuit(S_tot, theta) for theta in init_pts])
print("Energy at initial points in parameter space: ", exp_energy)

# %%

# ----------
# GREEDY NOISY SEARCH
# ----------

n_steps = 5
step_scale = 0.1

for i in range(n_steps + 1):
    if i > 80:
        step_scale /= (i // 80) + 1

    new_pts = init_pts + np.random.normal(0, step_scale, (optim_pts, S_tot, 3))
    new_exp_energy = np.array([circuit(S_tot, theta) for theta in new_pts])

    all_energy = np.concatenate([exp_energy, new_exp_energy])
    all_pts = np.concatenate([init_pts, new_pts], axis=0)
    idx = np.argsort(all_energy)[:6]

    lowest_energies = all_energy[idx]
    lowest_points = all_pts[idx]

    print("Lowest energies: ", lowest_energies, step_scale, i)
    exp_energy = lowest_energies
    init_pts = lowest_points
    step_scale = 0.1


# at this point exp_energy is the lowest possible energy we could find and init_pts are the corresponding points of theta
# print(init_pts, init_pts.shape)

# %%
# ------------
# POWELL METHOD
# ------------


def cost_function(theta_1d):
    theta_reshaped = theta_1d.reshape((S_tot, 3))

    energy = circuit(S_tot, theta_reshaped)
    return float(energy)


final_res = None

for i in range(6):
    result = minimize(
        fun=cost_function,
        x0=init_pts[i].flatten(),
        method="Powell",
        options={
            "disp": True,
            "maxiter": 100,
            "maxfev": 100,
            "xtol": 1e-9,
            "ftol": 1e-9,
        },
    )

    if final_res == None:
        final_res = result
    else:
        if result.fun < final_res.fun:
            final_res = result


# 4. Extract the results
best_energy = final_res.fun
best_params = final_res.x


print(f"\nOptimization Success: {final_res.success}")
print(f"Lowest Energy: {best_energy:.10f}")
print(f"Best parameters: {best_params}")


# %%
# -----------
# ALTERNATE BETWEEN GREEDY SEARCH AND POWELL
# -----------

n_steps = 150
step_scale = 0.001
prev_best_energy = 1000000

tol = 1e-9

acceptance_window = 30
acceptance_cutoff = 15
step_increase_factor = 1.2
step_decrease_factor = 0.8

best_energy_arr = []

for _ in range(10):
    init_pts = np.reshape(best_params, (S_tot, 3))
    exp_energy = circuit(S_tot, init_pts)

    current_step_scale = step_scale
    acceptance_count = 0

    for i in range(n_steps + 1):
        new_pts = init_pts + np.random.normal(0, current_step_scale, (S_tot, 3))
        new_exp_energy = circuit(S_tot, new_pts)

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

    # POWELL:
    result = minimize(
        fun=cost_function,
        x0=init_pts.flatten(),
        method="Powell",
        options={
            "disp": True,
            "maxiter": 100,
            "maxfev": 100,
            "xtol": 1e-9,
            "ftol": 1e-9,
        },
    )

    best_energy = result.fun
    best_params = result.x

    if np.abs(best_energy - prev_best_energy) < tol:
        break
    prev_best_energy = best_energy

    print("--------------")
    print(
        f"Best energy found ({_}): ",
        best_energy,
        " with step size:",
        current_step_scale,
    )
    print("--------------")
    best_energy_arr.append(float(best_energy))

# Target energy: -6.26500420602625


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
shot_list = [100, 1000, 10000, 50000, 100000]
n_shot_repeats = 30

shot_mean_energies = []
shot_std_energies = []
exact_energy_errors = []
noiseless_energy_errors = []

for shots in shot_list:
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

    sampled_energies = np.array(
        [circuit_shot_noise(S_tot, best_params_reshaped) for _ in range(n_shot_repeats)]
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
