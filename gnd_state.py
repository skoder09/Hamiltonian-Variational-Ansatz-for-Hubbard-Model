import pennylane as qml
import numpy as np
from numpy.linalg import eigh
def site_spin2qubit(site, spin):
    """
    Maps a site and spin to a qubit index. up=0, down = 1.
    """
    return 2 * site + spin

def generate_hubbard_terms(n_sites, PBC = True):
    """
    Generates hermitian terms for the Hubbard model on a two-leg ladder. In all code, here we assume N=4.
    Args:
        n_sites (int): Total number of sites (must be even for a 2-leg ladder).
    """
    
    # Generate h_h
    h_h = []

    for (u,v) in [(0,1), (2,3), (0,2), (1,3)]: #the last 2 elements are for h_v
        for s in [0,1]:
            term = qml.fermi.FermiWord({(0, site_spin2qubit(u,s)) : "+", (1, site_spin2qubit(v,s)) : "-"})
            term_conj = qml.fermi.FermiWord({(0, site_spin2qubit(v,s)) : "+", (1, site_spin2qubit(u,s)) : "-"})

            if (u,v)==(0,1) or (u,v) == (2,3): #PBC only in horizontal direction
                hermitian_term = qml.fermi.FermiSentence({term: -1.0-float(PBC), term_conj: -1.0-float(PBC)})   #having PBC same as horizontal coupling with double strength (for N=4 only)
            else:
                hermitian_term = qml.fermi.FermiSentence({term: -1.0, term_conj: -1.0})
            h_h.append(hermitian_term)
    


    # Generate h_v
    h_v = []
    h_v = h_h[4:]
    h_h=h_h[:4]

    #Generate h_U

    U = 2
    h_U = []
    for i in range(4):
        term = qml.fermi.FermiWord({(0, site_spin2qubit(i,0)) : "+", (1, site_spin2qubit(i,0)) : "-", (2, site_spin2qubit(i,1)) : "+", (3, site_spin2qubit(i,1)) : "-"})
        h_U.append(qml.fermi.FermiSentence({term: U}))
    return h_h, h_v, h_U


h_h, h_v, h_U = generate_hubbard_terms(4, False)

h_h_total = h_h[0]
for h_h_i in h_h[1:]:
    h_h_total += h_h_i

h_v_total = h_v[0]
for h_v_i in h_v[1:]: #adds together all terms of h_v
    h_v_total += h_v_i

h_U_total = h_U[0]
for h_U_i in h_U[1:]:
    h_U_total += h_U_i





####################
non_int_Ham = h_h_total + h_v_total
print("HAM: ", non_int_Ham)

n_orbitals = 8
n_electrons = 4

h = np.zeros((n_orbitals, n_orbitals), dtype=complex)

for word, coeff in non_int_Ham.items():
    ops = sorted(word.items(), key=lambda x: x[0][0])
    (_, p), op0 = ops[0]
    (_, q), op1 = ops[1]
    h[p, q] += coeff
print("H matrix: ", h)
eps, V = eigh(h)
U = V.T

#occupation = qml.qchem.hf_state(n_electrons, n_orbitals)
# occupation = np.zeros(n_orbitals, dtype=int)
# occupation[np.argsort(eps)[:n_electrons]] = 1

"""
occupation as in paper: 
"""



dev = qml.device("default.qubit", wires=n_orbitals)
qubit_H = qml.jordan_wigner(non_int_Ham)


def ground_state():
    qml.BasisState(occupation, wires=range(n_orbitals))
    qml.BasisRotation(wires=range(n_orbitals), unitary_matrix=U, check = True)
    return qml.state()

@qml.qnode(dev)
def ground_energy():
    qml.BasisState(occupation, wires=range(n_orbitals))
    qml.BasisRotation(wires=range(n_orbitals), unitary_matrix=U)
    return qml.expval(qubit_H)

#psi = ground_state()
E0 = ground_energy()

print("single-particle energies =", eps)
print("half-filled ground-state energy =", E0)
#print("statevector =", psi)


@qml.qnode(dev)
def circuit(S, theta):
    """
    S: what the paper calls steps in eq. 3
    theta: a vector of form [[theta_U^1, theta_h^1, theta_v^1], ..., [theta_U^S, theta_h^S, theta_v^S]]
    """

    ground_state()

    for step in range(S):
        """
        performance improvement possible:
        jw_U = [qml.jordan_wigner(t) for t in h_U_total]
        jw_h = [qml.jordan_wigner(t) for t in h_h_total]
        jw_v = [qml.jordan_wigner(t) for t in h_v_total]

        for term in jw_U:
            qml.exp(term, 1j * theta[step][0])

        """

        for i, term in enumerate(h_U_total):
            jw_term = qml.jordan_wigner(term)
            qml.exp(jw_term, 1j*theta[step][0]/2)

        #at this point in code, circ = e^{i*theta*h_U}
        for i, term in enumerate(h_h_total):
            jw_term = qml.jordan_wigner(term)
            qml.exp(jw_term, 1j*theta[step][1])
        #at this point in code, circ = e^{i*theta*h_U} @ e^{i*theta*h_h}
        for i, term in enumerate(h_v_total):
            jw_term = qml.jordan_wigner(term)
            qml.exp(jw_term, 1j*theta[step][2])
        #at this point in code, circ = e^{i*theta*h_U} @ e^{i*theta*h_h} @e^{i*theta*h_v}
        
        for i, term in enumerate(h_U_total):
            jw_term = qml.jordan_wigner(term)
            qml.exp(jw_term, 1j*theta[step][0]/2)
        
        #at this point in code, circ = e^{i*theta*h_U/2} @ e^{i*theta*h_h} @e^{i*theta*h_v} @ e^{i*theta*h_U/2}
    

    return qml.state()

