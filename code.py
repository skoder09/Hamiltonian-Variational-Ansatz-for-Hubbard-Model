import pennylane as qml

def site_spin2qubit(site, spin):
    """
    Maps a site and spin to a qubit index. up=0, down = 1.
    """
    return 2 * site + spin

def generate_hubbard_terms(n_sites):
    """
    Generates hermitian terms for the Hubbard model on a two-leg ladder. In all code, here we assume N=4.
    Args:
        n_sites (int): Total number of sites (must be even for a 2-leg ladder).
    """
    
    # Generate h_h
    h_h = []

    for (u,v) in [(0,1), (2,3),(0,2), (1,3)]: #the last 2 elements are for h_v
        for s in [0,1]:
            term = qml.fermi.FermiWord({(0, site_spin2qubit(u,s)) : "+", (1, site_spin2qubit(v,s)) : "-"})
            term_conj = qml.fermi.FermiWord({(0, site_spin2qubit(v,s)) : "+", (1, site_spin2qubit(u,s)) : "-"})

            hermitian_term = qml.fermi.FermiSentence({term: 1.0, term_conj: 1.0})
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


h_h, h_v, h_U = generate_hubbard_terms(4)

h_h_total = h_h[0]
for h_h_i in h_h[1:]:
    h_h_total += h_h_i

h_v_total = h_v[0]
for h_v_i in h_v[1:]:
    h_v_total += h_v_i

h_U_total = h_U[0]
for h_U_i in h_U[1:]:
    h_U_total += h_U_i

print(h_h_total)