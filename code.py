import pennylane as qml
from pennylane import numpy as np

def generate_hubbard_terms(n_sites):
    """
    Generates hermitian terms for the Hubbard model on a two-leg ladder.
    Args:
        n_sites (int): Total number of sites (must be even for a 2-leg ladder).
    """
    n_orbitals = 2 * n_sites  # Each site has spin-up and spin-down orbitals
    h_h_ops, h_v_ops, h_u_ops = [], [], []

    # Helper to get orbital index: site i, spin s (0 for up, 1 for down)
    def idx(site, spin):
        return 2 * site + spin

    half_n = n_sites // 2

    for i in range(half_n):
        for s in [0, 1]:
            # 1. h_h: Horizontal hopping (periodic boundary conditions)
            # Top leg: i to (i+1) % half_n; Bottom leg: (i+half_n) to ((i+1)%half_n + half_n)
            site_a, site_b = i, (i + 1) % half_n
            site_c, site_d = i + half_n, ((i + 1) % half_n) + half_n
            
            for (u, v) in [(site_a, site_b), (site_c, site_d)]:
                # Fermionic hopping: c*_u c_v + c*_v c_u
                op = qml.FermiC(idx(u, s)) @ qml.FermiA(idx(v, s))
                h_h_ops.append(op + qml.adjoint(op))

            # 2. h_v: Vertical hopping (open boundary conditions)
            # Between top leg (i) and bottom leg (i + half_n)
            op_v = qml.FermiC(idx(i, s)) @ qml.FermiA(idx(i + half_n, s))
            h_v_ops.append(op_v + qml.adjoint(op_v))

        # 3. h_U: Repulsion terms (on-site interaction)
        # U * n_{i, up} * n_{i, down} for both legs
        for site in [i, i + half_n]:
            n_up = qml.FermiC(idx(site, 0)) @ qml.FermiA(idx(site, 0))
            n_down = qml.FermiC(idx(site, 1)) @ qml.FermiA(idx(site, 1))
            h_u_ops.append(n_up @ n_down)

    return h_h_ops, h_v_ops, h_u_ops

generate_hubbard_terms(4)