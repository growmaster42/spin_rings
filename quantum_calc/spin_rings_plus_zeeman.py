def zeeman(vec, spin, number_of_spins, b_field):
    number_of_states = int(2 * spin + 1)
    dim = number_of_states ** number_of_spins
    num_spins = number_of_spins
    matrix = np.zeros((dim, dim), dtype=complex)
    basis_vectors = vec.copy()
    g = 2
    mu_b = 0.6717
    for k, ket in enumerate(basis_vectors):
        for l, bra in enumerate(basis_vectors):
            sz = 0
            for i in range(num_spins):
                # sz_sz operator
                if np.array_equal(ket, bra):
                    sz += s_z(vec, spin, k, i)
                else:
                    sz += 0

                # appending all the values to matrix (array) pre-factor 0.25 derives from unity matrix
                # transformation and the pre-factors when transforming sx and sy to s_plus and s_minus
                matrix[k, l] = g * mu_b * b_field * sz
    return matrix


def spin_ring(s, n, j_ij):
    vec = basvec(s, n)
    if j_ij == 0:
        spin_ring_matrix = dipole_dipole(s, n)
    else:
        spin_ring_matrix = heisenberg_ring(vec, s, n, j_ij) + dipole_dipole(s, n)
    return spin_ring_matrix

def spin_ring_zeeman(s, n, j_ij, b_field):
    vec = basvec(s, n)
    if j_ij == 0:
        spin_ring_matrix = dipole_dipole(s, n)
    else:
        spin_ring_matrix = heisenberg_ring(vec, s, n, j_ij) + dipole_dipole(s, n) + zeeman(vec, s, n, b_field)
    return spin_ring_matrix

def spin_ring_heisenberg_zeeman(s, n, j_ij, b_field):
    vec = basvec(s, n)
    if j_ij == 0:
        spin_ring_matrix = dipole_dipole(s, n)
    else:
        spin_ring_matrix = heisenberg_ring(vec, s, n, j_ij) + zeeman(vec, s, n, b_field)
    return spin_ring_matrix