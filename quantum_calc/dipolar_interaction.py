from operators import *
from distance_calculations import *
from basis_vectors import basvec



def s_j_s_heis(vec, spin, number_of_spins):
    """This function calculates the Heisenberg part of the dipole-dipole interaction,
    containing s_z * s_z, s+ * s- and s- * s+ for each pair of spins"""
    number_of_states = int(2 * spin + 1)
    dim = number_of_states ** number_of_spins
    num_spins = number_of_spins
    matrix = np.zeros((dim, dim), dtype=complex)
    basis_vectors = vec.copy()
    b_ij_factor = b_ij(num_spins)
    # creating the orientation matrices for each pair of spins which apparently is
    # a unity matrix, but this is done to make the code more readable and to transform
    # the matrix elements to the correct basis
    j_xx = 1
    j_xy = 0
    j_yx = 0
    j_yy = 1
    j_zz = 1
    # with k we iterate over all basis vectors; in this case they refer
    # to the row of the matrix
    for k, ket in enumerate(basis_vectors):
        # with l we iterate over all basis vectors; in this case they refer
        # to the column of the matrix
        for l, bra in enumerate(basis_vectors):
            # initializing the matrix elements to zero
            sz_sz = 0
            s_p_s_m = 0
            s_m_s_p = 0
            # iterating over all spins - 1, in order to avoid index errors since
            # the last spin is actually covered by the index j that starts at i + 1
            # the iteration over j returns all possible pairs of spins
            for i in range(num_spins - 1):
                for j in range(i + 1, num_spins):
                    if np.array_equal(ket, bra):
                        # sz_sz operator
                        sz_sz += s_z(vec, spin, k, i) * s_z(vec, spin, k, j) * b_ij_factor[(i, j)] * j_zz
                    else:
                        sz_sz += 0
                    # s_plus_s_minus_operator
                    if any(value > 2 * spin or value < 0 for value in s_plus_s_minus_ket_change(vec, k, i, j)):
                        s_p_s_m += 0
                    elif np.array_equal(bra, s_plus_s_minus_ket_change(vec, k, i, j)):
                        j_pm = (0.25 * j_xx + 0.25j * j_xy - 0.25j * j_yx + 0.25 * j_yy)
                        s_p_s_m += s_plus(vec, spin, k, i) * s_minus(vec, spin, k, j) * j_pm * b_ij_factor[(i, j)]
                    else:
                        s_p_s_m += 0
                    # s_minus_s_plus_operator
                    if any(value > 2 * spin or value < 0 for value in s_minus_s_plus_ket_change(vec, k, i, j)):
                        s_m_s_p += 0
                    elif np.array_equal(bra, s_minus_s_plus_ket_change(vec, k, i, j)):
                        j_mp = (0.25 * j_xx - 0.25j * j_xy + 0.25j * j_yx + 0.25 * j_yy)
                        s_m_s_p += s_minus(vec, spin, k, i) * s_plus(vec, spin, k, j) * j_mp * b_ij_factor[(i, j)]
                    else:
                        s_m_s_p += 0
                    # appending all the values to matrix (array) pre-factor 0.25 derives from unity matrix
                    # transformation and the pre-factors when transforming sx and sy to s_plus and s_minus
                    matrix[k, l] = sz_sz + s_p_s_m + s_m_s_p
    return matrix



def s_j_s_nhdw(vec, spin, number_of_spins):
    """This function calculates the non-Heisenberg part of the
    dipole-dipole interaction, containing s++, s+-, s-+, s--"""
    number_of_states = int(2 * spin + 1)
    num_spins = number_of_spins
    dim = number_of_states ** number_of_spins
    matrix = np.zeros((dim, dim), dtype=complex)
    basis_vectors = vec.copy()
    orient = orientation_matrices(num_spins)
    b_ij_factor = b_ij(num_spins)
    for k, ket in enumerate(basis_vectors):
        for l, bra in enumerate(basis_vectors):
            s_p_s_p = 0
            s_p_s_m = 0
            s_m_s_p = 0
            s_m_s_m = 0
            for i in range(num_spins - 1):
                for j in range(i + 1, num_spins):
                    j_xx = orient[(i, j)][0, 0]
                    j_xy = orient[(i, j)][0, 1]
                    j_yx = orient[(i, j)][1, 0]
                    j_yy = orient[(i, j)][1, 1]
                    # s_plus_s_plus_operator
                    if any(value > 2 * spin or value < 0 for value in s_plus_s_plus_ket_change(vec, k, i, j)):
                        s_p_s_p += 0
                    elif np.array_equal(bra, s_plus_s_plus_ket_change(vec, k, i, j)):
                        j_pp = (0.25 * j_xx - 0.25j * j_xy - 0.25j * j_yx - 0.25 * j_yy)
                        s_p_s_p += s_plus(vec, spin, k, i) * s_plus(vec, spin, k, j) * j_pp * b_ij_factor[(i, j)]
                    else:
                        s_p_s_p += 0
                    # s_plus_s_minus_operator
                    if any(value > 2 * spin or value < 0 for value in s_plus_s_minus_ket_change(vec, k, i, j)):
                        s_p_s_m += 0
                    elif np.array_equal(bra, s_plus_s_minus_ket_change(vec, k, i, j)):
                        j_pm = (0.25 * j_xx + 0.25j * j_xy - 0.25j * j_yx + 0.25 * j_yy)
                        s_p_s_m += s_plus(vec, spin, k, i) * s_minus(vec, spin, k, j) * j_pm * b_ij_factor[(i, j)]
                    else:
                        s_p_s_m += 0
                    # s_minus_s_plus_operator
                    if any(value > 2 * spin or value < 0 for value in s_minus_s_plus_ket_change(vec, k, i, j)):
                        s_m_s_p += 0
                    elif np.array_equal(bra, s_minus_s_plus_ket_change(vec, k, i, j)):
                        j_mp = (0.25 * j_xx - 0.25j * j_xy + 0.25j * j_yx + 0.25 * j_yy)
                        s_m_s_p += s_minus(vec, spin, k, i) * s_plus(vec, spin, k, j) * j_mp * b_ij_factor[(i, j)]
                    else:
                        s_m_s_p += 0
                    # s_minus_s_minus_operator
                    if any(value > 2 * spin or value < 0 for value in s_minus_s_minus_ket_change(vec, k, i, j)):
                        s_m_s_m += 0
                    elif np.array_equal(bra, s_minus_s_minus_ket_change(vec, k, i, j)):
                        j_mm = (0.25 * j_xx - 0.25j * j_xy - 0.25j * j_yx - 0.25 * j_yy)
                        s_m_s_m += s_minus(vec, spin, k, i) * s_minus(vec, spin, k, j) * j_mm * b_ij_factor[(i, j)]
                    else:
                        s_m_s_m += 0

                matrix[k, l] = s_p_s_p + s_m_s_p + s_p_s_m + s_m_s_m
    return matrix


# function that calls both dipole-dipole terms and adds them up correctly
def dipole_dipole(spin, number_of_spins):
    """This function calculates the dipole-dipole interaction:

    H_dd = mu_0*(g*mu_b)**2/(4*pi*1e-10) *
    SUM_i<j 1/(r_ij 4*pi*1e-10) * (s_i * s_j
    - 3(s_i * e_i * e_j * s_j))

    where e_i is the 3-D unit vector of the i-th spin
    and e_j is the 3-D unit vector of the j-th spin; the dyadic product of
    e_i and e_j, here depicted with an asterisk, is a 3x3 matrix with coupling
    constants for each pair of spins that depend on the
    distance between the spins and the orientation of the spins. """
    vec = basvec(spin, number_of_spins)
    matrix = prefactor_sum_dd * (s_j_s_heis(vec, spin, number_of_spins) - 3 * s_j_s_nhdw(vec, spin, number_of_spins))
    return matrix


# function that calculates matrix elements for heisenberg_hamiltonian spin-chain
def heisenberg_chain(vec, spin, number_of_spins, j_ij):
    number_of_states = int(2 * spin + 1)
    dim = number_of_states ** number_of_spins
    num_spins = number_of_spins
    matrix = np.zeros((dim, dim), dtype=complex)
    basis_vectors = vec.copy()
    for k, ket in enumerate(basis_vectors):
        for l, bra in enumerate(basis_vectors):
            sz_sz = 0
            s_p_s_m = 0
            s_m_s_p = 0
            for i in range(num_spins - 1):
                j = i + 1
                # sz_sz operator
                if np.array_equal(ket, bra):
                    sz_sz += s_z(vec, spin, k, i) * s_z(vec, spin, k, j)
                else:
                    sz_sz += 0
                # s_plus_s_minus_operator
                if any(value > 2 * spin or value < 0 for value in s_plus_s_minus_ket_change(vec, k, i, j)):
                    s_p_s_m += 0
                elif np.array_equal(bra, s_plus_s_minus_ket_change(vec, k, i, j)):

                    s_p_s_m += s_plus(vec, spin, k, i) * s_minus(vec, spin, k, j)
                else:
                    s_p_s_m += 0
                # s_minus_s_plus_operator
                if any(value > 2 * spin or value < 0 for value in s_minus_s_plus_ket_change(vec, k, i, j)):
                    s_m_s_p += 0
                elif np.array_equal(bra, s_minus_s_plus_ket_change(vec, k, i, j)):

                    s_m_s_p += s_minus(vec, spin, k, i) * s_plus(vec, spin, k, j)
                else:
                    s_m_s_p += 0
                # appending all the values to matrix (array) pre-factor 0.25 derives from unity matrix
                # transformation and the pre-factors when transforming sx and sy to s_plus and s_minus
                matrix[k, l] = - 2 * j_ij * (sz_sz + 0.5 * (s_p_s_m + s_m_s_p))
    return matrix


# function that calculates heisenberg_spin_ring hamilton matrix entries
def heisenberg_ring(vec, spin, number_of_spins, j_ij):
    number_of_states = int(2 * spin + 1)
    dim = number_of_states ** number_of_spins
    num_spins = number_of_spins
    if num_spins == 2:
        print("Spin-Chain used instead")
        heisenberg_chain(vec, spin, number_of_spins, j_ij)
    matrix = np.zeros((dim, dim), dtype=complex)
    basis_vectors = vec.copy()
    for k, ket in enumerate(basis_vectors):
        for l, bra in enumerate(basis_vectors):
            sz_sz = 0
            s_p_s_m = 0
            s_m_s_p = 0
            for i in range(num_spins):
                j = ((i + 1) % num_spins)
                if np.array_equal(ket, bra):
                    # sz_sz operator
                    sz_sz += s_z(vec, spin, k, i) * s_z(vec, spin, k, j)
                else:
                    sz_sz += 0
                # s_plus_s_minus_operator
                if any(value > 2 * spin or value < 0 for value in s_plus_s_minus_ket_change(vec, k, i, j)):
                    s_p_s_m += 0
                elif np.array_equal(bra, s_plus_s_minus_ket_change(vec, k, i, j)):

                    s_p_s_m += s_plus(vec, spin, k, i) * s_minus(vec, spin, k, j)
                else:
                    s_p_s_m += 0
                # s_minus_s_plus_operator
                if any(value > 2 * spin or value < 0 for value in s_minus_s_plus_ket_change(vec, k, i, j)):
                    s_m_s_p += 0
                elif np.array_equal(bra, s_minus_s_plus_ket_change(vec, k, i, j)):

                    s_m_s_p += s_minus(vec, spin, k, i) * s_plus(vec, spin, k, j)
                else:
                    s_m_s_p += 0
                # appending all the values to matrix (array) pre-factor 0.25 derives from unity matrix
                # transformation and the pre-factors when transforming sx and sy to s_plus and s_minus
                matrix[k, l] = - 2 * j_ij * (sz_sz + 0.5 * (s_p_s_m + s_m_s_p))

    return matrix


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


