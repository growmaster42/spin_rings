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


if __name__ == "__main__":





