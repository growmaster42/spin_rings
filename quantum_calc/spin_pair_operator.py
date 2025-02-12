def spin_pair_operator(spin, spin_pair, number_of_spins):
    vec = basvec(spin, number_of_spins)
    i, j = spin_pair
    number_of_states = int(2 * spin + 1)
    dim = number_of_states ** number_of_spins
    matrix = np.zeros((dim, dim), dtype=complex)
    mat_sz_sz = np.zeros((dim, dim), dtype=complex)
    mat_s_p_s_m = np.zeros((dim, dim), dtype=complex)
    mat_s_m_s_p = np.zeros((dim, dim), dtype=complex)
    basis_vectors = vec.copy()
    for k, ket in enumerate(basis_vectors):
        for l, bra in enumerate(basis_vectors):
            sz_sz = 0
            s_p_s_m = 0
            s_m_s_p = 0
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
            matrix[k, l] = sz_sz + 0.5 * (s_p_s_m + s_m_s_p)
            mat_sz_sz[k, l] = sz_sz
            mat_s_p_s_m[k, l] = s_p_s_m
            mat_s_m_s_p[k, l] = s_m_s_p

    return matrix, mat_sz_sz, mat_s_p_s_m, mat_s_m_s_p
