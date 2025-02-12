import numpy as np
from quantum_calc.operators import SOperators as Op
from hilbert_space.basis_vectors import BasisVectors as Bv
import time as tm


def heisenberg_chain(vec, spin, num_spins, j_ij):
    dim = int(2 * spin + 1) ** num_spins
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
                    sz_sz += Op.s_z(ket, i, spin) * Op.s_z(ket, j, spin)
                else:
                    sz_sz += 0
                # s_plus_s_minus_operator
                if any(value > 2 * spin or value < 0 for value in Op.s_plus_s_minus_ket_change(ket, i, j)):
                    s_p_s_m += 0
                elif np.array_equal(bra, Op.s_plus_s_minus_ket_change(ket, i, j)):

                    s_p_s_m += Op.s_plus(ket, i, spin) * Op.s_minus(ket, j, spin)
                else:
                    s_p_s_m += 0
                # s_minus_s_plus_operator
                if any(value > 2 * spin or value < 0 for value in Op.s_minus_s_plus_ket_change(ket, i, j)):
                    s_m_s_p += 0
                elif np.array_equal(bra, Op.s_minus_s_plus_ket_change(ket, i, j)):

                    s_m_s_p += Op.s_minus(ket, i, spin) * Op.s_plus(ket, j, spin)
                else:
                    s_m_s_p += 0
                # appending all the values to matrix (array) pre-factor 0.25 derives from unity matrix
                # transformation and the pre-factors when transforming sx and sy to s_plus and s_minus
                matrix[k, l] = - 2 * j_ij * (sz_sz + 0.5 * (s_p_s_m + s_m_s_p))

    return matrix


if __name__ == "__main__":
    num_spins = 8
    spin = 0.5
    start = tm.time()
    bs = Bv(num_spins, spin)
    basis = bs.computation_basis()
    matrix = heisenberg_chain(basis, spin, num_spins, 1)
    end = tm.time()
    print(f"Runtime s={spin}-Heisenberg_chain with num_spins={num_spins}: ", end - start)

