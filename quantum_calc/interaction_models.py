from quantum_calc.operators import SOperators as Op
from hilbert_space.basis_vectors import BasisVectors as Bv
import time as tm
import numpy as np

def heisenberg_chain(vec, spin, num_spins, j_ij):
    basis_vectors = vec.copy()
    double_spin = 2 * spin
    sz_entries = np.sum((basis[:, :-1] - spin) * (basis[:, 1:] - spin), axis=1)
    matrix = -2 * np.diag(sz_entries)
    for k, ket in enumerate(basis_vectors):
        for l, bra in enumerate(basis_vectors):
            s_p_m = 0
            for i in range(num_spins - 1):
                j = i + 1
                # s_plus_s_minus_operator
                s_plus_s_minus_ket_change = Op.s_plus_s_minus_ket_change(ket, i, j)
                if not any(value > double_spin or value < 0 for value in s_plus_s_minus_ket_change):
                    if np.array_equal(bra, Op.s_plus_s_minus_ket_change(ket, i, j)):
                        s_p_m += 0.5 * Op.s_plus(ket, i, spin) * Op.s_minus(ket, j, spin)

                # s_minus_s_plus_operator
                if not any(value > double_spin or value < 0 for value in Op.s_minus_s_plus_ket_change(ket, i, j)):
                    if np.array_equal(bra, Op.s_minus_s_plus_ket_change(ket, i, j)):
                        s_p_m += 0.5 * Op.s_minus(ket, i, spin) * Op.s_plus(ket, j, spin)

                # appending all the values to matrix (array) pre-factor 0.25 derives from unity matrix
                # transformation and the pre-factors when transforming sx and sy to s_plus and s_minus
                if s_p_m != 0:
                    matrix[k, l] = - 2 * j_ij * s_p_m
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



