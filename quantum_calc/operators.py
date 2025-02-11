import numpy as np
from hilbert_space import BasisVectors


class SOperators:
    """This class contains the spin operators s_z, s_plus, s_minus and their respective actions on the basis vectors.
    """

    def __init__(self, num_spins, spin, ket_number, state):
        self.state = state
        self.num_spins = num_spins
        self.spin = spin
        self.ket_number = ket_number
        self.basis = BasisVectors(self.num_spins, self.spin)
        self.ket = self.basis.computation_basis()[self.ket_number]

    def s_z(self):
        """This function calculates the action of the s_z operator on a given ket."""
        return self.ket[self.state] - self.spin

    def s_plus(self):
        """This function calculates the action of the s_plus operator on a given ket."""
        m_i = self.s_z()
        return np.sqrt(self.spin * (self.spin + 1) - m_i * (m_i + 1))

    def s_minus(self):
        """This function calculates the action of the s_plus operator on a given ket."""
        m_i = self.s_z()
        return np.sqrt(self.spin * (self.spin + 1) - m_i * (m_i - 1))

# this function changes the basis vector given the way s_plus * s_plus interact with the ket, i should max. be
# num_spins , since i refers to the first spin
def s_plus_s_plus_ket_change(vec, k, i, j):
    basis_vector = vec[k].copy()
    basis_vector[i] += 1
    basis_vector[j] += 1
    return basis_vector


# this function changes the basis vector given the way s_plus * s_plus interact with the ket, i should max. be
# num_spins , since i refers to the first spin
def s_plus_s_minus_ket_change(vec, k, i, j):
    basis_vector = vec[k].copy()
    basis_vector[i] += 1
    basis_vector[j] -= 1
    return basis_vector


# this function changes the basis vector given the way s_plus * s_plus interact with the ket, i should max. be
# num_spins , since i refers to the first spin
def s_minus_s_plus_ket_change(vec, k, i, j):
    basis_vector = vec[k].copy()
    basis_vector[i] -= 1
    basis_vector[j] += 1
    return basis_vector


# this function changes the basis vector given the way s_plus * s_plus interact with the ket, i should max. be
# num_spins , since i refers to the first spin
def s_minus_s_minus_ket_change(vec, k, i, j):
    basis_vector = vec[k].copy()
    basis_vector[i] -= 1
    basis_vector[j] -= 1
    return basis_vector


if __name__ == "__main__":
    s_operator = SOperators(3, 0.5, 1, 1)
    print(s_operator.s_plus())
    print(s_operator.s_minus())