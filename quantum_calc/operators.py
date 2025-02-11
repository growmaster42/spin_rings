import numpy as np
import hilbert_space as hs
class Operators:
    def __init__(self, spin, k, i):
        self.spin = spin
        self.k = k
        self.i = i

    def s_z(self, ket, spin, k, i):
        """This function acts as the s_z operator on a given ket and simply returns.
        the m_i which is quantum number of the i-th spin.
        Note: m_i = a_i - spin, since a computational basis is used.
        Example: s_z|0.5> = 0.5|0.5>,
        solution of the eigenvalue equation for s_z|s, m> = m|s, m>

        Args:
        vec (list): List of basis vectors
        spin (float): Spin of the system
        k (int): Index of the basis vector
        i (int): Index of the spin"""
        basis_vectors = vec.copy()
        vector = basis_vectors[k]
        m_i = vector[i] - spin
        return m_i


# this function returns a list with all pre-factors that s_plus produces for each ket
def s_plus(vec, spin, k, i):
    """This function acts as the s_plus operator on a given ket and returns
    the square root of some weird formula as seen in the code.

    Args:
        vec (list): List of basis vectors
        spin (float): Spin of the system
        k (int): Index of the basis vector
        i (int): Index of the spin"""
    basis_vectors = vec.copy()
    vector = basis_vectors[k]
    m = vector[i] - spin
    sqrt = np.sqrt(spin * (spin + 1) - m * (m + 1)) * h

    return sqrt


def s_minus(vec, spin, k, i):
    basis_vectors = vec.copy()
    vector = basis_vectors[k]
    m = vector[i] - spin
    sqrt = np.sqrt(spin * (spin + 1) - m * (m - 1)) * h
    return sqrt


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




