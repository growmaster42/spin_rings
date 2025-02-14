import numpy as np
from hilbert_space import BasisVectors
import time as tm

class SOperators:
    """This class contains the spin operators s_z, s_plus, s_minus and their respective actions on the basis vectors.
    :param num_spins: The number of spins in the system"""
    @staticmethod
    def s_z(ket, i, spin):
        """This function calculates the action of the s_z operator on a given ket.
        :returns: The magnetic quantum number of the given ket"""
        ket[i] -= spin
        return ket[i]

    @staticmethod
    def s_plus(ket, i, spin):
        """This function calculates the action of the s_plus operator on a given ket.
        :returns: the pre-factor of the s_plus operator after acting on the ket"""
        m_i = ket[i] - spin
        m_i_squared = np.square(m_i)
        return np.sqrt(spin * (spin + 1) - m_i_squared - m_i)

    @staticmethod
    def s_minus(ket, i, spin):
        """This function calculates the action of the s_plus operator on a given ket.
        :returns: the pre-factor of the s_minus operator after acting on the ket"""
        m_i = ket[i] - spin
        m_i_squared = np.square(m_i)
        return np.sqrt(spin * (spin + 1) - m_i_squared + m_i)

    @staticmethod
    def s_plus_s_plus_ket_change(ket, i, j):
        """This function calculates the change of the ket after acting with the s_plus*s_plus operator on two spins.
        :param i: The index of the first spin
        :param j: The index of the second spin
        :returns: The new ket after acting with the s_plus*s_plus operator"""
        changed_ket = ket.copy()
        changed_ket[i] += 1
        changed_ket[j] += 1
        return changed_ket

    @staticmethod
    def s_plus_s_minus_ket_change(ket, i, j):
        """This function calculates the change of the ket after acting with the s_plus*s_minus operator on two spins.
        :param i: The index of the first spin
        :param j: The index of the second spin
        :returns: The new ket after acting with the s_plus*s_minus operator"""
        changed_ket = ket.copy()
        changed_ket[i] += 1
        changed_ket[j] -= 1
        return changed_ket

    @staticmethod
    def s_minus_s_plus_ket_change(ket, i, j):
        """This function calculates the change of the ket after acting with the s_minus*s_plus operator on two spins.
        :param i: The index of the first spin
        :param j: The index of the second spin
        :returns: The new ket after acting with the s_minus*s_plus operator"""
        changed_ket = ket.copy()
        changed_ket[i] -= 1
        changed_ket[j] += 1
        return changed_ket

    @staticmethod
    def s_minus_s_minus_ket_change(ket, i, j):
        """This function calculates the change of the ket after acting with the s_minus*s_minus operator on two spins.
        :param i: The index of the first spin
        :param j: The index of the second spin
        :returns: The new ket after acting with the s_minus*s_minus operator"""
        changed_ket = ket.copy()
        changed_ket[i] -= 1
        changed_ket[j] -= 1
        return changed_ket


if __name__ == "__main__":
    start = tm.time()
    sz = SOperators.s_minus([1, 1, 1, 1], 1, 0.5)
    end = tm.time()
    print(f"Runtime s_z: ", end - start)
    print(sz)