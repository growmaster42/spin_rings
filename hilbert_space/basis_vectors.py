import numpy as np
import itertools


class BasisVectors:
    def __init__(self, num_spins, spin):
        self.num_spins = num_spins
        self.spin = spin
        self.basis = self.computation_basis()

    def computation_basis(self):
        """Creates the basis gets for a given number of spins and a given spin value.
            Since the basis of a spin system can be denoted with |m_1, m_2,..., m_N> where N is the number of spins and m_i is
            the magnetic quantum number, we can write the basis down easily. In order to keep the memory usage as low as
            possible, we transform the individual states using a_i = s_i + m_i.
            For instance, the basis ket |1/2, 1/2, 1/2, 1/2, 1/2> will be transformed to |1, 1, 1, 1, 1>
            For a spin system of 2 spins s=1/2 the basis will be:
            {|1/2, 1/2>, |1/2, -1/2>, |-1/2, 1/2>, |-1/2, -1/2>}
            or transformed:
            {|1, 1>, |1, 0>, |0, 1>, |0, 0>}
            Since it will simply be sufficient to fill the gets in ascending order up to the highest a_i, with all possible
             combinations of all m_i. We do not need to calculate the basis vectors in the manner described above. """
        a = int(2 * self.spin + 1)
        kets = np.array(list(itertools.product(range(a), repeat=self.num_spins)), dtype=int)
        return kets


if __name__ == "__main__":
    bv = BasisVectors(3, 1)
    print(bv.computation_basis())