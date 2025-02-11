import numpy as np
import time as tm
import itertools

def hilbert_basis(spin, number_of_spins):
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
    number_of_states = int(2 * spin + 1)
    dim = number_of_states ** number_of_spins
    # generates an array with zeros, dim x and number_of_spins entries per ket
    ketname = np.zeros((dim, number_of_spins), dtype=int)
    # makes rev_ketname globally available
    global rev_ketname
    # for loop that creates the parameter a that will be divided by number_of_states using divmod
    for i in range(dim - 1, -1, -1):
        # e.g. 5 spins 1/2 will result in dim = 32 basis vectors
        a = dim - 1 - i
        # for loop that runs through every entry of the ket
        for j in range(number_of_spins - 1, -1, -1):
            # creating a tuple that corresponds to modulus, x is a tuple that which contains the result of the
            # division and the rest (a/number_of_states, rest)
            x = divmod(a, number_of_states)
            # this puts the second value of the tuple (the rest) in the previously created ketname,
            ketname[i, j] = x[1]
            # this overwrites the previously generated a with the result of the division in divmod
            a = int(x[0])
            # ketname is being flipped for no reason just to have it nicely arranged
    rev_ketname = ketname[::-1]
    # this returns the ascending basis vectors as an array and terminates the function
    return rev_ketname


def hilbert_basis_new(spin, num_spins):
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
    a = int(2 * spin + 1)
    combinations = np.array(list(itertools.product(range(a), repeat=num_spins)), dtype=int)
    return combinations

# function that returns the vector number of any given basis vector
def vector_number(vector, spin):
    m = 0
    for i, entry in enumerate(vector[::-1]):
        m += entry * (2 * spin + 1) ** i
    return int(m)


if __name__ == "__main__":
    start = tm.time()
    basis = hilbert_basis(1, 10)
    end = tm.time()
    start_1 = tm.time()
    basis_new = hilbert_basis_new(1, 10)
    end_1 = tm.time()
    print("Runtime: ", end - start)
    print("type: ", type(basis))
    print("Runtime_new: ", end_1 - start_1)
    print("type_new: ", type(basis_new))
