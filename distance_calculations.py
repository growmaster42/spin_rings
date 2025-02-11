import numpy as np
import time as tm


class RingDistances:
    """This class calculates the distances between spins on a ring and the connection vectors"""
    def __init__(self, num_spins, x, y):
        self.num_spins = num_spins
        self.x = x
        self.y = y

    def cartesian_to_polar(self):
        """This function takes the x and y coordinates of the first spin
         on the ring and returns the polar coordinates of the spin.

         Args:
         x, y (float): Cartesian coordinates of the first spin
         returns:
         start_position (list): List with the polar coordinates of the spin
         :rtype: object"""
        r = np.sqrt(self.x ** 2 + self.y ** 2)
        phi = np.arccos(self.x / r)
        start_position = [r, phi]
        return start_position


def position_polar(num_spins, x, y):
    """This function takes the number of spins and returns a dictionary
    with the polar coordinates of each spin on the ring.

    Args:
    num_spins (int): Number of spins on the ring
    returns:
    positions (dict): Dictionary with the polar coordinates of each spin with key
    being the index of the spin"""
    start_position = cartesian_to_polar(x, y)
    start_angle = start_position[1]
    radius = start_position[0]
    positions = {}
    angle_step = 2 * np.pi / num_spins
    for i in range(num_spins):
        angle = (start_angle + i * angle_step) % (2 * np.pi)
        positions[i] = (radius, angle)
    return positions


def all_positions_cartesian(num_spins, x, y):
    """calculates all the positions of spins by using a start-spin
    whose position can be defined by user
    Args:
        num_spins (int): Number of spins on the ring
        x, y (float): Cartesian coordinates of the first spin
    Returns:
        all_positions (dict): Dictionary with the cartesian coordinates of each spin"""
    all_positions_polar = position_polar(num_spins, x, y)
    all_positions = {}
    for i in range(num_spins):
        x_c = all_positions_polar[i][0] * np.cos(all_positions_polar[i][1])
        y_c = all_positions_polar[i][0] * np.sin(all_positions_polar[i][1])
        z_c = 0
        all_positions[i] = np.array([x_c, y_c, z_c], dtype=complex)
    return all_positions


def connection_vectors(num_spins, x, y):
    """This function creates all the connection vectors for each possible spin pair
    Args:
        x, y (float): Cartesian coordinates of the first spin
        num_spins (int): Number of spins on the ring
    Returns:
        all_connection_vectors (dict): Dictionary with the connection vectors"""
    all_positions = all_positions_cartesian(num_spins, x, y)
    all_connection_vectors = {}
    for i in range(num_spins - 1):
        for j in range(i + 1, num_spins):
            connection_vector = all_positions[j] - all_positions[i]
            all_connection_vectors[(i, j)] = connection_vector
    return all_connection_vectors


def r_ij(num_spins, x, y):
    """This function calculates all distances between each spin using the connection vectors
    Args:
        num_spins (int): Number of spins on the ring
        x ,y (float): Cartesian coordinates of the first spin
    Returns:
        all_r_ij (dict): Dictionary with the distances between each spin"""
    all_connection_vectors = connection_vectors(num_spins, x, y)
    # initialize dictionary
    all_r_ij = {}
    for name, vector in all_connection_vectors.items():
        r = np.linalg.norm(vector)
        all_r_ij[name] = r
    return all_r_ij


def unit_connection_vectors(num_spins, x, y):
    """This function returns all normalised connection vectors
    Args:
        num_spins (int): Number of spins on the ring
        x, y (float): Cartesian coordinates of the first spin
    Returns:
        all_unit_connection_vectors (dict): Dictionary with the normalised connection vectors"""
    all_r_ij = r_ij(num_spins, x, y)
    all_unit_connection_vectors = {}
    for name, vector in connection_vectors(num_spins, x,
                                           y).items():  # enumerate returns a tuple that contains the index i (from 0
        # to n ) and the actual element of the list, so this loop runs through the index i and combines the element
        # from the list with the vectors from the function with all_connection_vectors
        e_ij = vector / all_r_ij[name]
        all_unit_connection_vectors[name] = e_ij
    return all_unit_connection_vectors


def orientation_matrices(num_spins, x, y):
    """This function creates all orientation matrices using the outer product
    Args:
        num_spins (int): Number of spins on the ring
        x, y (float): Cartesian coordinates of the first spin
    Returns:
        all_matrices (dict): Dictionary with the orientation matrices"""
    all_unit_connection_vectors = unit_connection_vectors(num_spins, x, y)
    all_matrices = {}
    for name, vector in all_unit_connection_vectors.items():
        matrix = np.outer(vector, vector)
        all_matrices[name] = matrix
    return all_matrices


def b_ij(num_spins, x, y):
    """This function calculates all b_ij pre-factors which are derived from r_ij
    Args:
        num_spins (int): Number of spins on the ring
        x, y (float): Cartesian coordinates of the first spin
    Returns:
        all_b_ij (dict): Dictionary with the b_ij pre-factors"""
    all_r_ij = r_ij(num_spins, x, y)
    all_b_ij = {}
    for name, entry in all_r_ij.items():
        b = 1 / (entry ** 3)
        all_b_ij[name] = b
    return all_b_ij


if __name__ == "__main__":
    print("-------TESTING WITH X=1, Y=1----------")
    ring = RingDistances(4, 0, 1)
    print(ring.cartesian_to_polar())
