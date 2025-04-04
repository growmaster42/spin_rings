import numpy as np
import time as time


class RingDistances:
    """This class calculates the distances between spins on a ring and the connection vectors
    """

    def __init__(self, num_spins, x, y):
        """Initializes the class with the number of spins and the x and y coordinates of the first spin, note that the
        rings center will always be at (0,0,0) in the cartesian coordinate system.
        Args:
            num_spins (int): Number of spins on the ring
            self.x, self.y (float): Cartesian coordinates of the first spin"""
        self.num_spins = num_spins
        self.x = x
        self.y = y

    def cartesian_to_polar(self):
        """This function takes the x and y coordinates of the first spin
         on the ring and returns the polar coordinates of the spin.

         returns:
             start_position (list): List with the polar coordinates of the spin"""
        r = np.sqrt(self.x ** 2 + self.y ** 2)
        # calculate the angle phi and getting rid of numerical errors
        if self.x / r == 0:
            phi = np.pi / 2
        elif self.x / r == - 1:
            phi = np.pi
        else:
            phi = np.arccos(self.x / r)
        start_position = [r, phi]
        return start_position

    def position_polar(self):
        """This function takes the number of spins and returns a dictionary
        with the polar coordinates of each spin on the ring.

        Args:
        num_spins (int): Number of spins on the ring
        returns:
        positions (dict): Dictionary with the polar coordinates of each spin with key
        being the index of the spin"""
        start_position = self.cartesian_to_polar()
        start_angle = start_position[1]
        radius = start_position[0]
        positions = {}
        angle_step = 2 * np.pi / self.num_spins
        for i in range(self.num_spins):
            angle = (start_angle + i * angle_step) % (2 * np.pi)
            positions[i] = (radius, angle)
        return positions

    def all_positions_cartesian(self):
        """calculates all the positions of spins by using a start-spin
        whose position can be defined by user
        Returns:
            all_positions (dict): Dictionary with the cartesian coordinates of each spin"""
        all_positions_polar = self.position_polar()
        all_positions = {}
        # getting rid of numerical errors and calculating the cartesian coordinates
        for i in range(self.num_spins):
            if all_positions_polar[i][1] == 0:
                x_c = all_positions_polar[i][0]
                y_c = 0
            elif all_positions_polar[i][1] == np.pi / 2:
                x_c = 0
                y_c = all_positions_polar[i][0]
            elif all_positions_polar[i][1] == np.pi:
                x_c = 0
                y_c = -all_positions_polar[i][0]
            else:
                x_c = all_positions_polar[i][0] * np.cos(all_positions_polar[i][1])
                y_c = all_positions_polar[i][0] * np.sin(all_positions_polar[i][1])
            z_c = 0
            all_positions[i] = np.array([x_c, y_c, z_c], dtype=complex)
        return all_positions

    def connection_vectors(self):
        """This function creates all the connection vectors for each possible spin pair
        Returns:
            all_connection_vectors (dict): Dictionary with the connection vectors"""
        all_positions = self.all_positions_cartesian()
        all_connection_vectors = {}
        for i in range(self.num_spins - 1):
            for j in range(i + 1, self.num_spins):
                connection_vector = all_positions[j] - all_positions[i]
                all_connection_vectors[(i, j)] = connection_vector
        return all_connection_vectors

    def r_ij(self):
        """This function calculates all distances between each spin using the connection vectors
        Returns:
            all_r_ij (dict): Dictionary with the distances between each spin"""
        all_connection_vectors = self.connection_vectors()
        all_r_ij = {}
        for name, vector in all_connection_vectors.items():
            r = np.linalg.norm(vector)
            all_r_ij[name] = r
        return all_r_ij

    def unit_connection_vectors(self):
        """This function returns all normalised connection vectors
        Returns:
            all_unit_connection_vectors (dict): Dictionary with the normalised connection vectors"""
        all_r_ij = self.r_ij()
        all_unit_connection_vectors = {}
        connection_vectors = self.connection_vectors()
        for name, vector in connection_vectors.items():  # enumerate returns a tuple that contains the index i (from 0
            # to n ) and the actual element of the list, so this loop runs through the index i and combines the element
            # from the list with the vectors from the function with all_connection_vectors
            e_ij = vector / all_r_ij[name]
            all_unit_connection_vectors[name] = e_ij
        return all_unit_connection_vectors

    def orientation_matrices(self):
        """This function creates all orientation matrices using the outer product
        Returns:
            all_matrices (dict): Dictionary with the orientation matrices"""
        all_unit_connection_vectors = self.unit_connection_vectors()
        all_matrices = {}
        for name, vector in all_unit_connection_vectors.items():
            matrix = np.outer(vector, vector)
            all_matrices[name] = matrix
        return all_matrices

    def b_ij(self):
        """This function calculates all b_ij pre-factors which are derived from r_ij
        Returns:
            all_b_ij (dict): Dictionary with the b_ij pre-factors"""
        all_r_ij = self.r_ij()
        all_b_ij = {}
        for name, entry in all_r_ij.items():
            b = 1 / (entry ** 3)
            all_b_ij[name] = b
        return all_b_ij


if __name__ == "__main__":
    print("-------TESTING WITH X=0, Y=1----------")
    ring = RingDistances(2, 0, 1)
    start = time.time()
    matrix = ring.orientation_matrices()
    ring.b_ij()
    end = time.time()
    print("Runtime: ", end - start)
    print(matrix)
