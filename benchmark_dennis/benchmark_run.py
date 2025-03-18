#!/usr/bin/env python

import numpy as np
#from matplotlib import pyplot as plt
from test_class import TestClass
from spin_class import *
from hamiltonian_class import *
from thermodyn_obs import *
import time as tm
from hilbert_space import basis_vectors as bv
from quantum_calc import ai_test as ai
def dennis_matrix(num_spins):
    test_obj = TestClass()
    test_obj.test_spin1()

    test_obj.test_dip()
    test_obj.test_zfs()

    spins = [1] * num_spins
    gfactor = 2
    h = -2
    heis_int = {(i, i+1): h for i in range(num_spins - 1)}

    start_time = tm.time()
    spin_sys = SpinSystem(spins, gfactor, heis_int)
    ham = SpinHamiltonian(spin_sys)
    end_time = tm.time()
    print("Time to generate Hamiltonian: ", end_time - start_time)
    return ham.hamilmat
    #eigvals = np.linalg.eigvalsh(ham.hamilmat)
    #print(eigvals[0])

matrix2 = dennis_matrix(19)
