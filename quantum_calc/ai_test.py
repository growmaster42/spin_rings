from quantum_calc.operators import SOperators as Op
from hilbert_space.basis_vectors import BasisVectors as Bv
import time as tm
import numpy as np
from multiprocessing import Pool, cpu_count
import os


def process_ket_chunk(args):
    ket_indices, ket_list, basis_vectors, spin, num_spins, j_ij = args
    double_spin = 2 * spin
    chunk_results = np.zeros((len(ket_indices), len(basis_vectors)))

    for idx, k in enumerate(ket_indices):
        ket = ket_list[k]
        for l, bra in enumerate(basis_vectors):
            s_p_m = 0
            for i in range(num_spins - 1):
                j = i + 1
                # Vectorize the boundary checks
                s_plus_s_minus_ket_change = Op.s_plus_s_minus_ket_change(ket, i, j)
                s_minus_s_plus_ket_change = Op.s_minus_s_plus_ket_change(ket, i, j)

                # Check boundaries vectorized
                plus_minus_valid = not any(value > double_spin or value < 0
                                           for value in s_plus_s_minus_ket_change)
                minus_plus_valid = not any(value > double_spin or value < 0
                                           for value in s_minus_s_plus_ket_change)

                if plus_minus_valid:
                    if np.array_equal(bra, s_plus_s_minus_ket_change):
                        s_p_m += 0.5 * Op.s_plus(ket, i, spin) * Op.s_minus(ket, j, spin)

                if minus_plus_valid:
                    if np.array_equal(bra, s_minus_s_plus_ket_change):
                        s_p_m += 0.5 * Op.s_minus(ket, i, spin) * Op.s_plus(ket, j, spin)

            if s_p_m != 0:
                chunk_results[idx, l] = -2 * j_ij * s_p_m

    return ket_indices, chunk_results


def parallel_heisenberg_chain(vec, spin, num_spins, j_ij):
    basis_vectors = vec.copy()

    # Vectorized computation of sz_entries
    sz_entries = np.sum((basis_vectors[:, :-1] - spin) * (basis_vectors[:, 1:] - spin), axis=1)
    matrix = -2 * np.diag(sz_entries)

    # Determine chunk size based on number of CPUs
    num_cpus = cpu_count()  # Use multiprocessing.cpu_count() instead of os.sched_getaffinity
    chunk_size = max(1, len(basis_vectors) // (num_cpus * 2))  # Ensure at least 1

    # Create chunks of indices
    ket_indices = list(range(len(basis_vectors)))
    chunks = [ket_indices[i:i + chunk_size]
              for i in range(0, len(ket_indices), chunk_size)]

    # Prepare arguments for parallel processing
    args = [(chunk, basis_vectors, basis_vectors, spin, num_spins, j_ij)
            for chunk in chunks]

    # Process chunks in parallel
    with Pool(processes=num_cpus) as pool:
        results = pool.map(process_ket_chunk, args)

    # Combine results
    for ket_indices, chunk_results in results:
        for idx, k in enumerate(ket_indices):
            matrix[k] += chunk_results[idx]

    return matrix


if __name__ == "__main__":
    # Set environment variables for optimal NumPy performance on M1
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(cpu_count())
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_count())

    num_spins = 10
    spin = 0.5

    # Test original version
    start = tm.time()
    bs = Bv(num_spins, spin)
    basis = bs.computation_basis()
    matrix_parallel = parallel_heisenberg_chain(basis, spin, num_spins, 1)
    time_parallel = tm.time() - start

    print(f"Parallel runtime: {time_parallel:.3f} seconds")
    print(f"Number of CPU cores used: {cpu_count()}")