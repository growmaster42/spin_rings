from quantum_calc.operators import SOperators as Op
from hilbert_space.basis_vectors import BasisVectors as Bv
import time as tm
import numpy as np
from scipy import sparse
import concurrent.futures
import numba
from functools import partial


@numba.njit(parallel=True)
def calculate_sz_entries(basis_vectors, spin):
    """Calculate the sz_sz interaction terms using Numba's parallelization"""
    return np.sum((basis_vectors[:, :-1] - spin) * (basis_vectors[:, 1:] - spin), axis=1)


def process_basis_chunk(chunk_data):
    """Process a chunk of basis vectors for parallel execution"""
    chunk_indices, basis_vectors, spin, num_spins, j_ij, basis_dict_bytes = chunk_data

    start_idx, end_idx = chunk_indices
    chunk = basis_vectors[start_idx:end_idx]

    # Reconstruct basis_dict from bytes
    basis_dict = {k: v for k, v in basis_dict_bytes}

    # Allocate arrays for this chunk
    est_nonzero = len(chunk) * (num_spins - 1) * 4  # Increased estimation to avoid resizing
    rows = np.zeros(est_nonzero, dtype=np.int32)
    cols = np.zeros(est_nonzero, dtype=np.int32)
    data = np.zeros(est_nonzero, dtype=np.float64)
    entry_idx = 0

    # Cached transition vectors
    transition_state = np.zeros_like(basis_vectors[0])
    double_spin = 2 * spin
    transitions = [(i, i + 1) for i in range(num_spins - 1)]

    # Process all basis vectors in this chunk
    for idx, ket in enumerate(chunk):
        k = start_idx + idx  # Global index

        for i, j in transitions:
            # s_plus_s_minus
            if not (ket[i] == double_spin or ket[j] == 0):
                s_plus_i = Op.s_plus(ket, i, spin)
                s_minus_j = Op.s_minus(ket, j, spin)

                if s_plus_i != 0 and s_minus_j != 0:
                    np.copyto(transition_state, ket)
                    transition_state[i] += 1
                    transition_state[j] -= 1

                    target_key = transition_state.tobytes()
                    if target_key in basis_dict:
                        l = basis_dict[target_key]
                        value = 0.5 * s_plus_i * s_minus_j

                        if entry_idx >= len(rows):
                            new_size = len(rows) * 2
                            rows = np.resize(rows, new_size)
                            cols = np.resize(cols, new_size)
                            data = np.resize(data, new_size)

                        rows[entry_idx] = k
                        cols[entry_idx] = l
                        data[entry_idx] = -2 * j_ij * value
                        entry_idx += 1

            # s_minus_s_plus
            if not (ket[i] == 0 or ket[j] == double_spin):
                s_minus_i = Op.s_minus(ket, i, spin)
                s_plus_j = Op.s_plus(ket, j, spin)

                if s_minus_i != 0 and s_plus_j != 0:
                    np.copyto(transition_state, ket)
                    transition_state[i] -= 1
                    transition_state[j] += 1

                    target_key = transition_state.tobytes()
                    if target_key in basis_dict:
                        l = basis_dict[target_key]
                        value = 0.5 * s_minus_i * s_plus_j

                        if entry_idx >= len(rows):
                            new_size = len(rows) * 2
                            rows = np.resize(rows, new_size)
                            cols = np.resize(cols, new_size)
                            data = np.resize(data, new_size)

                        rows[entry_idx] = k
                        cols[entry_idx] = l
                        data[entry_idx] = -2 * j_ij * value
                        entry_idx += 1

    # Trim arrays to actual size
    return rows[:entry_idx], cols[:entry_idx], data[:entry_idx]


def heisenberg_chain_m1_optimized(vec, spin, num_spins, j_ij, num_workers=None):
    """
    Optimized Heisenberg chain computation leveraging M1 Pro parallelism
    """
    basis_vectors = vec  # No need to copy
    double_spin = 2 * spin
    dim = len(basis_vectors)

    # Use optimal number of workers for M1 Pro
    if num_workers is None:
        # M1 Pro has 8 cores (efficiency + performance), but leaving 1-2 for system
        num_workers = 6

    # Calculate diagonal elements (sz_sz part) using Numba's parallelization
    sz_entries = calculate_sz_entries(basis_vectors, spin)
    diagonal = -2 * j_ij * sz_entries

    # Create basis dictionary for lookups - shared between all processes
    basis_dict = {b.tobytes(): idx for idx, b in enumerate(basis_vectors)}
    # Convert to a format that can be serialized for multiprocessing
    basis_dict_bytes = list(basis_dict.items())

    # Split work into chunks
    chunk_size = max(1, len(basis_vectors) // num_workers)
    chunks = [(i, min(i + chunk_size, len(basis_vectors)))
              for i in range(0, len(basis_vectors), chunk_size)]

    # Prepare chunk data for workers
    chunk_data = [(chunk, basis_vectors, spin, num_spins, j_ij, basis_dict_bytes)
                  for chunk in chunks]

    # Process chunks in parallel
    all_rows = []
    all_cols = []
    all_data = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_basis_chunk, chunk_data))

    # Combine results from all workers
    for rows, cols, data in results:
        all_rows.append(rows)
        all_cols.append(cols)
        all_data.append(data)

    if all_rows:  # Check if we have any entries
        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        data = np.concatenate(all_data)

        # Create sparse matrix and add diagonal
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(dim, dim))
        matrix = matrix + sparse.diags(diagonal, 0)

        # Convert to dense as in the original code
        return matrix.toarray()
    else:
        # Return identity matrix if no off-diagonal elements
        return np.diag(diagonal)


if __name__ == "__main__":
    num_spins = 8
    spin = 2.5
    start = tm.time()
    bs = Bv(num_spins, spin)
    basis = bs.computation_basis()

    # Original implementation
    # matrix = heisenberg_chain_optimized(basis, spin, num_spins, 1)

    # New M1 Pro optimized implementation
    matrix = heisenberg_chain_m1_optimized(basis, spin, num_spins, 1)

    end = tm.time()
    print(f"Runtime s={spin}-Heisenberg_chain with num_spins={num_spins}: {end - start:.6f} seconds")
    print(matrix)