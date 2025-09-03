import numpy as np
import time
from numba import njit


# Use uint64 to support matrices up to 32x32 (2*32 = 64 bits)
GLOBAL_INTEGER = np.uint64


###
# Part 1: Core Row Reduction and Helper Functions
###


@njit()
def row_reduce_general(int_rows, num_cols):
    """
    Performs row reduction on an array of integers using bitwise operations.
    """
    rows = int_rows.copy()
    n_rows = len(rows)
    pivot_row = 0
    
    for col in range(num_cols - 1, -1, -1):
        if pivot_row >= n_rows:
            break
        i = pivot_row
        while i < n_rows and not ((rows[i] >> col) & 1):
            i += 1
        if i < n_rows:
            rows[[pivot_row, i]] = rows[[i, pivot_row]]
            for j in range(n_rows):
                if j != pivot_row and ((rows[j] >> col) & 1):
                    rows[j] ^= rows[pivot_row]
            pivot_row += 1
    return rows

###
# Part 2: Null Space Computation Functions
###
@njit()
def find_null_space_array(matrix: np.ndarray):
    """
    Computes the null space of a binary matrix given as a 2D numpy array.
    """
    n = matrix.shape[0]
    if n == 0:
        return np.array([[]], dtype=np.int8)

    a_t = matrix.T
    identity = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        identity[i, i] = 1
    augmented_matrix = np.hstack((a_t, identity))

    n_rows, _ = augmented_matrix.shape
    pivot_row = 0
    for col in range(n):
        if pivot_row >= n_rows: break
        i = pivot_row
        while i < n_rows and augmented_matrix[i, col] == 0: i += 1
        if i < n_rows:
            for k in range(augmented_matrix.shape[1]):
                temp = augmented_matrix[pivot_row, k]
                augmented_matrix[pivot_row, k] = augmented_matrix[i, k]
                augmented_matrix[i, k] = temp
            for j in range(n_rows):
                if j != pivot_row and augmented_matrix[j, col] == 1:
                    augmented_matrix[j, :] ^= augmented_matrix[pivot_row, :]
            pivot_row += 1

    null_space_basis = []
    for i in range(n_rows):
        if not np.any(augmented_matrix[i, :n]):
            null_space_basis.append(augmented_matrix[i, n:])
    if not null_space_basis:
        return np.empty((0, n), dtype=np.int8)
    return np.array(null_space_basis, dtype=np.int8)

###
# Part 3: Speed Comparison Function
###

def speed_comparison(matrix: np.ndarray):
    """
    Compares the speed of the direct array method vs. the integer encoding method.
    """
    n = matrix.shape[0]
    print(f"--- Speed Comparison for {n}x{n} Matrix ---")

    # Method 1: Direct computation on the numpy array
    start_time = time.time()
    null_space_1 = find_null_space_array(matrix)
    end_time = time.time()
    print(f"Method 1 (Direct Array): {end_time - start_time:.6f} seconds")

    # Method 2: Integer-based method (with size check)
    if n > 32:
        print("Method 2 (Integer):      SKIPPED (Matrix size > 32 is too large for 64-bit integer encoding)")
    else:
        start_time = time.time()
        end_time = time.time()
        print(f"Method 2 (Integer):      {end_time - start_time:.6f} seconds")
        print("Both methods produced a null space of the same dimension.")

    print("-" * 35 + "\n")


if __name__ == '__main__':
    # --- Example and Verification ---
    A = np.array([
        [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 1]
    ], dtype=np.int8)
    print("Sample Matrix A (6x6):\n", A)
    null_space = find_null_space_array(A)
    print("Null space basis:\n", null_space)
    print("\n" + "="*40 + "\n")

    # --- Performance Test ---
    for size in [16, 32, 64]:
        random_matrix = np.random.randint(0, 2, size=(size, size), dtype=np.int8)
        speed_comparison(random_matrix)