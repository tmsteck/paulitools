
@njit
def Anull_space(A):
    """
    Computes the null space of matrix A over GF(2).
    
    Args:
        A (numpy.ndarray): An m x n matrix over GF(2).
        
    Returns:
        numpy.ndarray: A matrix whose rows form a basis for the null space of A.
    """
    m, n = A.shape
    A = A.copy()
    # Keep track of pivot columns
    pivot_columns = np.full(n, -1, dtype=GLOBAL_INTEGER)
    rank = 0
    for col in range(n):
        # Find a row with a leading 1 in this column
        pivot_row = -1
        for row in range(rank, m):
            if A[row, col] == 1:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        # Swap rows if needed
        if pivot_row != rank:
            temp = A[rank, :].copy()
            A[rank, :] = A[pivot_row, :]
            A[pivot_row, :] = temp
        # Eliminate entries in this column
        for row in range(m):
            if row != rank and A[row, col] == 1:
                A[row, :] = (A[row, :] + A[rank, :]) % 2
        pivot_columns[rank] = col
        rank += 1
        if rank == m:
            break
    # Identify free variables
    num_pivots = rank
    num_free = n - num_pivots
    free_columns = np.zeros(num_free, dtype=GLOBAL_INTEGER)
    idx = 0
    for col in range(n):
        is_pivot = False
        for i in range(num_pivots):
            if pivot_columns[i] == col:
                is_pivot = True
                break
        if not is_pivot:
            free_columns[idx] = col
            idx += 1
    # Allocate null space basis matrix
    null_space_basis = np.zeros((num_free, n), dtype=GLOBAL_INTEGER)
    for i in range(num_free):
        null_vector = np.zeros(n, dtype=GLOBAL_INTEGER)
        null_vector[free_columns[i]] = 1
        # Back-substitute
        for j in range(num_pivots - 1, -1, -1):
            pivot_col = pivot_columns[j]
            if pivot_col == -1:
                continue
            row = j
            s = 0
            for k in range(num_free):
                s = (s + A[row, free_columns[k]] * null_vector[free_columns[k]]) % 2
            null_vector[pivot_col] = s
        null_space_basis[i, :] = null_vector
    return null_space_basis
