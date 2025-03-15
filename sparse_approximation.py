import numpy as np

def orthogonal_matching_pursuit(A, b, sparsity):
    """ Orthogonal Matching Pursuit (OMP) for complex-valued data. """
    m, n = A.shape
    residual = b.copy()
    support = []
    x = np.zeros(n, dtype=np.complex128)
    
    for _ in range(sparsity):
        correlations = A.conj().T @ residual  # Compute correlations
        idx = np.argmax(np.abs(correlations))  # Find the index of the max correlation
        support.append(idx)
        
        A_restricted = A[:, support]  # Restricted dictionary
        x_restricted = np.linalg.lstsq(A_restricted, b, rcond=None)[0]  # Solve least squares
        
        residual = b - A_restricted @ x_restricted  # Update residual
    
    x[support] = x_restricted  # Assign values to x
    return x

def sequential_pursuit(A, b, sparsity):
    """ Sequential Pursuit (SP) for complex-valued data. """
    m, n = A.shape
    residual = b.copy()
    support = []
    x = np.zeros(n, dtype=np.complex128)
    
    for _ in range(sparsity):
        correlations = A.conj().T @ residual  # Compute correlations
        idx = np.argmax(np.abs(correlations))  # Find the index of the max correlation
        support.append(idx)
        x_subset = np.zeros(n, dtype=np.complex128)
        x_subset[support] = np.linalg.pinv(A[:, support]) @ b  # Compute solution using pseudo-inverse
        residual = b - A @ x_subset  # Update residual
    
    x = x_subset  # Assign values to x
    return x

if __name__ == "__main__":
    # Example usage:
    m, n, sparsity = 20, 40, 5
    A = np.random.randn(m, n) + 1j * np.random.randn(m, n)  # Complex dictionary matrix
    x_true = np.zeros(n, dtype=np.complex128)
    nonzero_indices = np.random.choice(n, sparsity, replace=False)
    x_true[nonzero_indices] = np.random.randn(sparsity) + 1j * np.random.randn(sparsity)
    b = A @ x_true  # Generate measurements

    x_omp = orthogonal_matching_pursuit(A, b, sparsity)
    x_sp = sequential_pursuit(A, b, sparsity)

    print("OMP Solution:", x_omp)
    print("SP Solution:", x_sp)
