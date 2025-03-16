import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


from sparse_approximation.omp import orthogonal_matching_pursuit
from sparse_approximation.subspace_p import subspace_pursuit
from sparse_approximation.sequential_p import sequential_pursuit


if __name__ == "__main__":
    np.random.seed(42)

    # Parameters
    # m, n = 50, 100  # A is m x n
    # num_groups = 10
    # group_size = n // num_groups
    # sparsity = 3  # Number of nonzero groups

    m = 146_100
    num_groups = 81
    group_size = 17
    n = num_groups * group_size
    sparsity = 18  # Number of nonzero groups

    # Generate random dictionary A
    A = np.random.randn(m, n) + 1j * np.random.randn(m, n)

    # Generate sparse ground truth x
    x_true = np.zeros(n, dtype=np.complex128)
    selected_groups = np.random.choice(num_groups, sparsity, replace=False)

    for g in selected_groups:
        indices = range(g * group_size, min((g + 1) * group_size, n))
        x_true[list(indices)] = (
            np.random.randn(len(indices)) +
            1j * np.random.randn(len(indices))
        )

    # Generate noisy measurements
    y = A @ x_true + 0.01 * np.random.randn(m)  # Add small noise

    # START
    start_time = time.time()
    # Apply OMP
    x_omp, support_omp = orthogonal_matching_pursuit(A, y, num_groups,
                                                     group_size, sparsity)
    omp_time = time.time()

    # Apply SP
    x_sp, support_sp = subspace_pursuit(A, y, num_groups, group_size, sparsity)
    subp_time = time.time()

    # Apply Sequential Pursuit
    x_seq, support_seq = sequential_pursuit(A, y, num_groups, group_size,
                                            sparsity)
    seqp_time = time.time()

    finish_time = time.time()
    print("Finish time in seconds:", finish_time - start_time)
    # Compare results
    print("\n=== COMPARISON ===")

    print(f"True Support Groups: {sorted(selected_groups)}")
    omp_total = omp_time - start_time
    print(f"    Orthogonal Matching Pursuit time in seconds:{omp_total} sec")
    omp_groups = sorted(set(s // group_size for s in support_omp))
    print(f"Orthogonal Matching Pursuit Recovered Groups: {omp_groups}")

    subp_total = subp_time - omp_time
    print(f"Subspace Pursuit time in seconds:{subp_total} sec")
    subp_groups = sorted(set(s // group_size for s in support_sp))
    print(f"Subspace Pursuit Recovered Groups: {subp_groups}")

    seqp_total = seqp_time - subp_time
    print(f"    Sequential Pursuit time in seconds:{seqp_total} sec")
    seqp_groups = sorted(set(s // group_size for s in support_seq))
    print(f"Sequential Pursuit Recovered Indices: {seqp_groups}")

    # Plot results
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 4, 1)
    plt.stem(np.abs(x_true))
    plt.title("True Sparse Signal")

    plt.subplot(1, 4, 2)
    plt.stem(np.abs(x_omp))
    plt.title("OMP")

    plt.subplot(1, 4, 3)
    plt.stem(np.abs(x_sp))
    plt.title("Sparse Pursuit")

    plt.subplot(1, 4, 4)
    plt.stem(np.abs(x_seq))
    plt.title("Sequential Pursuit")

    plt.tight_layout()
    plt.show()
