"Sequential Pursuit"
import numpy as np
from tqdm import tqdm


def sequential_pursuit(A, y, num_groups, group_size, sparsity, tol=1e-6):
    """
    Sequential Pursuit (SP) for complex-valued data.
    """
    _, n = A.shape
    residual = y.copy()
    support = set()  # Set to track selected indices
    x = np.zeros(n, dtype=np.complex128)

    # Compute group indices
    group_indices = [
        list(range(i * group_size, min((i + 1) * group_size, n)))
        for i in range(num_groups)
    ]

    for _ in tqdm(range(sparsity), desc="Sequential Group Pursuit Progress"):
        # Compute correlations for all groups
        residual_projection = A.conj().T @ residual

        # Select the group with the maximum correlation
        group_correlations = [
            (np.linalg.norm(residual_projection[g]), i)
            for i, g in enumerate(group_indices)
            # Avoid previously selected groups
            if not support.intersection(g)
        ]

        if not group_correlations:  # Stop if no more groups are available
            break

        # Select the best group based on the highest correlation
        best_group_idx = max(group_correlations, key=lambda x: x[0])[1]
        support.update(group_indices[best_group_idx])

        # Solve least squares problem using the selected groups
        support_list = sorted(support)
        A_restricted = A[:, support_list]
        x_restricted = np.linalg.lstsq(A_restricted, y, rcond=None)[0]

        # Update residual
        residual = y - A_restricted @ x_restricted

        # Early stopping if residual is small
        if np.linalg.norm(residual) < tol:
            break

    # Assign values to x at correct indices
    x[list(support)] = x_restricted
    return x, sorted(support)
