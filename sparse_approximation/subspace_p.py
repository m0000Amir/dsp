"""Subspace Pursuit"""
import numpy as np
from tqdm import tqdm


def subspace_pursuit(A, y, num_groups, group_size, sparsity, tol=1e-6):
    """
    Subspace Pursuit (Subspace P) for complex-valued data.
    """
    _, n = A.shape
    residual = y.copy()
    support = set()
    x = np.zeros(n, dtype=np.complex128)

    # Compute group indices
    group_indices = [
        list(range(i * group_size, min((i + 1) * group_size, n)))
        for i in range(num_groups)
    ]

    for _ in tqdm(range(sparsity), desc="SP Progress"):
        # Compute correlations for available groups
        group_correlations = []

        for i, g in enumerate(group_indices):
            # Avoid selecting previous groups
            if not any(idx in support for idx in g):
                correlation = np.linalg.norm(A[:, g].conj().T @ residual)
                group_correlations.append((correlation, i))

        if not group_correlations:  # Stop if no more groups are available
            break

        # Select best 'sparsity' number of groups
        best_groups = sorted(
            group_correlations, key=lambda x: -x[0]
        )[:sparsity]

        # Update support
        new_support = set()
        for _, g_idx in best_groups:
            new_support.update(group_indices[g_idx])

        support = new_support  # Keep only refined support

        # Solve least squares problem using selected support
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
