import numpy as np
from pathlib import Path


def transitive_closure(P):
    # Initialize the closure matrix with the given graph
    closure_matrix = np.copy(P)
    # Warshall's algorithm
    N = len(P)
    for k in range(N):
        for i in range(N):
            for j in range(N):
                closure_matrix[i, j] = closure_matrix[i, j] or \
                    (closure_matrix[i, k] and closure_matrix[k, j])

    return closure_matrix
