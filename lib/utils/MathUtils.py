from typing import List

import numpy as np


def solve_matrix(matrix: List[List[float]],
                 answer: List[float]) -> List[float]:
    """Solve linear system matrix depending on the answer provided
    <b>Ax = b</b>

    Example params:
    matrix:
    [[1,2,3]
    ,[4,5,6]
    ,[7,8,9]]

    answer:
    [1,2,3]

    :param matrix: matrix to solve. Pass lists as matrix rows.
    :param answer: answer vector to solve matrix (vertical)
    :returns: vector that contains the solution (vertical)
    """
    return np.linalg.solve(matrix, answer)
