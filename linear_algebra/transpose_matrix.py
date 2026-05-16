# --------------Without external libraries--------------

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    """
    n = len(a) 
    m = len(a[0]) 

    res = [[0 for _ in range(n)] for _ in range(m)]

    for i in range(n) :
        for j in range(m) :
            res[j][i] = a[i][j] 
    return res

# --------------With numpy--------------
import numpy as np
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    """
    a = np.array(a)
    return a.transpose().tolist()