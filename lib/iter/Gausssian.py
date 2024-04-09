import numpy as np
from numpy.linalg import norm


def gauss(n, a, b, x0, max, tol):
    k = 1
    while k <= max:
        x = [0. for _ in range(n)]
        for i in range(0, n):
            x[i] = (1 / a[i][i]) * (b[i] - sum(a[i][j] * (x[j] if j < i else x0[j] if j > i else 0) for j in range(0, n)))

        if norm(np.subtract(x, x0)) <= tol:
            return x

        k = k + 1
        for i in range(n):
            x0[i] = x[i]

    print("Max iteration count reached, stopping...")
    return None


if __name__ == '__main__':
    a = [[10, -1,  2,  0],
         [-1, 11, -1,  3],
         [2,  -1, 10, -1],
         [0,   3, -1,  8]]
    b = [6, 25, -11, 15]
    xo = [0, 0, 0, 0]
    n = 4
    tol = 10 ** (-3)
    max = 20

    print(gauss(4, a, b, xo, max, tol))

