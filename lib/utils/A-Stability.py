import numpy as np
import math

def R_s_stage(s, z):
    """
    Compute the stability function of the explicit RK method between 1 and 4.
    """
    # Check if s is between 1 and 4
    if s < 1 or s > 4:
        raise ValueError("s must be between 1 and 4")

    p = np.zeros_like(z)
    for k in range(s+1):
        a = 1 / math.factorial(k) * (z) ** k
        print("1 / " + str(math.factorial(k)) + " * z^" + str(k), end='')
        if k < s:
            print(" + ", end='')
        p += a
    return p

if __name__ == '__main__':

    # Setting up the grid points

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Plot the stability function of the explicit RK method between 1 and 4

    R_s_stage_4 = R_s_stage(4, Z)
