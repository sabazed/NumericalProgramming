import math
import warnings

import numpy as np
from scipy.optimize import bisect, newton, fsolve, fixed_point

from iter.Iteration import iterate

warnings.filterwarnings('ignore', 'The iteration is not making good progress')


# -y" + p(x)y' + q(x)y + r(x) = 0
# f = (np.power(np.e, -x) * np.sin(x**4) + np.log(1 + 2*x**2) - 5*x**4 * np.cos(x) * y2 - 3*np.cos(x*y1)) / 5
# f = np.power(np.e, -x) * np.sin(x**4) + np.log(1 + 2*x**2) - x**3 * np.cos(x) * y2 - *np.sin(y1 * x**2)
# y'' + x^3cos(x)y' + sin(x^2)y = e^(-x) * sin(x^4) + ln(1 + x^2)
# f = 1/8(32 + 2x**3 - y * y')

# Set to true if you want prints
log = True


def print_butcher(a, b, c):
    print()
    print("Butchereu Tableu:")
    for i in range(len(b)):
        print(f"{c[i]:.2f} |", end="")
        for j in range(len(b)):
            print(f"  {a[i][j]:.2f}", end="")
        print()
    for i in range(-1, len(b)):
        print("______", end="")
    print()
    for i in range(-1, len(b)):
        if i == -1:
            print("     |", end="")
        else:
            print(f"  {b[i]:.2f}", end="")
    print("\n")


def printKs(i, res1, res2):
    def _print(k):
        for i in range(len(k)):
            print(f"k{i+1}: {k[i]:.2f}", end="")
            if i < len(k) - 1:
                print(", ", end="")
        print()
    print(f"[{i+1}]" + ("\t  " if i < 9 else "  " if i < 99 else " ") + f"y1: {res1['y']:.2f}, ", end="")
    _print(res1['k'])
    print(f"\t  y2: {res2['y']:.2f}, ", end="")
    _print(res2['k'])


def find_butcher_coeffs(stages, order_sums, b=None):
    if b is None:
        # b = [(order_sums[0] / stages) for _ in range(stages)]
        # b = [(order_sums[0] / stages) for _ in range(stages)]  # Classical Kutta
        b = [1/6, 1/6, 2/3]
    c = np.array([0 for _ in range(stages)])
    # a = np.array([[0 for _ in range(stages)] for i in range(stages)])

    # Add all order equations
    def equations(c):
        eq1 = b[0] + b[1] + b[2] - 1
        eq2 = c[1] * b[1] + c[2] * b[2] - 1/2
        eq3 = c[1]**2 * b[1] + c[2]**2 * b[2] - 1/3
        # eq4 = c[0]**3 * b[1] + c[1]**3 * b[2] - 1/6
        return [eq1, eq2, eq3]
    c = fsolve(equations, c).tolist()

    # def equations_a(a):
    #     eq1 = b[0] + b[1] + b[2] - 1
    #     eq2 = c[0] + c[1] + c[2] - 1.5
    #     eq3 = b[0] * (a[0][0] * c[0] + a[0][1] * c[1] + a[0][2] * c[2])\
    #           + b[1] * (a[1][0] * c[0] + a[1][1] * c[1] + a[1][2] * c[2])\
    #           + b[2] * (a[2][0] * c[0] + a[2][1] * c[1] + a[2][2] * c[2])
    #     # eq4 = c[0]**3 * b[1] + c[1]**3 * b[2] - 1/6
    #     return [eq1, eq2, eq3]
    #
    # a = fsolve(equations_a, a).tolist()

    a = []
    for i in range(len(c)):
        # a.append([(c[i] / (i + 1)) for _ in range(i + 1)])
        # a.append([c[i] if j == i else 0 for j in range(i + 1)])
        a.append([c[i] if j == i - 1 else 0 for j in range(len(c))])

    a = [[0,   0,   0],
         [1,   0,   0],
         [1/4, 1/4, 0]]

    print(a, b, c)
    if log:
        print_butcher(a, b, c)
    return {'a': a, 'b': b, 'c': c}

# f = np.power(np.e, -x) * np.sin(x**4) + np.log(1 + 2*x**2) - x**3 * np.cos(x) * y2 - np.sin(y1 * x**2)

def p(x):
    return -(x**3) * math.cos(x)


def q(x, y):
    return -np.sin(y * x**2)


def r(x):
    return np.power(np.e, -x) * np.sin(x**4) + np.log(1 + 2*x**2)


# y1' = y2
def f1(x, y1, y2):
    return y2


# y2' = ...
def f2(x, y1, y2):
    # return 1/8 * (32 + 2 * x**3 - y1 * y2)
    return p(x) * y2 + q(x, y1) + r(x)


def get_next_y(order, x, y1, y2, a, b, c, h, f, derivative=False):
    ch = lambda i: 0 if i == 0 else c[i-1] * h
    ak = lambda i: 0 if i == 0 else h * sum(a[i][j] * k[j] for j in range(i - 1))
    if derivative:
        y1, y2 = y2, y1

    k = []
    for i in range(order):
        # k[i] = f(x + c[i] * h, y + h * (a[i][j] * k[j] for j in range(i - 1)))
        k.append(f(x + ch(i), y1 + ak(i), y2))

    y1 = y1 + h * sum(b[i] * k[i] for i in range(order))
    return {'y': y1, 'k': k}


def runge_kutta(order, butcher, x0, y0, xn, yn, steps, f1, f2, guess):
    a = butcher['a']
    b = butcher['b']
    c = butcher['c']
    h = (xn - x0) / steps
    y1 = y0
    y2 = guess
    x = x0

    for i in range(steps):
        old_y1 = y1
        old_y2 = y2
        res1 = get_next_y(order, x, old_y1, old_y2, a, b, c, h, f1)
        res2 = get_next_y(order, x, old_y1, old_y2, a, b, c, h, f2, True)
        y1 = res1['y']
        y2 = res2['y']

        if log:
            printKs(i, res1, res2)
    return y1


def runge_delta(order, butcher, x0, y0, xn, yn, steps, f1, f2, guess):
    res = runge_kutta(order, butcher, x0, y0, xn, yn, steps, f1, f2, guess)
    if log: print()
    print(f"[RK]: Guess {guess} => {res}")
    return res - yn

def runge_bisect(order, stage, order_sums, x0, y0, xn, yn, steps, f1, f2, guess_high, guess_low, max):
    print("Runge-bisect:")
    butcher = find_butcher_coeffs(stage, order_sums)
    delta_func = lambda initial: runge_delta(order, butcher, x0, y0, xn, yn, steps, f1, f2, initial)
    res = iterate(delta_func, bisect, guess_low, guess_high, max=max)
    y = runge_kutta(order, butcher, x0, y0, xn, yn, steps, f1, f2, res)
    return [res, y]


def runge_newton(order, stage, order_sums, x0, y0, xn, yn, steps, f1, f2, guess, max):
    print("Runge-newton:")
    butcher = find_butcher_coeffs(stage, order_sums)
    delta_func = lambda initial: runge_delta(order, butcher, x0, y0, xn, yn, steps, f1, f2, initial)
    res = iterate(delta_func, newton, guess, max=max)
    y = runge_kutta(order, butcher, x0, y0, xn, yn, steps, f1, f2, res)
    return [res, y]


def runge_fixed_point(order, stage, order_sums, x0, y0, xn, yn, steps, f1, f2, guess, max):
    print("Runge-fixed point:")
    butcher = find_butcher_coeffs(stage, order_sums)
    delta_func = lambda initial: runge_delta(order, butcher, x0, y0, xn, yn, steps, f1, f2, initial)
    res = iterate(delta_func, fixed_point, guess, max=max)
    y = runge_kutta(order, butcher, x0, y0, xn, yn, steps, f1, f2, res)
    return [res, y]


if __name__ == '__main__':
    stage = 3
    order = 3
    order_sums = [1, 1/2, 1/3]
    x0 = 1
    xn = np.pi
    y0 = 2
    yn = 3
    steps = 10
    max = 10

    # try:
    #     f = runge_fixed_point(order, stage, order_sums, x0, y0, xn, yn, steps, f1, f2, 0, max)
    #     print(f)
    # except Exception as e:
    #     print("Not enough iterations: ", e)
    #
    # print()

    try:
        b = runge_bisect(order, stage, order_sums, x0, y0, xn, yn, steps, f1, f2, -50, 50, max)
        print(b)
    except Exception as e:
        print("Not enough iterations: ", e)

    print()

    try:
        n = runge_newton(order, stage, order_sums, x0, y0, xn, yn, steps, f1, f2, 0, max)
        print(n)
    except Exception as e:
        print("Not enough iterations: ", e)

