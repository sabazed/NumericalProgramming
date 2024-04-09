import math
import warnings

import numpy as np
from scipy.optimize import bisect, newton, fixed_point

from iter.Iteration import iterate

warnings.filterwarnings('ignore', 'The iteration is not making good progress')


# -y" + p(x)y' + q(x)y + r(x) = 0
# f = (np.power(np.e, -x) * np.sin(x**4) + np.log(1 + 2*x**2) - 5*x**4 * np.cos(x) * y2 - 3*np.cos(x*y1)) / 5


def p(x):
    return -5 * (x ** 4) * math.cos(x) / 5


def q(x, y):
    return - 3 * np.cos(x * y) / 5


def r(x):
    return np.power(np.e, -x) * np.sin(x ** 4) + np.log(1 + 2 * x ** 2) / 5


# y1' = y2
def f1(x, y1, y2):
    return y2


# y2' = ...
def f2(x, y1, y2):
    return p(x) * y2 + q(x, y1) + r(x)


def get_next_y(x, y1, y2, h, f, derivative=False):
    if derivative:
        y1, y2 = y2, y1
    y1 = y1 + h * f(x, y1, y2)
    return y1


def forward_euler(x0, y0, xn, yn, steps, f1, f2, guess):
    h = (xn - x0) / steps
    y1 = y0
    y2 = guess
    x = x0

    for i in range(steps):
        old_y1 = y1
        old_y2 = y2
        y1 = get_next_y(x, old_y1, old_y2, h, f1)
        y2 = get_next_y(x, old_y1, old_y2, h, f2, True)

    return y1


def euler_delta(x0, y0, xn, yn, steps, f1, f2, guess):
    res = forward_euler(x0, y0, xn, yn, steps, f1, f2, guess)
    print(f"Euler: initial guess {guess} -> {res}")
    return res - yn


def euler_bisect(x0, y0, xn, yn, steps, f1, f2, guess_low, guess_high, max):
    print("Euler-bisect:")
    delta_func = lambda initial: euler_delta(x0, y0, xn, yn, steps, f1, f2, initial)
    res = iterate(delta_func, bisect, guess_low, guess_high, max=max)
    y = forward_euler(x0, y0, xn, yn, steps, f1, f2, res)
    return [res, y]


def euler_newton(x0, y0, xn, yn, steps, f1, f2, guess, max):
    print("Euler-newton:")
    delta_func = lambda initial: euler_delta(x0, y0, xn, yn, steps, f1, f2, initial)
    res = iterate(delta_func, newton, guess, max=max)
    y = forward_euler(x0, y0, xn, yn, steps, f1, f2, res)
    return [res, y]


def euler_fixed_point(x0, y0, xn, yn, steps, f1, f2, guess, max):
    print("Euler-fixed point:")
    delta_func = lambda initial: euler_delta(x0, y0, xn, yn, steps, f1, f2, initial)
    res = iterate(delta_func, fixed_point, guess, max=max)
    y = forward_euler(x0, y0, xn, yn, steps, f1, f2, res)
    return [res, y]


if __name__ == '__main__':
    x0 = np.pi
    xn = 2.7
    y0 = np.multiply(np.pi, 2)
    yn = -0.7
    steps = 50
    max = 100

    try:
        f = euler_fixed_point(x0, y0, xn, yn, steps, f1, f2, 1, max)
        print(f)
    except Exception as e:
        print("Not enough iterations: ", e)

    print()

    try:
        b = euler_bisect(x0, y0, xn, yn, steps, f1, f2, -100, 100, max)
        print(b)
    except Exception as e:
        print("Not enough iterations: ", e)

    print()

    try:
        n = euler_newton(x0, y0, xn, yn, steps, f1, f2, 1, max)
        print(n)
    except Exception as e:
        print("Not enough iterations: ", e)
