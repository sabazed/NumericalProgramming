import numpy as np
from numpy.linalg import norm


# linear       -y" + p(x)y' + q(x)y + r(x) = 0
# nonlinear     y" =  f(x, y, y')

# Set to true if you want prints
log = True


def p(x):
    return -2/x


def q(x):
    return 2 / x**2


def r(x):
    return np.divide(np.sin(np.log(x)), x**2)


def finite_difference_linear(x0, xn, y0, yn, steps):
    h = (xn - x0) / (steps + 1)
    x = x0 + h
    a = [2 + h**2 * q(x)]
    b = [-1 + (h/2) * p(x)]
    c = [0]
    d = [-(h**2 * r(x)) + (1 + (h/2) * p(x)) * y0]

    for i in range(2, steps):
        x = x0 + i * h
        a.append(2 + h**2 * q(x))
        b.append(-1 + (h/2) * p(x))
        c.append(-1 - (h/2) * p(x))
        d.append(-(h**2 * r(x)))

    x = xn - h
    a.append(2 + h ** 2 * q(x))
    c.append(-1 - (h/2) * p(x))
    d.append(-(h**2 * r(x)) + (1 - (h/2) * p(x)) * yn)

    l = [a[0]]
    u = [b[0] / a[0]]
    z = [d[0] / l[0]]

    for i in range(1, steps - 1):
        l.append(a[i] - c[i] * u[i - 1])
        u.append(b[i] / l[i])
        z.append((d[i] - c[i] * z[i - 1]) / l[i])

    l.append(a[-1] - c[-1] * u[-2])
    z.append((d[-1] - c[-1] * z[-2]) / l[-1])

    w = [y0, z[-1], yn]
    n = len(z) - 2

    for i in range(n, -1, -1):
        w.insert(1, z[i] - u[i] * w[1])

    res = []
    for i in range(0, steps + 2):
        res.append([x0 + i * h, w[i]])
        if log: print(f"{(x0 + i * h):.2f}: {w[i]:.2f}")

    return res


def f(x, y1, y2):
    return (1/8) * (32 + 2 * x**3 - y1 * y2)


def fy1(x, y1, y2):
    return (-1/8) * y2


def fy2(x, y1, y2):
    return (-1/8) * y1


def finite_difference_nonlinear(x0, xn, y0, yn, n, max, tol):
    h = (xn - x0) / (n + 1)

    w = [yn if i == n + 1 else y0 + i * ((yn - y0) / (xn - x0)) * h for i in range(0, n + 2)]
    # w = [y0]
    # for i in range(1, steps + 1):
    #     w.append(y0 + i * ((yn - y0) / (xn - x0)) * h)
    # w.append(yn)

    k = 1
    while k <= max:
        x = x0 + h
        t = (w[2] - y0) / (2*h)
        a = [0. for _ in range(n+1)]
        b = [0. for _ in range(n+1)]
        c = [0. for _ in range(n+1)]
        d = [0. for _ in range(n+1)]

        a[1] = 2 + h ** 2 * fy1(x, w[1], t)
        b[1] = -1 + (h / 2) * fy2(x, w[1], t)
        d[1] = -(2 * w[1] - w[2] - yn + h * f(x, w[1], t))

        for i in range(2, n):
            x = x0 + i * h
            t = (w[i + 1] - w[i - 1]) / (2 * h)
            a[i] = 2 + h**2 * fy1(x, w[i], t)
            b[i] = -1 + (h / 2) * fy2(x, w[i], t)
            c[i] = -1 - (h / 2) * fy2(x, w[i], t)
            d[i] = -(2 * w[i] - w[i + 1] - w[i - 1] + h**2 * f(x, w[i], t))

        x = xn - h
        t = (yn - w[n - 1]) / (2 * h)
        a[n] = 2 + h ** 2 * fy1(x, w[n], t)
        c[n] = -1 - (h / 2) * fy2(x, w[n], t)
        d[n] = -(2 * w[n] - w[n-1] - yn + h ** 2 * f(x, w[n], t))

        l = [0. for _ in range(n+1)]
        u = [0. for _ in range(n+1)]
        z = [0. for _ in range(n+1)]
        l[1] = a[1]
        u[1] = b[1] / a[1]
        z[1] = d[1] / l[1]

        for i in range(2, n):
            l[i] = a[i] - c[i] * u[i - 1]
            u[i] = b[i] / l[i]
            z[i] = (d[i] - c[i] * z[i - 1]) / l[i]

        l[n] = a[n] - c[n] * u[n-1]
        z[n] = (d[n] - c[n] * z[n-1]) / l[n]

        v = [0. for _ in range(n+1)]
        v[n] = z[n]
        w[n] = w[n] + v[n]

        for i in range(n - 1, 0, -1):
            v[i] = z[i] - u[i] * v[i + 1]
            w[i] = w[i] + v[i]

        if norm(v) <= tol:
            res = []
            for i in range(0, n + 2):
                res.append([x0 + i * h, w[i]])
                if log: print(f"{(x0 + i * h):.2f}: {w[i]:.2f}")

            return res

        k = k + 1

    print("Max iteration count reached, stopping...")
    return None


if __name__ == '__main__':
    x0 = 1
    xn = 3
    y0 = 17
    yn = 43/3
    steps = 20
    max = 100
    tol = 10 ** (-5)

    print(finite_difference_linear(x0, xn, y0, yn, steps))

    print()

    print(finite_difference_nonlinear(x0, xn, y0, yn, steps, max, tol))

