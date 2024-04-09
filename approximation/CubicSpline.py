import numpy as np

class CubicSpline:

    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._param = np.linspace(0, 1, len(x))
        self._smooth_param = np.linspace(0, 1, len(x))

    def _calc_coeffs_x(self):
        return self._calc_coeffs(self._param, np.array(self._x))
    

    def _calc_coeffs_y(self):
        return self._calc_coeffs(self._param, np.array(self._y))


    def _calc_coeffs(self, param, p):
        # A * M = d
        # A - tridiagonal matrix of coefficients
        # M - second derivatives of data points
        A = np.zeros((len(param), len(param)))
        d = np.zeros(len(param))

        diffs = np.diff(param)
        for i in range(0, len(param)):
            # Boundary conditions - endpoints are zero
            if (i == 0 or i == len(param) - 1):
                A[i, i] = 1
                continue
            d[i] = ((p[i + 1] - p[i]) / diffs[i] - (p[i] - p[i - 1]) / diffs[i - 1]) * 6
            A[i, i] = (diffs[i-1] + diffs[i]) * 2
            A[i, i + 1] = diffs[i]
            A[i, i - 1] = diffs[i-1]

        # Calculate M according to the system
        M = np.linalg.solve(A, d)
        # Finally compute each coefficient
        return {"A": (M[1:] - M[:-1]) / (6 * diffs), 
                "B":  M[:-1] / 2, 
                "C": ((p[1:] - p[:-1]) / diffs) - ((2 * M[:-1] + M[1:]) * diffs) / 6, 
                "D": p[:-1]}


    def apply(self):
        # Create two np arrays based on parameters
        x = np.zeros_like(self._smooth_param)
        y = np.zeros_like(self._smooth_param)
        # Calculate coefficients for both x and y points
        X = self._calc_coeffs_x()
        Y = self._calc_coeffs_y()

        # Fill the arrays
        for i, p in enumerate(self._smooth_param):
            # Find p
            for j in range(len(self._param) - 1):
                if self._param[j] <= p < self._param[j+1]:
                    break
                else:
                    j = len(self._param) - 2

            # Get position
            r = p - self._param[j]
            # Evaluate polynomials
            x[i] = ((X.get("A")[j] * r + X.get("B")[j]) * r + X.get("C")[j]) * r + X.get("D")[j]
            y[i] = ((Y.get("A")[j] * r + Y.get("B")[j]) * r + Y.get("C")[j]) * r + Y.get("D")[j]

        return x, y
