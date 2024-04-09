import numpy as np

class LeastSquares:

    def __init__(self, x, y, cols):
        self._x = x
        self._y = y
        self._cols = cols # Degree for columns in Vandermonde matrix
        self._param = np.linspace(0, 1, len(x))
        self._smooth_param = np.linspace(0, 1, len(x))

    def _calc_coeffs_x(self):
        return self._calc_coeffs(self._x)
    

    def _calc_coeffs_y(self):
        return self._calc_coeffs(self._y)

    def _calc_coeffs(self, y):
        # (AT ⋅ A) ⋅ a = AT ⋅ y for coefficients a
        # a = coefficients
        A = np.vander(self._param, self._cols, increasing=True)
        coefficients = np.linalg.solve(A.T @ A, A.T @ y)
        return coefficients
    
    def apply(self):
        # Calculate both x and y coefficients
        x_coeffs = self._calc_coeffs_x()
        y_coeffs = self._calc_coeffs_y()
        # Evaluate to get smooth coord
        x = np.vander(self._smooth_param, len(x_coeffs), increasing=True) @ x_coeffs
        y = np.vander(self._smooth_param, len(y_coeffs), increasing=True) @ y_coeffs
        return x, y