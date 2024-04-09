import numpy as np

class Lagrange:

    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._param = np.linspace(0, 1, len(x))
        self._smooth_param = np.linspace(0, 1, len(x))

    def _lagrange(self, x, y, a, b):
        # Calculate denominators
        ones = np.ones(len(x))
        for i in range(len(x)):
            for j in range(len(x)):
                ones[i] *= (x[i] - x[j]) if (i != j and x[i] != x[j]) else 1

        # Perform interpolation
        for i in range(len(x)):
            _ones = np.ones_like(a)
            for j in range(len(x)):
                _ones *= (a - x[j]) if (i != j and ones[i] != 0) else 1
            b += (_ones / ones[i]) * y[i] 

        return b
    
        
    def apply(self):
        # Calculate both for x and y coords
        x = self._lagrange(self._param, self._x, self._smooth_param, np.zeros_like(self._smooth_param))
        y = self._lagrange(self._param, self._y, self._smooth_param, np.zeros_like(self._smooth_param))
        return x, y