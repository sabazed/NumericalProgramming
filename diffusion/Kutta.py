import numpy as np

from utils import FuncUtils, ImageUtils


REPR = "Runge Kutta"


# Runge-Kutta uses iterative methods for function discretization
# Here RK2 is used
class Kutta:

    # Params:
    # - number of iterations,
    # - step-length
    # - lambda which is for 1/h^2
    # - diffusion constant coefficient
    def __init__(self, img, k, h, l, iter):
        self.__img = img
        self.__k = k
        self.__h = h
        self.__l = l
        self.__iter = iter

    # For implementation, an auxiliary function is used, documentation can be found in FuncUtils
    # Diffusion coefficient matrix is calculated from gradient image followed by 1/2 indexes
    def apply(self):
        img_res = self.__img
        img_tmp = np.zeros(self.__img.shape, dtype=self.__img.dtype)
        for t in range(0, self.__iter):
            k1 = self.__l * FuncUtils.apply(img_res, self.__k)

            img_res = ImageUtils.apply_trunc(img_res, lambda: k1)
            k2 = self.__l * FuncUtils.apply(img_res, self.__k)

            img_tmp[1:-1, 1:-1] = img_res[1:-1, 1:-1] + (self.__h / 2) * (k1 + k2)
            img_res = img_tmp
        return img_res
