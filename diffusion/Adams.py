from utils import FuncUtils

from Kutta import Kutta
from utils.ImageUtils import apply_trunc


REPR = "Adams Bashforth"


# Adam-Bashfort uses iterative methods for function discretization
# For first step RK2 is used
class Adams:

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

    def apply(self):
        img_res = None
        img_prev = self.__img
        img_tmp = Kutta(img_prev, self.__k, self.__h, self.__l, 1).apply()
        for i in range(0, self.__iter):
            step_1 = FuncUtils.apply(img_prev, self.__k)
            step_2 = FuncUtils.apply(img_tmp, self.__k)

            img_prev = img_tmp
            img_tmp = apply_trunc(img_tmp, lambda: self.__h * self.__l * 0.5 * (3 * step_1 - step_2))
            img_res = img_tmp
        return img_res
