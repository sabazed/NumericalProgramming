import numpy as np

from utils import ImageUtils, MathUtils


# Function for applying an auxiliary method
# Same as fj, taking all differences as whole matrices, dividing into 4 as per g direction, used for full discretization
def apply(img, k):
    img_trunc = img[1:-1, 1:-1]
    img_grad = ImageUtils.apply_gradient(img)
    img_grad_trunc = ImageUtils.apply_gradient(img_trunc)

    diff = np.exp((np.power(img_grad_trunc, 2)) * -1 / (np.power(k, 2)))

    u, d, r, l = MathUtils.get_mids_all(img_grad, diff)
    _u, _d, _r, _l = -img[:-2, 1:-1] + img_trunc, img[2:, 1:-1] - img_trunc, img[1:-1, 2:] - img_trunc, -img[1:-1, :-2] + img_trunc

    return -u * _u + d * _d + r * _r - l * _l
