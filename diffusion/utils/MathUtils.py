import numpy as np


# Calculates truncation error
def get_trunc_error(m1, m2):
    return np.sqrt(np.sum(np.square(m1 - m2)))


#  2 - up
# -2 - down
#  1 - right
# -1 - left
def __switch__(img, diff, pos, i, j):
    res = diff[i][j]
    match pos:
        case  2: res += img[i][j+1] if i == 0 else diff[i-1][j]
        case -2: res += img[len(diff)+1][j+1] if i == len(diff) - 1 else diff[i+1][j]
        case  1: res += img[i+1][len(diff[0])+1] if j == len(diff[0]) - 1 else diff[i][j+1]
        case -1: res += img[i+1][0] if j == 0 else diff[i][j-1]
    return float(res) / 2

# Used for getting the mid-points for the matrix based on the direction passed as `pos`
def get_mids(img, diff, pos):
    coeff = np.zeros_like(diff)
    for i in range(len(diff)):
        for j in range(len(diff[0])):
            coeff[i][j] = __switch__(img, diff, pos, i, j)
    return coeff


# Returns all 4 direction mid-points
def get_mids_all(img, diff):
    return get_mids(img, diff, 2), get_mids(img, diff, -2), get_mids(img, diff, 1), get_mids(img, diff, -1)
