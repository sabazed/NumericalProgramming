def iterate(func, method, guess_1: float, guess_2: float = None, max: int = None) -> float:
    """Use provided iteration method to derive the correct guess

    Iteration methods available: bisect, newton, fixed_point from <i>scipy.optimize</i>
    :param func: function which accepts an initial guess(es) and returns the difference between target y and derived y from initial guess
    :param method: iterative method to use
    :param guess_1: initial guess 1
    :param guess_2: initial guess 2
    :return: returns the correct guess
    """
    if guess_2 is None:
        if max is None:
            return method(func, guess_1)
        return method(func, guess_1, maxiter=max)
    if max is None:
        return method(func, guess_1, guess_2)
    return method(func, guess_1, guess_2, maxiter=max)
