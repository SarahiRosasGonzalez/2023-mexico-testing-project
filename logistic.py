import numpy as np
def logistic_function(x,r):

    f = r*x*(1-x)

    return f


def iterate_f(n,r,x0):
    x = x0
    xs = []
    for _ in range(n):
        f = logistic_function(x,r)
        xs.append(f)
    return np.array(xs)
