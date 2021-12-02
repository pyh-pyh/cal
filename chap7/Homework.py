import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace


def phi(degree, method='poly'):
    if method == 'poly':
        return x_data**degree
    else:
        return (np.log(x_data))**degree


def matrix(degree, method='poly'):
    mat = np.empty((degree + 1, degree + 1))
    for i in range(degree + 1):
        for j in range(degree + 1):
            mat[i, j] = np.inner(phi(i, method), phi(j, method))
    return mat


def right(degree, f, method='poly'):
    rightt = np.empty((degree + 1, 1))
    for i in range(degree + 1):
        rightt[i, 0] = np.inner(phi(i, method), f)
    return rightt


def fit(degree, x, f, method='poly'):
    result = 0
    if method == 'poly':
        a = np.dot(np.linalg.inv(matrix(degree)), right(degree, f))
        for i in range(degree + 1):
            result += a[i] * x**i
    if method == 'exp':
        a = np.dot(np.linalg.inv(matrix(degree)), right(degree, f))
        result = np.exp(a[0]) * np.exp(a[1] * x)
    if method == 'power':
        a = np.dot(np.linalg.inv(matrix(degree, method)),
                   right(degree, f, method))
        result = np.exp(a[0]) * x**a[1]
    return result


if __name__ == '__main__':
    x_data = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3, 6.8, 7.1]).T
    fx = np.array([
        102.56, 113.18, 130.11, 142.05, 167.53, 195.14, 224.87, 256.73, 299.50,
        326.72
    ]).T
    lnfx = np.log(fx)

    print(matrix(4), '\n', np.linalg.cond(matrix(4)))
    print(matrix(3), '\n', np.linalg.cond(matrix(3)))

    axis_x = linspace(4.0, 7.1, 50)
    Fity_4 = np.array([fit(4, x, fx) for x in axis_x])
    Fity_3 = np.array([fit(3, x, fx) for x in axis_x])
    Fitexp = np.array([fit(1, x, lnfx, method='exp') for x in axis_x])
    Fitpower = np.array([fit(1, x, lnfx, 'power') for x in axis_x])
    '''plt.plot(axis_x,
             Fity_3,
             linewidth=2.0,
             linestyle='--',
             color='red',
             label='degree = 3')
    plt.plot(axis_x,
             Fity_4,
             linewidth=1.2,
             linestyle='-',
             color='blue',
             label='degree = 4')
    plt.plot(axis_x,
             Fitexp,
             linewidth=1.2,
             linestyle='-',
             color='blue',
             label='exp')'''
    plt.plot(axis_x,
             Fitpower,
             linewidth=1.2,
             linestyle='-',
             color='blue',
             label='power')

    plt.legend()
    plt.scatter(x_data, fx, color='green')
    plt.show()
