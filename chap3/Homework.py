import numpy as np


def Jacobi(A, b):

    it = 0
    dim = len(A)
    D = np.zeros(A.shape)
    L = np.zeros(A.shape)
    U = np.zeros(A.shape)
    result = np.ones((1, dim))

    for i in range(dim):
        for j in range(dim):
            if i == j:
                D[i, j] = A[i, j]

    for i in range(dim):
        for j in range(dim):
            if i < j:
                U[i, j] = -A[i, j]

    for i in range(dim):
        for j in range(dim):
            if i > j:
                L[i, j] = -A[i, j]

    B = np.dot(np.linalg.inv(D), L + U)
    f = np.dot(np.linalg.inv(D), b)
    x = np.ones((dim, 1))
    inf = np.max(np.abs(x))

    while inf > 10**-8:
        it += 1
        x = np.dot(B, x) + f
        result = np.around(np.row_stack((result, x.T[0])), 8)
        inf = np.max(np.abs(x))

    return it, result


def SOR(A, b, omg):

    it = 0
    dim = len(A)
    D = np.zeros(A.shape)
    L = np.zeros(A.shape)
    U = np.zeros(A.shape)
    result = np.ones((1, dim))

    for i in range(dim):
        for j in range(dim):
            if i == j:
                D[i, j] = A[i, j]

    for i in range(dim):
        for j in range(dim):
            if i < j:
                U[i, j] = -A[i, j]

    for i in range(dim):
        for j in range(dim):
            if i > j:
                L[i, j] = -A[i, j]

    B = np.dot(np.linalg.inv((D - omg * L)), (1 - omg) * D + omg * U)
    f = omg * np.dot(np.linalg.inv((D - omg * L)), b)
    x = np.ones((dim, 1))
    inf = np.max(np.abs(x))

    while inf > 10**-8:
        it += 1
        x = np.dot(B, x) + f
        result = np.around(np.row_stack((result, x.T[0])), 8)
        inf = np.max(np.abs(x))

    return it, result


def make_A(dim):

    A = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i == j:
                A[i, j] = 20
            if i == j + 1 or j == i + 1:
                A[i, j] = -8
            if i == j + 2 or j == i + 2:
                A[i, j] = 1

    return A


def make_b(dim):

    b = np.zeros((dim, 1))

    return b


def Jacobi_main(dim):

    A = make_A(dim)
    b = make_b(dim)
    it, result = Jacobi(A, b)
    with open('Jacobi ' + 'dim ' + str(dim) + ' iter ' + str(it) + '.txt',
              'a') as f:
        for i in range(len(result)):
            if i != 0:
                print('', file=f)
            for j in range(dim):
                print(result[i, j], file=f, end=' ')


def SOR_main(dim, omg):

    A = make_A(dim)
    b = make_b(dim)
    it, result = SOR(A, b, omg)
    with open(
            'SOR ' + 'dim ' + str(dim) + ' omg ' + str(omg) + ' iter ' +
            str(it) + '.txt', 'a') as f:
        for i in range(len(result)):
            if i != 0:
                print('', file=f)
            for j in range(dim):
                print(result[i, j], file=f, end=' ')


if __name__ == '__main__':
    for dim in (10, 20, 40):
        Jacobi_main(dim)
    for dim in (10, 20, 40):
        for omg in (1, 1.25, 1.5, 1.75):
            SOR_main(dim, omg)
