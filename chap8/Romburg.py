import math


def f(x):
    return x**2 * math.exp(-x)


class Romburg:
    def __init__(self, func, dim, down, up) -> None:
        inter = 1
        self.T = [[] for _ in range(dim)]
        self.T[0] = [0 for _ in range(dim)]
        for i in range(dim):
            interlen = (up - down) / inter
            bottom = down
            for _ in range(inter):
                self.T[0][i] += (func(bottom) + func(bottom + interlen))
                bottom += interlen
            self.T[0][i] /= (inter * 2)
            inter *= 2
        for i in range(dim - 1):
            self.T[i + 1] = [0 for _ in range(dim - 1 - i)]
            for j in range(dim - 1 - i):
                self.T[i + 1][j] = (4**(i + 1) * self.T[i][j + 1] -
                                    self.T[i][j]) / (4**(i + 1) - 1)


if __name__ == '__main__':
    ro = Romburg(f, 3, 0, 1)
