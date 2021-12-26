import math
import numpy as np


def f(x, y):
    return -2 * y + 2 * x**2 + 2 * x


def y_hat(x):
    return math.exp(-2 * x) + x**2


class Base:
    def __init__(self, func, y0, xi, xf, h, y_hat) -> None:
        self.func = func
        self.y0 = y0
        self.xi = xi
        self.xf = xf
        self.h = h
        self.y_hat = y_hat

    def iter(self):
        yn = self.y0
        with open(self.name, 'w') as f:
            for xn in np.linspace(self.xi, self.xf,
                                  int((self.xf - self.xi) / self.h) + 1):
                print('%-15.4f%-15.4e%-15.4e%-15.4e' %
                      (xn, yn, self.y_hat(xn), self.y_hat(xn) - yn),
                      file=f)

                yn = self.ynplus(xn, yn, self.h)


class Euler(Base):
    def __init__(self, func, y0, xi, xf, h, y_hat, name='Euler.txt') -> None:
        super().__init__(func, y0, xi, xf, h, y_hat)
        self.name = name

    def ynplus(self, x, y, h):
        return y + h * self.func(x, y)


class ImprovedEuler(Base):
    def __init__(self,
                 func,
                 y0,
                 xi,
                 xf,
                 h,
                 y_hat,
                 name='ImprovedEuler.txt') -> None:
        super().__init__(func, y0, xi, xf, h, y_hat)
        self.name = name

    def ynplus(self, x, y, h):
        return y + h / 2 * (self.func(x, y) +
                            self.func(x + h, y + h * self.func(x, y)))


class ClassicRungeKutta(Base):
    def __init__(self,
                 func,
                 y0,
                 xi,
                 xf,
                 h,
                 y_hat,
                 name='Runge Kutta.txt') -> None:
        super().__init__(func, y0, xi, xf, h, y_hat)
        self.name = name

    def k1(self, x, y):
        return self.func(x, y)

    def k2(self, x, y, h):
        return self.func(x + h / 2, y + h * self.k1(x, y) / 2)

    def k3(self, x, y, h):
        return self.func(x + h / 2, y + h * self.k2(x, y, h) / 2)

    def k4(self, x, y, h):
        return self.func(x + h, y + h * self.k3(x, y, h))

    def ynplus(self, x, y, h):
        return y + h / 6 * (self.k1(x, y) + 2 * self.k2(x, y, h) +
                            2 * self.k3(x, y, h) + self.k4(x, y, h))


if __name__ == '__main__':
    eu = Euler(f, 1, 0, 0.5, 0.025, y_hat)
    ieu = ImprovedEuler(f, 1, 0, 0.5, 0.05, y_hat)
    rk = ClassicRungeKutta(f, 1, 0, 0.5, 0.1, y_hat)
    rk.iter()
