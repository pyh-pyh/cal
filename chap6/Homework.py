import numpy as np


def iteration(x1, x2, x3, it):
    x1_init, x2_init, x3_init = x1, x2, x3
    for i in range(it):
        x1_temp = (np.math.cos(x1 * x2) + 1 / 2) / 3
        x2_temp = np.math.sqrt((x1**2 + np.math.sin(x3) + 1.06) / 81) - 0.1
        x3_temp = (-(10 * np.math.pi - 3) / 3 - np.math.exp(-x1 * x2)) / 20
        x1 = x1_temp
        x2 = x2_temp
        x3 = x3_temp
        with open(
                './chap4/data/normal/Normal result init = ' +
                '( %.1f %.1f %.1f )' % (x1_init, x2_init, x3_init) +
                ' iter = ' + str(it) + '.txt', 'a') as f:
            print('iter = ',
                  '%-8d' % (i + 1),
                  '%-15f%-15f%-30f' % (x1, x2, x3),
                  file=f)


class newton:
    def __init__(self, x1, x2, x3, it) -> None:
        self.x1, self.x2, self.x3 = x1, x2, x3
        self.x1_init, self.x2_init, self.x3_init = x1, x2, x3
        self.it = it
        pass

    def F1(self, x1, x2):
        return 3 * x1 - np.math.cos(x1 * x2) - 1 / 2

    def F2(self, x1, x2, x3):
        return x1**2 - 81 * ((x2 + 0.1)**2) + np.math.sin(x3) + 1.06

    def F3(self, x1, x2, x3):
        return np.math.exp(-x1 * x2) + 20 * x3 + (10 * np.math.pi - 3) / 3

    def Fprime11(self, x1, x2):
        return 3 + np.math.sin(x1 * x2) * x2

    def Fprime12(self, x1, x2):
        return np.math.sin(x1 * x2) * x1

    def Fprime13(self):
        return 0

    def Fprime21(self, x1):
        return 2 * x1

    def Fprime22(self, x2):
        return -2 * 81 * (x2 + 0.1)

    def Fprime23(self, x3):
        return np.math.cos(x3)

    def Fprime31(self, x1, x2):
        return np.math.exp(-x1 * x2) * (-x2)

    def Fprime32(self, x1, x2):
        return np.math.exp(-x1 * x2) * (-x1)

    def Fprime33(self):
        return 20

    def iter(self):
        for i in range(self.it):
            x_mat = np.array([[self.x1], [self.x2], [self.x3]])
            Fprime = np.array([[
                self.Fprime11(self.x1, self.x2),
                self.Fprime12(self.x1, self.x2),
                self.Fprime13()
            ],
                               [
                                   self.Fprime21(self.x1),
                                   self.Fprime22(self.x2),
                                   self.Fprime23(self.x3)
                               ],
                               [
                                   self.Fprime31(self.x1, self.x2),
                                   self.Fprime32(self.x1, self.x2),
                                   self.Fprime33()
                               ]])
            F = np.array([[self.F1(self.x1, self.x2)],
                          [self.F2(self.x1, self.x2, self.x3)],
                          [self.F3(self.x1, self.x2, self.x3)]])
            delta_x = -np.dot(np.linalg.inv(Fprime), F)
            x_mat = x_mat + delta_x
            self.x1, self.x2, self.x3 = x_mat[0, 0], x_mat[1, 0], x_mat[2, 0]
            with open(
                    './chap4/data/newton/Newton result init = ' +
                    '( %.1f %.1f %.1f )' %
                (self.x1_init, self.x2_init, self.x3_init) + ' iter = ' +
                    str(self.it) + '.txt', 'a') as f:
                print('iter = ',
                      '%-8d' % (i + 1),
                      '%-15f%-15f%-15f' %
                      (round(self.x1, 8), round(self.x2, 8), round(self.x3, 8)),
                      file=f)


class broyden:
    def __init__(self, x1, x2, x3, it) -> None:
        self.x1, self.x2, self.x3 = x1, x2, x3
        self.x1_init, self.x2_init, self.x3_init = x1, x2, x3
        self.it = it
        pass

    def F1(self, x1, x2):
        return 3 * x1 - np.math.cos(x1 * x2) - 1 / 2

    def F2(self, x1, x2, x3):
        return x1**2 - 81 * ((x2 + 0.1)**2) + np.math.sin(x3) + 1.06

    def F3(self, x1, x2, x3):
        return np.math.exp(-x1 * x2) + 20 * x3 + (10 * np.math.pi - 3) / 3

    def Fprime11(self, x1, x2):
        return 3 + np.math.sin(x1 * x2) * x2

    def Fprime12(self, x1, x2):
        return np.math.sin(x1 * x2) * x1

    def Fprime13(self):
        return 0

    def Fprime21(self, x1):
        return 2 * x1

    def Fprime22(self, x2):
        return -2 * 81 * (x2 + 0.1)

    def Fprime23(self, x3):
        return np.math.cos(x3)

    def Fprime31(self, x1, x2):
        return np.math.exp(-x1 * x2) * (-x2)

    def Fprime32(self, x1, x2):
        return np.math.exp(-x1 * x2) * (-x1)

    def Fprime33(self):
        return 20

    def iter(self):
        F = np.array([[self.F1(self.x1, self.x2)],
                      [self.F2(self.x1, self.x2, self.x3)],
                      [self.F3(self.x1, self.x2, self.x3)]])
        Fprime = np.array([[
            self.Fprime11(self.x1, self.x2),
            self.Fprime12(self.x1, self.x2),
            self.Fprime13()
        ],
                           [
                               self.Fprime21(self.x1),
                               self.Fprime22(self.x2),
                               self.Fprime23(self.x3)
                           ],
                           [
                               self.Fprime31(self.x1, self.x2),
                               self.Fprime32(self.x1, self.x2),
                               self.Fprime33()
                           ]])
        B = np.linalg.inv(Fprime)

        for i in range(self.it):
            x_mat = np.array([[float(self.x1)], [float(self.x2)],
                              [float(self.x3)]])
            p = -np.dot(B, F)
            x_mat += p
            self.x1, self.x2, self.x3 = x_mat[0, 0], x_mat[1, 0], x_mat[2, 0]
            F_form = F
            F = np.array([[self.F1(self.x1, self.x2)],
                          [self.F2(self.x1, self.x2, self.x3)],
                          [self.F3(self.x1, self.x2, self.x3)]])
            q = F - F_form
            B += (np.dot(np.dot(
                (p - np.dot(B, q)), p.T), B)) / (np.dot(np.dot(p.T, B), q))

            with open(
                    './chap4/data/broyden/Broyden result init = ' +
                    '( %.1f %.1f %.1f )' %
                (self.x1_init, self.x2_init, self.x3_init) + ' iter = ' +
                    str(self.it) + '.txt', 'a') as f:
                print(
                    'iter = ',
                    '%-8d' % (i + 1),
                    '%-15f%-15f%-15f' %
                    (round(self.x1, 8), round(self.x2, 8), round(self.x3, 16)),
                    file=f)


if __name__ == '__main__':
    for i in (-0.5, 0, 0.5):
        for j in (-0.5, 0, 0.5):
            for k in (-0.5, 0, 0.5):
                # iteration(i, j, k, 100)
                newton(i, j, k, 100).iter()
                broyden(i, j, k, 100).iter()
    '''print(
        broyden(0.500000, 0.000000, -0.523599, 10).F1(0.500000, 0),
        broyden(0.500000, 0.000000, -0.523599, 10).F2(0.500000, 0, -np.math.pi/6),
        broyden(0.500000, 0.000000, -0.523599, 10).F3(0.500000, 0, -np.math.pi/6))'''
