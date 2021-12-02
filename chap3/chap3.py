import numpy as np

B = np.array([[0, 0.1, -2 / 10, 0], [1 / 11, 0, 1 / 11, -3 / 11],
              [-2 / 10, 1 / 10, 0, 1 / 10], [0, -3 / 8, 1 / 8, 0]])
f = np.array([6 / 10, 25 / 11, -11 / 10, 15 / 8]).T

x = [0, 0, 0, 0]
for i in range(5):
    x_temp = [0, 0, 0, 0]
    for j in range(len(x)):

        for k in range(len(x)):
            x_temp[j] += B[j, k] * x[k]
        x_temp[j] += f[j]
        x[j] = x_temp[j]
        x_temp = [0, 0, 0, 0]

    print(x)
