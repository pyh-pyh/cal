import numpy as np

# Ax=b
A = np.array([[4, 3, 0], [3, 4, -1], [0, -1, 4]])
b = np.array([[3], [5], [-5]])
it = 10

# Calculate
dim = len(b)
x = np.zeros((dim, 1))
r = b - np.dot(A, x)
p = r
for k in range(it):
    alpha = float(np.dot(r.T, r) / np.dot(np.dot(A, p).T, p))
    x = x + alpha * p
    r_form = r
    r = r - alpha * np.dot(A, p)
    beta = float(np.dot(r.T, r) / np.dot(r_form.T, r_form))
    p = r + beta * p
    print(x)
