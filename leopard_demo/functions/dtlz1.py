import numpy as np

TARGET = [0.06, 0.16, 0.38, 0.9, 2.0, 4.8, 10.8, 24, 51]
RECIPE = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # N=0 fitness=0.991
X_ortho_values = np.array([
    [0, 1.0],
    [0, 1.0],
    [0, 1.0],
    [0, 1.0],
    [0, 1.0],
    [0, 1.0],
    [0, 1.0],
    [0, 1.0],
    [0, 1.0],
])

Y_dim, X_dim = len(TARGET), X_ortho_values.shape[0]
X_input = [f'x{i}' for i in range(1, X_dim+1)]
Y_input = [f'y{i}' for i in range(1, Y_dim+1)]

p = np.zeros((X_dim, Y_dim))


def dtlz1(X, yi):
    if X_dim >= Y_dim:
        k = X_dim - Y_dim + 1
        gm = 100 * (k + sum(np.square(X[X_dim-k:] - 0.5) \
            - np.cos(2 * np.pi * (X[X_dim-k:] - 0.5))))
    else:
        k = 0
        gm = 0
        print(f"X_dim: {X_dim}, Y_dim: {Y_dim}")
    f = 0.5 * (1 + gm)
    f *= np.prod(X[:X_dim-yi-1], axis=0)
    if k > 0:
        f *= 1 - X[X_dim-yi-1]
    return f


def calculate_dtlz1(X, y_list=Y_input):
    
    performs=[]
    for y in y_list:
        k = (int)(y[1:])-1
        performs.append(dtlz1(X, k))
    return np.array(performs)
