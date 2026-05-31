import numpy as np

TARGET = [1.5, 3.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0, 150.0]
RECIPE = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # N=0 fitness=0.988
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


def dtlz3(X, yi):
    if X_dim >= Y_dim:
        k = X_dim - Y_dim + 1
        gm = 100 * (k + sum(np.square(X[X_dim-k:] - 0.5) \
            - np.cos(2 * np.pi * (X[X_dim-k:] - 0.5))))
    else:
        k = 0
        gm = 0
        print(f"X_dim: {X_dim}, Y_dim: {Y_dim}")
    f = 1 + gm    
    f *= np.prod(np.cos(X[:X_dim-yi-1] * np.pi / 2), axis=0)
    if k > 0:
        f *= np.sin(X[X_dim-yi-1] * np.pi / 2)

    return f


def calculate_dtlz3(X, y_list=Y_input):

    performs=[]
    for y in y_list:
        k = (int)(y[1:])-1
        performs.append(dtlz3(X, k))
    return np.array(performs)
