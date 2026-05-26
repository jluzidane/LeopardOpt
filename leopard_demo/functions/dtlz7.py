import numpy as np

TARGET = [5.9, 5.9, 5.9, 5.9, 5.9, 5.9, 5.9, 5.9, -121]
RECIPE = [0.0, 0.1, 0.7, 0.7, 0.7, 0.7, 0.7, 0.0, 0.0]  # N=1 fitness=0.354
X_ortho_values = np.array([
    [0, 0.75],
    [0, 0.75],
    [0, 0.75],
    [0, 0.75],
    [0, 0.75],
    [0, 0.75],
    [0, 0.75],
    [0, 0.75],
    [0, 0.75],
])

Y_dim, X_dim = len(TARGET), X_ortho_values.shape[0]
X_input = [f'x{i}' for i in range(1, X_dim+1)]
Y_input = [f'y{i}' for i in range(1, Y_dim+1)]

p = np.zeros((X_dim, Y_dim))


def dtlz7(X, yi):
    if X_dim >= Y_dim:
        k = X_dim - Y_dim + 1
    else:
        k = 0
        print(f"X_dim: {X_dim}, Y_dim: {Y_dim}")

    if k > 0:
        gm = 1 + 9 / (len(X[yi:])) * np.sum(X[yi:], axis=0)
        if yi < Y_dim-1:
            f = gm
        else:
            hm = np.sum(X[:yi] * (1 + np.sin(np.pi * X[:yi])), axis=0)
            f = Y_dim * (gm + 1) + Y_dim * hm
    
    return f


def calculate_dtlz7(X, y_list=Y_input):

    performs=[]
    for y in y_list:
        k = (int)(y[1:])-1
        if k == 8:  # F9 取相反数
            performs.append(-dtlz7(X, k))
        else:
            performs.append(dtlz7(X, k))

    return np.array(performs)
