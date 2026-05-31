import numpy as np

TARGET = [0.01, 0.03, 0.08, 0.15, 0.20, 0.40, 0.60, 0.98, 1.50]
RECIPE = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]  # N=3 fitness=0.291
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
scale = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
bias = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])


def dtlz6(X, yi):
    
    if X_dim >= Y_dim:
        k = X_dim - Y_dim + 1
        gm = sum(X[X_dim-k:] ** 0.1)
    else:
        k = 0
        gm = 0
        print(f"X_dim: {X_dim}, Y_dim: {Y_dim}")    
    f = 1 + gm
    theta = 0.5 * (1 + 2 * gm * X[:X_dim-yi-1]) / (1 + gm)
    f *= np.prod(np.cos(theta * np.pi / 2), axis=0)
    if k > 0:
        theta = 0.5 * (1 + 2 * gm * X[X_dim-yi-1]) / (1 + gm)
        f *= np.sin(theta * np.pi / 2)

    return f


def calculate_dtlz6(X, y_list=Y_input):
    
    performs=[]
    for y in y_list:
        k = (int)(y[1:])-1
        performs.append(dtlz6(X, k))
    return np.array(performs)
