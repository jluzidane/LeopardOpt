import numpy as np

TARGET = [0.02, 0.04, 0.06, 0.08, 0.10, 0.10, 0.20, 0.50, 1.00]
RECIPE = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.5, 0.2, 0.1]  # N=0 fitness=0.991
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


def dtlz2(X, yi):
    if X_dim >= Y_dim:
        k = X_dim - Y_dim + 1
        gm = sum((X[X_dim-k:] - 0.5)**2)
    else:
        k = 0
        gm = 0
        print(f"X_dim: {X_dim}, Y_dim: {Y_dim}")    
    f = 1 + gm    
    f *= np.prod(np.cos(X[:X_dim-yi-1] * np.pi / 2), axis=0)
    if k > 0:
        f *= np.sin(X[X_dim-yi-1] * np.pi / 2)

    return f*scale[yi]+bias[yi]


def calculate_dtlz2(X, y_list=Y_input):
    
    performs=[]
    for y in y_list:
        k = (int)(y[1:])-1
        performs.append(dtlz2(X, k))
    return np.array(performs)
