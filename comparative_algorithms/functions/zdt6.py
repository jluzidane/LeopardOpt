import numpy as np

TARGET = [1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45]
RECIPE = [0.77, 0.01, 0.84, 0.03, 0.77, 0.01, 0.77, 0.02, 0.80]  # N=4 fitness=0.119
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
scale = np.array([100, 1, 100, 1, 100, 1, 100, 1, 100])
bias = np.array([88, -9, 88, -9, 88, -9, 88, -9, 88])


def zdt6(X, yi):

    gm = 1 + 9 * pow((np.sum(X, axis=0) / 9), 0.25)
    sign = pow(-1, yi)
    
    if X_dim >= Y_dim:
        f = 1 - np.exp(-2.5*X[yi]) * pow(np.sin(np.pi*X[yi]/2), 6)
    else:
        f = 0
        print(f"X_dim: {X_dim}, Y_dim: {Y_dim}")
    f = 1 - (f / gm) ** sign
    
    return scale[yi] * f - bias[yi]


def calculate_zdt6(X, y_list=Y_input):

    performs=[]
    for y in y_list:
        k = (int)(y[1:])-1
        performs.append(zdt6(X, k))
    return np.array(performs)
