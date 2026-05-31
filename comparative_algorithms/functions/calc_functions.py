import numpy as np
from functions.rosenbrock1 import calculate_rosenbrock1
from functions.rosenbrock2 import calculate_rosenbrock2
from functions.dtlz1 import calculate_dtlz1
from functions.dtlz2 import calculate_dtlz2
from functions.dtlz3 import calculate_dtlz3
from functions.dtlz4 import calculate_dtlz4
from functions.dtlz5 import calculate_dtlz5
from functions.dtlz6 import calculate_dtlz6
from functions.dtlz7 import calculate_dtlz7
from functions.zdt6 import calculate_zdt6
from functions.ackley import calculate_ackley
from functions.griewank import calculate_griewank
from functions.rastrigin import calculate_rastrigin
from functions.zakharov import calculate_zakharov
from functions.levy import calculate_levy
from functions.schwefel import calculate_schwefel
from functions.perm import calculate_perm
from functions.trid import calculate_trid
from functions.styblinski import calculate_styblinski
from functions.dixon import calculate_dixon


def calculate_performance(X, Y_dim=9, formula='rosenbrock1'):
    """
    Evaluate the response values of a selected benchmark problem.

    Parameters
    ----------
    X : numpy.ndarray
        Input design vector to be evaluated.
    Y_dim : int, optional
        Dimension of response variables.
        Default is 9.
    formula : str, optional
        Name of the benchmark problem.

    Returns
    -------
    performances : numpy.ndarray
        Computed response values corresponding to ``y_list``.
    """
    
    X = np.array(X)
    y_list = [f'y{i}' for i in range(1, Y_dim+1)]
    
    # 实验反馈值接口
    if formula=='experiment':
        print("Recommended formulation：", X)
        performances = np.array(input("Please enter the performances: "))
    elif formula=='rosenbrock1':
        performances = calculate_rosenbrock1(X, y_list)
    elif formula=='rosenbrock2':
        performances = calculate_rosenbrock2(X, y_list)
    elif formula=='dtlz1':
        performances = calculate_dtlz1(X, y_list)
    elif formula=='dtlz2':
        performances = calculate_dtlz2(X, y_list)
    elif formula=='dtlz3':
        performances = calculate_dtlz3(X, y_list)
    elif formula=='dtlz4':
        performances = calculate_dtlz4(X, y_list)
    elif formula=='dtlz5':
        performances = calculate_dtlz5(X, y_list)
    elif formula=='dtlz6':
        performances = calculate_dtlz6(X, y_list)
    elif formula=='dtlz7':
        performances = calculate_dtlz7(X, y_list)
    elif formula=='zdt6':
        performances = calculate_zdt6(X, y_list)
    elif formula=='ackley':
        performances = calculate_ackley(X, y_list)    
    elif formula=='griewank':
        performances = calculate_griewank(X, y_list)
    elif formula=='rastrigin':
        performances = calculate_rastrigin(X, y_list)
    elif formula=='zakharov':
        performances = calculate_zakharov(X, y_list)
    elif formula=='levy':
        performances = calculate_levy(X, y_list)
    elif formula=='schwefel':
        performances = calculate_schwefel(X, y_list)
    elif formula=='perm':
        performances = calculate_perm(X, y_list)
    elif formula=='trid':
        performances = calculate_trid(X, y_list) 
    elif formula=='styblinski':
        performances = calculate_styblinski(X, y_list)
    elif formula=='dixon':
        performances = calculate_dixon(X, y_list)    
    else:
        raise ValueError(f"No evaluation function is available for '{formula}'.")
        
    return np.round(performances, 6)
