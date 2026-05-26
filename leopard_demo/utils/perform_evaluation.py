from utils.data_io import save_data
import numpy as np
import json
import os
import csv
import importlib
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


# Load benchmark configuration.
config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f) 
formula = config["settings"]["formula"]
try:
    func_config = importlib.import_module(f"functions.{formula}")
except ImportError:
    functions_dir = "./functions"
    valid_formulas = [filename[:-3] for filename in os.listdir(functions_dir)
        if filename.endswith(".py") and not filename.startswith("__")]
    raise ValueError(
        f"Configuration for '{formula}' was not found. "
        f"Please select a valid benchmark from the following options:\n"
        f"{valid_formulas}\n"
    )
targets = np.array(func_config.TARGET)
X0_input = func_config.X_input
Y0_input = func_config.Y_input


def calculate_performance(X, y_list=Y0_input, formula=formula):
    """
    Evaluate the response values of a selected benchmark problem.

    Parameters
    ----------
    X : numpy.ndarray
        Input design vector to be evaluated.
    y_list : list of str, optional
        List of response variables to be computed, e.g., ['y1', 'y2', 'y5'].
        Default is defined by the selected benchmark module.
    formula : str, optional
        Name of the benchmark problem.

    Returns
    -------
    performances : numpy.ndarray
        Computed response values corresponding to ``y_list``.
    """   

    if formula=='rosenbrock1':
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


def evaluate_fitness(X, epochs, formula=formula, epoch_trace=None):
    """
    Evaluate performance compliance and fitness metrics.

    Parameters
    ----------
    X : numpy.ndarray
        Current input formulation.
    epochs : int
        Current optimization epoch.
    formula : str, optional
        Name of the benchmark problem.
    epoch_trace : str, optional
        File path for saving the epoch-wise optimization trace.

    Returns
    -------
    Fit : list
        Fitness metrics in the form
        [fitness, distance over non-compliant performance, Euclidean distance].
    dist_err : numpy.ndarray
        Normalized distance errors. The error is set to zero for compliant performnances.
    Nc : int
        Number of compliant performances.
    performs : numpy.ndarray
        Response values of the current formulation.
    y_nonc : list of str
        List of non-compliant performances, e.g., ['y1', 'y2', 'y5'].
    """

    performs = calculate_performance(X, formula=formula)    
    # Normalized deviation from the target values.
    diff = (performs - targets) / abs(targets)

    indices = np.where(diff < 0)[0].tolist()
    Nnonc = len(indices)
    y_nonc = [f'y{i+1}' for i in indices]
    # Compute the number of compliant performance metrics (Nc).
    Nc = len(np.where(diff >= 0)[0])

    mask = diff < 0
    squared_diff = np.square(diff[mask])
    d_euclid = np.round(np.sqrt(np.sum(squared_diff)), 6)  # Euclidean distance.
    if len(Y0_input) == 0:
        raise ValueError(f"The response list '{Y0_input}' is empty."
                         f"Please check the benchmark configuration in './functions/{formula}.py'.")
    # Compute the fitness values.
    fitness = np.round(np.sqrt(np.sum(squared_diff) / len(Y0_input)), 6)
    if Nnonc == 0:  fitness2 = 0.
    else:  fitness2 = np.round(np.sqrt(np.sum(squared_diff) / Nnonc), 6)  # Average distance over non-compliant performances.
    Fit = [fitness, fitness2, d_euclid]
    # Compute relative error over non-compliant performances.
    dist_err = np.where(mask, np.abs(diff), 0)

    if epoch_trace:
        current_data = (
            list(np.round(X, 4)) + 
            list(np.round(performs, 2)) + 
            [Fit[0]] + 
            [Nc] + 
            list(np.round(dist_err, 4))
        )
        if epochs == 0:
            Y0_err = [f"dy_err{i}" for i in range(1, len(Y0_input) + 1)]
            with open(epoch_trace, mode='w', newline='') as outfile:
                writer = csv.writer(outfile, delimiter=',')            
                header = X0_input + Y0_input + ["Fitness", "Nc"] + Y0_err
                writer.writerow(header)
        
        save_data(current_data, epoch_trace)
        
    return Fit, dist_err, Nc, performs, y_nonc
