from bayes_opt import BayesianOptimization
import os
import csv
import pandas as pd
import numpy as np
from math import sqrt
import time
from datetime import datetime
import argparse
import importlib
from functions.calc_functions import calculate_performance
from scipy.spatial.distance import cdist
from scipy.stats import dirichlet, norm
import random


# Command-line argument interface
parser = argparse.ArgumentParser()
parser.add_argument("--randseed", type=int, help="Random seed value")
parser.add_argument("--outfile", type=str, default="results", help="Directory for saving output files")
parser.add_argument("--formula", type=str, default="rosenbrock2", help="Benchmark function for optimization")
parser.add_argument("--niter", type=int, default=100, help="Max iteration steps")
args = parser.parse_args()


def load_formula_config(formula_name):
    """Load the parameter configuration of the specified benchmark function."""
    try:
        config_module = importlib.import_module(f"functions.{formula_name}")
        return config_module.TARGET, config_module.X_ortho_values, config_module.X_input
    except ImportError:
        raise ValueError(f"Configuration for '{formula_name}' could not be found.")


# Configuration settings
formula = args.formula
TARGET, X_ortho_values, X_input = load_formula_config(formula)
NUM_VAR = X_ortho_values.shape[0]
NUM_OBJ = len(TARGET)
print(f"Optimization started: FUNC = '{formula}', NUM_VAR = {NUM_VAR}, NUM_OBJ = {NUM_OBJ}")

input_data_file = f'./input_data/input_{formula}.csv'  # Source orthogonal data file
weighted_data_file = f'./input_data/ParEGO_{formula}.csv'
X_title = [f'X{i}' for i in range(1, NUM_VAR+1)]
Y_title = [f'F{i}' for i in range(1, NUM_OBJ+1)]
# Domain of each formulation variable
all_bounds = {f'X{i+1}': (np.min(row), np.max(row))
              for i, row in enumerate(X_ortho_values)}

if args.randseed is not None:
    random.seed(args.randseed)
    np.random.seed(args.randseed)

params_history = []
results_history = []
weighted_sum_history = []
weighted_history = []
Fitness_history = []
Nc_history = []
result_file = os.path.join(f'{args.outfile}', f'parego_{formula}.csv')
with open(result_file, mode='w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    writer.writerow(['Iteration'] + X_title + Y_title + ['weighted_sum', 'Fitness', 'Nc'])


def normalize_objectives(Y):
    """Dynamically normalize objective values to the [0, 1] interval."""
    Y_min = np.min(Y, axis=0)
    Y_max = np.max(Y, axis=0)
    # Prevent division by zero when all values are identical.
    range_vals = Y_max - Y_min
    range_vals[range_vals == 0] = 1.0
    return (Y - Y_min) / range_vals


def read_orgothonal_data(infile, outfile):
    """Read and process the orthogonal experimental data."""
    try:
        data = pd.read_csv(infile)
    except FileNotFoundError:
        print(f"File not found: {infile}")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return

    x_columns = data.iloc[:, :NUM_VAR]
    y_columns = data.iloc[:, NUM_VAR:NUM_VAR+NUM_OBJ]

    data = pd.concat([x_columns, y_columns], axis=1)
    duplicates = data.iloc[:, :NUM_VAR].duplicated()
    data = data[~duplicates]    
    data.to_csv(outfile, index=False, header=False,  sep=',')
    
    return data


def generate_random_weights(num_obj):
    """Generate a random weight vector with nonnegative entries summing to one."""
    weights = np.random.rand(num_obj)
    weights /= np.sum(weights)
    return weights


def calculate_result(param_set):
    """Evaluate the multi-objective response vector for a given parameter set."""
    x = np.array([v for _, v in param_set])
    x = np.around(x, decimals=2)

    yi = calculate_performance(x, formula=formula)
    
    return np.round(yi, 6)


def black_box_function(**kwargs):
    """Black-box objective function for Bayesian optimization."""
    global params_history, params_history, weighted_sum_history, weighted_history, Fitness_history, Nc_history, Y_data
    
    param_set = sorted(kwargs.items())
    param = tuple(round(value, 4) for name, value in param_set)
    
    try:
        idx = params_history.index(param)
        return weighted_sum_history[idx]
    except ValueError:
        yi = calculate_result(param_set)
        Y_data = np.vstack((Y_data, yi))
        Y_norm = normalize_objectives(Y_data)
        weighted_sum = np.sum(weighted_history[-1] * Y_norm[-1])

        fitness = sqrt(sum(((i - j)/TARGET[index])**2/NUM_OBJ if i < j else 0 
                        for index, (i, j) in enumerate(zip(yi, TARGET))))
        Nc = sum(i > j for i, j in zip(yi, TARGET))
                
        params_history.append(param)
        results_history.append(yi)
        weighted_sum_history.append(weighted_sum)
        Fitness_history.append(round(fitness, 4))
        Nc_history.append(Nc)

        with open(result_file, mode='a', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            row = ([iteration] + list(param) + list(yi) + 
                   [round(weighted_sum, 4)] + 
                   [round(fitness, 4)] + [Nc])
            writer.writerow(row)
            if Nc >= NUM_OBJ:
                print("A feasible solution compliant with all objectives has been found:\n", row)
                
        return weighted_sum


def load_data_and_register(optimizer, filename, Y_norm):
    """
    Load historical data from file and register them into the Bayesian optimizer.
    """
    global params_history, results_history, weighted_sum_history, weighted_history

    
    with open(filename, mode='r') as infile:
        reader = csv.reader(infile, delimiter=',')
        # next(reader)

        for i, row in enumerate(reader):
            try:
                row_values = [float(row[i]) for i in range(NUM_VAR)]
                params = dict(zip(X_title, row_values))

                yi = [float(row[i]) for i in range(NUM_VAR, NUM_VAR+NUM_OBJ)]

                weights = generate_random_weights(NUM_OBJ)
                weighted_sum = np.sum(weights * Y_norm[i])

                optimizer.register(params=params, target=weighted_sum)

                params_history.append(tuple(row_values))
                results_history.append(np.round(yi, 6))
                weighted_sum_history.append(round(weighted_sum, 6))
                weighted_history.append(np.round(weights, 4))
                    
            except ValueError as e:
                print(f"Invalid row: {row}")
                print(f"Error: {e}")


# Read the initial data and define the search domain
orthogonal_data = read_orgothonal_data(input_data_file, weighted_data_file)
Y_data = orthogonal_data.iloc[:, NUM_VAR:NUM_VAR+NUM_OBJ].values    
Y_norm = normalize_objectives(Y_data)
# Initialize the Bayesian optimizer
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=all_bounds,
    verbose=0  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
)
load_data_and_register(optimizer, weighted_data_file, Y_norm)

# Start the optimization process
start_datetime = datetime.now()
time_monitor = []
for iteration in range(1, args.niter + 1):    
    start_time = time.process_time()  # CPU time for the current iteration

    new_weights = generate_random_weights(NUM_OBJ)
    weighted_history.append(new_weights)    
    
    optimizer.maximize(
        init_points=0,
        n_iter=1,
    )

    cpu_time_used = time.process_time() - start_time
    sys_time_taken = datetime.now() - start_datetime
    time_monitor.append([iteration, f'{cpu_time_used:.4f}', f'{sys_time_taken.total_seconds():.4f}'])

print(f"Total computation time {sys_time_taken.total_seconds():.4f} seconds.")


# Export the timing log
with open(os.path.join(f'{args.outfile}', f'parego-optTimes_{formula}.csv'), 'w', newline='') as times_file:
    writer = csv.writer(times_file)
    writer.writerow(['Iteration', 'cpu_time_iter', 'sys_time'])
    for row in time_monitor:
        writer.writerow(row)
