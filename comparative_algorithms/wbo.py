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
weighted_data_file = f'./input_data/WSBayesOpt_{formula}.csv'
X_title = [f'X{i}' for i in range(1, NUM_VAR+1)]
Y_title = [f'F{i}' for i in range(1, NUM_OBJ+1)]
# Domain of each formulation variable
all_bounds = {f'X{i+1}': (np.min(row), np.max(row))
              for i, row in enumerate(X_ortho_values)}
weights = TARGET
stop_flag = False
params_history = []
results_history = []
result_file = os.path.join(f'{args.outfile}', f'wbo_{formula}.csv')
with open(result_file, mode='w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    writer.writerow(['Iteration'] + X_title + Y_title + ['Y', 'Fitness', 'Nc'])


def read_orthogonal_data(infile, outfile):
    """
    Read the original orthogonal input data, compute the weighted objective value, 
    remove duplicate formulations, and export the processed data.
    """
    try:
        data = pd.read_csv(infile)
    except FileNotFoundError:
        print(f"File not found: {infile}")
    except ValueError as e:
        print(f"Error: {e}")

    x_columns = data.iloc[:, :NUM_VAR]
    y_columns = data.iloc[:, NUM_VAR:NUM_VAR+NUM_OBJ]
    data['Y'] = np.sum(np.sqrt(np.square((y_columns.values - TARGET) / weights)), axis=1) / NUM_OBJ

    data = pd.concat([x_columns, data['Y']], axis=1)
    duplicates = data.iloc[:, :-1].duplicated()
    data = data[~duplicates]
    data.to_csv(outfile, index=False, header=False,  sep=',')


def calculate_result(param_set):
    """
    Evaluate the response values and the aggregated weighted objective value
    for a given formulation.
    """

    x = np.array([v for _, v in param_set])
    x = np.around(x, decimals=2)
    yi = calculate_performance(x, NUM_OBJ, formula=formula)

    result = np.sum(np.sqrt(np.square((yi - TARGET) / weights))) / NUM_OBJ
    
    return np.round(yi, 4), np.round(result, 4)

        
def black_box_function(**kwargs):
    global params_history, results_history
    param_set = sorted(kwargs.items())
    param = tuple(round(value,4) for name,value in param_set)
    try:
        return results_history[params_history.index(param)]
    except ValueError:
        yi, result = calculate_result(param_set)
        params_history.append(param)
        results_history.append(result)
        # Calaulate the fitness value and the number of compliant objectives
        fitness = sqrt(sum(((i - j)/TARGET[index])**2/NUM_OBJ if i < j else 0 
                        for index, (i, j) in enumerate(zip(yi, TARGET))))
        Nc = sum(i > j for i, j in zip(yi, TARGET))
        
        # Write the evaluated result to the output file
        with open(result_file, mode='a', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            row = [iteration] + list(param) + list(yi) + [round(result,4)] + [round(fitness,4)] + [Nc]
            writer.writerow(row)
            if Nc >= NUM_OBJ:
                print("A feasible solution compliant with all objectives has been found:\n", row)
        
        return result


def load_data_and_register(optimizer, X, filename):
    """
    Load historical data from file and register them into the Bayesian optimizer.
    """
    with open(filename, mode='r') as infile:
        reader = csv.reader(infile, delimiter=',')        
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            try:
                row = [x.strip() for x in row if x.strip()]
                row_values = [float(row[i]) for i in range(len(X))]
                params = dict(zip(X, row_values))
                target = float(row[-1])

                optimizer.register(params=params, target=target)
                params_history.append(tuple(row_values))
                results_history.append(target)

            except ValueError as e:
                print(f"Invalid row: {row}")
                print(f"Error: {e}")


# Read the initial data and define the search domain
read_orthogonal_data(input_data_file, weighted_data_file)
# Initialize the Bayesian optimizer
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=all_bounds,
    verbose=0  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent    
)
load_data_and_register(optimizer, X_title, weighted_data_file)

# Start the optimization process
start_datetime = datetime.now()
time_monitor = []
for iteration in range(1, args.niter + 1):    
    start_time = time.process_time()  # CPU time for the current iteration
    
    optimizer.maximize(
        init_points=0,
        n_iter=1,
    )
    
    cpu_time_used = time.process_time() - start_time 
    sys_time_taken = datetime.now() - start_datetime   
    time_monitor.append([iteration, f'{cpu_time_used:.4f}', f'{sys_time_taken.total_seconds():.4f}'])

print(f"Total computation time {sys_time_taken.total_seconds():.4f} seconds.")


# Export the timing log
with open(os.path.join(f'{args.outfile}', f'wbo-optTimes_{formula}.csv'), 'w', newline='') as times_file:
    writer = csv.writer(times_file)
    writer.writerow(['Iteration', 'cpu_time_iter', 'sys_time'])
    for row in time_monitor:
        writer.writerow(row)
