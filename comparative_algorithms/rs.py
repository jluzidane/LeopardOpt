import os
import csv
import numpy as np
from math import sqrt
import random
import argparse
import importlib
from functions.calc_functions import calculate_performance
from datetime import datetime


# Command-line argument interface
parser = argparse.ArgumentParser()
parser.add_argument("--randseed", type=int, help="Random seed value")
parser.add_argument("--outfile", type=str, default="results", help="Directory for saving output files")
parser.add_argument("--formula", type=str, default="rosenbrock2", help="Benchmark function for optimization")
parser.add_argument("--niter", type=int, default=100, help="Max iteration steps")
parser.add_argument("--interval", type=int, default=1, help="Save intervals")
args = parser.parse_args()


def load_formula_config(formula_name):
    """Load the parameter configuration of the specified benchmark function."""
    try:
        config_module = importlib.import_module(f"functions.{formula_name}")
        return config_module.TARGET, config_module.X_ortho_values
    except ImportError:
        raise ValueError(f"Configuration for '{formula_name}' could not be found.")

# Configuration settings
formula = args.formula
TARGET, X_ortho_values = load_formula_config(formula)
NUM_VAR = X_ortho_values.shape[0]
NUM_OBJ = len(TARGET)
print(f"Optimization started: FUNC = '{formula}', NUM_VAR = {NUM_VAR}, NUM_OBJ = {NUM_OBJ}")

X_title = [f'X{i}' for i in range(1, NUM_VAR+1)]
Y_title = [f'F{i}' for i in range(1, NUM_OBJ+1)]
# Domain of each formulation variable
all_bounds = {f'x{i+1}': (np.min(row), np.max(row))
              for i, row in enumerate(X_ortho_values)}
weights = TARGET
stop_flag = False
    
   
def calculate_result(param_set):

    x = np.around(param_set, decimals=4)
    result = calculate_performance(x, formula=formula)
    
    return result

    
def random_search_formula():

    X = np.array([round(random.uniform(*bounds), 4) 
                  for bounds in all_bounds.values()])
    
    return X


# Start the optimization process
start_time = datetime.now()
csv_file = os.path.join(f'{args.outfile}', f'rs_{formula}.csv')
with open(csv_file, mode='w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    writer.writerow(['Generation'] + X_title + Y_title + ['Fitness', 'Nc'])

min_fit = float('inf')
best_param_set = None
best_yi = None
best_Nc = 0
random.seed(args.randseed if args.randseed else None)


for iteration in range(1, args.niter+1):
    param_set = random_search_formula()

    yi = calculate_result(param_set)
    fitness = sqrt(sum(((i - j)/weights[index])**2 if i < j else 0 
                    for index, (i, j) in enumerate(zip(yi, TARGET))))
    fitness = round(fitness, 4)
    Nc = sum(i > j for i, j in zip(yi, TARGET))  # Number of compliant objectives

    if fitness < min_fit:
        min_fit = fitness
        best_param_set = param_set
        best_yi = np.round(yi, 4)
        best_Nc = Nc
        
    # Store the best solution of the current generation
    if iteration % args.interval == 0  or  min_fit == 0.:        

        with open(csv_file, mode='a', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            row = [iteration] + list(best_param_set) + list(best_yi) + [min_fit] + [best_Nc]
            writer.writerow(row)

        # Terminate the optimization once a feasible target solution is found
        if min_fit == 0.:
            print(f"Conditions met at iteration: {iteration}. Exiting loop.")
            print(best_param_set)
            print(best_yi)

            with open(os.path.join(f'{args.outfile}', f'rs-Target_{formula}_components.txt'), mode="a") as file1, \
                 open(os.path.join(f'{args.outfile}', f'rs-Target_{formula}_properties.txt'), mode="a") as file2:
                print(iteration, best_param_set, file=file1)
                print(iteration, best_yi, file=file2)
            break

        min_fit = float('inf')
        

end_time = datetime.now()
elapsed = end_time - start_time
print(f"Total computation time: {elapsed.total_seconds():.4f} seconds.")
