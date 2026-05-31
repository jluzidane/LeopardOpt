import numpy as np
import pandas as pd
from platypus import SMPSO, Problem, Real, TerminationCondition, ParetoDominance
from math import sqrt
import os
import csv
import argparse
import importlib
from functions.calc_functions import calculate_performance
from datetime import datetime
import time


# Command-line argument interface
parser = argparse.ArgumentParser()
parser.add_argument("--randseed", type=int, help="Random seed value")
parser.add_argument("--outfile", type=str, default="results", help="Directory for saving output files")
parser.add_argument("--formula", type=str, default="rosenbrock2", help="Benchmark function for optimization")
parser.add_argument("--popsize", type=int, default=200, help="Population size")
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

swarm_size = args.popsize
max_iterations = args.niter


class MyProblem(Problem):

    def __init__(self):
        super().__init__(
            nvars=NUM_VAR,
            nobjs=NUM_OBJ,
            nconstrs=0,                 
        )    
        self.types[:] = [Real(min(X_ortho_values[i]), 
                              max(X_ortho_values[i]))for i in range(NUM_VAR)]

    def evaluate(self, solution):
        x = solution.variables[:]
        solution.objectives[:] = TARGET - calculate_performance(x, formula=formula)


# Define a callback class to capture data at each generation
data = []
save_interval = args.interval
temp_best_solutions = []
found_solution = False
opt_X = None
opt_F = None


def MyCallback(algorithm):
    global data, save_interval, temp_best_solutions, found_solution, opt_X, opt_F
    
    n_gen = algorithm.nfe // swarm_size
    if len(algorithm.leaders) == 0:
        return
    
    solutions = algorithm.leaders
    X = np.array([sol.variables for sol in solutions])
    F = np.array([sol.objectives for sol in solutions])

    # Identify the solution with the minimum fitness value
    distances = np.linalg.norm(F, axis=1)
    best_index = np.argmin(distances)
    num_obj = (F < 0).sum(axis=1)  # Number of compliant objectives
    fitness = sqrt(sum((k / j) ** 2 / len(TARGET) if k > 0 else 0 
                    for (k, j) in zip(F[best_index], TARGET)))

    current_best = ({
        "generation": n_gen,
        "X": np.round(X[best_index], 4),
        "F": np.round(-F[best_index] + TARGET, 4),
        "Fitness": round(fitness, 4),
        "Nc": num_obj[best_index],
    })
    temp_best_solutions.append(current_best)

    # Terminate the optimization once a feasible target solution is found
    if np.all(F[best_index] <= 0):
        print("Target solution found in generation", n_gen)
        print(f"X: {current_best['X']}")
        print(f"F: {current_best['F']}")
        found_solution = True
        opt_X = current_best['X']
        opt_F = current_best['F']

        data.append([
            current_best["generation"],
            *current_best["X"],
            *current_best["F"],
            current_best["Fitness"],
            current_best["Nc"]
        ])
        return
    
    # Store the best solution of the current generation
    if n_gen % save_interval == 0:
        if temp_best_solutions:
            best_solution = min(
               temp_best_solutions, 
                key=lambda x: x["Fitness"]
            )

            data.append([
                best_solution["generation"],
                *best_solution["X"],
                *best_solution["F"],
                best_solution["Fitness"],
                best_solution["Nc"]
            ])

            temp_best_solutions = []


class CustomTerminationCondition(TerminationCondition):
    """Terminating criterion based on objective function values."""
    def __init__(self, target=0.0):
        super().__init__()
        self.target = target
        self.start_nfe = 0      
        
    def initialize(self, algorithm):
        self.start_nfe = algorithm.nfe
        self.prev_solutions = None
        self.start_time = time.time()
    
    def shouldTerminate(self, algorithm):
        if found_solution:
            print(f"Early termination at NFE: {algorithm.nfe}")
            return True
        
        # Combined termination criterion: maximum evaluations and maximum runtime
        return (algorithm.nfe - self.start_nfe >= swarm_size*max_iterations or
                time.time() - self.start_time >= 600000)


termination = CustomTerminationCondition(target=np.zeros((NUM_OBJ)))
problem = MyProblem()

algorithm = SMPSO(
    problem=problem,
    swarm_size=swarm_size,
    leader_size=100,
    mutation_probability=0.1,
    mutation_perturbation=0.5,
    max_iterations=max_iterations
)


# Load the orthogonal input data as the initial population
ortho_data = pd.read_csv(f'input_data/input_{formula}.csv')
ortho_pop = ortho_data.iloc[:, :NUM_VAR].values

algorithm.initialize()
for i in range(swarm_size):
    algorithm.particles[i].variables = ortho_pop[i % len(ortho_pop)]
    problem.evaluate(algorithm.particles[i])


# Start the optimization process
start_time = datetime.now()
csv_file = open(os.path.join(f'{args.outfile}', f'smpso_{formula}.csv'), mode='w', newline='')
csv_writer = csv.writer(csv_file)
X_title = [f"X{i+1}" for i in range(NUM_VAR)]
Y_title = [f"F{i+1}" for i in range(NUM_OBJ)]
csv_writer.writerow(['Generation'] + X_title + Y_title + ['Fitness', 'Nc'])

found_solution = False
opt_X = None
opt_F = None
data = []
temp_best_solutions = []


algorithm.run(
    condition=termination,
    callback=MyCallback
)
  

solutions = algorithm.leaders
n_gen = algorithm.nfe // swarm_size
for row in data:
    csv_writer.writerow(row)
csv_file.close()

# Check whether a feasible target solution has been found
if found_solution:
    X = opt_X
    F = opt_F

    coe = 4
    with open(os.path.join(f'{args.outfile}', f'smpso-Target_{formula}_components.txt'), mode="a") as file1, \
         open(os.path.join(f'{args.outfile}', f'smpso-Target_{formula}_properties.txt'), mode="a") as file2:
        print(n_gen, np.round(X, coe), file=file1)
        print(n_gen, np.round(F, coe), file=file2)
    
    print(f"Conditions met at generation: {n_gen}. Exiting loop.")
    print("X: ", np.round(X, coe))
    print("Y: ", np.round(F, coe))


end_time = datetime.now()
elapsed = end_time - start_time
print(f"Total computation time: {elapsed.total_seconds():.4f} seconds.")
