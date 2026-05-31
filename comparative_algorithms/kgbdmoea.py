import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from math import sqrt
import os
import csv
import argparse
import importlib
from functions.calc_functions import calculate_performance


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
 

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(
            n_var=NUM_VAR,
            n_obj=NUM_OBJ,
            n_ieq_constr=0,
            xl=X_ortho_values.T[0],
            xu=X_ortho_values.T[-1],
        )

    def _evaluate(self, x, out, *args, **kwargs):
        
        out["F"] = TARGET-calculate_performance(x, formula=formula)


# Define a callback class to capture data at each generation
class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.csv_file = csv_file
        self.csv_writer = csv_writer
        self.save_interval = args.interval
        self.best_solutions = []
        self.found_solution = False
        self.solution_X = None
        self.solution_F = None
        

    def notify(self, algorithm):
        n_gen = algorithm.n_gen

        opt = algorithm.opt
        if opt is None:
            return
        X = opt.get("X")
        F = opt.get("F")
        if F is None or X is None:
            return
        
        # Identify the solution with the minimum fitness value
        min_fit = float('inf')
        best_index = -1
        num_obj = np.sum(F<0, axis=1).tolist()  # Number of compliant objectives
        for i in range(len(F)):
            fitness = sqrt(sum((k / j) ** 2 / len(TARGET) if k > 0 else 0 
                            for (k, j) in zip(F[i], TARGET)))

            if fitness < min_fit:
                min_fit = fitness
                best_index = i

            # Terminate the optimization once a feasible target solution is found
            if np.all(F[i] <= 0) and not self.found_solution:
                print("Target solution found in generation", algorithm.n_gen)
                self.found_solution = True
                self.solution_X = X[i]
                self.solution_F = F[i]                
                algorithm.termination.force_termination = True
                break

        if best_index >= 0:
            # Store the best Pareto solution of the current generation
            self.best_solutions.append({
                "n_gen": algorithm.n_gen,
                "X": np.round(X[best_index], 4),
                "F": np.round(-F[best_index] + TARGET, 4),
                "Fitness": round(min_fit, 4),
                "Nc": num_obj[best_index],      
            })
            
            if n_gen % self.save_interval == 0 or self.found_solution == True:
                if self.best_solutions:
                    # Select the solution with the minimum fitness value among the cached solutions
                    best_solution = min(
                        self.best_solutions, 
                        key=lambda x: x["Fitness"]
                    )
                
                n_gen_best = best_solution["n_gen"]
                best_X = best_solution["X"]
                best_Y = best_solution["F"]
                min_fit = best_solution["Fitness"]
                best_n = best_solution["Nc"]
                
                params = [n_gen_best] + list(best_X) + list(best_Y) + [min_fit, best_n]
                self.csv_writer.writerow(params)
                self.csv_file.flush()
                self.best_solutions = []


from pymoo.algorithms.moo.kgb import KGB
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.mutation.pm import PM
from datetime import datetime
import pandas as pd


# Load the orthogonal input data as the initial population
ortho_data = pd.read_csv(f'input_data/input_{formula}.csv')
ortho_pop = ortho_data.iloc[:, :NUM_VAR].values

algorithm = KGB(
    pop_size = len(ortho_pop),
    n_offsprings = args.popsize,
    sampling = ortho_pop,
    mutation = PM(prob=0.5, eta=20),
)

problem = MyProblem()
termination = get_termination("n_gen", args.niter)


# Start the optimization process
start_time = datetime.now()
csv_file = open(os.path.join(f'{args.outfile}', f'kgbdmoea_{formula}.csv'), mode='w', newline='')
csv_writer = csv.writer(csv_file)
X_title = [f"X{i+1}" for i in range(NUM_VAR)]
Y_title = [f"F{i+1}" for i in range(NUM_OBJ)]
csv_writer.writerow(['Generation'] + X_title + Y_title + ['Fitness', 'Nc'])

callback = MyCallback()
res = minimize(problem,
               algorithm,
               termination,
               seed=args.randseed,
               save_history=False,
               # verbose=True,
               callback=callback
              )
     
n_gen = res.algorithm.n_gen
# Check whether a feasible target solution has been found
if callback.found_solution:
    X = callback.solution_X
    F = callback.solution_F

    coe = 4
    with open(os.path.join(f'{args.outfile}', f'kgbdmoea-Target_{formula}_components.txt'), mode="a") as file1, \
         open(os.path.join(f'{args.outfile}', f'kgbdmoea-Target_{formula}_properties.txt'), mode="a") as file2:
        print(n_gen, np.round(X, coe), file=file1)
        print(n_gen, np.round(F, coe), file=file2)
    
    print(f"Conditions met at generation: {n_gen}. Exiting loop.")
    print("X: ", np.round(X, coe))
    print("Y: ", np.round(F, coe))

else:
    # Use the final population if no feasible target solution is identified
    X = res.X
    F = res.F

csv_file.close()


end_time = datetime.now()
elapsed = end_time - start_time
print(f"Total computation time: {elapsed.total_seconds():.4f} seconds.")
