# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:20:36 2025

@author: 91278

Leopard Project/
├── utils/
├──── data_io.py                  # Module 1: data I/O and preprocessing
├──── dimension_reduction.py      # Module 2: feature importance analysis and dimensionality reduction
├──── correlation_analysis.py     # Module 3: correlation analysis
├──── perform_evaluation.py       # Module 4: performance evaluation and fitness assessment
├──── bayesian_optimization.py    # Module 5: Bayesian optimization
├──── bayes_selected_strategy.py  # Module 6: candidate generation from Bayesian optimized solutions
├──── balance_strategy.py         # Module 7: balanced candidate selection strategy
├── config/                       # Configuration file (.json)
├── functions/                    # Benchmark functions
├── input_data/                   # Initial orthogonal design dataset
├── results/                      # Results of LeopardOpt
└── leopard.py                    # Main workflow controller of LeopardOpt
"""


# Pass the formula argument and map it to the active config file.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--formula", type=str, default="rosenbrock2", help="Functions for optimization")
args = parser.parse_args()
FORMULA = args.formula
import shutil
shutil.copy2(f"./config/config_{FORMULA}.json", "./config.json")


import os
import json
import numpy as np
import pandas as pd
import functools
import time
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")
from utils.data_io import load_data, create_exp_dir, save_data, plot_data
from utils.dimension_reduction import reduce_dimensionality
from utils.correlation_analysis import analyze_correlations
from utils.perform_evaluation import calculate_performance, evaluate_fitness
from utils.bayesian_optimization import bayesian_optimize
from utils.balance_strategy import select_composite_solution
import importlib.util


def timer(func):
    """Timing decorator that reports total runtime (hh:mm:ss.ms)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = timedelta(seconds=time.perf_counter() - start_time)
        print(f"Total runtime: {elapsed}\n")
        return result
    return wrapper


def load_config(config_path):
    """Load the global configuration and formula-specific settings."""
    with open(config_path, "r") as f:
        config = json.load(f)
    formula = config["settings"]["formula"]
    try:
        func_config = importlib.import_module(f"functions.{formula}")
        return config, func_config
    except ImportError:
        functions_dir = "./functions"
        valid_formulas = [filename[:-3] for filename in os.listdir(functions_dir)
            if filename.endswith(".py") and not filename.startswith("__")]
        raise ValueError(f"Configuration for {formula} was not found. Valid options are:\n{valid_formulas}\n")

 

@timer
def main(config, func_config):

    formula = config["settings"]["formula"]
    X0_input = func_config.X_input
    Y0_input = func_config.Y_input
    X_dim, Y_dim = func_config.X_dim, func_config.Y_dim
    X_bounds = {f"x{i+1}": [row[0], row[-1]] for i, row in enumerate(func_config.X_ortho_values)}
    IM_estimator = config["settings"]["IM_estimator"]
    Cor_estimator = config["settings"]["cor_estimator"]
    save_dir = config["file_paths"]["output_dir"]
    exp_dir = create_exp_dir(formula, save_dir)

    
    # 1) Load initial orthogonal experimental data.
    input_tab = config["file_paths"]["input_file"]
    data = pd.read_csv(input_tab)
    
    # 2) Random-forest-based dimensionality reduction.
    IM_order_filepath = os.path.join(exp_dir, config["file_paths"]["importance_order"])
    IM_matrix_filepath = os.path.join(exp_dir, config["file_paths"]["importance_matrix"])
    IM_matrix, IM_factors, RIF_matrix = reduce_dimensionality(
        data, 
        IM_order_filepath, 
        IM_matrix_filepath,
        config["thresholds"]["IM_threshold"], 
        X_dim, 
        Y_dim,
        mode=IM_estimator  # 'RandForest' or 'XGBoost'
    )

    # 3) Correlation analysis.
    Cor_matrix_filepath = os.path.join(exp_dir, config["file_paths"]["correlation_matrix"])
    Cor_matrix, Sign_matrix = analyze_correlations(
        data, 
        Cor_matrix_filepath, 
        X_dim, 
        Y_dim,
        mode=Cor_estimator  # 'spearman' or 'pearson'
    )
    
    # 4) Fitness evaluation and iterative update.
    X_current = np.array(func_config.RECIPE)  # Current formulation
    Y_target = np.array(func_config.TARGET)[:Y_dim]  # Target performance
    y_opt = Y0_input
    epoch = 0
    # Number of consecutive non-improving iterations
    cumulative_epoch = 0   # Used to trigger extremum-based balancing.
    cumulative_epoch1 = 0  # Used to restore the best historical solution.
    fitness_list = []
    Nc_list = []
    print ("epoch\t fitness\t #N compliant performances")
    
    while True:
        epoch_trace_path = os.path.join(exp_dir, config["file_paths"]["epoch_trace"])
        Fit, dist_err, Nc, Y_current, y_opt = evaluate_fitness(X_current, epoch, formula, epoch_trace=epoch_trace_path)

        fitness = Fit[0]
        if not fitness_list:
            fitness_min = fitness
            X_optimal = X_current.copy()
            Y_optimal = Y_current.copy()
            dist_optimal = dist_err
            Nc_optimal = Nc
        elif fitness < min(fitness_list):
            fitness_min = fitness
            X_optimal = X_current.copy()
            Y_optimal = Y_current.copy()
            dist_optimal = dist_err
            Nc_optimal = Nc
        elif Nc <= Nc_optimal:
            cumulative_epoch1 += 1
        fitness_list.append(fitness)
        Nc_list.append(Nc)
        print (f" {epoch} \t\t{fitness:.6f} \t\t{Nc}")
      
        if fitness <= config["thresholds"]["fitness_threshold"]:
            print(f"\nTarget formulation identified after {epoch} function evaluations.")            
            with open(os.path.join(exp_dir, "flag.txt"), "w") as f:
                X_current_str = " ".join([f"{x:.4f}" for x in X_current])
                f.write(f"{epoch} {X_current_str}\n")   # Success flag: iteration index and feasible formulation.
            break

        if epoch >= config["settings"]["stop_epochs"]:
            print(f"\nMaximum iteration limit reached {epoch}; optimization stopped.")
            break

        # If fitness stagnates, optionally restore the best historical solution.
        if cumulative_epoch1 >= config["thresholds"]["step_threshold1"]:
            # print("---------- callback ----------")
            if np.all(np.round(X_current, 2) == np.round(X_optimal, 2)):
                X_current = np.array(func_config.RECIPE)
                cumulative_epoch = config["thresholds"]["step_threshold2"]
            else:
                X_current = X_optimal
            cumulative_epoch1 = 0
            
        # 5) Bayesian recommendation for each objective.
        dX_recommended_array = np.zeros((Y_dim, X_dim))
        for y_input in y_opt:
            x_input = IM_factors[y_input]
            data_dimreduce = load_data(data, x_input, [y_input])

            dX_recommended = bayesian_optimize(                
                data_dimreduce,      # Reduced dataset for current objective yi and key factors [xj, xk, ...].
                X_current.copy(),    # Current formulation.
                {y_input: x_input},  # Mapping: current objective to key influencing variables.
                RIF_matrix,
                Cor_matrix,
                fitness,
                config["bayesian_optimization"]["mode"]
            )
            yi = int(y_input[1:])-1
            dX_recommended_array[yi] = dX_recommended
                    
        # 6) Composite balancing strategy.
        dX_current, cumulative_epoch = select_composite_solution(
            cumulative_epoch,
            fitness,
            fitness_min,
            y_opt,
            X_current.copy(),
            dX_recommended_array,
            IM_matrix,
            RIF_matrix*Sign_matrix,  # Combine importance and correlation-sign information for update direction.
            dist_err
        )

        X_current = np.round(X_current + dX_current, 4)

        epoch += 1

    # Final reporting and visualization.
    print(f"\nOptimal formulation: {np.round(X_optimal,2)}"
          f"\nFinal performance: {np.round(Y_optimal,2)}"
          f"\nFinal distance to target: {np.round(dist_optimal,2)}"
          f"\n# compliant performances: {Nc_optimal}"
          f"\nFitness: {fitness_min:.4f}")
    save_data(X_optimal, os.path.join(exp_dir, config["file_paths"]["optimal_X"]))
    save_data(Y_optimal, os.path.join(exp_dir, config["file_paths"]["optimal_Y"]))
    plot_data(Nc_list, exp_dir, "#N compliant performances")
    plot_data(fitness_list, exp_dir, "Fitness")
    for file in os.listdir(exp_dir):
        print(f"  - {file}")


if __name__ == "__main__":
    config, func_config = load_config("config.json")
    
    formula = config["settings"]["formula"]
    NUM_OBJ = func_config.Y_dim
    NUM_VAR = func_config.X_dim
    print(f"Optimization started: FUN = '{formula}', NUM_VAR = {NUM_VAR}, NUM_OBJ = {NUM_OBJ}")
    main(config, func_config)
