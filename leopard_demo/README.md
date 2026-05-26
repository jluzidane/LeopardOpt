# Demo: Out-of-the-Box Experience
This folder contains ready-to-run scripts to reproduce the core results presented in the paper.
Contents
- leopard.py: The proposed LeopardOpt algorithm.
- functions/: Scripts defining the 20 benchmark functions for testing.
- input_data/: Initial orthogonal experimental data for the 20 benchmark functions (.csv format).
- utils/: Subroutines called by the main program leopardopt.py.
- config/: Hyperparameter settings for running each benchmark function (JSON format).
# How to Run
1. Ensure you are in `leopard_demo/` directory.
2. Simply run the algorithm file you want to test: python leopard.py
3. The terminal will output the optimization time, iterations, optimized formulations, and their performance metrics.
4. The generated results will be automatically saved in the results/ folder, with the outputs for each benchmark function stored in a separate subfolder named after the function.
How to Switch Functions
All algorithms can be configured via command-line arguments. To change the target optimization function, run the desired Python script in the terminal with the `--formula` argument, for example:
`python leopard.py --formula rosenbrock`
or directly open the .py file to modify it manually.
