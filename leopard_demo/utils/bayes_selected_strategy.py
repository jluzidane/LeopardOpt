import numpy as np
import json
import importlib

# Load configuration.
config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f)
threshold1 = config["thresholds"]["fitness_threshold1"]
threshold2 = config["thresholds"]["fitness_threshold2"]
formula = config["settings"]["formula"]
try:
    func_config = importlib.import_module(f"functions.{formula}")
except ImportError:
    raise ValueError(
        f"Configuration for '{formula}' was not found. "
        f"Please check the corresponding function file."
    )
targets = np.array(func_config.TARGET)


def sorting_solutions(sorted_X, Dist, target, mode='Dy'):
    """Sort candidate updates according to the specified selection criterion."""
    if mode == 'Dy':
        above_target = Dist > target        
        indices_above = np.where(above_target)[0]
        sorted_above = indices_above[np.argsort(Dist[indices_above])]
        indices_below = np.where(~above_target)[0]
        sorted_below = indices_below[np.argsort(-Dist[indices_below])]
        sorted_order = np.concatenate([sorted_above, sorted_below])
    else:
        sign_matches = np.sum(np.sign(sorted_X) == np.sign(target), axis=1)
        sorted_order = np.lexsort((Dist, -sign_matches))
    return sorted_X[sorted_order]


def select_solution(sorted_X, p=0.8):
    """
    Select one candidate solution from a sorted list using a geometric prior.
        sorted_X: Sorted candidate solutions.
        p: Geometric selection probability (0<p<1). A larger value assigns higher
        probability to top-ranked candidates.
    """
    n = len(sorted_X)
    selected_idx = np.random.geometric(p) - 1
    if selected_idx > (n-1):
        selected_idx = 0
    
    return sorted_X[selected_idx]


def pick_optimal_solution(
        X_bayes,  # shape: (n_iter, num_key_x)
        Y_bayes, 
        X_0,  # shape: (n_iter, num_key_x)
        distX,  # list: length=n_iter
        distY,  # list: length=n_iter
        xy_input,  # list: {'y1': ['x1','x5','x6','x7']}
        IM_matrix,  # shape: (num_key_x,)
        Cor_matrix,  # shape: (num_key_x,)
        mode='Dy'
    ):
    """
    Select a recommended update from Bayesian optimization candidates.
        Selection criteria:
        1. First, check whether the X-variable updates follow the expected trend
           according to the variable-importance ranking. Candidate solutions for which
           the most important variable changes in the expected direction are prioritized.
        2. For mode='Dy', prioritize candidate solutions whose predicted response
           values exceed the target value.
           For mode='Dx', prioritize candidate solutions with the largest number of
           important variables whose update directions are consistent with the expected
           trend.
        3. A candidate solution is then selected with probability p, giving higher
           preference to solutions that are closer to the initial solution in design
           space in terms of Dy or Dx.
    """
    y_input, x_input = list(xy_input.keys())[0], list(xy_input.values())[0]
    DeltaX = X_bayes - X_0  # shape: (n_iter, num_key_x)

    # Prioritize candidates whose most important variables follow the expected update directions.
    key_indices = np.where(IM_matrix == 1)[0]
    sign_matches = np.all(
        np.sign(DeltaX[:, key_indices]) == np.sign(Cor_matrix[key_indices]),
        axis=1
    )
    sorted_indices = np.argsort(~sign_matches)
    sorted_DeltaX = DeltaX[sorted_indices]

    satisfied_DeltaX = sorted_DeltaX[:sum(sign_matches)]
    unsatisfied_DeltaX = sorted_DeltaX[sum(sign_matches):]

    if mode == 'Dy':
        target = targets[int(y_input[1:])-1]
        sorted_DistY = np.array(Y_bayes)[sorted_indices]
        satisfied_DistY = sorted_DistY[:sum(sign_matches)]
        unsatisfied_DistY = sorted_DistY[sum(sign_matches):]

        satisfied_DeltaX = sorting_solutions(satisfied_DeltaX, satisfied_DistY, target, mode='Dy')
        unsatisfied_DeltaX = sorting_solutions(unsatisfied_DeltaX, unsatisfied_DistY, target, mode='Dy')

        sorted_DeltaX = np.concatenate([satisfied_DeltaX, unsatisfied_DeltaX])

    else:  # mode == 'Dx'
        sorted_DistX = np.array(distX)[sorted_indices]
        satisfied_DistX = sorted_DistX[:sum(sign_matches)]
        unsatisfied_DistX = sorted_DistX[sum(sign_matches):]

        satisfied_DeltaX = sorting_solutions(satisfied_DeltaX, satisfied_DistX, Cor_matrix, mode='Dx')
        unsatisfied_DeltaX = sorting_solutions(unsatisfied_DeltaX, unsatisfied_DistX, Cor_matrix, mode='Dx')

        sorted_DeltaX = np.concatenate([satisfied_DeltaX, unsatisfied_DeltaX])
    
    # Accept the optimal solution in the sorted list with probability p.
    dX_selected = select_solution(sorted_DeltaX, p=0.85)
    # Convert the reduced-dimensional update into a full-dimensional update.
    dX_bayes = np.zeros(func_config.X_dim)
    for (k, v) in zip(x_input, dX_selected):
        index = int(k[1:])-1
        dX_bayes[index] = v

    return dX_bayes


def extreme_value_strategy(DeltaX, RIM_matrix):
    """
    Construct a candidate update using the extremum-value strategy.

    For positively correlated variables, larger positive updates are preferred.
    For negatively correlated variables, larger negative updates are preferred.
    Variables with zero signed importance are assigned zero updates.

    Parameters
    ----------
    DeltaX : numpy.ndarray
        Candidate update matrix with shape ``(n_iter, num_key_x)``.
    RIM_matrix : numpy.ndarray
        Signed importance vector with shape ``(num_key_x,)``.

    Returns
    -------
    dX_bayes : numpy.ndarray
        Combined reduced-dimensional update.
    """
    dim_x = RIM_matrix.shape[0]
    dX_bayes = np.zeros(dim_x)
    
    for var_idx in range(dim_x):
        rim_val = RIM_matrix[var_idx]
        var_values = DeltaX[:, var_idx]
        
        if rim_val > 0:
            sorted_var = np.sort(var_values)[::-1]
            dX_bayes[var_idx] = max(select_solution(sorted_var, p=0.8), 0)
        elif rim_val < 0:
            sorted_var = np.sort(var_values)        
            dX_bayes[var_idx] = min(select_solution(sorted_var, p=0.8), 0)
        else:
            dX_bayes[var_idx] = 0.0

    return dX_bayes


def boudary_value_strategy(DeltaX, RIM_matrix):
    """
    Construct a candidate update using the boundary-extreme-value strategy.
    
    For positively correlated variables, the smallest positive update is preferred. 
    For negatively correlated variables, the largest negative update is preferred. 
    This strategy therefore selects the update closest to the initial solution 
    while preserving the expected update direction.

    Parameters
    ----------
    DeltaX : numpy.ndarray
        Candidate update matrix with shape ``(n_iter, num_key_x)``.
    RIM_matrix : numpy.ndarray
        Signed importance vector with shape ``(num_key_x,)``.
    
    Returns
    -------
    dX_bayes : numpy.ndarray
    """
    dim_x = RIM_matrix.shape[0]
    dX_bayes = np.zeros(dim_x)
    
    for var_idx in range(dim_x):
        rim_val = RIM_matrix[var_idx]
        var_values = DeltaX[:, var_idx]
        
        if rim_val > 0:
            positive_vals = var_values[var_values > 0]
            if len(positive_vals) > 0:
                sorted_var = np.sort(positive_vals)
                dX_bayes[var_idx] = select_solution(sorted_var, p=0.8)
            else:
                dX_bayes[var_idx] = 0.0
        elif rim_val < 0:
            negative_vals = var_values[var_values < 0]
            if len(negative_vals) > 0:
                sorted_var = np.sort(negative_vals)[::-1]
                dX_bayes[var_idx] = select_solution(sorted_var, p=0.8)
            else:
                dX_bayes[var_idx] = 0.0
        else:
            dX_bayes[var_idx] = 0.0

    return dX_bayes


def pick_optimal_solution_ex(
        X_bayes,
        X_0,
        xy_input,
        IM_matrix,
        Cor_matrix,
        fitness
    ):
    """
    Select a recommended update using signed-importance-based strategies.

    Candidate updates are combined according to the signed importance matrix.
    When the fitness value is large, the extremum-value strategy is used to
    encourage larger corrective updates. Otherwise, the boundary-value strategy
    is used to obtain a conservative update that is closer to the initial
    solution.
    """
    y_input, x_input = list(xy_input.keys())[0], list(xy_input.values())[0]
    DeltaX = X_bayes - X_0  # X_bayes shape: (dim, iter)

    RIM_matrix = IM_matrix * np.sign(Cor_matrix)

    if fitness > threshold2:
        dX_selected = extreme_value_strategy(DeltaX, RIM_matrix)
    else:
        dX_selected = boudary_value_strategy(DeltaX, RIM_matrix)
    
    dX_bayes = np.zeros(func_config.X_dim)
    for (k, v) in zip(x_input, dX_selected):
        index = int(k[1:])-1
        dX_bayes[index] = v
        
    return dX_bayes
