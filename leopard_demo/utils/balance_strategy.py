import numpy as np
import json
from utils.perform_evaluation import calculate_performance, evaluate_fitness
import importlib

# Load configuration.
config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f)
threshold1 = config["thresholds"]["fitness_threshold1"]
threshold2 = config["thresholds"]["fitness_threshold2"]
stepshold1 = config["thresholds"]["step_threshold1"]
stepshold2 = config["thresholds"]["step_threshold2"]
weighted_mode = config["settings"].get("weighted_mode", None)
num_crement = config["bayesian_optimization"]["x_level"]
formula = config["settings"]["formula"]
try:
    func_config = importlib.import_module(f"functions.{formula}")
except ImportError:
    raise ValueError(        
        f"Configuration for '{formula}' was not found. "
        f"Please check the corresponding function file."
    )
targets = np.array(func_config.TARGET)
x_bounds = {f"x{i+1}": [row[0], row[-1]] for i, row in enumerate(func_config.X_ortho_values)}
EPS = 0.0000001


def extreme_value_method(dX_matrix, RIM_matrix):
    """
    Construct a composite solution using the extremum-value strategy.
    
    Parameters
    ----------
    dX_matrix : numpy.ndarray
        Matrix of recommended updates, with shape ``(Y_dim, X_dim)``.
    RIM_matrix : numpy.ndarray
        Signed relative importance matrix, with the same shape as
        ``dX_matrix``.
    
    Returns
    -------
    dX_recommended : numpy.ndarray
        Composite recommended update, with shape ``(X_dim,)``.
    """
    num_solutions, num_variables = dX_matrix.shape
    dX_recommended = np.zeros(num_variables) 

    # Process each design variable independently.
    for var_idx in range(num_variables):
        var_values = dX_matrix[:, var_idx]
        im_values = RIM_matrix[:, var_idx]

        positive_im_indices =  np.where(im_values > 0 + EPS)[0]
        negative_im_indices = np.where(im_values < 0 - EPS)[0]

        positive_count = len(positive_im_indices)
        negative_count = len(negative_im_indices)
                
        # If more responses require an increase, select the maximum update.
        if positive_count > negative_count:
            dX_recommended[var_idx] = np.max(var_values[positive_im_indices])
        # If more responses require a decrease, select the minimum update.
        elif negative_count > positive_count:            
            dX_recommended[var_idx] = np.min(var_values[negative_im_indices])
        # If the numbers are equal, compare the total importance magnitudes.
        elif positive_count > 0 or negative_count > 0:
            positive_sum = np.sum(im_values[positive_im_indices])
            negative_sum = np.sum(np.abs(im_values[negative_im_indices]))            
            if positive_sum >= negative_sum:
                dX_recommended[var_idx] = np.max(var_values[positive_im_indices])
            else:
                dX_recommended[var_idx] = np.min(var_values[negative_im_indices])
        # If the variable is unimportant for all responses, keep it unchanged.
        else:
            dX_recommended[var_idx] = 0.

        # If all candidate updates conflict with the dominant direction, use a bounded one-step update.
        if dX_recommended[var_idx] == 0:
            x_input = 'x' + str(var_idx + 1)
            x_min, x_max = x_bounds[x_input]
            crement = (x_max - x_min) / num_crement
            if np.sum(dX_matrix[:, var_idx])<0:
                dX_recommended[var_idx] = max(min(dX_matrix[:, var_idx]), -crement)
            else:
                dX_recommended[var_idx] = min(max(dX_matrix[:, var_idx]), crement)
                
    return dX_recommended


def weighted_value_method(dX_matrix, RIM_matrix, y_opt, mode=weighted_mode):
    """
    Construct a composite update using a weighted averaging strategy.
    
    Parameters
    ----------
    dX_matrix : numpy.ndarray
        Matrix of recommended updates, with shape ``(Y_dim, X_dim)``.
    RIM_matrix : numpy.ndarray
        Signed relative importance matrix, with the same shape as
        ``dX_matrix``.
    y_opt : list of str
        List of non-compliant objectives.
    
    Returns
    -------
    dX_recommended : numpy.ndarray
        Composite recommended update, with shape ``(X_dim,)``.    
    """
    
    n = func_config.Y_dim
    weighted_sum = np.sum(dX_matrix * RIM_matrix, axis=0)    
    if mode == "mean":
        dX_recommended = weighted_sum / len(y_opt)
    else:
        dX_recommended = weighted_sum / n

    return dX_recommended


def weighted_value_method2(dX_matrix, X0_matrix, IM_matrix, Relative_Err, mask):
    """
    Construct a composite solution using an alternative weighted-voting strategy.
    This method uses the absolute importance matrix rather than the relative
    importance matrix to avoid amplifying weak factors.

    Parameters
    ----------
    dX_matrix : numpy.ndarray
        Matrix of recommended updates, with shape ``(Y_dim, X_dim)``.
    RIM_matrix : numpy.ndarray
        Signed relative importance matrix, with the same shape as
        ``dX_matrix``.

    Returns
    -------
    dX_recommended : numpy.ndarray
        Composite recommended update, with shape ``(X_dim,)``.
    """
    
    X_matrix = X0_matrix + dX_matrix
    expanded_err = np.repeat(Relative_Err[:, np.newaxis], func_config.X_dim, axis=1)
    
    # Weight matrix combining importance and relative target distance.
    weighted = IM_matrix + expanded_err * mask
    weighted_norm = X_matrix * weighted / weighted.sum(axis=0)
    
    dX_recommended = weighted_norm.sum(axis=0) - X0_matrix

    return dX_recommended


def select_composite_solution(
        cumulative_epoch, 
        fitness,
        fitness_min,
        y_opt,
        X0_matrix,
        dX_matrix,
        IM_matrix,
        RIM_matrix,
        RelativeErr
    ):
    """
    Select a solution update from multiple response-specific recommendations.

    The selection strategy depends on the current optimization stage and the
    fitness value. In the early stage, either the extremum-value method or the
    weighted-value method is used. If the optimization stagnates for multiple
    consecutive iterations, the strategy switches to a more aggressive
    extremum-value exploration.
    """
    
    if cumulative_epoch < stepshold2:
        if fitness > threshold1:
            # Extremum-value strategy.
            dX_recommended = extreme_value_method(dX_matrix, RIM_matrix)
        else:
            # Weighted-balanced-value strategy
            mask = (RIM_matrix != 0)
            RIM_matrix = abs(RIM_matrix) + RelativeErr.reshape(-1, 1)
            RIM_matrix[~mask] = 0
            candidate_methods = [
                lambda: weighted_value_method(dX_matrix, RIM_matrix, y_opt),
                lambda: weighted_value_method2(dX_matrix, X0_matrix, IM_matrix, RelativeErr, mask)
            ]            
            weighted_candidates = [method() for method in candidate_methods]            
            fitness_values = [
                evaluate_fitness(candidate + X0_matrix, 1, epoch_trace=False)[0][0]
                for candidate in weighted_candidates
            ]            
            dX_recommended = weighted_candidates[np.argmin(fitness_values)]
            
    else:
        # Switch to extremum-value strategy if the fitness does not improve consistently.    
        dX_recommended = extreme_value_method(dX_matrix, RIM_matrix)                
        cumulative_epoch = 0

    # Accumulate the stagnation counter if the fitness does not improve.
    if fitness > fitness_min:
        cumulative_epoch += 1
    else:
        cumulative_epoch = 1
        
    return dX_recommended, cumulative_epoch
