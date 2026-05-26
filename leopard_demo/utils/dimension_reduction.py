import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json


config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f)


def reduce_dimensionality(
        data, 
        importance_file1, 
        importance_file2, 
        IM_threshold, 
        dx, 
        dy,
        min_key_fact_num = config["settings"]["key_fact_num"],
        rand_seed = config["settings"]["randseed"],
        mode='RandForest'
    ):
    """
    Perform dimensionality reduction based on feature importance estimated by
    Random Forest or XGBoost.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing design variables and response variables.
    importance_file1 : str
        Path for saving the feature-importance ranking.
    importance_file2 : str
        Path for saving the feature-importance matrix.
    IM_threshold : float
        Cumulative importance threshold for selecting key factors.
    dx : int
        Number of design variables.
    dy : int
        Number of response variables.
    min_key_fact_num : int, optional
        Minimum number of selected key factors for each response.
    rand_seed : int, optional
        Random seed used for model training.
    mode : str, optional
        Feature-importance estimator. Options are "RandForest" and "XGBoost".

    Returns
    -------
    IM_matrix : numpy.ndarray
        Feature-importance matrix.
    IM_factors : dict
        Dictionary of selected key factors for each response.
    RIF_matrix : numpy.ndarray
        Relative importance factor matrix.
    """

    X = data.iloc[:, :dx]
    IM_matrix = np.empty((dy, dx))
    IM_factors = {}

    # open(importance_file1, 'w')
    # open(importance_file2, 'w')
        
    for i in range(dy):
        col = data.iloc[:, dx+i] 
        mask = ~col.isna()
        y = col[mask]
        
        if mode == "RandForest":
            model = RandomForestRegressor(
                n_estimators=1000, 
                random_state=rand_seed
            )
        elif mode == "XGBoost":
            model = xgb.XGBRegressor(
                n_estimators=1000,
                random_state=rand_seed,
                objective='reg:squarederror'
            )
        else:
            print(f"Error: Unsupported model mode: {mode}.")
            import sys
            sys.exit(0)

        model.fit(X[mask], y)

        importances = model.feature_importances_
        indices = importances.argsort()[::-1]
        IM_matrix[i, :] = importances

        IM_factors[f"y{i+1}"] = select_IM_factors(importances, indices, IM_threshold, min_key_fact_num)
    
    RIF_matrix = calc_relative_IM(IM_matrix, IM_factors)

    IM_matrix = np.around(IM_matrix, decimals=6)
    print("Key factors for each objective:")
    for key in IM_factors:
        print(f"{key}: {IM_factors[key]}")

    plot_heatmap(IM_matrix, os.path.dirname(importance_file1))

    return IM_matrix, IM_factors, RIF_matrix



def write_importance(outfile1, outfile2, importances, indices, target_index):        
    """Save feature-importance rankings and the corresponding importance matrix."""    
    
    with open(outfile1, 'a') as f:        
        f.write(f"Feature Importance Ranking of y{target_index+1}:\n")
        for rank, index in enumerate(indices, start=1):
            f.write(f"{rank}  x{index+1}  {importances[index]:.4f}\n")
    with open(outfile2, 'a') as f:
        f.write(' '.join(['{:.4f}'.format(i) for i in importances]) + '\n')


def plot_heatmap(IM_matrix, save_dir):
    """Plot and save the heatmap of the feature-importance matrix."""

    data = pd.DataFrame(IM_matrix)
    data.index = [f'$Y_{i+1}$' for i in range(data.shape[0])]
    data.columns = [f'$X_{i+1}$' for i in range(data.shape[1])]

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 8))
    plot = sns.heatmap(
        data, 
        annot=True, 
        cmap='viridis', 
        vmax=0.5, 
        fmt='.2f', 
        linewidths=0.8, 
        xticklabels=True, 
        yticklabels=True, 
        annot_kws={"size": 12, 'weight': 'bold'}
    )

    for item in plot.get_xticklabels():
        item.set_fontsize(22)
        item.set_fontweight('bold')
    for item in plot.get_yticklabels():
        item.set_fontsize(22)
        item.set_fontweight('bold')

    plt.savefig(os.path.join(save_dir, 'Importance_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()


def select_IM_factors(importances, indices, threshold, min_key_num=1):
    """
    Select key input factors according to cumulative feature importance.

    Features are first ranked in descending order of importance. The selected
    feature set is expanded until the cumulative importance exceeds the specified
    threshold and the minimum number of key factors is satisfied.

    Parameters
    ----------
    importances : numpy.ndarray
        Feature-importance values.
    indices : numpy.ndarray
        Feature indices sorted by descending importance.
    threshold : float
        Cumulative importance threshold.
    min_key_num : int, optional
        Minimum number of selected key factors. Default is 1.

    Returns
    -------
    selected_variables : list of str
        Names of selected key input variables.
    """
    sorted_importances = importances[indices]
    
    cumulative_sum = 0
    selected_indices = []
    for i, value in enumerate(sorted_importances):
        cumulative_sum += value
        selected_indices.append(indices[i])
        if cumulative_sum > threshold and len(selected_indices)>=min_key_num:
            break

    variables = [f"x{i+1}" for i in range(len(importances))]
    selected_variables = [variables[i] for i in selected_indices]    
    selected_variables = sorted(selected_variables)
    
    return selected_variables


def calc_relative_IM(IM_matrix, IM_factors):
    """
    Compute the relative importance factor matrix.

    Non-selected factors are first masked out. The remaining importance values
    are then normalized by the maximum importance value within each response
    variable, yielding a row-wise relative importance representation.

    Parameters
    ----------
    IM_matrix : numpy.ndarray
        Original feature-importance matrix.
    IM_factors : dict
        Dictionary of selected key factors for each response variable.

    Returns
    -------
    RIF_matrix : numpy.ndarray
        Relative importance factor matrix.
    """
    IF_matrix= IM_matrix.copy()
    index_map = {f'x{i+1}': i for i in range(IF_matrix.shape[1])}
    # Retain only the selected key factors for each response variable.
    for i, factors in enumerate(IM_factors.values()):
        indices_keep = [index_map[x] for x in factors]
        IF_matrix[i] = np.array([IF_matrix[i, j] if j in indices_keep else 0 for j in range(IM_matrix.shape[1])])

    # Normalized by the maximum of key factors of each objective.
    row_max = np.max(IF_matrix, axis=1, keepdims=True)
    RIF_matrix = IF_matrix / row_max
    
    return RIF_matrix
