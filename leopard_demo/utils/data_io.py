import sys
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


def load_data(data, X, y):

    columns_to_select = X + y  
    data = data[columns_to_select]
    
    # Remove duplicated design points while retaining the corresponding response column.
    duplicates = data.iloc[:, :-1].duplicated()
    data = data[~duplicates]
    
    return data


def save_data(data, filename):
    """
    Save array-like data to a CSV file.

    Parameters
    ----------
    data : numpy.ndarray or list
        Data to be written. A list is written as a single row, whereas a
        NumPy array can be written as a matrix.
    filename : str
        Path to the output file.
    """
    
    if isinstance(data, np.ndarray):
        pd.DataFrame(data).to_csv(filename, mode='a', header=False, index=False)
    
    if isinstance(data, list):
        with open(filename, mode='a', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            row = data
            writer.writerow(row)


def plot_data(loss_list, save_dir, ylabel_name="Fitness"):
    """
    Plot the evolution of a metric over optimization epochs.

    Parameters
    ----------
    loss_list : list
        Sequence of metric values recorded at each epoch.
    save_dir : str
        Directory for saving the figure.
    ylabel_name : str, optional
        Label of the y-axis and output figure name. Default is "Fitness".
    """
    epochs = range(len(loss_list))
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_list, 'b-', linewidth=5, label=f'{ylabel_name}')
    plt.xlabel('Number of function evaluations', fontsize=22)
    plt.ylabel(ylabel_name, fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', linewidth=1.5, alpha=0.8)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f'{ylabel_name}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    

def create_exp_dir(formula, save_dir):
    """
    Create a new experiment directory for the current run.

    Parameters
    ----------
    formula : str
        Name of the benchmark function or formulation system.
    save_dir : str
        Root directory for saving experimental results.

    Returns
    -------
    str
        Path to the newly created experiment directory.
    """
    base_dir = os.path.join(save_dir, formula)    
    os.makedirs(base_dir, exist_ok=True)
    
    # Identify existing experiment directories.
    existing_exp_dirs = []
    for item in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, item)) and item.startswith("exp"):
            try:
                num = int(item[3:])
                existing_exp_dirs.append(num)
            except ValueError:
                continue
    next_num = max(existing_exp_dirs) + 1 if existing_exp_dirs else 1    
    new_exp_dir = os.path.join(base_dir, f"exp{next_num}")
    os.makedirs(new_exp_dir, exist_ok=True)
    print(f"Created new directory {new_exp_dir} for storing results of the current run...")
    
    return new_exp_dir
