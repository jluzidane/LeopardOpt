import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


def analyze_correlations(data, correlation_file, dx, dy, mode='spearman'):
    """
    Compute the correlation matrix between input variables and response variables.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing input variables followed by response variables.
    correlation_file : str
        Path for saving the rounded correlation matrix.
    dx : int
        Number of input variables.
    dy : int
        Number of response variables.
    mode : str, optional
        Correlation method. Options are 'pearson' and 'spearman'.
        Default is 'spearman'.

    Returns
    -------
    Cor_matrix : numpy.ndarray
        Correlation coefficient matrix with shape (dy, dx).
    Sign_matrix : numpy.ndarray
        Sign matrix of the correlation coefficients, where positive values are
        denoted by 1, negative values by -1, and zero values by 0.
    """

    if mode == 'pearson':
        Cor_matrix = pearson_correlation_coefficients(data, dx, dy)
    elif mode == 'spearman':
        Cor_matrix = spearman_correlation_coefficients(data, dx, dy)
    else:
        print(f"Error: unsupported correlation method {mode}.")
        import sys
        sys.exit(0)

    Sign_matrix = np.sign(Cor_matrix)

    Cor_matrix_rounded = np.round(Cor_matrix, decimals=2)
    write_data(correlation_file, Cor_matrix_rounded)
    plot_heatmap(Cor_matrix_rounded, os.path.dirname(correlation_file))
    
    return Cor_matrix, Sign_matrix


def pearson_correlation_coefficients(data,dx,dy):
    """Compute Pearson correlation coefficients between each X and Y."""
    matrix = np.zeros((dy, dx))
    for i in range(dy):
        for j in range(dx):            
            col1 = data.iloc[:, dx+i]
            col2 = data.iloc[:, j]
            mask = ~(col1.isna() | col2.isna())
            valid_col1 = col1[mask]
            valid_col2 = col2[mask]         
            
            matrix[i][j] = np.corrcoef(valid_col1, valid_col2)[0, 1]
       
    return matrix


def spearman_correlation_coefficients(data,dx,dy):
    """Compute Spearman correlation coefficients between each X and Y."""
    matrix = np.zeros((dy, dx))
    for i in range(dy):
        for j in range(dx):
            col1 = data.iloc[:, dx+i]
            col2 = data.iloc[:, j]
            mask = ~(col1.isna() | col2.isna())
            valid_col1 = col1[mask]
            valid_col2 = col2[mask]
            
            corr, _ = spearmanr(valid_col1, valid_col2)
            matrix[i][j] = corr
    return matrix


def write_data(filename, data):
    """Save a numerical matrix to a text file."""
    with open(filename, 'w') as file:
        for row in data:
            values = ' '.join(['{:>5.2f}'.format(x) for x in row])
            file.write(values + '\n')


def plot_heatmap(np_data, save_dir):
    """Plot and save the heatmap of the correlation matrix."""
    data = pd.DataFrame(np_data)
    data.index = [f'$Y_{i+1}$' for i in range(data.shape[0])]
    data.columns = [f'$X_{i+1}$' for i in range(data.shape[1])]

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 8))

    from matplotlib.colors import LinearSegmentedColormap
    colors_list = [
        "#023B72",
        "#0E5696",
        "#4090C5",
        "#9ECDEA",
        "#FFFFFF",
        "#F7CBCC",
        "#F1A5A9",
        "#EB8386",
        "#D34B49"
    ]
    my_cmap = LinearSegmentedColormap.from_list("my_custom", colors_list, N=256)
    plot = sns.heatmap(
        data, 
        annot=True, 
        cmap=my_cmap, 
        vmin=-0.6, 
        vmax=0.6, 
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

    plt.savefig(os.path.join(save_dir, 'Correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
