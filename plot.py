import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist, cdist
from generations import rotate_matrix

def plot_cov_ellipse(center, cov, ax=None, **kwargs):
    """
    Plot covariance ellipse for a 2D distribution.

    Parameters:
        points (array-like): Data points.
        cov (array-like): Covariance matrix.
        ax (matplotlib.axes.Axes): Plotting axes. If None, creates a new figure.
        kwargs: Additional keyword arguments passed to Ellipse.

    Returns:
        matplotlib.patches.Ellipse: Ellipse patch representing the covariance ellipse.
    """
    if ax is None:
        ax = plt.gca()

    # Compute eigenvalues and eigenvectors of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Create ellipse
    ellipse = Ellipse(xy=center, 
                      width=2 * np.sqrt(5.991*eigenvalues[0]),
                      height=2 * np.sqrt(5.991*eigenvalues[1]),
                      angle=angle, **kwargs)

    ax.add_patch(ellipse)
    return ellipse

    

def plot_data(data, centers):
    plt.scatter(data[:, 0], data[:, 1], s = 8, label='пятна')
    plt.scatter(centers[:, 0], centers[:, 1], marker='s', label="центры")
    plt.title(f"Сгенерировано {len(centers)} групп, число точек = {len(data)}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def plot_result(data, model, true_covs=None, print_labels=False):
    """
    Plots the data points and clusters along with their covariance ellipses.

    Parameters:
    data (numpy.ndarray): Input data points.
    model: Trained Gaussian Mixture Model.
    true_covs (numpy.ndarray, optional): True covariance matrices for evaluation.
    print_labels (bool, optional): Whether to print labels on the plot.

    Returns:
    None
    """
    
    title = f"ll = {round(model.best_attempt['ll'],2)}"
    if true_covs:
        acc = round(np.mean(true_covs == np.argmax(model.cov_weights_, axis=1)), 2)
        title += f"| accuracy = {acc}"
    labels, covs = model.predict()
    
    plt.title(title)
    plt.scatter(data[:, 0], data[:, 1], s = 8, label='points', c=labels)
    plt.scatter(model.means_[:, 0], model.means_[:, 1], marker='s')

    covs = np.array(covs)
    matrix = model.E_step()

    for i in range(model.n_components):
        cov_type = np.argmax(model.cov_weights_[i])
        if print_labels:
            text = str(cov_type) + "\n" +str(round(max(model.cov_weights_[i]), 2))
            plt.text(model.means_[i][0], model.means_[i][1], text, fontsize=10, c = 'r')
    
        cov = rotate_matrix(model.covariances_[cov_type], model.phi[i, cov_type])
        plot_cov_ellipse(model.means_[i], cov, alpha=0.2, color='red')


