import numpy as np

def create_covariances(k : int, min_val : float, max_val : float) -> list:
    """
    Create a list of covariance matrices with specified range and step size.

    Parameters:
    k (int): Number of covariance matrices to generate.
    min_val (float): Minimum value for the range.
    max_val (float): Maximum value for the range.

    Returns:
    list: List of covariance matrices.
    """

    step = (max_val - min_val) / (k - 1)
    covariances = []
    for i in range(k):
        covariances.append([[(min_val + step*i)**2, 0], [0, (min_val + step*i)**2]])

    return np.array(covariances)

def create_long_covariances(k : int, min_val : float, max_val : float) -> list:
    """
    Create a list of long-form covariance matrices with specified range and step size.

    Parameters:
    k (int): Number of covariance matrices to generate.
    min_val (float): Minimum value for the range.
    max_val (float): Maximum value for the range.

    Returns:
    list: List of long-form covariance matrices.
    """

    step = (max_val - min_val) / (k - 1)
    covariances = []
    for i in range(1, k+1):
        covariances.append([[1, 0], [0, (min_val + step*i)**2]])

    return np.array(covariances).reshape()


def rotate_matrix(matrix, phi):
    """
    Rotate a 2x2 matrix by a specified angle.

    Parameters:
    matrix (np.ndarray): Input 2x2 matrix.
    phi (int or float): Rotation angle in degrees.

    Returns:
    np.ndarray: Rotated 2x2 matrix.
    """
    phi = np.radians(phi)
    rotation_matrix = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    
    return rotation_matrix @ matrix @ rotation_matrix.T

def generate_groups(covariances, groups=20, l=10, r=20, a=0, b=100):
    """
    Generate synthetic data points based on specified parameters.

    Parameters:
    covariances: np.array of covariance matrices.
    groups (int): Number of groups.
    l (int): Minimum number of data points per group.
    r (int): Maximum number of data points per group.
    a (float): Minimum value for data point range.
    b (float): Maximum value for data point range.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing generated data points, cluster centers and cov indices.
    """
    
    centers = np.random.uniform(a, b, [groups,2])
    data = []
    covs_ind = np.random.randint(0, len(covariances), size=groups)
    covs = covariances[covs_ind]

    for i in range(groups):
        angle = np.random.randint(0, 360)
        data.extend(np.random.multivariate_normal(centers[i], rotate_matrix(covs[i], angle), size=np.random.randint(l, r)))
    data = np.array(data)

    return data, centers, covs_ind