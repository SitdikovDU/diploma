import numpy as np
from scipy.stats import multivariate_normal
from generations import rotate_matrix
import random


def rotate_matrix(matrix, phi):
    phi = 2*np.pi/360*phi
    rotation_matrix = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

    return rotation_matrix @ matrix @ rotation_matrix.T

class EM:
    def __init__(self, n_components=2, max_iter=100, reg_covar=1e-6, eps=1e-312):
        self.max_iter = max_iter
        self.n_components = n_components
        self.data = None
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.cov_weights_ = []
        self.phi = None
        self.eps = eps
        self.reg_covar = reg_covar
        self.fixed_means = 0
        self.ll_history = []
        self.best = {"ll": -np.inf}

    def get_cluster_type(self, k):
         return np.argmax(self.cov_weights_[k])
        
    def log_likelihood(self):
        num_samples = len(self.data)
        ll = 0
        labels, covs = self.predict()
        for i in range(num_samples):
            value = self.weights_[labels[i]] * self.cov_weights_[labels[i], covs[i]]* multivariate_normal.pdf(self.data[i], mean=self.means_[labels[i]], cov=rotate_matrix(self.covariances_[covs[i]], self.phi[labels[i],covs[i]]))
            ll += np.log(value + self.eps)
        return ll

    def E_step(self):
        num_samples = len(self.data)
        n_covariances = len(self.covariances_)
        responsibilities = np.zeros((num_samples, self.n_components, n_covariances))
        
        
        for k in range(self.n_components):
            for j in range(n_covariances):
                responsibilities[:, k, j] = self.weights_[k] * self.cov_weights_[k, j] * multivariate_normal.pdf(self.data, mean=self.means_[k], cov=rotate_matrix(self.covariances_[j], self.phi[k, j]))
        
        for i in range(num_samples):   
            responsibilities[i, :, :] /= self.eps + np.sum(responsibilities[i, :, :])
        
        return responsibilities

    def M_step(self, responsibilities):
        num_samples = len(self.data)
        n_covariances = len(self.covariances_)

        self.weights_ = np.sum(responsibilities, axis=(0,2)) / num_samples
        self.cov_weights_ = np.divide(np.sum(responsibilities, axis=0), np.sum(responsibilities, axis=(0, 2))[:, np.newaxis] + self.eps)

        for l in range(self.fixed_means, self.n_components):
            c = self.eps * np.ones((2, 1))
            denominator = self.eps
            for j in range(n_covariances):
                for i in range(num_samples):
                    c += np.linalg.inv(self.covariances_[j]) @ self.data[i].reshape((-1, 1)) *  responsibilities[i, l, j]
                
                denominator += np.linalg.inv(self.covariances_[j]) * np.sum(responsibilities[:, l, j])

            self.means_[l] = np.array(np.linalg.inv(denominator) @ c).flatten()


        for k in range(self.n_components):
            for j in range(n_covariances):
                c = (self.data - self.means_[k])[:, 0]
                d = (self.data - self.means_[k])[:, 1]
                numerator = 2* np.dot(responsibilities[:, k, j], np.multiply(c, d))
                denominator = np.dot(responsibilities[:, k, j], c**2 - d**2)

                if abs(denominator) < self.eps:
                    self.phi[k, j] = 45
                    continue
                self.phi[k, j] = np.arctan(numerator/denominator)/2*180/np.pi
                if denominator > 0:
                    self.phi[k, j] += 90


    def fit(self, data, fixed_means=None, means=None, weights=None, covariances=None, verbose=0):
        self.data = data
        if means is not None:
            self.means_ = np.array(means).reshape(-1, 2)
            self.means_ = np.append(self.means_, np.array(random.sample(list(data), k=self.n_components - len(self.means_))).reshape(-1,2), axis=0)
            if fixed_means is not None:
                self.fixed_means = len(fixed_means)
        elif fixed_means is None:
            self.means_ = np.array(random.sample(list(data), k=self.n_components))
        else:
            self.means_ = np.array(fixed_means).reshape(-1,2)
            self.means_ = np.append(self.means_, np.array(random.sample(list(data), k=self.n_components - len(fixed_means))).reshape(-1,2), axis=0)
            self.fixed_means = len(fixed_means)

        if covariances is None:
            self.covariances_ = [np.identity(data.shape[1]) for _ in range(self.n_components)]
        else:
            self.covariances_ = np.array(covariances)

        if weights is None:
            self.weights_ = np.ones(self.n_components) / self.n_components
        else:
            self.weights_ = np.array(weights)

        self.cov_weights_ = np.ones((self.n_components, len(self.covariances_))) / self.n_components
        self.phi = np.random.randint(0, 360, size=[self.n_components, len(self.covariances_)]) 
        
        for i in range(self.max_iter):
            responsibilities = self.E_step()
            self.M_step(responsibilities)
            if verbose:
                ll = self.log_likelihood()
                self.ll_history.append(ll)
                print(f"iteration {i}, ll = {ll}")
  
        self.best["ll"] = self.log_likelihood()

    def density(self):
        num_samples = len(self.data)
        labels, covs = self.predict()
        
        densities = np.array([
            self.weights_[labels[i]] *
            self.cov_weights_[labels[i], covs[i]] *
            multivariate_normal.pdf(
                self.data[i],
                mean=self.means_[labels[i]],
                cov=rotate_matrix(self.covariances_[covs[i]], self.phi[labels[i], covs[i]])
            )
            for i in range(num_samples)
        ])

        return densities

    def cluster_density(self, points, k):
        cov_type = self.get_cluster_type(k)
        cov = self.covariances_[cov_type]
        
        densities = np.array([
            self.weights_[k] *
            self.cov_weights_[k, cov_type] *
            multivariate_normal.pdf(
                point,
                mean=self.means_[k],
                cov=rotate_matrix(cov, self.phi[k, cov_type])
            )
            for point in points
        ])

        return densities

    def predict(self):
        num_samples = len(self.data)
        covs = []
        
        matrix = self.E_step()
        labels = np.argmax(np.sum(matrix, axis=2), axis=1)
        for i in range(num_samples):
            _, b = np.unravel_index(np.argmax(matrix[i], axis=None), matrix[i].shape)
            covs.append(b)

        return labels, covs



class multiEM(EM):
    def __init__(self, n_components=2, max_iter=100, reg_covar=1e-6, eps=1e-312):
        super().__init__(n_components=n_components, max_iter=max_iter, reg_covar=reg_covar, eps=eps)
        self.best_attempt = {"ll": -np.inf}
        self.history = []
        
    def fit(self, data, fixed_means=None, means=None, weights=None, covariances=None, verbose=0, n_init=10):
        for i in range(n_init):
            super().fit(data, fixed_means=fixed_means, means=means, weights=weights, covariances=covariances, verbose=verbose % 2)
    
            ll = super().log_likelihood()
            self.history.append(ll)
            
            if verbose:
                print(f"run {i} ll = {ll}")

            if self.best_attempt["ll"] < ll:
                self.best_attempt["weights"] = self.weights_
                self.best_attempt["cov_weights"] = self.cov_weights_
                self.best_attempt["phi"] = self.phi
                self.best_attempt["means"] = self.means_
                self.best_attempt["ll"] = ll
            
        self.weights_ = self.best_attempt["weights"]
        self.cov_weights_ = self.best_attempt["cov_weights"]
        self.phi = self.best_attempt["phi"]
        self.means_ = self.best_attempt["means"]