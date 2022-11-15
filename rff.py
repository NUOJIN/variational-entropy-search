#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 21:36:31 2022

@author: nokicheng
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def sample_rff(x, kernel, lengthscale, coefficient, num_functions, num_features):
    '''
    A function sampling random Fourier features given kernel info and evaluation points x
    Input: 
        x: x grid, num_x * d
        kernel: type of kernel rbf/laplace/cauchy
        lengthscale: hyperparameter l of kernel 
        coefficient: hyperparameter sigma of kernel
        num_functions: number of functions to generate
        num_features: number of fourier features
    Output:
        features: a tensor containing features evaluation at x, num_function * num_features * num_x
    '''
    
    # Dimension of data space
    x_dim = x.shape[-1]
    omega_shape = (num_functions, num_features, x_dim)
    
    # Handle each of three possible kernels separately
    if kernel == 'rbf':
        omega = np.random.normal(size=omega_shape)
        
    elif kernel == 'laplace':
        omega =  np.random.standard_cauchy(size=omega_shape)
        
    elif kernel == 'cauchy':
        omega = np.random.laplace(size=omega_shape)
        
    elif kernel == 'matern32':
        omega = np.random.standard_t(3,size=omega_shape)
        
    elif kernel == 'matern52':
        omega = np.random.standard_t(5,size=omega_shape)
        
    # Scale omegas by lengthscale -- same operation for all three kernels
    omega = omega / lengthscale
    
    phi = np.random.uniform(low=0.,
                            high=(2 * np.pi),
                            size=(num_functions, num_features, 1))
    
    features = np.cos(np.einsum('sfd, nd -> sfn', omega, x) + phi)
    features = (2 / num_features) ** 0.5 * features * coefficient
    
    return features

def rff_posterior(x_data, y_data, x_pred, kernel, lengthscale, coefficient, 
                  num_functions, num_features, noise):
    '''
    A function sampling Gaussian process posterior based on random Fourier features
    Input:
        x_data: samples at x, train_size * d
        y_data: samples at y, train_size
        x_pred: x grid, num_x * d
        kernel: type of kernel rbf/laplace/cauchy
        lengthscale: hyperparameter l of kernel 
        coefficient: hyperparameter sigma of kernel
        num_functions: number of functions to generate
        num_features: number of fourier features
        noise: a positive number denoting noise level
    Output:
        functions_pred: a matrix containing functions evaluation at x, num_function * num_x
    '''
    
    num_data = x_data.shape[0]
    x_full = np.concatenate([x_pred, x_data])
    
    features = sample_rff(x=x_full, 
                             kernel=kernel, 
                             lengthscale=lengthscale, 
                             coefficient=coefficient, 
                             num_functions=num_functions, 
                             num_features=num_features)
    
    features_pred = features[:,:, :-num_data]
    features_data = features[:,:, -num_data:]
    
    # sample posterior weights
    weights = np.empty(shape=(num_functions,num_features))
    for func in range(num_functions):
        K = features_data[func] @ features_data[func].T + noise*np.eye(num_features)
        weights[func] = np.random.multivariate_normal(np.linalg.solve(K,features_data[func]@y_data), 
                                                     noise*np.linalg.pinv(K))
    functions_pred = np.einsum('ij, ijk -> ik',
                               weights,
                               features_pred)
    
    return functions_pred

def rff_mle(x_data, y_data, kernel, lengthscale_init, coefficient_init,
            num_features, noise):
    '''
    A function optimizing hyperparameter lengthscale and coefficient based on MLE
    Input:
        x_data: samples at x, train_size * d
        y_data: samples at y, train_size
        kernel: type of kernel rbf/laplace/cauchy
        lengthscale_init: initial lengthscale of kernel 
        coefficient_init: initial coefficient of kernel
        num_features: number of fourier features
        noise: a positive number denoting noise level
    Output:
        lengthscale_opt: optimal lengthscale
        coefficient_opt: optimal coefficient
    '''
    def mle(hypa):
        lengthscale = hypa[0]
        coefficient = hypa[1]
        features=sample_rff(x_data, kernel, lengthscale, coefficient, 1, num_features)[0]
        KI = features.T@features + noise*np.eye(len(y_data))
        a = np.linalg.solve(KI,y_data)
        return a@y_data + np.log(np.linalg.det(KI))
    
    x0 = np.array([lengthscale_init, coefficient_init])
    res = minimize(mle, x0, method='BFGS')
    return res.x

def rff_sgd(x_data, y_data, x_grid, x_init, max_iter, step_size, 
            kernel, lengthscale, coefficient, num_functions, num_features, noise, return_x=False):
    '''
    A function maximizing variational entropy search acquisition function via SGD
    Input:
    Input:
        x_data: samples at x, train_size * d
        y_data: samples at y, train_size
        x_grad: x grid, num_x * d
        x_init: initial x point, 1 * 1
        max_iter: maximal number of iteration
        step_size: step size for SGD
        kernel: type of kernel rbf/laplace/cauchy
        lengthscale: hyperparameter l of kernel 
        coefficient: hyperparameter sigma of kernel
        num_functions: number of functions to generate
        num_features: number of fourier features
        noise: a positive number denoting noise level
        return_x (OPTIONAL): return the list of all argmax_x satisfying y>y*_t if True
    Output:
        x: value of x that maximize VES acquisition function
    '''
    t = 0
    y_max = np.max(y_data)
    x_lst = np.array([])
    lowb = np.min(x_grid) # only for 1D
    higb = np.max(x_grid) # only for 1D
    while True:
        if t > max_iter:
            if return_x:
                return x_init,np.array(x_lst).flatten()
            return x_init
        x_new_grid = np.vstack((x_grid,x_init))
        functions = rff_posterior(x_data, y_data, x_new_grid, kernel, lengthscale, coefficient, 
                    num_functions, num_features, noise)
        y_init = functions[:,-1]
        functions_grid = functions[:,:-1]
        x_argmax = x_grid[np.argmax(functions_grid,axis=1)]
        eff_x_argmax = x_argmax[y_init>y_max].flatten()
        x_lst = np.hstack((x_lst,eff_x_argmax))
        grad = 2*step_size*np.mean((x_init - x_argmax).flatten()*(y_init>y_max))
        if np.abs(grad) < x_grid[1] - x_grid[0]:
            if return_x:
                return x_init - grad,np.array(x_lst).flatten()
            return x_init - grad
        x_init = np.clip(x_init - grad,lowb,higb)
        t += 1
    return 
    
if __name__ == '__main__':
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.gaussian_process import GaussianProcessRegressor
    
    def f(x):
        return (-x**2+np.sin(x)).flatten()
    
    kernel_name = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
    
    
    # Kernel parameters, # of functions to sample, # of features for each function
    lengthscale = 1.
    coefficient = 1.
    noise = 0.1
    num_functions = 100
    num_features = 10
    
    # Input locations
    x = np.linspace(-3., 3., 800)[:, None]
    
    x_data = np.array([-2.5,-0.1,2.5])[:,None]
    y_data = f(x_data)+noise*np.random.normal(size=len(x_data))
    x_pred = x
    kernel = 'matern32'
    functions_pred = rff_posterior(x_data, y_data, x_pred, kernel, lengthscale, coefficient,
                                   num_functions, num_features, noise)  
 
    gpr = GaussianProcessRegressor(kernel=kernel_name, random_state=0)
    gpr.fit(x_data,y_data)
    gpr_pred,gpr_std = gpr.predict(x,return_std=True)
    rff_pred = np.mean(functions_pred,axis=0)
    rff_std = np.std(functions_pred,axis=0)
    
    plt.plot(x,gpr_pred,label='GPR',c='b')
    plt.fill_between(x.flatten(), gpr_pred+1.94*gpr_std, gpr_pred-1.94*gpr_std, color='b', alpha=0.4)
    plt.plot(x,rff_pred,label='RFF',c='r')
    plt.fill_between(x.flatten(), rff_pred+1.94*rff_std, rff_pred-1.94*rff_std, color='r', alpha=0.4)
    plt.legend()
    
    new_lengthscale,new_coefficient = rff_mle(x_data, y_data, kernel, lengthscale, coefficient,
                                              num_features, noise)
    print('New hyperparameter results',new_lengthscale,new_coefficient)

# =============================================================================
#     x_init = np.array([[0.0]])
#     y_max = np.max(y_data)
#     x_grid = x
#     step_size = 0.1
#     max_iter = 50
#     
#     t = 0
#     while True:
#         if t > max_iter:
#             break
#         x_new_grid = np.vstack((x_grid,x_init))
#         functions = rff_posterior(x_data, y_data, x_new_grid, kernel, lengthscale, coefficient, 
#                     num_functions, num_features, noise)
#         y_init = functions[:,-1]
#         functions_grid = functions[:,:-1]
#         x_argmax = x_grid[np.argmax(functions_grid,axis=1)]
#         eff_x_argmax = x_argmax[y_init>y_max].flatten()
#         x_lst = np.hstack((x_lst,eff_x_argmax))
#         grad = 2*step_size*np.mean((x_init - x_argmax).flatten()*(y_init>y_max))
#         x_init = x_init - grad
#         t += 1
# =============================================================================
