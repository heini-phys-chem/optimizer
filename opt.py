#!/usr/bin/env python3

import sys
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs

import matplotlib.pyplot as plt
#import seaborn as sns

from numdifftools import Jacobian, Hessian


''' Functions '''
def trid(X):
	x = X[0]
	y = X[1]
	return (x-1)**2 + (y-1)**2 - x*y

def rosen(X):
	x = X[0]
	y = X[1]
	return (1-x)**2 + 100*(y - x**2)**2

def beale(X):
	x = X[0]
	y = X[1]
	return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.65 - x + x*y**3)**2

def booth(X):
	x = X[0]
	y = X[1]
	return (x + 2*y - 7)**2 + (2*x + y - 5)**2

def himmelblau(X):
	x = X[0]
	y = X[1]
	return (x**2 + y -11)**2 + (x + y**2 -7)**2

''' First derivation '''
def func_der(x, func=booth):
    return Jacobian(lambda x: func(x))(x).ravel()

''' Second derivation '''
def func_hess(x, func=rosen):
    return Hessian(lambda x: func(x))(x)

''' Plotting the stuff '''
def plot_function(func, func_name):

	if func_name == 'rosen':
		xmin, xmax, xstep = -2.0, 2.0, .05
		ymin, ymax, ystep = -1.0, 3.0, .05

	else:
		xmin, xmax, xstep = -5., 5., .05
		ymin, ymax, ystep = -5., 5., .05

	X = np.arange(xmin, xmax + xstep, xstep)
	Y = np.arange(ymin, ymax + ystep, ystep)

	x, y = np.meshgrid(X, Y)

	z = func(np.array([x, y]))

	plt.imshow(z, extent=[0, len(x[0])-1, 0, len(y[0])-1], origin='lower', cmap='viridis')
	plt.colorbar()
	plt.contour(z, 15, colors='white');

	plt.show()

def opt_the_shit(x0, func):
	#res = minimize(func, x0, method='BFGS', jac=func_der, hess=func_hess)
	minimize(func, x0, method='BFGS', jac=func_der, options={'disp': True})
#	fmin_bfgs(func, x0, fprime=func_der(, rosen))


if __name__ == "__main__":


	x = float(sys.argv[1])
	y = float(sys.argv[2])

	starting_point = np.array([x, y])
	opt_the_shit(starting_point, booth)

	#plot_function(himmelblau, 'himmelblau')
