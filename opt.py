#!/usr/bin/env python3

import sys
import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt

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
def func_der(x, func=himmelblau):
    return Jacobian(lambda x: func(x))(x).ravel()

''' Second derivation '''
def func_hess(x, func=himmelblau):
    return Hessian(lambda x: func(x))(x)

''' Plotting the stuff '''
def plot_function(func, func_name):

	if func_name == 'Rosenbrock':
		xmin, xmax, xstep = -2.0, 2.0, .05
		ymin, ymax, ystep = -1.0, 3.0, .05

	else:
		xmin, xmax, xstep = -5., 5., .05
		ymin, ymax, ystep = -5., 5., .05

	X = np.arange(xmin, xmax + xstep, xstep)
	Y = np.arange(ymin, ymax + ystep, ystep)

	#x, y = np.meshgrid(X, Y)
	x, y = np.mgrid[-2:2:100j, -1:3:100j]
	#x, y = np.mgrid[-5:5:100j, -5:5:100j]

	z = func(np.array([x, y]))

	plt.imshow(z, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap='viridis')
	#plt.imshow(z, extent=[xmin, xmax, ymin, ymax], cmap='viridis')
#	plt.imshow(z, origin='lower', cmap='viridis')
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=20)

	#i, j = np.unravel_index(z.argmin(), z.shape)

	plt.scatter(1,1, marker='*', color='white')
#	plt.scatter(2,3, marker='*', color='white')
#	plt.scatter(3.131312,-2.805118, marker='*', color='white')
#	plt.scatter(-3.2283186,-3.779310, marker='*', color='white')
#	plt.scatter(-1.848126,3.584428,  marker='*', color='white')

	plt.contour(z, 10, extent=[xmin, xmax, ymin, ymax], colors='white');
	#plt.contour(z, 15, extent=[xmin, xmax, ymin, ymax], colors='white');

	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
#	plt.scatter(i, j, marker='*', color='white')


	plt.title(func_name, fontsize=30)
#	plt.axis('off')

	plt.show()

def opt_the_shit(x0, func):
	#res = minimize(func, x0, method='BFGS', jac=func_der, hess=func_hess)
	minimize(func, x0, method='BFGS', jac=func_der, options={'disp': True})
#	fmin_bfgs(func, x0, fprime=func_der(, rosen))


if __name__ == "__main__":


	x = float(sys.argv[1])
	y = float(sys.argv[2])

	starting_point = np.array([x, y])
	#opt_the_shit(starting_point, himmelblau)

	plot_function(rosen, 'Rosenbrock')
	#plot_function(himmelblau, 'Himmelblau')
