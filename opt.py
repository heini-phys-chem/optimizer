#!/usr/bin/env python3

import sys
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs

import matplotlib.pyplot as plt
import seaborn as sns

def rosen(x):
	return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

#def rosen_der(x):
#	xm    = x[1:-1]
#	xm_m1 = x[:-2]
#	xm_p1 = x[2:]
#
#	der = np.zeros_like(x)
#	der[1:-1] = 200/(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
#	der[0]    = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
#	der[-1]   = 200*(x[-1]-x[-2]**2)
#
#	return der

def plot_function():

	X0 = np.arange(start=-1.5, stop=2.0, step=0.1)
	Y0 = np.arange(start=-0.5, stop=2.9, step=0.1)

	data = np.array([])

	for i in range(len(X0)):
		for j in range(len(Y0)):
			value = rosen( np.array([ float(X0[i]), float(Y0[j]) ]) )
			data = np.append(data, float(value))

	print(len(data))
	data = data.reshape(len(X0), len(Y0))
	sns.heatmap(data, cmap='coolwarm')
	plt.show()

def callbackF(Xi):
    print("{}\t\t{}\t\t{}".format(Xi[0], Xi[1], rosen(Xi)))

def minimize(starting_point):
	#[xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] = fmin_bfgs(rosen, starting_point, callback=callbackF, full_output=True)
	fmin_bfgs(rosen, starting_point)

#	print()
#	print("xopt:       {}".format(xopt))
#	print("fopt:       {}".format(fopt))
#	print("gopt:       {}".format(gopt))
##	print("Bopt:       {}".format(Bopt))
#	print("func_calls: {}".format(func_calls))
#	print("grad_calls: {}".format(grad_calls))
#	print("warnflg:    {}".format(warnflg))
#	print()

if __name__ == "__main__":

	x = float(sys.argv[1])
	y = float(sys.argv[2])

	starting_point = np.array([x, y])
	minimize(starting_point)
	print(x,y)
#	plot_function()

'''
Nfeval = 1

def rosen(X): #Rosenbrock function
    return (1.0 - X[0])**2 + 100.0 * (X[1] - X[0]**2)**2 + \
           (1.0 - X[1])**2 + 100.0 * (X[2] - X[1]**2)**2

def callbackF(Xi):
    global Nfeval
    print '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], rosen(Xi))
    Nfeval += 1

print  '{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)')
x0 = np.array([1.1, 1.1, 1.1], dtype=np.double)
[xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] = \
    fmin_bfgs(rosen,
              x0,
              callback=callbackF,
              maxiter=2000,
              full_output=True,
              retall=False)
'''
