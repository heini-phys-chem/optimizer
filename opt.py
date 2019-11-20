#!/usr/bin/env python3

import sys
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_ncg
from scipy.optimize import fmin_cg

import matplotlib.pyplot as plt
import seaborn as sns

def rosen(x):
	return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
	xm    = x[1:-1]
	xm_m1 = x[:-2]
	xm_p1 = x[2:]

	der = np.zeros_like(x)
	der[1:-1] = 200/(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
	der[0]    = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
	der[-1]   = 200*(x[-1]-x[-2]**2)

	return der

def rosen_hess(x):
	x = np.asarray(x)
	H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
	diagonal = np.zeros_like(x)
	diagonal[0] = 1200*x[0]**2-400*x[1]+2
	diagonal[-1] = 200
	diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
	H = H + np.diag(diagonal)

	return H


def plot_function():

	X0 = np.arange(start=-2.0, stop=2.0, step=0.05)
	Y0 = np.arange(start=-1.0, stop=3.0, step=0.05)

	data = np.array([])

	fig, ax = plt.subplots()

	for i in range(len(X0)):
		for j in range(len(Y0)):
			value = rosen( np.array([ float(X0[i]), float(Y0[j]) ]) )
			data = np.append(data, float(value))

	data = data.reshape(len(X0), len(Y0))
	sns.heatmap(data, cmap='gist_rainbow', ax=ax)

	ax.yaxis.set_major_locator(plt.NullLocator())
	ax.xaxis.set_major_formatter(plt.NullFormatter())

	plt.show()

def minimize(x0):
#	res = minimize(rosen, starting_point, method='BFGS', jac=rosen_der,options={'disp': True})
#	fmin_bfgs(rosen, starting_point)
#	fmin_ncg(rosen, x0, rosen_der, fhess=rosen_hess, avextol=1e-8)
	fmin_ncg(rosen, x0, rosen_der)

if __name__ == "__main__":


	x = float(sys.argv[1])
	y = float(sys.argv[2])

	starting_point = np.array([x, y])
	#minimize(starting_point)
	#print(x,y)


	plot_function()
