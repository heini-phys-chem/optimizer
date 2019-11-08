#!/usr/bin/env python3

import sys
import numpy as np
from scipy.optimize import minimize

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

if __name__ == "__main__":

	X0 = np.arange(start=-1.5, stop=2.0, step=0.3)
	Y0 = np.arange(start=-0.5, stop=3.0, step=0.3)

	data = np.array([])

	for i in range(len(X0)):
		for j in range(len(Y0)):
			value = rosen( np.array([ float(X0[i]), float(Y0[j]) ]) )
			data = np.append(data, float(value))

	data = data.reshape(len(X0), len(Y0))

	res = minimize(rosen, [-0.5, 0.5], method='BFGS', jac=rosen_der,options={'disp': True})
	sns.heatmap(data, cmap='coolwarm')
	plt.show()
