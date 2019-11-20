import sys
import time
from datetime import datetime
import random
import numpy as np
from copy import deepcopy
import qml
from qml.representations import *
from qml.kernels import gaussian_kernel
from qml.kernels import laplacian_kernel
from qml.math import cho_solve
import itertools
from time import time
from scipy.optimize import minimize
from collections import defaultdict


def get_energies(filename):
	""" returns dic with energies for xyz files
	"""
	f = open(filename, "r")
	lines = f.readlines()
	f.close()

	energies = dict()

	for line in lines:
		try:
			tokens = line.split()
			xyz_name = tokens[0] + "_" + tokens[1]
			Ebind = float(tokens[2])
			energies[xyz_name] = Ebind
		except: energies[tokens[0] + "_" + tokens[1]] = 0

	return energies


def get_rep(names):
	rep = np.array([])

	for name in names:
		tokens = name.split('_')
		rep = np.append(rep, np.array([float(tokens[0]), float(tokens[1])]))

	return rep

if __name__ == "__main__":

	data = get_energies("train_sd.txt")
	data2 = get_energies("test_sd.txt")

	mols = []
	mols_test = []
	names = []
	names2 = []

	for xyz_file in sorted(data.keys()):
		try:
			mol = qml.Compound()
			mol.properties = data[xyz_file]
			names.append(xyz_file)
			mols.append(mol)
		except: continue


	for xyz_file in sorted(data2.keys()):
		try:
			mol = qml.Compound()
			mol.properties = data2[xyz_file]
			names2.append(xyz_file)
			mols_test.append(mol)
		except: continue

	N = [50,100,200,len(mols)]
	total = len(mols)
	nModels = 10
	sigma = [0.1*2**i for i in range(15)]
	ll = [1e-1,1e-3,1e-5, 1e-7, 1e-9, 1e-11]

	X = get_rep(names)
	X = X.reshape(len(mols),2)
	X_test = get_rep(names2)
	X_test = X_test.reshape(len(mols_test),2)

	Yprime = np.asarray([ mol.properties for mol in mols ])
	Y_test = np.asarray([ mol.properties for mol in mols_test ])

	random.seed(667)

	for j in range(len(sigma)):
		print('\n')
		for l in ll:
			print()
			K = laplacian_kernel(X, X, sigma[j])
			K_test = laplacian_kernel(X, X_test, sigma[j])
			for train in N:
				maes = []
				for i in range(nModels):
					split = list(range(total))
					random.shuffle(split)

					training_index	= split[:train]

					Y = Yprime[training_index]

					C = deepcopy(K[training_index][:,training_index])
					C[np.diag_indices_from(C)] += l

					alpha = cho_solve(C, Y)

					Yss = np.dot((K_test[training_index]).T, alpha)
					diff = Yss	- Y_test
					mae = np.mean(np.abs(diff))
					maes.append(mae)

				s = np.std(maes)/np.sqrt(nModels)
				print(str(l) + '\t' + str(sigma[j]) +	"\t" + str(train) + "\t" + str(sum(maes)/len(maes)) + " " + str(s))
