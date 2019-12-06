#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.set_style('ticks')
sns.set_context("poster")

labelsize = 30
fontsize  = 35
legend_fontsize = 30

label = ['Nelder-Mead', 'BFGS', 'Newton-CG', 'Nelder-Mead', 'BFGS', 'Newton-CG','SD', 'BoB', 'SLATM', 'FCHL19', 'one-hot']
colors = ['C0', 'C1', 'C2', 'C0', 'C1', 'C2','C3', 'C0', 'C1', 'C2', 'C3']
markers = ['o', 'd', 's', 'o', 'd', 's','x', 'o', 'd', 's', 'x']

def get_data(f):
	lines = open(filename, 'r').readlines()

	N = np.array([])

	for line in lines:
		tokens = line.split()
		N = np.append(N, int(tokens[2]))

	return N

def set_stuff(ax):
#	ax[0].set_xlabel('Number of optimization steps', fontsize=fontsize)
	ax[0].set_ylabel('Frequency [a.u.]', fontsize=fontsize)


	leg = ax[1].legend(fontsize=legend_fontsize)
	ax[0].tick_params(labelsize=labelsize)
	ax[1].tick_params(labelsize=labelsize)

	ax[0].set_xlim([0,390])
	ax[1].set_xlim([390,410])

	ax[0].spines['right'].set_visible(False)
	ax[1].spines['left'].set_visible(False)
	ax[0].yaxis.tick_left()
	ax[0].tick_params(labelright='off')
	ax[1].yaxis.tick_right()

	plt.subplots_adjust(wspace=-1)


def plot_density(number_of_steps, label, ax, color):
	sns.kdeplot(number_of_steps, color=color, shade=True, ax=ax[0])
	sns.kdeplot(number_of_steps, label=label, color=color, shade=True, ax=ax[1])

if __name__ == "__main__":

	filenames = [ sys.argv[i+1] for i in range(len(sys.argv) - 1)]

#	f, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
	f, axes = plt.subplots(1,2, sharey=True, facecolor='w', figsize=(8,8))

	for i, filename in enumerate(filenames):
		number_of_steps = get_data(filename)

		plot_density(number_of_steps, label[i], axes, colors[i])

	f.text(.5, .01, "Number of optimization steps", ha='center', fontsize=fontsize)
	set_stuff(axes)
	plt.yticks([])
	plt.tight_layout()


	f.savefig('density_rosen.pdf')
	#plt.show()
