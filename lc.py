#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_style('whitegrid')
sns.set_style('ticks')
sns.set_context("poster")

numLC     = 3
labelsize = 20
fontsize  = 25
legend_fontsize = 15

label = ['Nelder-Mead', 'BFGS', 'Newton-CG', 'Nelder-Mead', 'BFGS', 'Newton-CG','SD', 'BoB', 'SLATM', 'FCHL19', 'one-hot']
colors = ['C0', 'C1', 'C2', 'C0', 'C1', 'C2','C3', 'C0', 'C1', 'C2', 'C3']
markers = ['o', 'd', 's', 'o', 'd', 's','x', 'o', 'd', 's', 'x']

def get_data(filename):
	lines = open(filename, 'r').readlines()

	N = np.array([])
	energies = np.array([])

	for line in lines:
		tokens = line.split()
		N = np.append(N, int(tokens[2]))
		energies = np.append(energies, float(tokens[3]))

	return N, energies

def plot_lc(N, energies, label, ax, color, marker, counter):
	N = np.log(N)
	energies = np.log(energies)

	if counter < 3:
		sns.scatterplot(N, energies, label=label, color=color, marker=marker, ax=ax)
		sns.regplot(N, energies, ci=0, color=color, marker=marker, ax=ax)
	else:
		sns.scatterplot(N, energies, color=color, marker=marker, label=label,  linestyle='--', ax=ax)
		sns.regplot(N, energies, ci=0, color=color, marker=marker, ax=ax, line_kws={'linestyle': '--'})

def set_ticks(ax):
	ax.set_xticks(np.array([np.log(N[0]), np.log(N[1]), np.log(N[2]), np.log(N[3]), np.log(N[4])  ]))
	ax.set_xticklabels(['20', '40','80','160', '320'])
	ax.set_yticks(np.array([np.log(1.0), np.log(2.0), np.log(4.0), np.log(8.0), np.log(16.0), np.log(32.0), np.log(64.0)   ]))
	ax.set_yticklabels(['1.0', '2.0', '4.0', '8.0', '16.0', '32.0', '64.0'])

	ax.tick_params(labelsize=labelsize)

def set_labels(ax, title):
#	ax.set_title(title, fontsize=fontsize)
	ax.set_xlabel('training set size $N$', fontsize=fontsize)
	ax.set_ylabel('MAE [# steps]', fontsize=fontsize)

#def set_legend(ax, colors, numLC):
	R_NM    = Line2D([],[],label='Rosen NM', color='C0', marker='o', linestyle='-')
	R_BFGS  = Line2D([],[],label='Rosen BFGS', color='C1', marker='s', linestyle='-') 
	R_NCG   = Line2D([],[],label='Rosen NM', color='C2', marker='d', linestyle='-')
	H_NM    = Line2D([],[],label='Himmelblau NM', color='C0', marker='o', linestyle='--')
	H_BFGS  = Line2D([],[],label='Himmelblau BFGS', color='C1', marker='s', linestyle='--')
	H_NCG   = Line2D([],[],label='Himmelblau N-CG', color='C2', marker='d', linestyle='--')
	handles = [R_NM,R_BFGS,R_NCG,H_NM,H_BFGS,H_NCG]

	#leg = ax.legend(loc='lower left', handlelength=6)
	leg = ax.legend(handles=handles, frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.25),ncol=2)
#	for i in range(numLC):
#		leg.legendHandles[i].set_color(colors[i])

def set_acc(ax):
	ax.axhline(np.log(1.0), color='C9', ls='--')
	ax.axhline(np.log(3.0), color='C9', ls='--')

	ax.text(np.log(90.), np.log(3.), 'MP2 acc.', fontsize=20)
	ax.text(np.log(90.), np.log(1.), 'Chem acc.',fontsize=20)


if __name__ == '__main__':

	filenames = [ sys.argv[i+1] for i in range(len(sys.argv) - 1)]

	f, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 10))

	for i, filename in enumerate(filenames):
		N, energies = get_data(filename)

		plot_lc(N, energies, label[i], axes, colors[i], markers[i], i)
		set_ticks(axes)
		set_labels(axes, r'Himmelblau')
#		set_acc(axes)

#	set_legend(axes[0], colors, numLC, "Rosen")

	plt.tight_layout()

	f.savefig("figs/LCs.png")
	f.savefig("figs/LCs.pdf")
