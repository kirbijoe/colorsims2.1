### Module containing functions for generating results from simulations run. Results can be used for theoretical inference. ###

import numpy as np
import pandas as pd 
from os import path, mkdir, listdir
import math
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.patheffects as pe
from scipy import stats
from colorsims_stimuli import ColorGrid
from colorsims_utils import find_ksim, bct_lookup


#All WCS languages except the ones omitted due to transcription errors
all_langs = list(range(1,111))
all_langs.pop(92)
all_langs.pop(91)
all_langs.pop(61)
all_langs.pop(44)

#Subset of WCS languages used for testing. Subset is representative of the original distribution of number of BCTs.
langs_subset = [17,106,38,19,27,80,5,25,39,9,10,52,70,83,1,47,59,89,101,64,66,74,14,18,104]


def clean_data(df):
	'''Cleans a dataframe of imported chip agreement data.'''

	df = df.dropna(axis=0, thresh=220)	#drop any rounds which have primarily nan values
	df = df.fillna(0)					#fill any remaining nan values with 0
	df = df.round(decimals=5)			#round off values to prevent weird values
	df[df < 0] = 0						#replace any negative values with 0
	df[df > 1] = 1						#replace any unusually large values with 1
	return df

def rand_index(sys1, sys2):
	'''Computes the Rand Index between two naming systems. The systems are assumed to be a 1x320 vector containing the names for each chip.'''
	
	a = 0
	b = 0
	for i in range(320):
		for j in range(i+1,320):
			sys1_name_i = sys1[i]
			sys1_name_j = sys1[j]
			sys2_name_i = sys2[i]
			sys2_name_j = sys2[j]

			if sys1_name_i == sys1_name_j and sys2_name_i == sys2_name_j:
				a += 1
			elif sys1_name_i != sys1_name_j and sys2_name_i != sys2_name_j:
				b += 1

	rand = float(a + b)/((320*319)/2)
	return rand

def rand_index_from_sim(title):
	'''Computes the average Rand Index between the participants in the converged simulation state.
	title is the of the form 'Lang[lang_num]theta[theta_val] ([n_c] centroids)'.''' 
	
	lang_num = int(title[4:title.find("theta")])

	wcs_df = pd.read_csv("C:\\Users\\Kirbi Joe\\Documents\\colorsims_evol\\WCSParticipantData\\ParticipantData\\ParticipantDataLang"+str(lang_num)+".csv", header=None)
	wcs_df = wcs_df - 1		#-1 accounts for indexing
	sim_df = pd.read_csv(path.abspath(path.join("Lang "+str(lang_num), title, "system_solution.csv")), header=None)

	all_agent_rand = []
	for agent in range(len(sim_df)):
		a = 0
		b = 0
		wcs_part = wcs_df.iloc[agent]
		conv_agent = sim_df.iloc[agent]
		agent_rand = rand_index(conv_agent, wcs_part)
		all_agent_rand.append(agent_rand)

	return np.average(all_agent_rand)

def rand_indices_lang(lang_num, theta1, theta2):
	'''Computes Rand Indices between two simulations from the same language but with different thetas.'''

	title1 = "Lang"+str(lang_num)+"theta"+str(theta1)+" (3 centroids)"
	title2 = "Lang"+str(lang_num)+"theta"+str(theta2)+" (3 centroids)"
	theta1_df = pd.read_csv(path.abspath(path.join("Run 2", "Lang "+str(lang_num), title1, "system_solution.csv")), header=None)
	theta2_df = pd.read_csv(path.abspath(path.join("Run 2", "Lang "+str(lang_num), title2, "system_solution.csv")), header=None)

	theta1_mode = [theta1_df[i].value_counts().keys()[0] for i in range(320)]	#modal map of the converged solution of simulation with theta1
	theta2_mode = [theta2_df[i].value_counts().keys()[0] for i in range(320)]	#modal map of the converged solution of simulation with theta2

	rand = rand_index(theta1_mode, theta2_mode)
	return rand

def all_rand_indices_lang(lang_num):
	'''Computes the mean and standard deviation of Rand Indices between all combinations of theta for lang_num.'''

	theta = np.linspace(0,1,11)
	all_rand = []
	for i in range(11):
		for j in range(i+1,11):
			r = rand_indices_lang(lang_num, round(theta[i], 1), round(theta[j], 1))
			all_rand.append(r)
	return (np.average(all_rand), np.std(all_rand))

def plot_heatmaps(title):
	'''Plots the heatmaps depicting the chip agreement across the simulation rounds. title is of the form 'Lang[lang_num]theta[theta_val] ([n_c] centroids)'.'''

	lang_num = int(title[4:title.find("theta")])
	chip_agree_df = pd.read_csv(path.abspath(path.join("Run 2", "Lang " + str(lang_num), title, "chip_agreement.csv")))
	chip_agree_df = clean_data(chip_agree_df)

	image_path = path.abspath(path.join("Lang " + str(lang_num), title, "Heatmaps"))
	if not path.exists(image_path):
		mkdir(image_path)

	for i in range(len(chip_agree_df)):
		data = np.array(chip_agree_df.iloc[i])
		game_num = data[0]
		data = data[1:]
		heat_map = np.array(data).reshape((8,40))

		ax = None
		fig, ax = plt.subplots()

		ax.matshow(heat_map, cmap="hot_r")

		ax.set_yticks([])
		ax.set_yticklabels([])
		ax.set_xticks([])
		ax.set_xticklabels([])

		plt.savefig(path.join(image_path, str(game_num)+".png"))	    #file name should be game_num.png
		plt.close()

def get_agreement_from_sim(title):
	'''Returns the list of global agreements (A_r) aross all rounds of a simulation. (This list represents the learning curve of the population.'''

	lang_num = int(title[4:title.find("theta")])
	agree_df = pd.read_csv(path.abspath(path.join("Run 2", "Lang " + str(lang_num), title, "agreement_levels.csv")))
	ar = agree_df["A_r"]
	return agree_df["A_r"]

def final_agree_all_sims(lang_list):
	'''Returns the average final agreement level for sets of simulations run for the langs in lang_list.'''

	all_final_agree = np.zeros(shape=[11, len(lang_list)])
	for i in range(len(lang_list)):
		a = 0
		for theta in np.linspace(0,1,11):
			title = "Lang"+str(lang_list[i])+"theta"+str(round(theta,1))+" (3 centroids)"
			final_agree = np.array(get_agreement_from_sim(title))[-1]
			all_final_agree[a, i] = final_agree
			a += 1
	return np.average(all_final_agree)

def initial_agree_all_sims(lang_list):
	'''Returns the average initial agreement level for sets of simulations run for the langs in lang_list.'''

	all_initial_agree = np.zeros(shape=[11, len(lang_list)])
	for i in range(len(lang_list)):
		a = 0
		for theta in np.linspace(0,1,11):
			title = "Lang"+str(lang_list[i])+"theta"+str(round(theta,1))+" (3 centroids)"
			initial_agree = np.array(get_agreement_from_sim(title))[0]	
			all_initial_agree[a, i] = initial_agree
			a += 1
	return np.average(all_initial_agree)

def count_unnamed(system):
	'''Counts the number of color chips in a system that are unamed. system is a word_map which contains the names that an agent/population are
	assigning to each color chips. system can be a 1x320 vector or 8x40 matrix.'''

	system = np.array(system)
	return sum(system == -1)

def final_named_all_sims(lang_list):
	'''Returns the proportion of color chips that had a name in the final solutions of the simulations run for langs in lang_list.'''

	all_named = np.zeros(shape=[11, len(lang_list)])
	for i in range(len(lang_list)):
		a = 0
		for theta in np.linspace(0,1,11):
			title = "Lang"+str(lang_list[i])+"theta"+str(round(theta,1))+" (3 centroids)"
			name_sys_df = pd.read_csv(path.abspath(path.join("Run 2", "Lang " + str(lang_list[i]), title, "system_solution.csv")))
			named_prop = [float(320 - count_unnamed(name_sys_df.iloc[i]))/320 for i in range(len(name_sys_df))]
			all_named[a, i] = np.average(named_prop)
			a += 1
	return np.average(all_named)

def get_empty_type(lang_num, theta):
	'''Returns a data frame which shows the "empty type" classification of each chip over the course of the simulation.
	-2 = no name in the population
	-1 = ambiguously named; many competing names
	 0 = population has salient name for chip'''

	chip_agree_df = pd.read_csv(path.abspath(path.join("Run 2", "Lang "+str(lang_num), "Lang"+str(lang_num)+"theta"+str(theta)+" (3 centroids)", "chip_agreement.csv")), index_col=0)
	chip_agree_df = clean_data(chip_agree_df)
	low_agree = chip_agree_df <= 0.5 	#identify which chips in which rounds have population agreement less than 50%
	empty_types = pd.DataFrame(columns=low_agree.columns, index=low_agree.index) 	#initialize a blank data frame to store empty type classifications

	for game_num in low_agree.index:
		chips = low_agree.columns[low_agree.loc[game_num]].astype(int)
		sys = pd.read_csv(path.abspath(path.join("Run 2", "Lang "+str(lang_num), "Lang"+str(lang_num)+"theta"+str(theta)+" (3 centroids)", "Population Systems", str(game_num)+".csv")), header=None)
		sys = np.array(sys)
		for chip in chips:
			agent_names = sys[:,chip]
			most_freq_name = stats.mode(agent_names).mode[0]
			if most_freq_name == -1:
				empty_types[str(chip)].loc[game_num] = -2
			else:
				empty_types[str(chip)].loc[game_num] = -1

	empty_types = empty_types.fillna(0)

	return empty_types	

def plot_empty(lang_nu, theta, write=True, plot=True):
	'''Plots the empty type classifications of color chips in heatmaps organized by simulation round.
	write determines whether the classifications are written out into a file (rows: simulation rounds (snapshots from every 10,000 rounds), columns: 320 color chips)
	plot determines whether figures are generated for each population snapshot'''

	empty_df = get_empty_type(lang_num, theta)
	image_path = path.abspath(path.join("Run 2", "Lang "+str(lang_num), "Lang"+str(lang_num)+"theta"+str(theta)+" (3 centroids)", "Empty Space Figures"))
	if not path.exists(image_path):
		mkdir(image_path)

	for game_num in empty_df.index:
		data = empty_df.loc[game_num]
		data = np.array(data).reshape([8,40])

		if plot:
			fig, ax = plt.subplots()
			ax.matshow(data, cmap="Greys")
			ax.set_yticks([])
			ax.set_yticklabels([])
			ax.set_xticks([])
			ax.set_xticklabels([])
			plt.savefig(path.join(image_path, str(game_num)+".png"))
			plt.close()

	if write:
		empty_df.to_csv(path.join("Run 2", "Lang "+str(lang_num), "Lang"+str(lang_num)+"theta"+str(theta)+" (3 centroids)", "empty_space_systems.csv"))

def all_plot_empty(lang_list, write=True, plot=True):
	'''Runs plot_empty() for all langs in lang_list.'''

	for lang_num in lang_list:
		for theta in np.linspace(0,1,11):
			theta = round(theta, 1)
			plot_empty(lang_num, theta, write=write, plot=plot)

def empty_matshow(lang_num, theta):
	'''Plots a matrix visualization of the empty space classifications over time. Shows how individual chips change types over time.'''

	empty_df = pd.read_csv(path.join("Run 2", "Lang "+str(lang_num), "Lang"+str(lang_num)+"theta"+str(theta)+" (3 centroids)", "empty_space_systems.csv"), index_col=0)
	fig, ax = plt.subplots()
	ax.matshow(empty_df, cmap="Greys")
	ax.set_yticks([])
	ax.set_yticklabels([])
	ax.set_xticks([])
	ax.set_xticklabels([])
	plt.show()
	plt.close()

def learn_trajectory(lang_num, theta):
	'''Returns the correlation between the trajectory of the color chips and the round number. This determines if there is a relationship between the 
	transition of a chip from unnamed-->ambiguous-->named and time.'''

	empty_df = pd.read_csv(path.join("Run 2", "Lang "+str(lang_num), "Lang"+str(lang_num)+"theta"+str(theta)+" (3 centroids)", "empty_space_systems.csv"), index_col=0)
	pearson = []

	for chip in empty_df.columns:
		if len(np.unique(empty_df[chip])) == 1:
			pass
		else:
			r = stats.pearsonr(empty_df.index, empty_df[chip])[0]
			pearson.append(r)

	return np.average(pearson)

def avg_learn_traj(lang_list):
	'''Returns the average correlation between empty type and round number for all langs in lang_list.'''

	pearson = []
	for lang_num in lang_list:
		for theta in np.linspace(0,1,11):
			theta = round(theta, 1)
			r = learn_trajectory(lang_num, theta)
			pearson.append(r)
	return (np.average(pearson), np.std(pearson))



### Functions used to generate the boundary value/probability files for each simulation. These files are used to compute the Boundary Probability of a chip. ###

#Initialize some global variables
bct_dict = bct_lookup()
G = ColorGrid()
conv_file = open(path.abspath("MunsellCIE.txt"), "r")
LAB_table = G.LAB_lookup(conv_file)
LUV_table = G.LUV_lookup(LAB_table)
row_convert = {0:'B', 1:'C', 2:'D', 3:'E', 4:'F', 5:'G', 6:'H', 7:'I'}

def boundary_chips(lang_num, theta, n_centroids=3, num_of_ksims=1):
	'''Generates a file with the boundary value of each chip.'''

	solution_data = pd.read_csv(path.abspath(path.join("Run 2", "Lang "+str(lang_num), "Lang"+str(lang_num)+"theta"+str(theta)+" (3 centroids)", "system_solution.csv")), header=None)

	bnd_chip_path = path.abspath(path.join("Run 2", "Boundary Analysis", "Lang"+str(lang_num)+"theta"+str(theta)+"_bound_chips_"+str(num_of_ksims)+"ksims.csv"))
	bnd_chip_df = pd.DataFrame(columns=[str(i) for i in range(320)], index=solution_data.index)

	num_BCT = bct_dict[lang_num]
	k_sim = find_ksim(num_BCT)

	for i in solution_data.index:
		word_map = np.array(solution_data.iloc[i])
		props = []
		for chip in range(320):
			stim_within_ksim = []
			same_name = 0

			chip_row = chip // 40
			chip_col = chip % 40 
			Mun_stim1 = (row_convert[chip_row], str(chip_col+1))	#stimulus form: (row, col) 
			LAB_stim1 = LAB_table[Mun_stim1]	#stimulus form: (L,a,b)
			LUV_stim1 = LUV_table[LAB_stim1]	#stimulus form: (L,u,v)

			for other_chip in range(320):
				other_chip_row = other_chip // 40
				other_chip_col = other_chip % 40
				Mun_stim2 = (row_convert[other_chip_row], str(other_chip_col+1))	#stimulus form: (row, col) 
				LAB_stim2 = LAB_table[Mun_stim2]	#stimulus form: (L,a,b)
				LUV_stim2 = LUV_table[LAB_stim2]	#stimulus form: (L,u,v)

				if G.Euclid_distance_LUV(LUV_stim1, LUV_stim2) <= num_of_ksims*k_sim and chip != other_chip:
					stim_within_ksim.append(other_chip)

			for chip_ksim in stim_within_ksim:
				if word_map[chip] == word_map[chip_ksim]:
					same_name += 1
			try:
				prop_same_name = float(same_name) / len(stim_within_ksim)
			except ZeroDivisionError:
				prop_same_name = 0
			props.append(prop_same_name)

		bnd_chip_df.iloc[i] = props

	bnd_chip_df.to_csv(bnd_chip_path)

def all_langs_boundary(lang_list):
	'''Generates all the boundary value files for the langs in lang_list.'''

	for lang_num in lang_list:
		for theta in np.linspace(0,1,11):
			theta = round(theta, 1)
			print("LANG NUM: " + str(lang_num) + " // THETA = " + str(theta))
			boundary_chips(lang_num, theta, 1)

def gen_prob_func(x_vals, y_vals):
	'''Generates the probability density function used to convert boundary values into boundary probabilities.'''
	step_function = dict()
	for i in range(len(x_vals)):
		step_function[x_vals[i]] = y_vals[i]

	def prob_function(bound_val):
		truncate = math.floor(bound_val*10**2) / 10**2
		return step_function[truncate]

	return prob_function
    
def get_bound_prob(lang_num, theta, picture=False):
	'''Generates a file with the boundary probability for each chip.'''

	bound_data = pd.read_csv(path.abspath(path.join("Run 2", "Boundary Analysis", "Lang"+str(lang_num)+"theta"+str(theta)+"_bound_chips_1ksims.csv")), index_col=0)
	averages = np.average(bound_data, axis=0)
	x_values = []
	y_values = []
	for i in range(101):
		x_values.append(round(1 - i*0.01, 2))
	for x in x_values:
		num_chips = (averages > x).sum()
		prob = float(num_chips)/320
		y_values.append(prob)

	step_function = gen_prob_func(x_values, y_values)
	all_chip_prob = []
	bound_prob_path = path.abspath(path.join("Run 2", "Boundary Analysis", "Lang"+str(lang_num)+"theta"+str(theta)+"_bound_probs.csv"))
	bound_prob_df = pd.DataFrame(columns=["Chip Num", "Boundary Value", "Boundary Probability"])
	chip_num = 1
	for bound_val in averages:
		chip_prob = step_function(bound_val)
		all_chip_prob.append(chip_prob)
		bound_prob_df = bound_prob_df.append({"Chip Num": chip_num, "Boundary Value": bound_val, "Boundary Probability": chip_prob}, ignore_index=True) 
		chip_num += 1  

	bound_prob_df.to_csv(bound_prob_path)     

	if picture:
		'''Plot the boundary probability of each chip as a heatmap and save as a figure.'''

		prob_grid_form = np.reshape(all_chip_prob, [8,40])
		image_path = path.abspath(path.join("Run 2", "Boundary Analysis", "Pictures"))

		ax = None
		fig, ax = plt.subplots()
		ax.matshow(prob_grid_form, cmap="hot")
		ax.set_yticks([])
		ax.set_yticklabels([])
		ax.set_xticks([])
		ax.set_xticklabels([])
		plt.savefig(path.join(image_path, str(lang_num)+"_bound_probs.png"))

def all_langs_probbound(lang_list):
	'''Generates all the boundary probability files for the langs in lang_list.'''

	for lang_num in lang_list:
		for theta in np.linspace(0,1,11):
			theta = round(theta, 1)
			print("LANG NUM: " + str(lang_num) + " // THETA = " + str(theta))
			get_bound_prob(lang_num, theta)


def correl_bound_learn(lang_num, theta):
	'''Returns the correlation between the learning of a color chip and its location relative to category boundaries. Provides insight into whether
	location affects the learning trajectory of a color chip.'''

	empty_df = pd.read_csv(path.join("Run 2", "Lang "+str(lang_num), "Lang"+str(lang_num)+"theta"+str(theta)+" (3 centroids)", "empty_space_systems.csv"), index_col=0)
	bound_prob_df = pd.read_csv(path.join("Run 2", "Boundary Analysis", "Lang"+str(lang_num)+"theta"+str(theta)+"_bound_probs.csv"), index_col=0)

	bound_prob = bound_prob_df['Boundary Probability']
	rounds_named = []

	for chip in empty_df.columns:
		begin_named = est_loc_named(empty_df, chip)
		rounds_named.append(begin_named)

	return stats.pearsonr(bound_prob, rounds_named)

def avg_correl_bound(lang_list):
	'''Returns the average correlation between boundary location and time it takes for a chip to become salient for all langs in lang_list.'''

	pearson = []
	for lang_num in lang_list:
		for theta in np.linspace(0,1,11):
			theta = round(theta, 1)
			r = correl_bound_learn(lang_num, theta)[0]
			pearson.append(r)
	return (np.average(pearson), np.std(pearson))

def est_loc_named(empty_df, chip):
	'''Returns an estimate of the round in which a chip transitions to being named (becoming salient to the population).'''

	across_rounds = empty_df[chip]
	named_rounds = empty_df.index[across_rounds == 0]
	est_start = across_rounds.index[-1]
	for r in named_rounds:
		i = (r // 10000) - 1
		try:
			decision = np.all(across_rounds[i:i+5] == 0)	#if a chip stays named for 50,000 rounds, consider it "saliently named"
		except: #i+5 exceeds end of data frame
			break
		if decision:
			est_start = r
			break
	return est_start





def do_all(lang_list, write_emp=True, plot_emp=True):
	'''Generates all of the data at once for all the langs in lang_list.'''

	for lang_num in lang_list:
		for theta in np.linspace(0,1,11):
			theta = round(theta, 1)
			print("LANG NUM: " + str(lang_num) + " // theta = " + str(theta))
			plot_empty(lang_num, theta, write=write_emp, plot=plot_emp)
			boundary_chips(lang_num, theta, 1)
			get_bound_prob(lang_num, theta)