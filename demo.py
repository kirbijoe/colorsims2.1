from colorsims_stimuli import ColorGrid
from colorsims_populations import AgentPopulation
from colorsims_simulations import LearningAndCommunicationGame_Grid
from colorsims_agents import RLAgent
from colorsims_utils import SimulationVisualizer, find_ksim
from os import path, mkdir, listdir
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import colors


#All World Color Survey languages, excluding the ones omitted due to transcription errors.
all_langs = list(range(1,111))
all_langs.pop(92)
all_langs.pop(91)
all_langs.pop(61)
all_langs.pop(44)

#Subset of WCS languages used for testing. Subset of representative of the original distribution of BCTs.
lang_subset = [17,106,38,19,27,80,5,25,39,9,10,52,70,83,1,47,59,89,101,64,66,74,14,18,104]


def sim(lang_num, theta=0, num_centroids=1, n_games=1000000, save_every=10000, forced_rounds=100000, epsilon=0.002, fill_method="wcs", save_dir=None):
    '''Runs an example Levinson-Emergence simulation.
    num_centroids is the number of focal chips per category to initialize.
    n_games is the number of games to play. If n_games=None, then run simulation until converged.'''

    G = ColorGrid()
    P = AgentPopulation(G, lang_num, theta=theta, num_centroids_per_cat=num_centroids, fill_method=fill_method)
    Sim = LearningAndCommunicationGame_Grid(P, G, ksim=P.get_agent(0).k_sim, forced_rounds=forced_rounds, epsilon=epsilon)

    if save_dir != None:
        save_path = path.join(save_dir, 'Lang'+str(lang_num)+'theta'+str(theta)+' ('+str(num_centroids)+'_centroids)')
    else:
        save_path = 'Lang'+str(lang_num)+'theta'+str(theta)+' ('+str(num_centroids)+'_centroids)'
	
    if n_games != None:
        Sim.play_random_games(n_games, save_to_disk=True, save_every=save_every, save_path=save_path)
    else:
        Sim.play_all_games(save_to_disk=True, save_every=save_every, save_path=save_path)
    
    #Create movie of simulation
    V = SimulationVisualizer(Sim.folder)

    print ("Creating a movie out of the snapshots")
    V.movie_from_snapshots(filename=path.join(Sim.folder, 'Lang'+str(lang_num)+'theta'+str(theta)+' ('+str(num_centroids)+'_centroids)'))

def all_sim(lang_set, num_centroids=1, n_games=1000000, save_every=10000, forced_rounds=300000, epsilon=0.002, fill_method="wcs"):
    '''Runs simulations for all langs in lang_set.'''

    for lang in lang_set:
        if not path.exists(path.abspath("Lang " + str(lang))):		
            mkdir(path.abspath("Lang " + str(lang)))

        for theta in np.linspace(0, 1, 11):
            print("-------- LANG " + str(lang) + " // theta: " + str(theta) + " --------")
            sim(lang, theta, num_centroids, n_games, save_every, forced_rounds, epsilon, fill_method, save_dir="Lang " + str(lang))




#Functions for checking the simulation results. Checks for simulations which did not converge correctly.
def lang_check1(lang_list):
    '''Prints out the simulation parameters which only ran the forced number of rounds. Used to determine if simulations actualy converged.'''
    
    for lang_num in lang_list:
        for theta in np.linspace(0,1,11):
            theta = round(theta,1)
            filepath = path.abspath(path.join("Run 2", "Lang "+str(lang_num), "Lang"+str(lang_num)+"theta"+str(theta)+" (3_centroids)", "snapshots"))
            num_files = len([f for f in listdir(filepath) if path.isfile(path.join(filepath, f))])
            if num_files == 49:
                print(lang_num, theta)

def lang_check2(lang_list):
    '''Prints how many files are in the snapshots folder of a simulation. Used to determine if simulations actually converged.
    lang_list will usually be a list compiled after running lang_check1.'''
    
    for lang_num in lang_list:
        for theta in np.linspace(0,1,11):
            theta = round(theta,1)
            filepath = path.abspath(path.join("Run 2", "Lang "+str(lang_num), "Lang"+str(lang_num)+"theta"+str(theta)+" (3_centroids)", "snapshots"))
            num_files = len([f for f in listdir(filepath) if path.isfile(path.join(filepath, f))])
            print(lang_num, theta, num_files)



	
	
