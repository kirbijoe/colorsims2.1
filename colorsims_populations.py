import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pickle
from colorsims_agents import RLAgent
from os import path
import pandas as pd
import math



class AgentPopulation():

    def __init__(self, hue_space, lang, theta=0, num_centroids_per_cat=1, fill_method="prob"):
        '''Initializes a population of agents.
        Assumes vocabulary is a list of the vocabulary words (typically a list of integers).
        Assumes the hue space is either a color_circle or color_grid, where color_circle is a DiscreteCircle object and
        color_grid is a ColorGrid object.'''

        self.lang_num = lang
        raw_wcs_data = np.array(pd.read_csv(path.abspath(path.join("WCSParticipantData", "ParticipantData", "ParticipantDataLang"+str(self.lang_num)+".csv")), header=None))
        self.hue_space_type = "color_grid"
        self.color_grid = hue_space 
        
        self.num_agents = np.shape(raw_wcs_data)[0]
        self.init_network()
        self.agents = {}

        #Populate network with appropriate agent types
        for node in self.network.nodes():
            self.agents[node] = RLAgent(self.color_grid, lang=self.lang_num, p_id=node+1, theta=theta, num_centroids_per_cat=num_centroids_per_cat, fill_method=fill_method)  #+1 is to correct for indexing

        self.vocabulary = np.arange(self.get_agent(0).num_words)

    def init_network(self):
        '''Initializes a complete network.
        Override this method in order to init different network types.'''
        
        self.network = nx.complete_graph(self.num_agents)

    def get_agent(self, agent_key):
        '''Returns the agent object specified by the agent_key.'''
        
        return self.agents[agent_key]

    def neighbors(self, agent_key):
        '''Returns the neighbors of the agent specified by agent_key.'''
        
        return self.network.neighbors(agent_key)

    def random_agent(self):
        '''Returns the agent_key of a random agent from the population.'''
        
        return self.network.nodes()[ np.random.choice( np.arange( len(self.network.nodes()) ) ) ]

    def random_neighbor(self, agent_key):
        '''Returns a random neighbor of the agent specified by agent_key'''
        
        # return np.random.choice(self.network.neighbors(agent))
        return self.network.neighbors(agent_key)[ np.random.choice( np.arange( len(self.network.neighbors(agent_key)) ) ) ]

    def get_random_pair(self):
        '''Returns the agent objects for two randomly selected agents who are neighbors of each other'''
        
        agent_one_key = self.random_agent()
        agent_two_key = self.random_neighbor(agent_one_key)

        # return self.get_agent(agent_one_key), self.get_agent(agent_two_key)
        return agent_one_key, agent_two_key

    def random_generation(self):
        '''Pre-generate all of the random agent pairs for a complete generation.'''
        
        pass

    def plot_population(self, filename=None):
        '''Plots the population on a grid. The grid is comprised of the visual representation of each agent's word map (line graph 
        for DiscreteCircle and heat map for ColorGrid).
        Override this if you need something different.
        gray: prints word map in gray scale if True, in color if False'''
        
        grid_size = int(math.ceil(np.sqrt(self.num_agents)))
        fig, axarr = plt.subplots(grid_size, grid_size, sharex='col', sharey='row')

        i = 0
        for r in range(grid_size):
            for c in range(grid_size):
                if i < self.num_agents:
                    agent = list(self.agents.keys())[i]
                    self.get_agent(agent).plot_word_map(ax=axarr[r,c])
                    i += 1
                else:
                    axarr[r,c].axis('off')

        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename)
            plt.close()

    def save(self, path=None, suffix=None):
        '''Saves the population as a binary pickled file.
        Default is in current working directory unless path is specified.
        If suffix is specified it is a string that is appended to the end of the filename.'''

        filename = 'pop'
        
        if path is not None:
            filename = path + '/' + filename

        if suffix is not None:
            filename += suffix

        filename += '.pkl'

        f = open(filename, 'wb')
        pickle.dump(self,f)
        f.close()
