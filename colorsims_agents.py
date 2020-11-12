import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy import stats
import colorsims_utils
from os import path
import pandas as pd


class RLAgent:

    def __init__(self, color_grid, lang, p_id, theta=0, num_centroids_per_cat=1, fill_method="prob"):
        '''Initializes a Reinforcement Learning Agent representing the participant p_id from lang.'''
        
        self.color_grid = color_grid
        self.lang_num = lang
        self.p_id = p_id
        self.theta = theta  #probability that chip needs to be named at initialization (theta=0 --> Em Hyp; theta=1 --> Par Hyp)
        self.num_centroids_per_cat = num_centroids_per_cat  #number of centroids to initialize in each category

        raw_wcs_data = np.array(pd.read_csv(path.abspath(path.join("WCSParticipantData", "Single Participant Matrices", "Lang"+str(self.lang_num), "Lang"+str(self.lang_num)+"Participant"+str(self.p_id)+".csv")), header=None))
        self.num_words = np.shape(raw_wcs_data)[0]        
        self.wcs_word_map = np.reshape(raw_wcs_data.argmax(axis=0), (8,40))  #WCS word map for participant
        self.bcts = self._get_bcts()
        self.k_sim = colorsims_utils.find_ksim(len(self.bcts))
        
        self.reinforcement_units = self.num_words * 4
        self.reinforcement_delta = 1
        self.unnorm_naming_strategy = np.zeros([self.num_words, self.color_grid.num_chips], dtype=int)
        self._init_naming_matrix(fill_method)

    def _get_bcts(self):
        '''Returns a list of the BCTs in self.lang_num.'''

        bct_path = path.abspath("all_bcts.txt")
        bct_df = pd.read_table(bct_path)

        bcts = bct_df["Term Num"][bct_df["Lang Num"]==self.lang_num] - 1    #-1 to correct for indexing

        return bcts

    def _init_naming_matrix(self, fill_method="prob"):
        '''Initialize the agent's initial probability matrix. Assigns probability 1 to centroid chips and their associated name
        given by the identified agent (via lang_num and agent_id) in WCS data and 0 to every other cell in the matrix.
        fill_method = "wcs" or "prob"; "wcs" will fill in non-centroid chips using the terms WCS participants gave, "prob" will
        assign names to non-centroid chips by taking a probabilistic draw based on the distance of the chip from the initialized centroids.'''
        
        #First, initialize the centroid colors.
        self.get_centroids()

        init_index = []     #list of chip indices which have already been initialized (i.e. the indices of the centroids)

        for i in self.centroids:
            for f in self.centroids[i]:
                f_index = f[0]*40 + f[1]    #chip number of the centroid (between 0 and 319)
                self.unnorm_naming_strategy[i, f_index] = self.reinforcement_units
                init_index.append(f_index)

        #Next, determine whether all other chips should be given a name. If they are given a name, the name is chosen according to the chip's distance from the centroids.
        chip_indices = np.delete(np.arange(320), init_index)

        for chip in chip_indices:
            give_name = np.random.binomial(1, self.theta)    #coin flip to decide whether chip should be named

            if give_name:   #name given will be a function of the distance of chip to the centroids   
                chosen_name = self._get_chip_name((chip//40, chip%40), fill_method)
                self.unnorm_naming_strategy[chosen_name, chip] = self.reinforcement_units

        #return self.unnorm_naming_strategy

    def naming_strategy(self):
        return self.unnorm_naming_strategy / float(self.reinforcement_units)

    def _get_chip_name(self, chip, fill_method="prob"):
        '''Performs a probabilistic draw based on the distance between chip and the current defined centroids. The result of the draw determines the name the agent will 
        use for chip. chip is given in the form (row, col).
        fill_method = "wcs" or "prob"; "wcs" will fill in non-centroid chips using the terms WCS participants gave, "prob" will
        assign names to non-centroid chips by taking a probabilistic draw based on the distance of the chip from the initialized centroids.'''

        if fill_method == "prob":
            term_order = sorted(self.centroids.keys())     #sorted to make sure the probabilities align with the right terms
            centroids_list = np.array([y for x in term_order for y in self.centroids[x]])   
            all_sims = np.array([self.color_grid.similarity(centroid, (chip[0], chip[1])) for centroid in centroids_list])  #probabilistic draw taken over "similarity" values, which are a function of the Euclidean distance between chip and the centroids    

            xk = sorted(term_order*self.num_centroids_per_cat) 
            pk = [float(i)/sum(all_sims) for i in all_sims]  #turn similarity measures into a probability distribution
            sim_rv = stats.rv_discrete(name='sim_rv', values=(np.arange(len(xk)), pk))
            chosen_name = xk[sim_rv.rvs()]
        else:
            chosen_name = self.wcs_word_map[chip[0], chip[1]]
        
        return chosen_name

    def _get_category_centroid(self, state, cat_name):
        '''Identifies most centroid chip(s) for a given category. Number of chips to identify per category is given by self.num_centroids_per_cat.'''

        #Gets all the chips given name cat_name by the agent
        if state == "wcs":
            word_map_flat = self.wcs_word_map.flatten()
        else: #state == "current"
            word_map_flat = self.word_map().flatten()
        chips_in_cat = np.argwhere(word_map_flat == cat_name)[:,0]   

        #Get boundary values for all chips in the category
        bound_vals = np.array([self._bound_val((chip//40, chip%40)) for chip in chips_in_cat])     #boundary values of all chips in category cat_name (see Gooyabadi & Joe 2019 for formula)

        #Get "centroid" chips, i.e. chips in category with highest boundary values
        zipped = np.column_stack((chips_in_cat, bound_vals))  #match the chip indicies with their corresponding boundary values
        zipped = zipped[np.argsort(zipped[:,-1])]    #sort the array according to boundary values
        centroid_index = zipped[-1*self.num_centroids_per_cat:][:,0]  #get the chips with highest boundary values (number of chips is determined by self.num_centroids_per_cat)
        centroids = [(int(i//40), int(i%40)) for i in centroid_index]  

        try:  
            if len(chips_in_cat) < self.num_centroids_per_cat:  #if there are fewer chips in cat_name than the number of centroids, then repeat one of the centroids to fill the gap
                for i in range(self.num_centroids_per_cat - len(chips_in_cat)):
                    centroids.append(centroids[-1])       
        except:     #should only get here if centroids=[] (i.e. participant didn't use one of the BCTs)
            pass

        return centroids

    def get_centroids(self):
        '''Get centroids for all BCT categories.
        Returns a dictionary where the keys are the category names and the values are the list of centroid chips for the corresponding category.'''

        self.centroids = {w:self._get_category_centroid("wcs", w) for w in self.bcts if w in self.wcs_word_map}   
        # return self.centroids

    def _bound_val(self, chip):
        '''Returns the boundary value of a chip. Chip is given in the form (row, col).'''

        chip_index = chip[0]*40 + chip[1]

        word_map_flat = self.wcs_word_map.flatten()
        stim_within_ksim = word_map_flat[np.array([self.color_grid.Euclid_distance(chip, (i//40, i%40)) for i in np.arange(320)]) <= self.k_sim]
        same_name = np.sum(stim_within_ksim == word_map_flat[chip_index]) - 1   #-1 is to take away the extra count given for a chip being named the same thing as itself

        try:
            prop_same_name = float(same_name) / len(stim_within_ksim)
        except ZeroDivisionError:
            prop_same_name = 0

        return prop_same_name

    def response(self, stimulus):
        '''Returns the name of stimulus.
        Assumes stimulus is an 2-tuple of integer values, (hue, saturation), that is an index of the color grid.
        self.centroids is a dictionary where key=stimuli position and value=stimulus name'''

        stim_index = 40*stimulus[0] + stimulus[1]
        p = self.unnorm_naming_strategy[:,stim_index]    #column of probability matrix corresponding to stimulus

        if np.count_nonzero(p) == 0:
            #No idea what to name stimulus --> pure probabilistic draw based on distance from foci
            chosen_name = self._get_chip_name(stimulus)
            return chosen_name

        else:
            #Check if there are any nonzero entries in column of probability matrix corresponding to stimulus.
            #Any probability mass not reflected in this column becomes the probability that the agent "assigns no name" to stimulus.
            #When an agent chooses to "assign no name", they take a probabilistic draw over similarity values to the centroids.
            nonzero_vals = p[p > 0]
            sum_vals = np.sum(nonzero_vals, dtype=int)
            prob_no_name = (self.reinforcement_units - sum_vals) / float(self.reinforcement_units)

            xk = np.append(np.arange(len(p))[p > 0], -1)    #indices of p with nonzero probability plus -1 for "assigns no name"
            pk = np.append(nonzero_vals / float(self.reinforcement_units), prob_no_name)
            choice_rv = stats.rv_discrete(name='choice_rv', values=(xk, pk))

            #Choice will either be an index representing an existing word with a nonzero entry or -1 to represent "unsure what to name chip"
            choice = choice_rv.rvs()    

            if choice != -1:
                return choice
            else:
                chosen_name = self._get_chip_name(stimulus)
                return chosen_name
                

    def reinforce(self, stimulus, response, blocked_responses=[]):
        '''Reinforces the stimulus-response pair. (stimulus is a chip represented by (i,j))
        Decrements a randomly chosen alternative response to offset the reward and conserve the number
        of reinforcement units for the stimulus.
        The reinforced response and any responses specified in blocked_responses list are not used
        as the decremented response.

        Returns the response that was decremented to offset reinforcement or None if nothing changed.'''

        stimulus_index = 40*stimulus[0] + stimulus[1]

        # Check if the stimulus already has max reinforecement units
        if self.unnorm_naming_strategy[response, stimulus_index] < self.reinforcement_units:
            # Doesn't have the max yet so go ahead and reinforce.

            # Figure out the maximum amount of probability that can be added
            delta = self.reinforcement_units - self.unnorm_naming_strategy[response, stimulus_index]
            
            # Reinforcement amount will be the equal to reinforcement_delta or however many units need
            # to be added to get to the max for this stimulus -- whichever is lower.
            delta = min(delta, self.reinforcement_delta)

            if np.sum(self.unnorm_naming_strategy[:, stimulus_index]) < self.reinforcement_units:
                # Do increment for reinforced response
                self.unnorm_naming_strategy[response, stimulus_index] += delta
                return None
            else:
                # Find the responses that can be decremented.
                valid_responses_to_decrement = self.unnorm_naming_strategy[:, stimulus_index] >= delta
                # Can't use the response that is being reinforced
                valid_responses_to_decrement[response] = False
                #Can't use any of the blocked responses
                valid_responses_to_decrement[blocked_responses] = False

                if not np.any(valid_responses_to_decrement):
                    # No responses can be decremented so can't proceed
                    return None

                # Choose one of the valid responses to decrement
                response_to_decrement = np.random.choice(np.arange(self.num_words)[valid_responses_to_decrement])
                # Do the decrement
                self.unnorm_naming_strategy[response_to_decrement, stimulus_index] -= delta

                # Do increment for reinforced response
                self.unnorm_naming_strategy[response, stimulus_index] += delta

        else:
            # Already at the max reinforcement units for this stimulus.
            return None

        return response_to_decrement

    def punish(self, stimulus, response, blocked_responses=[], response_to_learn=None):
        '''Punishes the stimulus-response pair.  (stimulus is a chip represented by (i,j))
        Increments a randomly chosen alternative response to offset the punishment and conserve the number
        of reinforcement units for the stimulus.
        The reinforced response and any responses specified in blocked_responses list are not used
        as the incremented response.

        Returns the response that was incremented to offset punishment or None if nothing changed.

        If response_to_learn pre-specifies which alternative response should be incremented to offset the punishment.
        This is used when an agent is "learning" a response from another agent, for example. '''

        stimulus_index = 40*stimulus[0] + stimulus[1]

        # Check if the stimulus already has zero probability
        if self.unnorm_naming_strategy[response, stimulus_index] > 0:
            # Doesn't have the min yet so go ahead and punish.

            # Figure out the maximum number of reinforcement units that can be subtracted
            delta = self.unnorm_naming_strategy[response, stimulus_index]
            
            # Punishment amount will be the equal to reinforcement_delta or however many units need
            # to be subtracted to get to the min for this stimulus -- whichever is lower.
            delta = min(delta, self.reinforcement_delta)

            # Is there a pre-specified response that should be incremented?
            if response_to_learn is not None:
                response_to_increment = response_to_learn

            else:
                #No pre-specified response. Choose a random one.

                # Find the responses that can be incremented.
                valid_responses_to_increment = self.unnorm_naming_strategy[:, stimulus_index] <= (self.reinforcement_units - delta)
                # Can't use the response that is being punished
                valid_responses_to_increment[response] = False
                #Can't use any of the blocked responses
                valid_responses_to_increment[blocked_responses] = False
                
                if not np.any(valid_responses_to_increment):
                    # No responses can be incremented so can't proceed
                    return None

                # Choose one of the valid responses to increment
                response_to_increment = np.random.choice(np.arange(self.num_words)[valid_responses_to_increment])

            # Do the increment
            self.unnorm_naming_strategy[response_to_increment, stimulus_index] += delta

            # Do decrement for punished response
            self.unnorm_naming_strategy[response, stimulus_index] -= delta

##        elif self.unnorm_naming_strategy[response, stimulus_index] == 0 and np.sum(self.unnorm_naming_strategy[:, stimulus_index]) == 0:
##            delta = self.reinforcement_delta
##            
##            # Already at zero probability for this stimulus, but whole column is zero, so pick random chip to increment.
##            response_to_increment = np.random.choice(np.arange(self.num_words))
##            self.unnorm_naming_strategy[response_to_increment, stimulus_index] += delta

        else:
            # Already at zero probability.
            return None

        return response_to_increment

    def visualize_text_matrix(self):
        '''Returns a text representation of the naming strategy where the word (row)
        with highest probability in each column is a "1" and everything else is "0".'''
        
        m = self.naming_strategy()
        return (m == m.max(axis=0)).astype(int)

    def word_map(self):
        '''Returns the index of the most probable word for each stimulus (color chip).
        If two or more words are tied for highest probability, the word with the lowest row index is used.'''

        word_map = np.full(320, -1, dtype=int)     #This is a 1x320 array filled with -1 (-1 means agent has no name for the chip yet)
        for i in range(320):
            col = self.naming_strategy()[:,i]
            if max(col) >= 0.5:     #A chip is vizualized to have a name when the probability of assigning at least one name is greater than 50%
                word_map[i] = np.where(col==max(col))[0][0]
        word_map = word_map.reshape([8,40])     #This is an 8x40 matrix with entries being the chosen word for each stimulus.
        
        return word_map	

    def plot_word_map(self, ax=None, filename=None, show=False):
        '''Plots the agent's naming strategy using a heat map, where each of the colors in the map is representative of one of the words 
        in the agent's vocabulary. The map shows what each chip's most probable name is at any instance in the simulation.
        Pass filename to save file.
        Pass optional matplotlib axis object (ax) in order to have plot drawn on the axis.
        gray: prints word map in gray scale if True, in color if False'''
        
        if ax is None:
            fig, ax = plt.subplots()

        #Plot the color map
        #List of 90 colors (list of 30 x3) First color needs to be white to represent "unsure name" space
        all_colors = ['#ffffff','#8000ff', '#00ff00', '#0040ff', '#cc4400', '#0080ff', '#ff0040', '#ffff00', '#a67c00', '#bf9b30', '#5e3c58', '#bf00ff', '#00ffbf', '#ff0080', 
'#ffdbac', '#000000', '#ff8000', '#d4d4d4', '#555555', '#aafd96', '#00bfff', '#ff00ff', '#ff93ac', '#ffbf00', '#400000', '#204c39', '#997a8d', '#063b79', 
'#757906', '#70330b', '#00ffff', '#8000ff', '#00ff00', '#0040ff', '#cc4400', '#0080ff', '#ff0040', '#ffff00', '#a67c00', '#bf9b30', '#5e3c58', '#bf00ff', 
'#00ffbf', '#ff0080', '#ffdbac', '#000000', '#ff8000', '#d4d4d4', '#555555', '#aafd96', '#00bfff', '#ff00ff', '#ff93ac', '#ffbf00', '#400000', '#204c39', 
'#997a8d', '#063b79', '#757906', '#70330b', '#00ffff', '#8000ff', '#00ff00', '#0040ff', '#cc4400', '#0080ff', '#ff0040', '#ffff00', '#a67c00', '#bf9b30', 
'#5e3c58', '#bf00ff', '#00ffbf', '#ff0080', '#ffdbac', '#000000', '#ff8000', '#d4d4d4', '#555555', '#aafd96', '#00bfff', '#ff00ff', '#ff93ac', '#ffbf00', 
'#400000', '#204c39', '#997a8d', '#063b79', '#757906', '#70330b', '#00ffff']
        cmap = colors.ListedColormap(all_colors)
        bounds = np.arange(-1,len(all_colors))
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax.matshow(self.word_map(), cmap=cmap, norm=norm)
            
        #Remove axis ticks and labels
        ax.set_yticks([])
        ax.set_yticklabels([])

        ax.set_xticks([])
        ax.set_xticklabels([])               

        #Save to file?
        if filename is not None:
            plt.savefig(filename)

        if show:
            plt.show()
		

		
		
		
		
		
		
		
		
		
		
		
		
		

