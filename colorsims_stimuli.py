import numpy as np
import random
from os import path


class ColorGrid:
    def __init__(self):
        '''Initialize a 2-dimensional list representing the grid of color chips. The grid is comprised of 8x40 
        distinct color chips. Hue varies across columns and saturation varies across rows. This color grid is 
        modeled after the set of stimuli that was used in the World Color Survey.'''
        
        self.saturation_steps = 8	#i.e. number of rows
        self.distinct_hues = 40		#i.e. number of columns
        self.num_chips = self.saturation_steps * self.distinct_hues

        self.conv_file = open(path.abspath("MunsellCIE.txt"), "r")
        self.LAB_table = self.LAB_lookup(self.conv_file)
        self.LUV_table = self.LUV_lookup(self.LAB_table)
        self.row_convert = {0:'B', 1:'C', 2:'D', 3:'E', 4:'F', 5:'G', 6:'H', 7:'I'}
        
    def sample(self, sets=1, n=2):
        '''Returns an array of chips sampled from the grid.
        Example: To sample 2 chips use sets=1, n=2.
        For efficiency you may want to sample many sets of 2 chips --
        i.e. for 1000 sets of 2 chips use sets=1000, n=2'''
        
        all_chips = []

        for i in range(sets):
            chip_1 = (random.randint(0, 7), random.randint(0, 39))
            chip_2 = (random.randint(0, 7), random.randint(0, 39))
            all_chips.append([chip_1, chip_2])
        
        return all_chips 		
    
    def distance(self, chip_1, chip_2):
        '''Returns the distance between chip 1 and chip 2. Distance is defined as a 2-tuple (d1, d2) where d1 
        represents the absolute value distance in hue from chip 1 and chip 2 and d2 represents the absolute value 
        distance in brightness from chip 1 and chip 2. '''
                        
        return (abs(chip_1[1] - chip_2[1]), abs(chip_1[0] - chip_2[0]))
            
    def city_block_metric(self, chip_1, chip_2):
        '''Returns the distance between chip 1 and chip 2. Distance is defined by adding the number of 
        horizontal steps to the number of vertial steps that it takes to travel from chip 1 to chip 2.'''
        return abs(chip_1[0] - chip_2[0]) + abs(chip_1[1] - chip_2[1])
    
    def LAB_lookup(self, conv_file):
        '''Returns a lookup table that converts chips from row-column form, (R, C), to CIELAB space, (L,a,b)
        (key: (R,C), value: LAB).'''
        
        file_line = conv_file.readline()	#ignore header of the conversion file
        file_line = conv_file.readline()	#read first line of data
        LAB_conv = {}
        
        while file_line != '' and file_line != '\n':
            line_content = file_line.split('\t')

            if line_content[2] == 0:
                pass
            else:
                LAB_conv[(line_content[1], line_content[2])] = (float(line_content[6]), float(line_content[7]), float(line_content[8].replace('\n','')))
            
            file_line = conv_file.readline()
        
        conv_file.close()
        return LAB_conv
    
    #fw, fy, fz, xr, yr, and xr are all helper functions that are used in the calculations used to
    #convert a chip from CIELAB space to CIELUV space. These functions were taken from the ColorCalculator
    #at brucelindblom.com.
    def _fx(self, L, a):
        return (a/500) + self._fy(L)
    
    def _fy(self, L):
        return (L+16)/116
    
    def _fz(self, L, b):
        return self._fy(L) - (b/200)
    
    def _xr(self, epsilon, K, L, a):
        if (self._fx(L, a))**3 > epsilon:
            return (self._fx(L, a))**3
        else:
            return (116*self._fx(L, a) - 16)/K
            
    def _yr(self, epsilon, K, L):
        if L > K*epsilon:
            return ((L+16)/116)**3
        else:
            return L/K
            
    def _zr(self, epsilon, K, L, b):
        if (self._fz(L, b))**3 > epsilon:
            return (self._fz(L, b))**3
        else:
            return (116*self._fz(L, b) - 16)/K
            
    
    def _CIELUV_convert(self, chip):
        '''Returns a chip in the form of (L,u,v), representing its position in CIELUV space. 
        chip is originally given as a 3-tuple (L,a,b).
        '''
        L = chip[0]
        a = chip[1]
        b = chip[2]
        
        e = 0.008856	#Actual CIE Standard
        K = 903.3		#Actual CIE Standard
        
        Xr = 0.9504		#CIE X value for standard white reference D65
        Yr = 1.00		#CIE Y value for standard white reference D65
        Zr = 1.0888		#CIE Z value for standard white reference D65
        
        X = self._xr(e, K, L, a) * Xr
        Y = self._yr(e, K, L) * Yr
        Z = self._zr(e, K, L, b) * Zr
        
        if Y/Yr > e:
            L = 116 * (Y)**(float(1)/3) - 16
        else:
            L = K*Y
                
        u_p = (4*X)/(X + 15*Y + 3*Z)
        v_p = (9*Y)/(X + 15*Y + 3*Z)
        ur_p = (4*Xr)/(Xr + 15*Yr + 3*Zr)
        vr_p = (9*Yr)/(Xr + 15*Yr + 3*Zr)		
        u = 13*L*(u_p - ur_p)
        v = 13*L*(v_p - vr_p)
        
        return (L, u, v)
    
    def LUV_lookup(self, LAB_table):
        '''Returns a lookup table that converts chips from (L,a,b) coordinates to (L,u,v) coordinates, 
        (key: LAB, value: LUV) representing the chips' position in CIELUV space. All chips' (L,a,b) values 
        were taken from another lookup table that converts (row, col) to (L,a,b).'''
        
        LUV_conv = {}
        for chip in LAB_table.values():
            LUV_conv[chip] = self._CIELUV_convert(chip)
        
        return LUV_conv
            
    def Euclid_distance_LUV(self, chip_1, chip_2):
        '''Returns the Euclidean distance between chip 1 and chip 2. chip_1 and chip_2 are 3-tuples 
        representing the colors' positions in CIELUV space, (L,u,v).'''
        
        dL = chip_1[0] - chip_2[0]
        du = chip_1[1] - chip_2[1]
        dv = chip_1[2] - chip_2[2]
        
        return (dL**2 + du**2 + dv**2)**(0.5)
    
    def Euclid_distance(self, chip_1, chip_2):
        '''Returns the Euclidean distance between chip 1 and chip 2. chip_1 and chip_2 are 2-tuples 
        representing the colors' positions in the color grid.'''

        Mun_stim1 = (self.row_convert[chip_1[0]], str(chip_1[1]+1))	#stimulus form: (row, col) 
        Mun_stim2 = (self.row_convert[chip_2[0]], str(chip_2[1]+1))

        LAB_stim1 = self.LAB_table[Mun_stim1]	#stimulus form: (L,a,b)
        LAB_stim2 = self.LAB_table[Mun_stim2]
        
        LUV_stim1 = self.LUV_table[LAB_stim1]	#stimulus form: (L,u,v)
        LUV_stim2 = self.LUV_table[LAB_stim2]
        
        dL = LUV_stim1[0] - LUV_stim2[0]
        du = LUV_stim1[1] - LUV_stim2[1]
        dv = LUV_stim1[2] - LUV_stim2[2]
        
        return (dL**2 + du**2 + dv**2)**(0.5)

    def similarity(self, chip_1, chip_2, c=0.0003):
        '''Computes the similarity between two chips. Similarity is inversely related to Euclidean distance.'''
        
        dist = self.Euclid_distance(chip_1, chip_2)
        return np.exp(-c*(dist**2))
        
		
		

	
	
		
		
		
		
		
