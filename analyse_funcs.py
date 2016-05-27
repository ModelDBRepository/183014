# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 15:39:39 2015

@author: salvadord
"""

import pickle
from pylab import *
import csv

#%% plot filled error bars
def errorfill(x, y, yerr, lw=1, elinewidth=1, color=None, alpha_fill=0.2, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, lw=lw)
    ax.fill_between(x, ymax, ymin, color=color, lw= elinewidth, alpha=alpha_fill)

#%% function to obtain unique list of lists
def uniqueList(seq): 
    seen = {}
    result = []
    indices = []
    for index,item in enumerate(seq):
        marker = tuple(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
        indices.append(index)
    return result,indices
				
#%% function to read data				
def loadData(folder, islands, dataFrom):
	#%% Load data from files
	if islands > 1:
		ind_gens_isl=[] # individuals data for islands
		ind_cands_isl=[]
		ind_fits_isl=[]
		ind_cs_isl=[]
			
		stat_gens_isl=[] # statistics.csv for islands
		stat_worstfits_isl=[]
		stat_bestfits_isl=[]
		stat_avgfits_isl=[]
		stat_stdfits_isl=[]
		
		fits_sort_isl=[] #sorted data
		gens_sort_isl=[] 
		cands_sort_isl=[]
		params_sort_isl=[]
	
	for island in range(islands):
		ind_gens=[] # individuals data
		ind_cands=[]
		ind_fits=[]
		ind_cs=[]
		
		eval_gens=[] # error files for each evaluation
		eval_cands=[]
		eval_fits=[]
		eval_params=[]
		
		stat_gens=[] # statistics.csv 
		stat_worstfits=[]
		stat_bestfits=[]
		stat_avgfits=[]
		stat_stdfits=[]
		
		if islands > 0:
			folderFinal = folder+"_island_"+str(island)
		else: 
			folderFinal = folder
			
		with open('data/%s/individuals.csv'% (folderFinal)) as f: # read individuals.csv
			reader=csv.reader(f)
			for row in reader:
				ind_gens.append(int(row[0]))
				ind_cands.append(int(row[1]))
				ind_fits.append(float(row[2]))
				cs = [float(row[i].replace("[","").replace("]","")) for i in range(3,len(row))]
				ind_cs.append(cs)
		
		with open('data/%s/statistics.csv'% (folderFinal)) as f: # read statistics.csv
			reader=csv.reader(f)
			for row in reader:
				stat_gens.append(float(row[0]))
				stat_worstfits.append(float(row[2]))
				stat_bestfits.append(float(row[3]))
				stat_avgfits.append(float(row[4]))
				stat_stdfits.append(float(row[6]))
		
		# unique generation number (sometimes repeated due to rerunning in hpc)
		stat_gens, stat_gens_indices = unique(stat_gens,1) # unique individuals
		stat_worstfits, stat_bestfits, stat_avgfits, stat_stdfits = zip(*[[stat_worstfits[i], stat_bestfits[i], stat_avgfits[i], stat_stdfits[i]] for i in stat_gens_indices])
		
		if dataFrom == 'fitness':		
			for igen in range(max(ind_gens)): # read error files from evaluations
				print 'Loading data from island %d, generation %d'%(island,igen)
				for ican in range(max(ind_cands)):
					try:
						f=open('data/%s/gen_%d_cand_%d_errortmp'%(folderFinal, igen,ican)); 
						eval_fits.append(pickle.load(f))
						f=open('data/%s/gen_%d_cand_%d_params'%(folderFinal, igen,ican)); 
						eval_params.append(pickle.load(f))
						eval_gens.append(igen)
			        		eval_cands.append(ican)
					except:
			                 pass
						#eval_fits.append(0.15)
						#eval_params.append([])
		
		# find x corresponding to smallest error from function evaluations	
		if dataFrom == 'fitness':
			#fits_sort, fits_sort_indices, fits_sort_origind = unique(eval_fits, True, True)
			fits_sort_indices = sorted(range(len(eval_fits)), key=lambda k: eval_fits[k])
			fits_sort = [eval_fits[i] for i in fits_sort_indices]
			gens_sort = [eval_gens[i] for i in fits_sort_indices]
			cands_sort = [eval_cands[i] for i in fits_sort_indices]
			params_sort = [eval_params[i] for i in fits_sort_indices]
		# find x corresponding to smallest error from individuals file
		elif dataFrom == 'individuals':
			params_unique, unique_indices = uniqueList(ind_cs) # unique individuals
			fits_unique = [ind_fits[i] for i in unique_indices]
			gens_unique = [ind_gens[i] for i in unique_indices]
			cands_unique = [ind_cands[i] for i in unique_indices]
			
			sort_indices = sorted(range(len(fits_unique)), key=lambda k: fits_unique[k]) # sort fits
			fits_sort = [fits_unique[i] for i in sort_indices]
			gens_sort = [gens_unique[i] for i in sort_indices]
			cands_sort = [cands_unique[i] for i in sort_indices]
			params_sort = [params_unique[i] for i in sort_indices]
		
		# if multiple islands, save data for each
		if islands > 1:
			ind_gens_isl.append(ind_gens) # individuals data for islands
			ind_cands_isl.append(ind_cands)
			ind_fits_isl.append(ind_fits)
			ind_cs_isl.append(ind_cs)
				
			stat_gens_isl.append(stat_gens) # statistics.csv for islands
			stat_worstfits_isl.append(stat_worstfits)
			stat_bestfits_isl.append(stat_bestfits)
			stat_avgfits_isl.append(stat_avgfits)
			stat_stdfits_isl.append(stat_stdfits)
			
			fits_sort_isl.append(fits_sort) #sorted data
			gens_sort_isl.append(gens_sort) 
			cands_sort_isl.append(cands_sort)
			params_sort_isl.append(params_sort)
			
	if islands > 1:
		return ind_gens_isl, ind_cands_isl, ind_fits_isl, ind_cs_isl, stat_gens_isl, \
			stat_worstfits_isl, stat_bestfits_isl, stat_avgfits_isl, stat_stdfits_isl, \
			fits_sort_isl, gens_sort_isl, cands_sort_isl, params_sort_isl
