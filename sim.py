# sim.py -- Python script for running the simulation interactively
#
# Usage:
#   ipython --pylab -i sim.py
# 
# Last update: 13/09/10 (salvadord)

#
# Global parameters
#

# Use the NEURON GUI?
use_NEURON_GUI = True

# Index to grvec panel object for latest loaded sim.
curr_grvec_pind = 0

# for ipython debugging
from IPython.core.debugger import Tracer; debug_here = Tracer()

##############
# Misc functions
##############

def ctypandind2cind (ctyp,ctind):
   cell_inds = find(cell_types == get_ctyp_num(ctyp))
   return cell_inds[ctind]

def cind2ctypandind (cell_num):
   # Get the cell type number.
   ctypnum = cell_types[cell_num]

   # Get the first cell index for that type.
   ctypfind = find(cell_types == ctypnum)[0]

   # Show the cell type string and the relative cell index.
   return (get_ctyp_str(ctypnum), cell_num - ctypfind)

######################
# Data analysis functions
######################

def get_ctyp_fire_rate (ctyp,min_t=0.0,max_t=-1.0,allcellrates=False):
   if (max_t == -1.0):
      max_t = sim_duration
   tvec = nqscol2narr('spknq','t')
   idvec = nqscol2narr('spknq','id')
   tvec,idvec = get_vec_subset(tvec,idvec,mintvec=min_t,maxtvec=max_t)
   freqs = get_ave_fire_freqs(tvec,idvec,num_cells,max_t - min_t)
   cell_inds = find(cell_types == get_ctyp_num(ctyp))
   if (allcellrates):
      return freqs[cell_inds]
   else:
      return freqs[cell_inds].mean()

# get_curr_fire_rate() -- get an estimate of a firing rate for a target cell at a 
#    particular time.  By default, the counting window is 100 ms and starts at time t.
#    Requires that snq table is loaded.
def get_curr_fire_rate (targtype='',targind=0,t=0.0,spkwindwid=100.0,spkwindoffset=0.0):
   # Remember the old snq table verbosity.
   oldverbose = h.spknq.verbose

   # Turn off the table verbosity.
   h.snq.verbose = 0

   # Default the cell index to targind.
   cell_ind = targind

   # If we have a cell type, get the cell IDs and get the indexed value.
   if (targtype != ''):
      targ_cells = find(cell_types == get_ctyp_num(targtype)) 
      cell_ind = targ_cells[targind]     
   
   # Get number all of the desired cell spikes in the desired time window.
   num_spks = h.spknq.select('id',cell_ind,'t','[]',t+spkwindoffset, \
      t+spkwindoffset+spkwindwid)

   # Get the firing rate from the spike count and window width.
   fire_rate = num_spks * (1000.0 / spkwindwid)

   # Reset the snq table.
   h.spknq.tog()
   h.spknq.verbose = oldverbose

   return fire_rate

###########################
# Analysis functions by salvadord
##########################
   
def DPparams():
	for o,i in zip(h.cedp, range(0, int(h.cedp.count()))): print o.id, o.interval, o.number, o.start, o.noise, o.mlenmin, o.mlenmax, o.zloc

def DPspikes(t1,t2):
	h.snq[0].select("type",h.DP,"t","()",t1,t2)
	#h('snq[0].select("type",DP)')
	h.snq[0].pr()

#def CSTIMparams():
	#h.ls = h.LEMNoise()
	#for o,i in zip (h.lcstim.o(0).nsl, range(0, int(h.lcstim.o(0).nsl.count()))): print i, o.interval, o.number, o.start, o.noise, h('nqE.v(1).x(%d)' % i)#, h('nqE.v(3).x(%d)' % i), h('lcstim.o(0).ncl.o(%d).weight[4]'%i), h('lcstim.o(0).ncl.o(%d).weight[5]'%i),h('lcstim.o(0).ncl.o(i).weight[2]'%i)
	
def EMStimParams():
	h('objref lemlist')
	h('lemlist = lem.o(0)')
	for o,i in zip (h.lemlist, range(0, int(h.lemlist.count()))): print i, o.interval, o.number, o.start, o.noise, h('nqE.v(1).x(%d)' % i), h('nqE.v(3).x(%d)' % i), h('lcstim.o(0).ncl.o(%d).weight[4]' % i), h('lcstim.o(0).ncl.o(%d).weight[5]' % i),h('lcstim.o(0).ncl.o(%d).weight[2]' % i)

def plotTraj():
	{
	 h('{drxytraj() g=new Graph() g.size(40,tstop,-1,4) drshouldertrajectory() drelbowtrajectory()}')
	}
###################
# Simulation functions
##################

# Run the simulation and save the spikes.
def runsim ():
   h.run()
   h.skipsnq = 0   # flag to create NQS with spike times, one per column
   h.initAllMyNQs()  # setup of NQS objects with spike/other information
   #h('if (spknq != nil) nqsdel(spknq)')
   #h('spknq = new NQS()')
   #h('spknq.copy(snq[0])')


#
# Script code (run always)
#

# Load the sys and os packages.
import sys, os

# Load the interpreter object.
from neuron import h

# Load the NEURON GUI.
if use_NEURON_GUI:
   from neuron import gui

# Load default parameters and initialize the network.
h.xopen("main.hoc")

# Load functions for interfacing with hoc data structures.
from hocinterface import *

# Load the neuroplot namespace.
from neuroplot import *

# Load the numpy namespace.
from numpy import *

# Load analysis
from analysis import *


# Set up cellsnq and connsnq for display functions. 
h('objref cellsnq')
h('cellsnq = col[0].cellsnq')
h('objref connsnq')
h('connsnq = col[0].connsnq')

# Set up hoc objects for more NQS tables
h('objref spknq')

# Remember the simulation duration (in ms).
sim_duration = h.mytstop	


