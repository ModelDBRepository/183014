# analysis.py -- Python script for running the simulation interactively
#
# Usage:
#   ipython --pylab -i sim.py
# 
# Last update: 13/09/10 (salvadord)

#
# Global parameters
#

# Load the sys and os packages.
import sys, os
from neuron import h # Load the interpreter object.
from numpy import *
from pylab import *
#from pylab import figure, array, size, zeros, subplot, ion, show, pause, hold, sum
from os import system,listdir
from copy import deepcopy
import scipy.io
import numpy as np
import pickle
from bicolormap import bicolormap
import time
import analyse_funcs
    
# for ipython debugging
#from IPython.core.debugger import Tracer; debug_here = Tracer()


# show graphs with batch simulations error results 
def batchErrors(simdatadir, param1Arg=None):
    # Set up the simulation data directory.
    #simdatadir = 'data/13sep06_sim1'

    invert_axis = 1;
    
    # seed values
    wseedvals =[120456, 398115]#, 534031, 796321, 895199]
    iseedvals = [1235, 2837]#, 3955, 4506, 6789]
    
    # parameter values
    #arange(10,110,10)# #[0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]#[10,20,30,40,50,60,70,80,90,100]#[50, 100,250,500,750,1000]#arange(20,180,20)#[25, 50,75, 100, 125, 150, 175, 200, 225, 250]#
    if param1Arg==None:
        param1_range =arange(0.8,1.24,0.04)#arange(1,9)# [0,1]#arange(50,550,50)
    else:
        param1_range=param1Arg
    param2_range = [0,1,2,3,4,5,6,7]
    
    # number of errror values = 6 = (ang vs xy) x (10%, 50%, 90%)
    err_vals = 6;
    
    # create array for all results (2 errors, 10 train times, 5 wseeds, 5 inseed, 4 error values)
    error_all = zeros((len(param1_range),len(param2_range),len(wseedvals),len(iseedvals),err_vals))
    
    # create array for results avg'd over seeds (2 errors, 10 train times, 4 error values)
    error_avg = zeros((len(param1_range),len(param2_range),err_vals))
    error_std = zeros((len(param1_range),len(param2_range),err_vals))
    error_p1_avg = zeros((len(param1_range),err_vals))
    
    # Loop over param1 values
    iparam1 = -1
    for param1 in param1_range:
        iparam1 = iparam1 + 1
        iparam2 = -1
        skipValue = 0
        # Loop over param2 values
        for param2 in param2_range:
            iparam2 = iparam2 + 1
            iwseed = -1
            # Loop over wiring seed...
            for wseed in wseedvals:
                iwseed = iwseed + 1
                iiseed = -1
                # Loop over input seed...
                for iseed in iseedvals:
                    iiseed = iiseed + 1
                # set filename
                outfilestem = '%s/p1-%d_p2-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, iseed, wseed)
                if os.path.isfile(outfilestem):
                    print "loading "+str(outfilestem)
                else:
                    outfilestem = '%s/p1-%.2f_p2-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, iseed, wseed)
                    if os.path.isfile(outfilestem):
                        print "loading "+str(outfilestem)
                    else: 
                        outfilestem = '%s/p1-%.3f_p2-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, iseed, wseed)
                        if os.path.isfile(outfilestem):
                            print "loading "+str(outfilestem)
                        else:
                            skipValue=1

                # get errors from nqs
                if skipValue==0:
                    tmp = h.calcErrorFromNQ(outfilestem).to_python()
                    tmp =array(tmp)
                    error_all[iparam1, iparam2, iwseed, iiseed, :] = tmp
                #print tmp
            # calculate avg error over seeds
            if skipValue==0:
                for i in arange(err_vals):
                    error_avg[iparam1, iparam2, i] = mean(error_all[iparam1, iparam2, :, :, i])
                    error_std[iparam1, iparam2, i]= std(error_all[iparam1, iparam2, :, :, i])
    
        # calculate average over all targets (param2) for each value of param1
        if skipValue==0:
            for i in arange(err_vals):
                error_p1_avg[iparam1, i] = mean(error_avg[iparam1,:,i])
                
    # plot results    
    figsize=[800,500]
    fig = figure(figsize=(figsize[0]/100,figsize[1]/100),dpi=100)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    param_xaxis = 'param2'
    #debug_here()
    
    if invert_axis:
        param_xaxis='param1'
        tmp=param1_range
        param1_range = param2_range
        param2_range = tmp
        error_avg = swapaxes(error_avg, 0, 1)
        
    xaxis=array(param2_range)
    colorlist = ['black','darkgrey','blue', 'cyan', 'green', 'brown', 'red', 'magenta', 'orange','yellow']
    
    # plot cartesian error
    iparam1=-1
    for param1 in param1_range:
        iparam1 = iparam1 + 1
        #ax1.errorbar(xaxis, error_avg[iparam1,:,0], yerr=error_std[0,:,0], fmt='s', color = colorlist[iparam1%10],)
        #ax1.errorbar(xaxis, error_avg[iparam1,:,1], yerr=error_std[0,:,1], fmt='o', color = colorlist[iparam1%10], )
        #ax1.errorbar(xaxis, error_avg[iparam1,:,2], yerr=error_std[0,:,2], fmt='x', color = colorlist[iparam1%10], )
        #ax1.plot(xaxis, error_avg[iparam1,:,0], color = colorlist[iparam1%10], label=str(param1)+', last 10%')
        ax1.plot(xaxis, error_avg[iparam1,:,0], linestyle = "--",color = colorlist[iparam1%10],label=str(param1)+', last 20%')
        ax1.plot(xaxis, error_avg[iparam1,:,2], linestyle = "-", color = colorlist[iparam1%10], label=str(param1)+', last 90%')
    ax1.plot(xaxis, error_p1_avg[:,0], linestyle = "--",color = 'black',linewidth=4, label='AVG - 20%')
    ax1.plot(xaxis, error_p1_avg[:,2], linestyle = "-", color = 'black', linewidth=4, label='AVG - 90%')
    
    ax1.set_ylabel('cartesian error (m)')
    ax1.set_xlabel(param_xaxis)
    ax1.set_title('Cartesian error as a func of param1 and param2')
    #ax1.legend(loc='upper center', bbox_to_anchor=(1, 0.2),  borderaxespad=0., prop={'size':12})
    ax1.grid(True)
    
    # plot angular error
    iparam1=-1
    for param1 in param1_range:
        iparam1 = iparam1 + 1
        #ax2.errorbar(xaxis, error_avg[iparam1,:,3], yerr=error_std[0,:,3], fmt='s', color = colorlist[iparam1%10],)
        #ax2.errorbar(xaxis, error_avg[iparam1,:,4], yerr=error_std[0,:,4], fmt='o', color = colorlist[iparam1%10],)
        #ax2.errorbar(xaxis, error_avg[iparam1,:,5], yerr=error_std[0,:,5], fmt='x', color = colorlist[iparam1%10], )
        #ax2.plot(xaxis, error_avg[iparam1,:,3], color = colorlist[iparam1%10], label=str(param1)+', last 10%')
        ax2.plot(xaxis, error_avg[iparam1,:,3], linestyle = "--", color = colorlist[iparam1%10], label=str(param1)+', last 20%')
        ax2.plot(xaxis, error_avg[iparam1,:,5], linestyle = "-",color = colorlist[iparam1%10], label=str(param1)+', last 90%')
    ax2.plot(xaxis, error_p1_avg[:,3], linestyle = "--",color = 'black',linewidth=4, label='AVG - last 20%')
    ax2.plot(xaxis, error_p1_avg[:,5], linestyle = "-", color = 'black', linewidth=4, label=' AVG - last 90%')
        
    ax2.set_ylabel('angular error (rad)')
    ax2.set_xlabel(param_xaxis)
    ax2.set_title('Angular error as a func of param1 and param2')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 2.0),  borderaxespad=0., prop={'size':12})
    ax2.grid(True)

    fig.tight_layout()    
    show()

# show the receptive field of a neuron or group of neurons (targetCell); 
# nSyn = number of synaptic connections to take into account; eg. if 1: ES->EM; if 2, DP->ES->EM
# animate = include synaptic changes over time; make graph

# show error graphs for 8 sims of same day
def batchErrors8(simdatadirRoot):
    numSims = 8

    param1_range =  []#[None] * numSims
    param1_range.append([10,20,30,40,50])
    param1_range.append([10,20,30,40,50,60,70,80,90,100])
    param1_range.append(arange(0.8,1.24,0.04))#([0.012, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15])
    param1_range.append([10,20,30,40,50,60,70,80,90,100])
    param1_range.append([25, 50,75, 100, 125, 150, 175, 200, 225, 250])
    param1_range.append([20,40,60,80,100,120,140,160])
    param1_range.append([200,225,250,275,300,325,350])
    param1_range.append([50, 100,250,500,750,1000])
    
    for i in range(numSims):
        batchErrors(simdatadirRoot+'_sim'+str(i+1), param1_range[i])
    
# show the RF of targetCells
# if nSyn2: use polysynaptic chains of length =2
# if animate: show RF change over time, overimposed with spikes
def receptiveFieldAndSpikes(targetCells, nSyn2, animate, dosave):    
    #dosave=1<
    if dosave:
        moviedir='data/movies/'
        moviename='13oct15b.mpg'
        maxmovieframes = 500 # Maximum number of movie frames
    view3d=0
    showSpikes=1
    
    print "Calculating receptive field of cell group starting at id %d " % targetCells[0]
    targetCells = array(targetCells)
    popnames=['P','ES','IS','ILS','EM','IM','IML'];
    popinds=[ 2, 41, 42, 43, 44, 45, 46];
    popshape=['o',  '^',  'h', '*',  '^',  'h',  '*', '8'];
    
    # convert connectivity to python arrays
    conpreid=array(h.col.connsnq.getcol("id1"), 'i')
    conpostid=array(h.col.connsnq.getcol("id2"), 'i')
    #condelay=array(h.col.connsnq.getcol("del"))
    #condistance=array(h.col.connsnq.getcol("dist"))
    conweight1=array(h.col.connsnq.getcol("wt1"))
    conweight2=array(h.col.connsnq.getcol("wt2"))
    
    order = lexsort((conpreid, conpostid)) # order by post and then pre
    conpreid = conpreid[order]
    conpostid = conpostid[order]
    #condelay = condelay[order]
    #condistance = condistance[order]
    conweight1 = conweight1[order]
    conweight2 = conweight2[order]
    
    maxW = max(conweight1.max(), conweight2.max())
    conweight1 = conweight1/maxW    # normalize so can multiply together
    conweight2 = conweight2/maxW # 
    
    conweight1Original = deepcopy(conweight1) # make copy of weights at t=0
    
    # convert locations to python arrays
    h('objref cellList')
    h('cellList=col.ce')
    h('objref cellListDP') 
    h('cellListDP = cedp')
    n=h.cellList.count() # Number of cells
    ndp=h.cellListDP.count() # number of DP cells
    #cellLocations = zeros((n+ndp,5)) # number of cells and attributes
    cellids =zeros(n+ndp)
    celltypes = zeros(n+ndp)
    xlocs =zeros(n+ndp)
    ylocs =zeros(n+ndp)
    zlocs =zeros(n+ndp)
    
    for i in range(int(n)): # Loop over each cell
        cellids[i]=i # Cell ID
        celltypes[i]=h.cellList.o(i).type() # Cell population
        xlocs[i]=h.cellList.o(i).xloc # X position
        ylocs[i]=h.cellList.o(i).yloc # Y position
        zlocs[i]=h.cellList.o(i).zloc # Z position

    for i in range(int(ndp)): # Loop over each cell
        cellids[n+i]=n+i # Cell ID
        celltypes[n+i]=h.cellListDP.o(i).type # Cell population
        xlocs[n+i]=h.cellListDP.o(i).xloc # X position
        ylocs[n+i]=h.cellListDP.o(i).yloc # Y position
        zlocs[n+i]=h.cellListDP.o(i).zloc # Z position
    
    # arrange neurons in spatial grid and according to muscle group
    spatialGrid = 1
    if spatialGrid:
        x = [0, 0, 0, 0, 0] # initial starting positions
        y = [0, 0, 0, 0, 0]        
        xstep = 1 # step increase
        ystep = 1.5
        xmax = 10
        for i in range(len(xlocs)):
            if celltypes[i] == popinds[1]:    #ES
                xmax = 24
                xoffset = [0, 0, 0, 0, 0]
                yoffset = [0, 0, 0, 0, 14]
                zlocs[i] = 0 # fix to same muscle group
            elif celltypes[i] == popinds[2]: #IS
                zlocs[i] = 4 # do not divide in muscle groups
            elif celltypes[i] == popinds[3]:    #ISL
                zlocs[i] = 4 # do not divide in muscle groups
            elif celltypes[i] == popinds[4]:    #EM
                xoffset = [0, 14, 0, 14, 5] #offset for different muscle populations + inhibitory cells
                yoffset = [0, 0, 14, 14, 7]
                xmax = 12
            elif celltypes[i] == popinds[5]: #IM
                zlocs[i] = 4 # do not divide in muscle groups
                xmax = 16
            elif celltypes[i] == popinds[6]:    #IML
                zlocs[i] = 4 # do not divide in muscle groups
                xmax = 16
            elif celltypes[i] == popinds[0]: #P
                xmax = 12
                xoffset = [0, 14, 0, 14, 3] #offset for different muscle populations + inhibitory cells
                yoffset = [0, 0, 12, 12, 9]    
            
            zloc = int(zlocs[i])
            x[zloc] = x[zloc] + xstep # increase step

            if x[zloc] > xmax: # if reached x limit
                y[zloc] = y[zloc] + ystep
                x[zloc] = 1
            
            xlocs[i] = x[zloc] + xoffset[zloc] # set positions
            ylocs[i] = y[zloc] + yoffset[zloc]
            
            if (i==(192+44+20-1) or (i==(2*(192+44+20)-1))): # when change subplot, restart xy positions
                x = [0, 0, 0, 0, 0] # initial starting positions
                y = [0, 0, 0, 0, 0]
                
    ########################
    # animation data and params
    if animate:
        # convert synaptic changes over time (since connsnq doesn't change with t, can add)
        h('objref synChanges') 
        #h('nqsy.tog()')
        h('nqsy.select("t",">=", 0)')
        h('synChanges = nqsy')
        synpreid=array(h.synChanges.getcol("id1"), 'i')
        synpostid=array(h.synChanges.getcol("id2"), 'i')
        synweight=array(h.synChanges.getcol("wg"))
        syntime=array(h.synChanges.getcol("t"))

        order = lexsort((synpreid, synpostid)) # order syn by post and then pre
        synpreid = synpreid[order]
        synpostid = synpostid[order]
        synweight=synweight[order]        
        syntime=syntime[order]    
        
        # convert spiking data to python arrays
        h('objref spikes')
        h('snq.select("t",">=", 0)')
        h('spikes = snq')
        spikeId=array(h.spikes.getcol("id"))
        spikeType=array(h.spikes.getcol("type"))
        spikeTime=array(h.spikes.getcol("t"))
        spikeMid=array(h.spikes.getcol("mid"))
        
        weightInc = h.plastEEinc / maxW # set weight increases and normalize
    
    def makeplot():
        # visualization options
        figsize = [720,800] # Figure size in pixels
        targetColor = 'green';#array([(1,0.4,0) , (0,0.2,0.8)]) # Define excitatory and inhibitory colors -- orange and turquoise
        normalColor = 'white'#'lightyellow'
        targetSizeFactor = 2
        weightsCmap = 'YlOrRd' #'jet'#'hot'#'autumn'
        cellSize = 50;
        cellBorder = 1;
        
        # create subplots for M, S and P populations
        ion()
        fig = figure(figsize=(figsize[0]/100,figsize[1]/100),dpi=100)
        fig.subplots_adjust(left=0.02) # Less space on left
        fig.subplots_adjust(right=0.93) # Less space on right
        fig.subplots_adjust(bottom=0.08) # Less space on bottom
        #fig.subplots_adjust(wspace=0.25) # More space between
        #fig.subplots_adjust(hspace=0.30) # More space between
        if view3d: 
            proj='3d'
        else:
            proj='rectilinear'
            
        rasterMsubplot = subplot(311, projection=proj) # create subplots
        rasterSsubplot = subplot(312, projection=proj)
        rasterPsubplot = subplot(313, projection=proj)
        
        rasterMsubplot.set_title('Motor population', fontsize=12) # titles
        rasterSsubplot.set_title('Somatosensory population', fontsize=12)
        rasterPsubplot.set_title('Proprioceptive population', fontsize=12)
        
        rasterMsubplot.text(-0.7, 1, "sho\next", fontsize=10, fontweight='bold') # muscle labels M
        rasterMsubplot.text(26.6, 1, "sho\nflex", fontsize=10, fontweight='bold')
        rasterMsubplot.text(-0.7, 15, "elb\next", fontsize=10, fontweight='bold')
        rasterMsubplot.text(26.6, 15, "elb\nflex", fontsize=10, fontweight='bold')
        
        rasterPsubplot.text(-0.7, 1, "sho\next", fontsize=10, fontweight='bold') # muscle labels P
        rasterPsubplot.text(26.6, 1, "sho\nflex", fontsize=10, fontweight='bold')
        rasterPsubplot.text(-0.7, 13, "elb\next", fontsize=10, fontweight='bold')
        rasterPsubplot.text(26.6, 13, "elb\nflex", fontsize=10, fontweight='bold')
         
        setp(rasterMsubplot.get_xticklabels(), visible=False) # hide x and y ticks
        setp(rasterSsubplot.get_xticklabels(), visible=False)
        setp(rasterPsubplot.get_xticklabels(), visible=False)
        setp(rasterMsubplot.get_yticklabels(), visible=False)
        setp(rasterSsubplot.get_yticklabels(), visible=False)
        setp(rasterPsubplot.get_yticklabels(), visible=False)
        
        border = 2
        rasterMsubplot.set_xlim([xlocs.min()-border, xlocs.max()+border]) # set x-y lims
        rasterMsubplot.set_ylim([ylocs.min()-border, ylocs.max()+border])
        rasterSsubplot.set_xlim([xlocs.min()-border, xlocs.max()+border])
        rasterSsubplot.set_ylim([ylocs.min()-border, ylocs.max()+border])
        rasterPsubplot.set_xlim([xlocs.min()-border, xlocs.max()+border])
        rasterPsubplot.set_ylim([ylocs.min()-border, ylocs.max()+border])
        
        ######################
        # plot cells in gray (static)
        cellsScatter = [None]*len(popinds);
        for p in range(len(popinds)):
            #cells=where(cellLocations[:,1]==popinds[p])
            cells=(celltypes==popinds[p])
            if view3d:
                if (p == 0):
                    rasterPsubplot.scatter(xlocs[cells], ylocs[cells], zlocs[cells], c=normalColor, marker=popshape[p], linewidth=cellBorder, s=cellSize) 
                elif (p>=1 and p<=3):
                    rasterSsubplot.scatter(xlocs[cells], ylocs[cells], zlocs[cells], c=normalColor, marker=popshape[p], linewidth=cellBorder, s=cellSize) 
                elif (p>=4 and p<=6):
                    rasterMsubplot.scatter(xlocs[cells], ylocs[cells], zlocs[cells], c=normalColor, marker=popshape[p], linewidth=cellBorder, s=cellSize) 
            else:
                if (p == 0):
                    cellsScatter[p] = rasterPsubplot.scatter(xlocs[cells], ylocs[cells],  c=normalColor, marker=popshape[p], linewidth=cellBorder, s=cellSize, label=popnames[p]) 
                elif (p>=1 and p<=3):
                    cellsScatter[p] = rasterSsubplot.scatter(xlocs[cells], ylocs[cells],  c=normalColor, marker=popshape[p], linewidth=cellBorder, s=cellSize, label=popnames[p]) 
                elif (p>=4 and p<=6):
                    cellsScatter[p] = rasterMsubplot.scatter(xlocs[cells], ylocs[cells],  c=normalColor, marker=popshape[p], linewidth=cellBorder, s=cellSize, label=popnames[p]) 
            
        #############################
        # show target cells in different color
        targetLabel = 'target cell'
        for i in range(len(targetCells)):
            for p in range(len(popinds)):
                # find target cells for each layer
                #targetLayerCells = intersect1d(targetCells, where(cellLocations[:,1]==popinds[p])[0])
                targetLayerCells = (cellids==targetCells[i]) * (celltypes==popinds[p]) # obtain indices for population p
                if targetLayerCells.any(): # it at least one cell was found
                    if (p == 0):
                        rasterPsubplot.scatter(xlocs[targetLayerCells], ylocs[targetLayerCells], c=targetColor, marker=popshape[7], linewidth=cellBorder*2, s=cellSize*targetSizeFactor) 
                    elif (p>=1 and p<=3):
                        rasterSsubplot.scatter(xlocs[targetLayerCells], ylocs[targetLayerCells], c=targetColor, marker=popshape[7],  linewidth=cellBorder*2, s=cellSize*targetSizeFactor) 
                    elif (p>=4 and p<=6):
                        if i == 0: targetPlot =rasterMsubplot.scatter(xlocs[targetLayerCells], ylocs[targetLayerCells], c=targetColor, marker=popshape[7], linewidth=cellBorder*2, s=cellSize*targetSizeFactor, label = targetLabel) 
                        else: rasterMsubplot.scatter(xlocs[targetLayerCells], ylocs[targetLayerCells], c=targetColor, marker=popshape[7], linewidth=cellBorder*2, s=cellSize*targetSizeFactor) 
                    
        legendLabels=['E', 'I', 'ILS', 'P', 'target']
        rasterSsubplot.legend([cellsScatter[1], cellsScatter[2],  cellsScatter[3], cellsScatter[0], targetPlot], legendLabels, columnspacing=0,markerscale=0.8, loc='upper center', prop={'size':9}, scatterpoints=1, bbox_to_anchor=(0.94, 1.05),  borderpad=0.07)#, borderpad=0.05, labelspacing=0.1)
        
        ######################
        # show afferent cells weights (RF)
        rf = [None]*len(popinds);
        
        def calculateRF(mode,rf):
            onlyAMPA = 1
            adaptScale = 0
            scaleMax =[0.5,1,1] 
            # find all cells projecting to the target cell for each layer
            afferentCells = array([], 'i')
            weightColors = []
            for i in range(len(targetCells)):
                targetAfferents = array(conpreid[where(conpostid==targetCells[i])], 'i') # get afferent of next target cell
                if targetAfferents.any(): # if any 
                    for iAfferent in targetAfferents: # for each new afferent 
                        if (iAfferent in afferentCells): # check if already included in list
                            if onlyAMPA: weightColors2 = (conweight1[iAfferent])
                            else: weightColors2 = ((conweight1[iAfferent]) +(conweight2[iAfferent]))
                            weightColors[afferentCells==iAfferent] +=  weightColors2
                        else: # if not included in list
                            afferentCells=append(afferentCells, iAfferent) # add to list of afferent cells
                            if onlyAMPA: weightColors = append(weightColors, conweight1[targetAfferents]); # calculate weigths
                            else: weightColors = append(weightColors, conweight1[afferentCells]-conweight2[afferentCells]); # calculate weigths
                    
                    if nSyn2: # 2 synaptic connections
                        for targetCell2 in targetAfferents:  # loop through all afferent cells (which now become target cells)
                            targetAfferents2 = array(conpreid[where(conpostid==targetCell2)], 'i') # get new set of afferent cells
                            if targetAfferents2.any():
                                for iAfferent in targetAfferents2: # loop through all afferent cells
                                    if (iAfferent in afferentCells): # if already included in list
                                        if onlyAMPA: weightColors2 = (conweight1[targetCell2]) * (conweight1[iAfferent])
                                        else: weightColors2 = (conweight1[targetCell2]) * (conweight1[iAfferent])
                                        weightColors[afferentCells==iAfferent] +=  weightColors2
                                    else:
                                        afferentCells=append(afferentCells, iAfferent) # add to list of afferent cells
                                        if onlyAMPA: weightColors2 = (conweight1[targetCell2]) * (conweight1[iAfferent])
                                        else: weightColors2 = (conweight1[targetCell2]+conweight2[targetCell2]) * (conweight1[iAfferent]+conweight2[iAfferent])
                                        weightColors=append(weightColors, weightColors2)
                                            
            weightColors = weightColors/weightColors.max() # normalize weights to 1
            #print afferentCells
                    
            for p in range(len(popinds)):
                # find all cells projecting to the target cell for each layer
                afferentCellsL = intersect1d(afferentCells, where(celltypes==popinds[p])[0]) # find indices for population p
                #print afferentCellsL
                if afferentCellsL.any():
                    weightColorsL = weightColors[in1d(afferentCells, afferentCellsL)]; # obtain weight colors for this population
                    #print weightColorsL
                    if mode == "new":
                        if (p == 0):
                            rf[p] = rasterPsubplot.scatter(xlocs[afferentCellsL], ylocs[afferentCellsL], cmap=weightsCmap, c=weightColorsL, vmin=0, vmax=scaleMax[0], marker=popshape[p], linewidth=cellBorder, s=cellSize) 
                            if adaptScale and ('c1' in locals() or 'c1' in globals()):
                                rf[p].set_clim([weightColorsL.min(), weightColorsL.max()])
                                c1.set_clim([weightColorsL.min(), weightColorsL.max()])
                                c1.draw_all()
                        elif (p>=1 and p<=3):
                            rf[p] = rasterSsubplot.scatter(xlocs[afferentCellsL], ylocs[afferentCellsL], cmap=weightsCmap,  c=weightColorsL, vmin=0, vmax=scaleMax[1], marker=popshape[p], linewidth=cellBorder, s=cellSize) 
                            if adaptScale and ('c2' in locals() or 'c2' in globals()):
                                rf[p].set_clim([weightColorsL.min(), weightColorsL.max()])
                                c2.set_clim([weightColorsL.min(), weightColorsL.max()])
                                c2.draw_all()
                        elif (p>=4 and p<=6):
                            rf[p] = rasterMsubplot.scatter(xlocs[afferentCellsL], ylocs[afferentCellsL], cmap=weightsCmap,  c=weightColorsL,vmin=0, vmax=scaleMax[0], marker=popshape[p], linewidth=cellBorder, s=cellSize) 
                            if adaptScale and ('c3' in locals() or 'c1' in globals()):
                                rf[p].set_clim([weightColorsL.min(), weightColorsL.max()])
                                c3.set_clim([weightColorsL.min(), weightColorsL.max()])
                                c3.draw_all()
                    elif mode == "update":
                            rf[p].set_color(weightColorsL)
                            rf[p].set_cmap(weightsCmap)
                            
                    #print afferentCells
            
        calculateRF("new",rf)
        tight_layout()
        
        cfraction = 0.09
        cpad = 0.01
        cshrink = 0.75
        cfontsize=9
        if rf[0] != None: 
            c1=colorbar(rf[0], ax=rasterPsubplot, fraction=cfraction, pad=cpad, shrink=cshrink)
            c1.ax.tick_params(labelsize=cfontsize) 
        c2=colorbar(rf[1], ax=rasterSsubplot, fraction=cfraction, pad=cpad, shrink=cshrink)
        c2.set_label('normalized weight to target cell', fontsize=cfontsize+1)
        c2.ax.tick_params(labelsize=cfontsize) 
        c3=colorbar(rf[4], ax=rasterMsubplot,  fraction=cfraction, pad=cpad, shrink=cshrink)
        c3.ax.tick_params(labelsize=cfontsize) 
        
        ########################################
        # ANIMATE SPIKES AND SYNAPTIC CHANGES
        if animate:
            # spike parameters
            raster = [None]*len(popinds);
            spikeColor = 'chartreuse'#'DarkMagenta'#'lime'#'purple'
            spikeSizeFactor = 1.5

            maxframes = 1000 # Maximum number of non-movie frames
            binsize = 50 # Bin size in ms
            ncells = len(xlocs)
            xmax = round(xlocs.max())
            ymax = round(xlocs.max())
            tmax = spikeTime.max()
            totalspikes = len(spikeTime)

            nbins = int(tmax/binsize)+1 # Calculate number of bins; +1 for edge effects
            spikecounts=zeros((ncells,nbins)) # Number of spike counts
            timebins = array(range(nbins))*binsize/1e3 # Calculate time bins
            spikebins = array(spikeTime/binsize,dtype=int) # Convert times to bins

            for s in range(totalspikes): # Loop over each spike
                spikecounts[spikeId[s],spikebins[s]] += 1 # Increment this bin
            spikingrate = sum(spikecounts,axis=0)*1e3/binsize/ncells # Get the total spike counts over time
            
            maxspikes = 5*sum(spikecounts,axis=0).mean() # Find out the maximum number of spikes in any given timestep
            maxiters = int(min(maxframes,maxmovieframes,len(timebins))) if dosave else int(min(maxframes,len(timebins))) # See how many iterations to go for

            # plot spikes and syn changes for every frame
            for b in range(maxiters): # for each bin
                
                rasterMsubplot.set_title('Motor population, t = %0.3f s' % timebins[b],  fontsize=12)
                rasterSsubplot.set_title('Somatosensory population, t = %0.3f s' % timebins[b], fontsize=12)
                rasterPsubplot.set_title('Proprioceptive population = %0.3f s' % timebins[b], fontsize=12)
                
                #############################
                # show spikes
                if showSpikes:
                    fired = array(spikecounts[:,b]>0).flatten() # Find which neurons fired in this timestep
                    print('  bin = %i of %i; %i spikes' % (b, maxiters, sum(fired)))
                    numfired = sum(fired) # Total number of cells that fired
                    
                    for p in range(len(popinds)):
                        firedL = fired * (celltypes==popinds[p]) # calculate cells that fired for population p
                        #print firedL
                        if (p == 0):
                            raster[p] = rasterPsubplot.scatter(xlocs[firedL], ylocs[firedL],  c=spikeColor, marker=popshape[p], linewidth=cellBorder-1, s=cellSize*spikeSizeFactor) 
                        elif (p>=1 and p<=3):
                            raster[p] = rasterSsubplot.scatter(xlocs[firedL], ylocs[firedL],  c=spikeColor, marker=popshape[p], linewidth=cellBorder-1, s=cellSize*spikeSizeFactor) 
                        elif (p>=4 and p<=6):
                            raster[p] = rasterMsubplot.scatter(xlocs[firedL], ylocs[firedL],  c=spikeColor, marker=popshape[p], linewidth=cellBorder-1, s=cellSize*spikeSizeFactor) 
                    
                    show()
                else:
                    print('  bin = %i of %i' % (b, maxiters))
                    
                ############################
                # show changes in synaptic weights
                synBin = where(syntime == b*binsize)[0]
                if (size(synBin)>0 and size(synBin)<size(conweight1)):
                    # update connection matrix based on syn wieght gains
                    conweight1[range(len(synBin))] = conweight1Original[range(len(synBin))] * synweight[synBin]
                    
                    for p in range(len(popinds)): 
                        if rf[p] != None: rf[p].remove() # remove previous scatter    
                    calculateRF("new",rf) # plot new RF

                if dosave: fig.savefig('tmpmovie%04i.png' % b) # Save PNG files for movie
                
                if dosave!=1: pause(1e-10) 
                if showSpikes: 
                    for p in range(len(popinds)): raster[p].remove() # remove previous scatter
                if dosave!=1: pause(1e-10)
    
    makeplot()

    if dosave:
        print('Making movie...')
        system('mkdir -p ' + moviedir)
        system("mencoder 'mf://tmpmovie*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %s" % (moviedir+moviename))
        print('  Cleaning up...')
        system('rm tmpmovie*.png')

# save potentiated/depressed synapses at each time step (for video)
def saveRLsyn(t1,t2):
    # convert synaptic changes over time (since connsnq doesn't change with t, can add)
        h('objref synChanges') 
        #h('nqsy.tog()')
        #h('nqsy.select("t",">", 0)')
        h('nqsy.select("t","[]", %d, %d)'%(t1,t2))
        h('synChanges = nqsy')
        synpreid=array(h.synChanges.getcol("id1"), 'i')
        synpostid=array(h.synChanges.getcol("id2"), 'i')
        synweight=array(h.synChanges.getcol("wg"))
        syntime=array(h.synChanges.getcol("t"))
        
        savetime=h.syDT
        prevWeight=synweight[syntime==t1]
        if t1<50:
            prevWeight=prevWeight[:len(prevWeight)/2]
            t1=2*savetime
        else: 
            t1=t1+savetime
        synChanged=[0,0,0,0]
        bins=0
        
        for t in arange(t1, syntime.max(), savetime):
            print t
            bins=bins+1
            # get data for this time
            currsynpreid = synpreid[syntime==t] 
            currsynpostid = synpostid[syntime==t]
            currsyntime = syntime[syntime==t]
            
            # find and save connections that have changed
            currWeight = synweight[syntime==t]
            diffWeight = currWeight-prevWeight
            changedWeight = where(diffWeight != 0)[0]
            # CHECK IF CAN CHANGE LINE BELOW TO AVOID FOR!
            #for i in range(len(changedWeight)):
                #synChanged = vstack((synChanged, [currsyntime[changedWeight[i]], currsynpreid[changedWeight[i]], currsynpostid[changedWeight[i]], diffWeight[changedWeight[i]]]))
            synChanged = vstack((synChanged, array([currsyntime[changedWeight], currsynpreid[changedWeight], currsynpostid[changedWeight], diffWeight[changedWeight]]).T))
            
            prevWeight=currWeight
        
        scipy.io.savemat('arch3d/synChanged.mat', mdict={'synChanged': synChanged})

# Calculate average weight of all incoming connections (2nd order) to each EM muscle subpopulation over time
# Useful to analyse the effect that RL has on the weights as a function of  - WORK IN PROGRESS--
def synWeightsEM():
    
    print "Calculating average weight to each EM muscle subpopulation"
    
    EMstart=256 # starting cell index of EM population
    EMlength=192 # number of EM cells
    EMsubpopo = [None]*4
    for i in range(4): EMsubpop[i] = [arange(EMstart+i,EMstart+EMlength,4)] # create array with list of ids for each EM cell subpopulation  
    popnames=['P','ES','IS','ILS','EM','IM','IML'];
    popinds=[ 2, 41, 42, 43, 44, 45, 46];
    popshape=['o',  '^',  'h', '*',  '^',  'h',  '*', '8'];
    
    # convert connectivity to python arrays
    conpreid=array(h.col.connsnq.getcol("id1"), 'i')
    conpostid=array(h.col.connsnq.getcol("id2"), 'i')
    #condelay=array(h.col.connsnq.getcol("del"))
    #condistance=array(h.col.connsnq.getcol("dist"))
    conweight1=array(h.col.connsnq.getcol("wt1"))
    conweight2=array(h.col.connsnq.getcol("wt2"))
    
    order = lexsort((conpreid, conpostid)) # order by post and then pre
    conpreid = conpreid[order]
    conpostid = conpostid[order]
    #condelay = condelay[order]
    #condistance = condistance[order]
    conweight1 = conweight1[order]
    conweight2 = conweight2[order]
    
    maxW = max(conweight1.max(), conweight2.max())
    conweight1 = conweight1/maxW    # normalize so can multiply together
    conweight2 = conweight2/maxW 
    
    # convert synaptic changes over time (since connsnq doesn't change with t, can add)
    h('objref synChanges') 
    #h('nqsy.tog()')
    h('nqsy.select("t",">=", 0)')
    h('synChanges = nqsy')
    synpreid=array(h.synChanges.getcol("id1"), 'i')
    synpostid=array(h.synChanges.getcol("id2"), 'i')
    synweight=array(h.synChanges.getcol("wg"))
    syntime=array(h.synChanges.getcol("t"))
    
    order = lexsort((synpreid, synpostid)) # order syn by post and then pre
    synpreid = synpreid[order]
    synpostid = synpostid[order]
    synweight=synweight[order]        
    syntime=syntime[order]    
    
    # define t
    tMax = syntime.max()
    tstep = 50
    T = arange(0, tMax, tstep)
    
    # create array to store average weight change for each subpopulation
    weightSubpops = array((T.len(), 4))        
    
    def showRF(targetCells):
        onlyAMPA = 1
        # find all cells projecting to the target cell for each layer
        afferentCells = array([], 'i')
        weightColors = []
        for i in range(len(targetCells)):
            targetAfferents = array(conpreid[where(conpostid==targetCells[i])], 'i') # get afferent of next target cell
            if targetAfferents.any(): # if any 
                for iAfferent in targetAfferents: # for each new afferent 
                    if (iAfferent in afferentCells): # check if already included in list
                        if onlyAMPA: weightColors2 = (conweight1[targetCell2]) * (conweight1[iAfferent])
                        else: weightColors2 = (conweight1[targetCell2]) * (conweight1[iAfferent])
                        weightColors[afferentCells==iAfferent] +=  weightColors2
                    else: # if not included in list
                        afferentCells=append(afferentCells, iAfferent) # add to list of afferent cells
                        if onlyAMPA: weightColors = append(weightColors, conweight1[targetAfferents]); # calculate weigths
                        else: weightColors = append(weightColors, conweight1[afferentCells]-conweight2[afferentCells]); # calculate weigths
                
                if nSyn2: # 2 synaptic connections
                    for targetCell2 in targetAfferents:  # loop through all afferent cells (which now become target cells)
                        targetAfferents2 = array(conpreid[where(conpostid==targetCell2)], 'i') # get new set of afferent cells
                        if targetAfferents2.any():
                            for iAfferent in targetAfferents2: # loop through all afferent cells
                                if (iAfferent in afferentCells): # if already included in list
                                    if onlyAMPA: weightColors2 = (conweight1[targetCell2]) * (conweight1[iAfferent])
                                    else: weightColors2 = (conweight1[targetCell2]) * (conweight1[iAfferent])
                                    weightColors[afferentCells==iAfferent] +=  weightColors2
                                else:
                                    afferentCells=append(afferentCells, iAfferent) # add to list of afferent cells
                                    if onlyAMPA: weightColors2 = (conweight1[targetCell2]) * (conweight1[iAfferent])
                                    else: weightColors2 = (conweight1[targetCell2]+conweight2[targetCell2]) * (conweight1[iAfferent]+conweight2[iAfferent])
                                    weightColors=append(weightColors, weightColors2)
                
                #weightColors = weightColors/weightColors.max() # normalize weights to 1
                weight = weightColors.sum()
                return weight
    

    for b in T:
        # calculate sum of weight for each subpopulation
        
        for i in range(4):
            targetCells = EMsubpop[i]
            weight=calculateRF(targetCells) # plot new RF
            weightSubpops[b, i] = weight
            
            # update weights 
            synBin = where(syntime == b )[0]
            if (size(synBin)>0 and size(synBin)<size(conweight1)):
                # update connection matrix based on syn wieght gains
                conweight1[range(len(synBin))] = conweight1[range(len(synBin))] * synweight[synBin]
                                            
    figure()
    plot(T, weightSubpops)
    show()

# plot raster of saved sim
def raster(nqsdir, nqsparams, tstop=4):
    wseedvals =[120456, 398115, 534031, 796321, 895199] # seed values for filename
    iseedvals = [1235, 2837, 3955, 4506, 6789]
    # load connectivity file
    outfilestem = '"%s/p1-%.2f_p2-%.2f_p3-%.2f_p4-%.2f_p5-%d_i-%d_w-%d_test-spk.nqs"' % (nqsdir, nqsparams[0], nqsparams[1], nqsparams[2], nqsparams[3], nqsparams[4], iseedvals[nqsparams[5]], wseedvals[nqsparams[6]])
    filename = '%s/p1-%.2f_p2-%.2f_p3-%.2f_p4-%.2f_p5-%d_i-%d_w-%d_test-spk.nqs' % (nqsdir, nqsparams[0], nqsparams[1], nqsparams[2], nqsparams[3], nqsparams[4], iseedvals[nqsparams[5]], wseedvals[nqsparams[6]])

    if os.path.isfile(filename):
        print "loading "+outfilestem
        # get errors from nqs
        h('objref nqaload')
        h('nqaload = new NQS(%s)'%outfilestem)
        h('snq[0]=nqaload')
        h.tstop=tstop;
        h('{gg() drit(0, tstop) grlines()}')
    else:
        print "file not found: "+outfilestem

# plot motor spike rate for each muscls subpopulation, from saved sim
def plotvEM(nqsdir, nqsparams,xmin=4,xmax=400, ymax=60):
    wseedvals =[120456, 398115, 534031, 796321, 895199] # seed values for filename
    iseedvals = [1235, 2837, 3955, 4506, 6789]
    # load connectivity file
    outfilestem = '"%s/p1-%.2f_p2-%.2f_p3-%.2f_p4-%.2f_p5-%d_i-%d_w-%d_test-nqaupd.nqs"' % (nqsdir, nqsparams[0], nqsparams[1], nqsparams[2], nqsparams[3], nqsparams[4], iseedvals[nqsparams[5]], wseedvals[nqsparams[6]])
    filename = '%s/p1-%.2f_p2-%.2f_p3-%.2f_p4-%.2f_p5-%d_i-%d_w-%d_test-nqaupd.nqs' % (nqsdir, nqsparams[0], nqsparams[1], nqsparams[2], nqsparams[3], nqsparams[4], iseedvals[nqsparams[5]], wseedvals[nqsparams[6]])

    if os.path.isfile(filename):
        print "loading "+outfilestem
        # get errors from nqs
        h('objref nqaload')
        h('nqaload = new NQS(%s)'%outfilestem)
        shext=array(h.nqaload.getcol("shext"))
        shflex=array(h.nqaload.getcol("shflex"))
        elext=array(h.nqaload.getcol("elext"))
        elflex=array(h.nqaload.getcol("elflex"))

        figsize=[800,500]
        fig = figure(figsize=(figsize[0]/100,figsize[1]/100),dpi=100)
        colorlist = ['black','darkgrey','blue', 'cyan', 'green', 'brown', 'red', 'magenta', 'orange','yellow']
        #plot(shext,color = colorlist[3],label='shext')
        #plot(shflex,color = colorlist[5],label='shflex')
        #plot(elext,color = colorlist[6],label='elext')
        #plot(elflex,color = colorlist[9],label='elflex')
        x=arange(xmin,xmax)
        width=1
        bar(x,shext[xmin:xmax],width, color = colorlist[2],label='shext')
        bar(x,shflex[xmin:xmax],width, bottom=shext[xmin:xmax],color = colorlist[4],label='shflex')
        bar(x,elext[xmin:xmax],width, bottom=shflex[xmin:xmax]+shext[xmin:xmax],color = colorlist[6],label='elext')
        bar(x,elflex[xmin:xmax],width,bottom=elext[xmin:xmax]+shflex[xmin:xmax]+shext[xmin:xmax], color = colorlist[9],label='elflex')

        #xlim([xmin, xmax])
        ylim([0, ymax])
        legend(loc='upper right')

    else:
        print "file not found: "+outfilestem

# Show graphs of weight vs. time for each cell subpopulation (subdivided into muscle groups eg. Pshext->ES, Pshflex->ES)
# One subplot for each layer/pop - 1) Pshflex->,Pshext->, ... 2) ES-> 3) IS, ISL, 4) EMshflex->, EMshext->, 5) IM, IML 
def popWeights(loadnqs=0, nqsfile = "", nqsdir="", nqsparams=[0,0,0,0,0,0,0], savefig=0, savenpy=0, loadnpy=0, animate=0, saveanim=0, tinterval=0):
    
    def index2d(myList, v):
        for i, x in enumerate(myList):
            if v in x:
                return (i)
        
    def calculateWeightMatrix():
        print "calculating final weight matrix..."
        #weightMatrix = ndarray((sum(popslen), sum(popslen)))        
        w, h = len(p), len(p)
        weightMatrix = [[0] * w for i in range(h)]
        onlyweight1 = 1
        for i in range(len(synBin)):
            if (conweight1[i]>0):# || conweight2[i]>0):
                if onlyweight1: weightMatrix[index2d(subpops,conpreid[i])][index2d(subpops,conpostid[i])] += conweight1[i]
                else: weightMatrix[index2d(subpops,conpreid[i])][index2d(subpops,conpostid[i])]  += conweight1[i] + conweight2[i]
        weightMatrix=array(weightMatrix)
        for i in range(len(subpops)):
            for j in range(len(subpops)):
                weightMatrix[i][j]=weightMatrix[i][j]/(len(subpops[i])*len(subpops[j]))
        return weightMatrix
        
    # alternative method under construction
    def calculateWeightMatrix2():
        #weightMatrix = ndarray((sum(popslen), sum(popslen)))        
        w, h = len(p), len(p)
        weightMatrix = [[0] * w for i in range(h)]
        onlyAMPA = 1
        for pre in range(len(subpops)):
            for post in range(len(subpops)):
                indicespre=[]
                indicespost=[]
                for i in conpreid: indicespre.append(i in subpops[pre])
                for i in conpostid: indicespost.append(i in subpops[post])
                indices = indicespre * indicespost
                if (conweight1[i]>0):# || conweight2[i]>0):
                    if onlyAMPA: weightMatrix[pre][post] += conweight1[indices].sum()
                    else: weightMatrix[pre][post]  += conweight1[indices].sum() + conweight2[indices].sum()
        return weightMatrix

    # visualization options
    figsize = [1000,800] # Figure size in pixels
    weightsCmap = 'hot_r'# 'jet'#'YlOrRd' #'jet'#'hot'#'autumn'

    # create figures
    ion()
    if animate: # only create figure 1 (evolution over time) if aniamte=1
        fig = figure(figsize=(figsize[0]/100,figsize[1]/100),dpi=100)
        #fig.subplots_adjust(left=0.02) # Less space on left
        #fig.subplots_adjust(right=0.93) # Less space on right
        fig.subplots_adjust(bottom=0.08) # Less space on bottom    
        subplot1 = subplot(221) # create subplots
        subplot2 = subplot(222)
        subplot3 = subplot(223)
        subplot4 = subplot(224)

    #border = 2
    #weightplot.set_xlim([xlocs.min()-border, xlocs.max()+border]) # set x-y lims
    
    # cell populations parameters
    popslen = [192, 44, 20, 192, 44, 20,192]
    #popnames=['ES','IS','ILS','EM','IM','IML', 'P'];
    popinds=[41, 42, 43, 44, 45, 46, 2];
    popshape=['o',  '^',  'h', '*',  '^',  'h',  '*', '8'];
    
    # cell subpopulations (incuding muscle groups)
    subpops= [None]*13
    p = {'Pse':0, 'Psf':1,'Pee':2,'P':3,'ES':4,'IS':5,'ILS':6,'EMse':7,'EMsf':8,'EMee':9,'EMef':10,'IM':11,'IML':12}
    p2 = {0:'Pse', 1:'Psf',2:'Pee',3:'Pef',4:'ES',5:'IS',6:'ILS',7:'EMse',8:'EMsf',9:'EMee',10:'EMef',11:'IM',12:'IML'}
    
    cellslist=[] # list of cells with muscle groups together
    popstart=0
    index=0
    for i in range(len(popslen)):
        popend = popstart+popslen[i]
        if i==3 or i==6: # for P and EM divide into 4 groups 
            for j in range(4): 
                subpops[index] = list(arange(popstart+j,popend,4)) # create array with list of ids for each cell subpopulation  
                index+=1
        else:
            subpops[index]=list(arange(popstart,popend))
            index+=1
        popstart+=popslen[i] # increase popstart
    
    # reorder to have P first
    neworder=[9,10,11,12,0,1,2,3,4,5,6,7,8]
    subpops = [ subpops[i] for i in neworder]

    # load data from neuron variables
    if (not loadnpy):
        wseedvals =[120456, 398115, 534031, 796321, 895199] # seed values for filename
        iseedvals = [1235, 2837, 3955, 4506, 6789]
        if loadnqs: # load data from nqs saved file
            print "loading data from nqs files..."
            # load connectivity file
            if (nqsfile !=""):
                outfilestem = '"%s-con.nqs"' % (nqsfile)
                filename = '%s-con.nqs' % (nqsfile) 
            else:
                outfilestem = '"%s/p1-%.2f_p2-%.2f_p3-%.2f_p4-%.2f_p5-%d_i-%d_w-%d_test-con.nqs"' % (nqsdir, nqsparams[0], nqsparams[1], nqsparams[2], nqsparams[3], nqsparams[4], iseedvals[nqsparams[5]], wseedvals[nqsparams[6]])
                filename = '%s/p1-%.2f_p2-%.2f_p3-%.2f_p4-%.2f_p5-%d_i-%d_w-%d_test-con.nqs' % (nqsdir, nqsparams[0], nqsparams[1], nqsparams[2], nqsparams[3], nqsparams[4], iseedvals[nqsparams[5]], wseedvals[nqsparams[6]])

            if os.path.isfile(filename):
                print "loading "+outfilestem
                # get errors from nqs
                h('objref nqaload')
                h('nqaload = new NQS(%s)'%outfilestem)
                # convert connectivity to python arrays
                conpreid=array(h.nqaload.getcol("id1"), 'i')
                conpostid=array(h.nqaload.getcol("id2"), 'i')
                #condelay=array(h.col.connsnq.getcol("del"))
                #condistance=array(h.col.connsnq.getcol("dist"))
                conweight1=array(h.nqaload.getcol("wt1"))
                conweight2=array(h.nqaload.getcol("wt2"))

            # load weights file
            if (nqsfile != ""):
                outfilestem = '"%s-syn.nqs"' % (nqsfile)
                filename = '%s-syn.nqs' % (nqsfile) 
            else:
                outfilestem = '"%s/p1-%.2f_p2-%.2f_p3-%.2f_p4-%.2f_p5-%d_i-%d_w-%d_test-syn.nqs"' % (nqsdir, nqsparams[0], nqsparams[1], nqsparams[2], nqsparams[3], nqsparams[4], iseedvals[nqsparams[5]], wseedvals[nqsparams[6]])
                filename = '%s/p1-%.2f_p2-%.2f_p3-%.2f_p4-%.2f_p5-%d_i-%d_w-%d_test-syn.nqs' % (nqsdir, nqsparams[0], nqsparams[1], nqsparams[2], nqsparams[3], nqsparams[4], iseedvals[nqsparams[5]], wseedvals[nqsparams[6]])
    
            if os.path.isfile(filename):
                print "loadding "+outfilestem
                # get errors from nqs
                h('objref nqaload2')
                h('nqaload2 = new NQS(%s)'%outfilestem)
                # convert connectivity to python arrays
                #h('objref synChanges') 
                #h('nqaload2.select("t",">=", 0)')
                #h('synChanges = nqaload2')
                synpreid=array(h.nqaload2.getcol("id1"), 'i')
                synpostid=array(h.nqaload2.getcol("id2"), 'i')
                synweight=array(h.nqaload2.getcol("wg"))
                syntime=array(h.nqaload2.getcol("t"))


            else: # load from current simulation environment
                print "loading data from current sim..."
                # convert connectivity to python arrays
                conpreid=array(h.col.connsnq.getcol("id1"), 'i')
                conpostid=array(h.col.connsnq.getcol("id2"), 'i')
                #condelay=array(h.col.connsnq.getcol("del"))
                #condistance=array(h.col.connsnq.getcol("dist"))
                conweight1=array(h.col.connsnq.getcol("wt1"))
                conweight2=array(h.col.connsnq.getcol("wt2"))
    
                h('objref synChanges') 
                h('nqsy.select("t",">=", 0)')
                h('synChanges = nqsy')
                synpreid=array(h.synChanges.getcol("id1"), 'i')
                synpostid=array(h.synChanges.getcol("id2"), 'i')
                synweight=array(h.synChanges.getcol("wg"))
                syntime=array(h.synChanges.getcol("t"))

            if 'conpreid' in locals():
                order = lexsort((conpostid, conpreid)) # order by post and then pre
                #condelay = condelay[order]
                #condistance = condistance[order]
                conpreid = conpreid[order]
                conpostid = conpostid[order]
                conweight1 = conweight1[order]
                conweight2 = conweight2[order]
                
                # normalize conweight1 and conweight2 separately
                #maxW = max(conweight1.max(), conweight2.max())
                #conweight1 = conweight1/maxW    # normalize so can multiply together
                #conweight2 = conweight2/maxW 
                
                #conweight1Original = deepcopy(conweight1) # make copy of weights at t=0
                
                # sum conweight1+conweight2 and normalize
                conweight = conweight1+conweight2
                conweightOriginal = conweight / max(conweight)
                
                # convert synaptic changes over time (since connsnq doesn't change with t, can add)
                #synpreid=synpreid[0:len(conpreid)]
                order = lexsort((synpostid, synpreid, syntime)) # order syn by post and then pre
                synpreid = synpreid[order]
                synpostid = synpostid[order]
                synweight=synweight[order]        
                syntime=syntime[order]    
            else: 
                print "Coulndn't load file: "+outfilestem
        

    fig2 = figure(figsize=(figsize[0]/100,figsize[1]/100),dpi=100)
    fontsiz=14
    matrixsubplot = subplot(111)
    matrixsubplot.set_xlabel('postsynaptic population',fontsize=fontsiz)
    matrixsubplot.set_ylabel('presynaptic population',fontsize=fontsiz)

    # plot static weight matrix - use last set of weights
    if (not animate): 
        if loadnpy:
            weightMatrix=np.load('weightMatrix.npy')
        else:
            # load last set of weights
            synBin = where(syntime == syntime.max())[0]
            if (size(synBin)>0 and size(synBin)<size(conweight1)):
                # update connection matrix based on syn wieght gains
                #conweight1[range(len(synBin))] = (conweight1Original[range(len(synBin))] * synweight[synBin]) - conweight1Original[range(len(synBin))] # absolute weight increase
                conweight1[range(len(synBin))] = synweight[synBin] # relative increase 
                #conweight1[range(len(synBin))] = conweight1Original[range(len(synBin))] * synweight[synBin] # final weight
                #conweight1 = conweightOriginal # original weights (includes P population)
                #synBin = zeros(len(conweight1)) # TEMPORARY LINE to include P population weights - remove after!

            # calculate sum of weight for each subpopulation
            weightMatrix=calculateWeightMatrix()
            weightMatrix = weightMatrix/weightMatrix.max()
            EorI = [1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1] # define excitatory vs inhibitory pops
            for (x,y),value in ndenumerate(weightMatrix): # make weight negative if inhib
                if (EorI[x] == -1):
                    weightMatrix[x,y] = -1 * value 
        

        
        # plot 2dmap
        #weightMatrix[6,6] = weightMatrix[5,5]
        im=matrixsubplot.pcolor(array(weightMatrix), cmap=bicolormap(gap=0.05, mingreen=-0.6,redbluemix=0.0,epsilon=0.005), edgecolors='k', linewidths=1, vmin=-1, vmax=1)
        xticks([float(i)+0.5 for i in p.values()], p.keys())
        yticks([float(i)+0.5 for i in p.values()], p.keys())
            
        matrixsubplot.axis([3, len(p), 3,len(p)])  # set the limits of the plot to the limits of the data
        #matrixsuplot.tick_params(labelsize=8)
        
        c=colorbar(im, ax=matrixsubplot, label='normalized effective connectivity',fontsize=fontsiz)#, fraction=0.09,pad=0.01)     
        try:
            filename
        except:    
            fig2.canvas.set_window_title('Data loaded from saved array')
        else:
            fig2.canvas.set_window_title(filename)
        show()
        #fig.tight_layout()
        if savenpy:
            np.save("weightMatrix.npy", weightMatrix)
        if savefig:
            fig2.savefig("connectivity.pdf",format='pdf')            
            #fig2.savefig('gif/%s_p1-%d_p2-%d_p3-%d_p4-%d_p5-%d_i-%d_w-%d_wmat.png' % (nqsdir[5:], nqsparams[0], nqsparams[1], nqsparams[2], nqsparams[3], nqsparams[4], nqsparams[5], nqsparams[6])) # Save PNG file
        
    if animate:
        # define t
        if 'syntime' in globals():
            tMax = syntime.max()
        else:
            tMax = 35000
        tstep = tinterval
        T = arange(0, tMax+tstep, tstep)
        print T
        if load: 
            weightMatrixT=np.load('weightMatrixT.npy')
        else:
            weightMatrixT = [None] * len(T)
        t=0
        for b in T:
            if load:
                weightMatrix=weightMatrixT[t]
            else:
                # calculate sum of weight for each subpopulation
                weightMatrix=calculateWeightMatrix() # plot new RF
                weightMatrixT[t]=weightMatrix
            
            t+=1    
            if t==1: #fixed max value 
                plotvmax=1.2*weightMatrix.max()
            
            #plot 2dmap  , save and wait ; make video
            im=matrixsubplot.pcolor(weightMatrix, cmap=weightsCmap, vmin=0, vmax=plotvmax)
            xticks(p.values(), p.keys())
            yticks(p.values(), p.keys())
            matrixsubplot.axis([0, len(p), 0,len(p)]) 
            if t==1:
                c=colorbar(im, ax=matrixsubplot)
            else:
                c.set_clim([0,plotvmax])
                c.draw_all()
            show()
            #fig.tight_layout()
            if dosave: fig.savefig('tmpmovie%04i.png' % b) # Save PNG files for movie
            if dosave!=1: pause(1)#(1e-10) 
            # update weights 
            if (not load):
                synBin = where(syntime == b )[0]
                if (size(synBin)>0 and size(synBin)<size(conweight1)):
                    # update connection matrix based on syn wieght gains
                    conweight1[range(len(synBin))] = conweight1Original[range(len(synBin))] * synweight[synBin]
                    #conweight1[range(len(synBin))] = synweight[synBin]
        
        # create video from images            
        if saveanim:
            moviedir='data/movies/'
            moviename='14jan28_weights.mpg'
            print('Making movie...')
            system('mkdir -p ' + moviedir)
            system("mencoder 'mf://tmpmovie*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %s" % (moviedir+moviename))
            print('  Cleaning up...')
            system('rm tmpmovie*.png')

        if savenpy:
            np.save("weightMatrixT.npy", weightMatrixT)
    
        # show plots with evolution of weights over time
        if (not load): weightMatrixT=array(weightMatrixT);
        colorlist = ['black','darkgrey','blue', 'cyan', 'green', 'brown', 'red', 'magenta', 'orange','yellow']
        legendx=0.3
        legendy=1.0
        legendfontsize=8
        xlabel = 'time (ms)'
        ylabel = 'summed weights'
        #p = {'Pse':0, 'Psf':1,'Pee':2,'Pef':3,'ES':4,'IS':5,'ILS':6,'EMse':7,'EMsf':8,'EMee':9,'EMef':10,'IM':11,'IML':12}
        
        #subplot 1: Px4 (0:3) -> ES (4); IS (5)->ES (4); IM (11)->EM x4 (7:10) total: 9
        subplot1.set_title('P->ES, IS->ES, IM->EM', fontsize=12) # titles
        icolor=0
        for pre in arange(0,4):
            for post in arange(4,5):
                subplot1.plot(T,weightMatrixT[:, pre, post], colorlist[icolor], label=str(p2[pre])+'->'+str(p2[post]))
                icolor+=1
        for pre in arange(5,6):
            for post in arange(4,5):
                subplot1.plot(T,weightMatrixT[:, pre, post], colorlist[icolor], label=str(p2[pre])+'->'+str(p2[post]))
                icolor+=1
        for pre in arange(11,12):
            for post in arange(7,11):
                subplot1.plot(T,weightMatrixT[:, pre, post], colorlist[icolor], label=str(p2[pre])+'->'+str(p2[post]))
                icolor+=1
        subplot1.legend(loc='upper right', bbox_to_anchor=(legendx, legendy),  borderaxespad=0., prop={'size':legendfontsize})
        subplot1.set_ylabel(ylabel)
        subplot1.set_xlabel(xlabel)
        subplot1.grid(True)

        # ES (4)-> ES, EMx4, IS, ISL (4:10) total: 8
        subplot2.set_title('ES', fontsize=12)
        icolor=0
        for pre in arange(4,5):
            for post in arange(4,11):
                subplot2.plot(T,weightMatrixT[:, pre, post], colorlist[icolor], label=str(p2[pre])+'->'+str(p2[post]))
                icolor+=1
        subplot2.legend(loc='upper right', bbox_to_anchor=(legendx, legendy),  borderaxespad=0., prop={'size':legendfontsize})
        subplot2.set_ylabel(ylabel)
        subplot2.set_xlabel(xlabel)    
        subplot2.grid(True)

        # EMx4 (7:10)-> EMx4 (7:10) total:16 (use dotted vs line))
        subplot3.set_title('EM recurrent', fontsize=12)
        icolor=0
        linesty='-'
        for pre in arange(7,11):
            for post in arange(7,11):
                subplot3.plot(T,weightMatrixT[:, pre, post], colorlist[icolor],  linestyle = linesty, label=str(p2[pre])+'->'+str(p2[post]))
                icolor+=1
                if icolor>7:
                    icolor=0
                    linesty='--'
        subplot3.legend(loc='upper right', bbox_to_anchor=(legendx, legendy),  borderaxespad=0., prop={'size':legendfontsize})
        subplot3.set_ylabel(ylabel)
        subplot3.set_xlabel(xlabel)
        subplot3.grid(True)

        # EMx4 (7:10)->  ES, IM, IML (3, 11:12) total:12
        subplot4.set_title('EM', fontsize=12)
        icolor=0
        linesty='-'
        for pre in arange(7,11):
            for post in (4,11,12):
                subplot4.plot(T,weightMatrixT[:, pre, post], colorlist[icolor], linestyle = linesty, label=str(p2[pre])+'->'+str(p2[post]))
                icolor+=1
                if icolor>5:
                    icolor=0
                    linesty='--'
        subplot4.legend(loc='upper right', bbox_to_anchor=(legendx, legendy),  borderaxespad=0., prop={'size':legendfontsize})
        subplot4.set_ylabel(ylabel)
        subplot4.set_xlabel(xlabel)
        subplot4.grid(True)
        show()

# call popWeight to plot the weights of the 4 different targets in a specific sim (with specific params)
def popWeights4Targ(nqsdir, nqsparams=[0,0,0,0], savefig=0):
    for i in range(4):
        popWeights(1, nqsdir, [nqsparams[0],nqsparams[1],nqsparams[2],nqsparams[3],i, 0,0], savefig)


# Compare population weights for each target after training
def popMuscles(nqsdir="", plotfig = 0, savefig = 0, save2Matlab = 1):
    
    def index2d(myList, v):
        for i, x in enumerate(myList):
            if v in x:
                return (i)
        
    def calculateWeightMatrix():
        print "calculating final weight matrix..."
        #weightMatrix = ndarray((sum(popslen), sum(popslen)))        
        w, h = len(p), len(p)
        weightMatrix = [[0] * w for i in range(h)]
        onlyweight1 = 1
        for i in range(len(synBin)):
            if (conweight1[i]>0):# || conweight2[i]>0):
                if onlyweight1: weightMatrix[index2d(subpops,conpreid[i])][index2d(subpops,conpostid[i])] += conweight1[i]
                else: weightMatrix[index2d(subpops,conpreid[i])][index2d(subpops,conpostid[i])]  += conweight1[i] + conweight2[i]
        weightMatrix=array(weightMatrix)
        for i in range(len(subpops)):
            for j in range(len(subpops)):
                weightMatrix[i][j]=weightMatrix[i][j]/(len(subpops[i])*len(subpops[j]))
        return weightMatrix
        
    # visualization options
    figsize = [1000,800] # Figure size in pixels
    weightsCmap = 'hot_r'# 'jet'#'YlOrRd' #'jet'#'hot'#'autumn'

    # create figures
    ion()

    # cell populations parameters
    popslen = [192, 44, 20, 192, 44, 20,192]
    #popnames=['ES','IS','ILS','EM','IM','IML', 'P'];
    popinds=[41, 42, 43, 44, 45, 46, 2];
    popshape=['o',  '^',  'h', '*',  '^',  'h',  '*', '8'];
    
    # cell subpopulations (incuding muscle groups)
    subpops= [None]*13
    p = {'Pse':0, 'Psf':1,'Pee':2,'P':3,'ES':4,'IS':5,'ILS':6,'EMse':7,'EMsf':8,'EMee':9,'EMef':10,'IM':11,'IML':12}
    p2 = {0:'Pse', 1:'Psf',2:'Pee',3:'Pef',4:'ES',5:'IS',6:'ILS',7:'EMse',8:'EMsf',9:'EMee',10:'EMef',11:'IM',12:'IML'}
    
    cellslist=[] # list of cells with muscle groups together
    popstart=0
    index=0
    for i in range(len(popslen)):
        popend = popstart+popslen[i]
        if i==3 or i==6: # for P and EM divide into 4 groups 
            for j in range(4): 
                subpops[index] = list(arange(popstart+j,popend,4)) # create array with list of ids for each cell subpopulation  
                index+=1
        else:
            subpops[index]=list(arange(popstart,popend))
            index+=1
        popstart+=popslen[i] # increase popstart
    
    # reorder to have P first
    neworder=[9,10,11,12,0,1,2,3,4,5,6,7,8]
    subpops = [ subpops[i] for i in neworder]

    # load data from neuron variables
    wseedvals =[120456, 398115, 534031, 796321, 895199] # seed values for filename
    iseedvals = [1235, 1235+(2*17),2837, 3955, 4506, 6789]

    print "loading data from nqs files..."
    # load connectivity file
    targets=[0,1,2,3]
    loaded = 0
    for itarget in targets:
        iseedval = iseedvals[0]
        for wseedval in wseedvals:
            loaded = 0
            outfilestem = '"%s/target-%d_i-%d_w-%d_train-con.nqs"' % (nqsdir, itarget, iseedval, wseedval)
            filename = '%s/target-%d_i-%d_w-%d_train-con.nqs' % (nqsdir,itarget, iseedval, wseedval)

            if os.path.isfile(filename):
                print "loading "+outfilestem
                # get errors from nqs
                h('objref nqaload')
                h('nqaload = new NQS(%s)'%outfilestem)
                # convert connectivity to python arrays
                conpreid=array(h.nqaload.getcol("id1"), 'i')
                conpostid=array(h.nqaload.getcol("id2"), 'i')
                #condelay=array(h.col.connsnq.getcol("del"))
                #condistance=array(h.col.connsnq.getcol("dist"))
                conweight1=array(h.nqaload.getcol("wt1"))
                conweight2=array(h.nqaload.getcol("wt2"))
                loaded += 1

            # load weights file

            outfilestem = '"%s/target-%d_i-%d_w-%d_train-syn.nqs"' % (nqsdir, itarget, iseedval, wseedval)
            filename = '%s/target-%d_i-%d_w-%d_train-syn.nqs' % (nqsdir, itarget, iseedval, wseedval)

            if os.path.isfile(filename):
                print "loading "+outfilestem
                # get errors from nqs
                h('objref nqaload2')
                h('nqaload2 = new NQS(%s)'%outfilestem)
                # convert connectivity to python arrays
                #h('objref synChanges') 
                #h('nqaload2.select("t",">=", 0)')
                #h('synChanges = nqaload2')
                synpreid=array(h.nqaload2.getcol("id1"), 'i')
                synpostid=array(h.nqaload2.getcol("id2"), 'i')
                synweight=array(h.nqaload2.getcol("wg"))
                syntime=array(h.nqaload2.getcol("t"))
                loaded += 1


            if loaded == 2: #'conpreid' in locals():
                order = lexsort((conpostid, conpreid)) # order by post and then pre
                #condelay = condelay[order]
                #condistance = condistance[order]
                conpreid = conpreid[order]
                conpostid = conpostid[order]
                conweight1 = conweight1[order]
                conweight2 = conweight2[order]
                
                # normalize conweight1 and conweight2 separately
                #maxW = max(conweight1.max(), conweight2.max())
                #conweight1 = conweight1/maxW    # normalize so can multiply together
                #conweight2 = conweight2/maxW 
                
                #conweight1Original = deepcopy(conweight1) # make copy of weights at t=0
                
                # sum conweight1+conweight2 and normalize
                conweight = conweight1+conweight2
                conweightOriginal = conweight / max(conweight)
                
                # convert synaptic changes over time (since connsnq doesn't change with t, can add)
                #synpreid=synpreid[0:len(conpreid)]
                order = lexsort((synpostid, synpreid, syntime)) # order syn by post and then pre
                synpreid = synpreid[order]
                synpostid = synpostid[order]
                synweight=synweight[order]        
                syntime=syntime[order]    

        
                if plotfig:
                    fig2 = figure(figsize=(figsize[0]/100,figsize[1]/100),dpi=100)
                    fontsiz=14
                    matrixsubplot = subplot(111)
                    matrixsubplot.set_xlabel('postsynaptic population',fontsize=fontsiz)
                    matrixsubplot.set_ylabel('presynaptic population',fontsize=fontsiz)

                # plot static weight matrix - use last set of weights
                # load last set of weights
                synBin = where(syntime == syntime.max())[0]
                if (size(synBin)>0 and size(synBin)>size(conweight1)):
                    synBin = synBin[0:(len(synBin)/2)-1]
                #synBin = where(syntime == 4000)[0]
                if (size(synBin)>0 and size(synBin)<size(conweight1)):
                    # update connection matrix based on syn wieght gains
                    conweight1[range(len(synBin))] = (conweightOriginal[range(len(synBin))] * synweight[synBin]) - conweightOriginal[range(len(synBin))] # absolute weight increase
                    #conweight1[range(len(synBin))] = synweight[synBin] # relative increase 
                    #conweight1[range(len(synBin))] = conweight1Original[range(len(synBin))] * synweight[synBin] # final weight
                    #conweight1 = conweightOriginal # original weights (includes P population)
                    #synBin = zeros(len(conweight1)) # TEMPORARY LINE to include P population weights - remove after!

                    # calculate sum of weight for each subpopulation
                    weightMatrix=calculateWeightMatrix()
                    #weightMatrix = weightMatrix/weightMatrix.max()
                    EorI = [1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1] # define excitatory vs inhibitory pops
                    for (x,y),value in ndenumerate(weightMatrix): # make weight negative if inhib
                        if (EorI[x] == -1):
                            weightMatrix[x,y] = -1 * value 

                    # plot 2dmap
                    if plotfig:
                        im=matrixsubplot.pcolor(array(weightMatrix), cmap=bicolormap(gap=0.05, mingreen=-0.6,redbluemix=0.0,epsilon=0.005), edgecolors='k', linewidths=1)#, vmin=-1, vmax=1)
                        xticks([float(i)+0.5 for i in p.values()], p.keys())
                        yticks([float(i)+0.5 for i in p.values()], p.keys())
                            
                        matrixsubplot.axis([3, len(p), 3,len(p)])  # set the limits of the plot to the limits of the data
                        #matrixsuplot.tick_params(labelsize=8)
                        
                        c=colorbar(im, ax=matrixsubplot, label='normalized effective connectivity')#,fontsize=fontsiz)#, fraction=0.09,pad=0.01)     
                        try:
                            filename
                        except:    
                            fig2.canvas.set_window_title('Data loaded from saved array')
                        else:
                            fig2.canvas.set_window_title(filename)
                        show()
                    #fig.tight_layout()
                    
                    if savefig:
                        fig2.savefig("connectivity.pdf",format='pdf')            
                        #fig2.savefig('gif/%s_p1-%d_p2-%d_p3-%d_p4-%d_p5-%d_i-%d_w-%d_wmat.png' % (nqsdir[5:], nqsparams[0], nqsparams[1], nqsparams[2], nqsparams[3], nqsparams[4], nqsparams[5], nqsparams[6])) # Save PNG file

                    if save2Matlab:
                        print "saving matlab file with weights..."
                        scipy.io.savemat(('%s/target-%d_i-%d_w-%d_weights.mat'%(nqsdir, itarget, iseedval, wseedval)), \
                                        mdict={'weightMatrix': weightMatrix})
            else: 
                print "Coulndn't load file: "+outfilestem
                              


# Show 2D pcolor of 704x704 cells with connectivity/weights - over time?
def cellWeightsMatrix(animate, dosave, tinterval):
    # visualization options
    figsize = [800,800] # Figure size in pixels
    weightsCmap = 'jet'#'YlOrRd' #'jet'#'hot'#'autumn'

    # create figure
    ion()
    fig = figure(figsize=(figsize[0]/100,figsize[1]/100),dpi=100)
    #fig.subplots_adjust(left=0.02) # Less space on left
    #fig.subplots_adjust(right=0.93) # Less space on right
    fig.subplots_adjust(bottom=0.08) # Less space on bottom    
    weightplot = subplot(111) # create subplot        
    weightplot.set_title('Weight Matrix', fontsize=12) # titles
    weightplot.set_xlabel('post (Pse,Psf,Pee,Pef,ES,IS,ILS,EMse,EMsf,EMee,EMef,IM,IML)')
    weightplot.set_ylabel('pre (Pse,Psf,Pee,Pef,ES,IS,ILS,EMse,EMsf,EMee,EMef,IM,IML)')
    
    #setp(weightplot.get_xticklabels(), visible=False) # hide x and y ticks
    #setp(weightplot.get_yticklabels(), visible=False)
    
    #border = 2
    #weightplot.set_xlim([xlocs.min()-border, xlocs.max()+border]) # set x-y lims
    
    # cell populations parameters
    popslen = [192, 44, 20, 192, 44, 20,192]
    #popnames=['P','ES','IS','ILS','EM','IM','IML'];
    #popinds=[ 2, 41, 42, 43, 44, 45, 46];
    popshape=['o',  '^',  'h', '*',  '^',  'h',  '*', '8'];
    
    # cell subpopulations (incuding muscle groups)
    subpops= [None]*13
    cellslist=[] # list of cells with muscle groups together
    popstart=0
    index=0
    for i in range(len(popslen)):
        popend = popstart+popslen[i]
        if i==3 or i==6: # for P and EM divide into 4 groups 
            for j in range(4): 
                subpops[index] = list(arange(popstart+j,popend,4)) # create array with list of ids for each EM cell subpopulation  
                cellslist=cellslist+subpops[index]
                index+=1
        else:
            subpops[index]=list(arange(popstart,popend))
            cellslist=cellslist+subpops[index]
            index+=1
        # increase popstart
        popstart+=popslen[i]
    #move P to the beginning
    cellslist=cellslist[-192:]+cellslist[:-192]


    # convert connectivity to python arrays
    conpreid=array(h.col.connsnq.getcol("id1"), 'i')
    conpostid=array(h.col.connsnq.getcol("id2"), 'i')
    #condelay=array(h.col.connsnq.getcol("del"))
    #condistance=array(h.col.connsnq.getcol("dist"))
    conweight1=array(h.col.connsnq.getcol("wt1"))
    conweight2=array(h.col.connsnq.getcol("wt2"))
    
    order = lexsort((conpreid, conpostid)) # order by post and then pre
    conpreid = conpreid[order]
    conpostid = conpostid[order]
    #condelay = condelay[order]
    #condistance = condistance[order]
    conweight1 = conweight1[order]
    conweight2 = conweight2[order]
    
    maxW = max(conweight1.max(), conweight2.max())
    conweight1 = conweight1/maxW    # normalize so can multiply together
    conweight2 = conweight2/maxW 
    
    conweight1Original = deepcopy(conweight1) # make copy of weights at t=0
    
    # convert synaptic changes over time (since connsnq doesn't change with t, can add)
    h('objref synChanges') 
    #h('nqsy.tog()')
    h('nqsy.select("t",">=", 0)')
    h('synChanges = nqsy')
    synpreid=array(h.synChanges.getcol("id1"), 'i')
    synpostid=array(h.synChanges.getcol("id2"), 'i')
    synweight=array(h.synChanges.getcol("wg"))
    syntime=array(h.synChanges.getcol("t"))
    
    order = lexsort((synpreid, synpostid)) # order syn by post and then pre
    synpreid = synpreid[order]
    synpostid = synpostid[order]
    synweight=synweight[order]        
    syntime=syntime[order]    
    
        
    def calculateWeightMatrix():
        #weightMatrix = ndarray((sum(popslen), sum(popslen)))        
        w, h = sum(popslen), sum(popslen)
        weightMatrix = [[0] * w for i in range(h)]
        onlyAMPA = 0
        for i in range(len(conweight1)):
            if onlyAMPA: weightMatrix[conpreid[i]][conpostid[i]] = conweight1[i]
            else: weightMatrix[conpreid[i]][conpostid[i]] = conweight1[i] + conweight2[i]
        #weightMatrix = [ mylist[i] for i in myorder] # reorder
        weightMatrix2=deepcopy(weightMatrix)
        for i,sublist in enumerate(weightMatrix):
            weightMatrix2[i]=[sublist[j] for j in cellslist] #reorder rows
        
        weightMatrix2=[weightMatrix2[j] for j in cellslist]  # reorder columns
            
        weightMatrix2=array(weightMatrix2)            
        return weightMatrix2
    
    # calculate sum of weight for each subpopulation
    weightMatrix=calculateWeightMatrix() # plot new RF
    
    # plot 2dmap
    im=weightplot.imshow(weightMatrix, cmap=weightsCmap)
    #im=imshow(weightMatrix, cmap=weightsCmap)
    weightplot.axis([0, sum(popslen), 0,sum(popslen)])  # set the limits of the plot to the limits of the data
    c=colorbar(im, ax=weightplot)#, fraction=0.09,pad=0.01)     
    show()
    #fig.tight_layout()
    
    if animate:
        # define t
        tMax = syntime.max()
        tstep = tinterval
        T = arange(0, tMax, tstep)
        for b in T:
            # calculate sum of weight for each subpopulation
            weightMatrix=calculateWeightMatrix() # plot new RF
            
            #plot 2dmap  , save and wait ; make video
            im=weightplot.imshow(weightMatrix, cmap=weightsCmap, vmin=0, vmax=1.5)
            #im=imshow(weightMatrix, cmap=weightsCmap, vmin=0, vmax=1.5)
            #plt.imshow(z)
            #plt.pcolor(Z)
        
            show()
            #fig.tight_layout()
            if dosave: fig.savefig('tmpmovie%04i.png' % b) # Save PNG files for movie
            if dosave!=1: pause(1e-10) 
            
            # update weights 
            synBin = where(syntime == b )[0]
            if (size(synBin)>0 and size(synBin)<size(conweight1)):
                # update connection matrix based on syn wieght gains
                conweight1[range(len(synBin))] = conweight1Original[range(len(synBin))] * synweight[synBin]
                
    if dosave:
        moviedir='data/movies/'
        moviename='14jan28_weights.mpg'
        print('Making movie...')
        system('mkdir -p ' + moviedir)
        system("mencoder 'mf://tmpmovie*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %s" % (moviedir+moviename))
        print('  Cleaning up...')
        system('rm tmpmovie*.png')
                                            
# calculate the position of center-out targets in joe's BMI experiments to use in sims                    
def calculateTargetsCenterOut():
    monkeyArmLen=[0.0839, 0.2038]
    varmLen=[0.4634 - 0.173, 0.7169 - 0.4634] # added average hand size (included in monkey dimensions)
    factor = 2*(varmLen[0]+varmLen[1])/(monkeyArmLen[0]+monkeyArmLen[1])
    monkeyCenter = [0.050536, 0.095114]
    varmAngleCenter = [0.62, 1.53] # 0.53, 1.41
    varmCenter = angles2pos(varmAngleCenter[0], varmAngleCenter[1], varmLen[0], varmLen[1])
    targetDist = 0.04
    targetDist45deg = 0.0282

    print "\nvarmCenter:"
    print varmCenter
    
    targets = zeros((8,2))
    targets_pos = zeros((8,2))

    targets_pos[0] = varmCenter[0]+targetDist*factor, varmCenter[1]+0 #right
    targets_pos[1] = varmCenter[0]-targetDist*factor, varmCenter[1]+0 #left
    targets_pos[2] = varmCenter[0], varmCenter[1]+targetDist*factor #top
    targets_pos[3] = varmCenter[0], varmCenter[1]-targetDist*factor #bottom
    targets_pos[4] = varmCenter[0]+targetDist45deg*factor, varmCenter[1]+targetDist45deg*factor #right-top
    targets_pos[5] = varmCenter[0]-targetDist45deg*factor, varmCenter[1]+targetDist45deg*factor #left-top
    targets_pos[6] = varmCenter[0]+targetDist45deg*factor, varmCenter[1]-targetDist45deg*factor #left-bottom
    targets_pos[7] = varmCenter[0]-targetDist45deg*factor, varmCenter[1]-targetDist45deg*factor #rightbottom

    print "\ntarget positions:"
    print targets_pos

    print "\ntarget displacements:"
    print targets_pos - varmCenter

    targets[0] = pos2angles(varmCenter[0]+targetDist*factor, varmCenter[1]+0, varmLen[0], varmLen[1]) #right
    targets[1] = pos2angles(varmCenter[0]-targetDist*factor, varmCenter[1]+0, varmLen[0], varmLen[1])#left
    targets[2] = pos2angles(varmCenter[0], varmCenter[1]+targetDist*factor, varmLen[0], varmLen[1]) #top
    targets[3] = pos2angles(varmCenter[0], varmCenter[1]-targetDist*factor, varmLen[0], varmLen[1]) #bottom
    targets[4] = pos2angles(varmCenter[0]+targetDist45deg*factor, varmCenter[1]+targetDist45deg*factor, varmLen[0], varmLen[1]) #right-top
    targets[5] = pos2angles(varmCenter[0]-targetDist45deg*factor, varmCenter[1]+targetDist45deg*factor, varmLen[0], varmLen[1]) #left-top
    targets[6] = pos2angles(varmCenter[0]+targetDist45deg*factor, varmCenter[1]-targetDist45deg*factor, varmLen[0], varmLen[1]) #left-bottom
    targets[7] = pos2angles(varmCenter[0]-targetDist45deg*factor, varmCenter[1]-targetDist45deg*factor, varmLen[0], varmLen[1]) #rightbottom
    
    print "\ntarget angular displacements:"
    print targets - varmAngleCenter

    print "\nabsolute target angular displacements:"
    print sum(abs(targets - varmAngleCenter),1)
    
    print "\ntarget angles:"
    print targets

    return targets    
    
# convert cartesian position to joint angles
def pos2angles(x,y,l1,l2):
    
    elang = abs(2*arctan(sqrt(((l1 + l2)**2 - (x**2 + y**2))/((x**2 + y**2) - (l1 - l2)**2)))); 

    phi = arctan2(y,x); 
    psi = arctan2(l2 * sin(elang), l1 + (l2 * cos(elang)));

    shang = phi - psi; 
    
    return [shang,elang]

# convert joint angles to cartesian position
def angles2pos(shang,elang,l1,l2):
    armAng=zeros(2)
    armLen=zeros(2)
    armAng[0]=shang
    armAng[1]=elang
    armLen[0]=l1
    armLen[1]=l2
    
    shoulderPosx = 0
    shoulderPosy = 0
    elbowPosx = armLen[0] * cos(armAng[0]) # end of elbow
    elbowPosy = armLen[0] * sin(armAng[0])
    wristPosx = elbowPosx + armLen[1] * cos(+armAng[0]+armAng[1]) # wrist=arm position
    wristPosy = elbowPosy + armLen[1] * sin(+armAng[0]+armAng[1])
    
    return wristPosx,wristPosy    

# calculates deviation of trajectory from ideal (straight line) traj to target
def trajDeviation(x,y,center,targetInd, targetXdist):
    # subtract center position from trajectory
    x=x-center[0]
    y=y-center[1]

    # set angle of rotation as a function of target
    targetAngles = [0, - np.pi, -np.pi/2, np.pi/2, -np.pi/4, -3*np.pi/4, np.pi/4, 3*np.pi/4] 
    rotAng = targetAngles[targetInd]
    
    # rotate trajectory to position over positive x-axis (=target 0, center-right)
    xrot = x*cos(rotAng) - y*sin(rotAng)
    yrot = x*sin(rotAng) + y*cos(rotAng)

    # include only trajectory points within trajDist*target x pos
    # enables comparison of different targets irrespective of time to target
    trajDist=1.1
    xlim=where(xrot<trajDist*(targetXdist-center[0]))[0]

    # add penalty for displacements going in wrong direction
    xneg=where(xrot<0)
    yrot[xneg]=yrot[xneg]+abs(xrot[xneg])

    # calculate deviation as the sum of the absolute y displacements
    #ydisp = sum(abs(yrot)) #sum
    ydisp = mean(abs(yrot[xlim])) #mean
    return ydisp

# calculate the average (replaced by min!) distance between the trajectory and the target during trange
def trajAccuracy(x,y,targetPos,varmLen, trange, method):
    #method = 'min'#

    if method == 'sum':
        dist=0
    elif method == 'min':
        dist=1
    for i in trange:
        if method == 'sum':
            dist += sqrt((x[i]-targetPos[0])**2 + (y[i]-targetPos[1])**2) # sum
        elif method == 'min':
            dist = min(dist, sqrt((x[i]-targetPos[0])**2 + (y[i]-targetPos[1])**2)) # min
    #print "min dist:"+str(dist)
    if method == 'sum':
        dist = dist/len(trange) # average 
    return dist

# plot center out reaching trajectory for single target
def plotCenterOutTrajSingleParam(simdatadir, trajStartArg, trajStopArg, niseeds=100):

    #######################
    ##### obtain target locations
    #########################
    targetsAng = calculateTargetsCenterOut()
    # convert to cartesian position
    varmLen=[0.4634 - 0.173, 0.7169 - 0.4634]
    varmCenter = angles2pos(0.62, 1.53, varmLen[0], varmLen[1])
    elbowCenter = [varmLen[0] * cos(0.62), varmLen[0] * sin(0.62)]
    targetsPos = zeros((8,2))
    for i in range(len(targetsAng)):
        targetsPos[i]=angles2pos(targetsAng[i][0], targetsAng[i][1], varmLen[0], varmLen[1])
    
    
    ###########################
    # read and plot trajectory data (t,x,y from nqa) 
    ########################
    wseedvals =[120456, 398115, 534031, 796321, 895199]
    iseedvals = arange(1235,  1235+(17*niseeds), 17)
    
    # parameter values
    param1_range = arange(1,2,1)#  arange(0.8,1.24,0.04)#arange(200,375,25)#arange(20,180,20)##arange(10,110,10)# [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]##arange(0.8,1.24,0.04)# arange(1,9)#arange(50,550,50)# arange(10,110,10)## #[0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]#[10,20,30,40,50,60,70,80,90,100]#[50, 100,250,500,750,1000]#arange(20,180,20)##
    param2_range = [0,1,2,3] # [0,1,2,3,4,5,6,7]

    # Visualization options
    figsize =[800,800]# [1300,500] # Figure size in pixels
    targetColor = 'green';#array([(1,0.4,0) , (0,0.2,0.8)]) # Define excitatory and inhibitory colors -- orange and turquoise
    targetLine = 2*2 #??
    targetSize = 30 *2
    targetMarker = 'x'
    colorlist = ['black','blue', 'red', 'green', 'brown', 'cyan','darkgrey', 'magenta', 'orange','yellow']
    trajLine = 1
    trajSize = 1
    trajMarker = '.'    
    trajStart = trajStartArg#2 # sample number to start from (each sample = 10 ms)
    trajStop = trajStopArg#400 # sample number to finish on (max 400=4sec)
        
    # create subplots for each of the param values tested
    ion()
    fig = figure(figsize=(figsize[0]/100,figsize[1]/100),dpi=100)
    fig.subplots_adjust(left=0.02) # Less space on left
    fig.subplots_adjust(right=0.93) # Less space on right
    fig.subplots_adjust(bottom=0.08) # Less space on bottom
    subplotsx = 1#5
    subplotsy = 1#2
    maxSubplots = subplotsx*subplotsy
    param1Subplots =  [None] * min(len(param1_range), maxSubplots)
    border = 0.1
    for i in range(min(len(param1_range), maxSubplots)):
        param1Subplots[i] = subplot(subplotsy, subplotsx, i+1)
        #setp(param1Subplots[i].get_xticklabels(), visible=False) # hide x and y ticks
        #setp(param1Subplots[i].get_yticklabels(), visible=False)
        param1Subplots[i].set_xlim([targetsPos[1][0]-border*1.5, targetsPos[0][0]+border]) # set x-y lims
        param1Subplots[i].set_ylim([targetsPos[3][1]-border*2.5, targetsPos[2][1]+border])
    
    # plot data for each of the 4 starting conditions in same color; different color for each target
    # Loop over param1 values
    h('objref nqaload')
    iparam1=0
    param1Subplots[iparam1].set_title('Trajectories for multiple input seeds')
    # Loop over param2 values
    iparam2=-1
    for param2 in param2_range:
        iparam2 = iparam2 + 1
        iwseed = -1
        # plot target location
        param1Subplots[iparam1].scatter(targetsPos[iparam2][0], targetsPos[iparam2][1],  c=colorlist[iparam2], marker=targetMarker, linewidth=targetLine, s=targetSize)
        # Loop over wiring seed...
        for wseed in wseedvals:
            iwseed = iwseed + 1
            iiseed = -1
            # Loop over input seed...
            for iseed in iseedvals:
                iiseed = iiseed + 1
                # set filename
                #outfilestem = '"%s/p1-%.2f_p2-%d_i-%d_w-%d_train-nqa.nqs"' % (simdatadir, param1, param2, iseed, wseed)
                #outfilestem = '"%s/p1-%d_p2-%d_i-%d_w-%d_test-nqa.nqs"' % (simdatadir, param1, param2, iseed, wseed)
                outfilestem = '"%s/target-%d_i-%d_w-%d_test-nqa.nqs"' % (simdatadir, param2, iseed, wseed)
                filename = '%s/target-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param2, iseed, wseed)
                if os.path.isfile(filename):
                    print outfilestem
                    # get errors from nqs
                    h('nqaload = new NQS(%s)'%outfilestem)
                    t=array(h.nqaload.getcol("t"))
                    x=array(h.nqaload.getcol("x"))
                    y=array(h.nqaload.getcol("y"))
                    shAng = array(h.nqaload.getcol("ang0"))
                    #print shAng
                    elAng = array(h.nqaload.getcol("ang1"))
                    #print elAng
                    param1Subplots[iparam1].scatter(x[trajStart:trajStop], y[trajStart:trajStop],  edgecolor=colorlist[iparam2], marker=trajMarker, linewidth=trajLine, s=trajSize) 
                
    # draw arm
    plot([0,elbowCenter[0]], [0, elbowCenter[1]],'k-',lw=4) #elbow
    plot([elbowCenter[0], varmCenter[0]], [elbowCenter[1], varmCenter[1]],'k-',lw=4)
    tight_layout()
    fig.savefig('gif/%s_traj_%d.png' % (simdatadir[5:], trajStop))

# plot center out reaching trajectories
def plotCenterOutTraj(simdatadir, trajStartArg, trajStopArg, param1Arg=None):

    #######################
    ##### obtain target locations
    #########################
    targetsAng = calculateTargetsCenterOut()
    # convert to cartesian position
    varmLen=[0.4634 - 0.173, 0.7169 - 0.4634]
    varmCenter = angles2pos(0.62, 1.53, varmLen[0], varmLen[1])
    targetsPos = zeros((8,2))
    for i in range(len(targetsAng)):
        targetsPos[i]=angles2pos(targetsAng[i][0], targetsAng[i][1], varmLen[0], varmLen[1])
    
    ###########################
    # read and plot trajectory data (t,x,y from nqa) 
    ########################
    wseedvals =[120456, 398115]#, 534031, 796321, 895199]
    iseedvals = [1235, 2837]#, 3955, 4506, 6789]
    
    # parameter values
    if param1Arg==None:
        param1_range =  arange(0.8,1.24,0.04)#arange(200,375,25)#arange(20,180,20)##arange(10,110,10)# [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]##arange(0.8,1.24,0.04)# arange(1,9)#arange(50,550,50)# arange(10,110,10)## #[0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]#[10,20,30,40,50,60,70,80,90,100]#[50, 100,250,500,750,1000]#arange(20,180,20)##
    else:
        param1_range = param1Arg
    param2_range = [0,1,2,3,4,5,6,7]

    # Visualization options
    figsize =[1300,500] # Figure size in pixels
    targetColor = 'green';#array([(1,0.4,0) , (0,0.2,0.8)]) # Define excitatory and inhibitory colors -- orange and turquoise
    targetLine = 2 #
    targetSize = 30
    targetMarker = 'x'
    colorlist = ['black','darkgrey','blue', 'cyan', 'green', 'brown', 'red', 'magenta', 'orange','yellow']
    trajLine = 1
    trajSize = 1
    trajMarker = '.'    
    trajStart = trajStartArg#2 # sample number to start from (each sample = 10 ms)
    trajStop = trajStopArg#400 # sample number to finish on (max 400=4sec)
        
    # create subplots for each of the param values tested
    ion()
    fig = figure(figsize=(figsize[0]/100,figsize[1]/100),dpi=100)
    fig.subplots_adjust(left=0.02) # Less space on left
    fig.subplots_adjust(right=0.93) # Less space on right
    fig.subplots_adjust(bottom=0.08) # Less space on bottom
    subplotsx = 5
    subplotsy = 2
    maxSubplots = subplotsx*subplotsy
    param1Subplots =  [None] * min(len(param1_range), maxSubplots)
    border = 0.1
    for i in range(min(len(param1_range), maxSubplots)):
        param1Subplots[i] = subplot(subplotsy, subplotsx, i+1)
        setp(param1Subplots[i].get_xticklabels(), visible=False) # hide x and y ticks
        setp(param1Subplots[i].get_yticklabels(), visible=False)
        param1Subplots[i].set_xlim([targetsPos[1][0]-border, targetsPos[0][0]+border]) # set x-y lims
        param1Subplots[i].set_ylim([targetsPos[3][1]-border, targetsPos[2][1]+border])
    
    # plot data for each of the 4 starting conditions in same color; different color for each target
    # Loop over param1 values
    h('objref nqaload')
    iparam1 = -1
    for param1 in param1_range:
        iparam1 = iparam1 + 1
        iparam2 = -1
        param1Subplots[iparam1].set_title('param1= '+str(param1))
        # Loop over param2 values
        for param2 in param2_range:
            iparam2 = iparam2 + 1
            iwseed = -1
            # plot target location
            param1Subplots[iparam1].scatter(targetsPos[iparam2][0], targetsPos[iparam2][1],  c=colorlist[iparam2], marker=targetMarker, linewidth=targetLine, s=targetSize)
            # Loop over wiring seed...
            for wseed in wseedvals:
                iwseed = iwseed + 1
                iiseed = -1
                # Loop over input seed...
                for iseed in iseedvals:
                    iiseed = iiseed + 1
                    skipValue=0
                    # set filename and check if file exists
                    outfilestem = '%s/p1-%d_p2-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, iseed, wseed)
                    if os.path.isfile(outfilestem):
                        print "loading "+str(outfilestem)
                    else:
                        outfilestem = '%s/p1-%.2f_p2-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, iseed, wseed)
                        if os.path.isfile(outfilestem):
                            print "loading "+str(outfilestem)
                        else: 
                            outfilestem = '%s/p1-%.3f_p2-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, iseed, wseed)
                            if os.path.isfile(outfilestem):
                                print "loading "+str(outfilestem)
                            else:
                                skipValue=1

                            
                    if skipValue==0:
                        # get errors from nqs
                        h('nqaload = new NQS("%s")'%outfilestem)
                        print("extracting trajectory data...")
                        t=array(h.nqaload.getcol("t"))
                        x=array(h.nqaload.getcol("x"))
                        y=array(h.nqaload.getcol("y"))
                        #print t,x,y
                    
                        param1Subplots[iparam1].scatter(x[trajStart:trajStop], y[trajStart:trajStop],  edgecolor=colorlist[iparam2], marker=trajMarker, linewidth=trajLine, s=trajSize) 
                
    tight_layout()

# plot center out reaching trajectories for 8 sims with same date
def plotCenterOutTraj8(simdatadirRoot, trajStartArg, trajStopArg):    
# call plotCenterOutTraj for each sim in same date
    
    numSims = 8
    
    param1_range =  []#[None] * numSims
    param1_range.append([10,20,30,40,50])
    param1_range.append([10,20,30,40,50,60,70,80,90,100])
    param1_range.append(arange(0.8,1.24,0.04))#([0.012, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15])
    param1_range.append([10,20,30,40,50,60,70,80,90,100])
    param1_range.append([25, 50,75, 100, 125, 150, 175, 200, 225, 250])
    param1_range.append([20,40,60,80,100,120,140,160])
    param1_range.append([200,225,250,275,300,325,350])
    param1_range.append([50, 100,250,500,750,1000])
    
    for i in range(numSims):
        plotCenterOutTraj(simdatadirRoot+'_sim'+str(i+1), trajStartArg, trajStopArg, param1_range[i])

# plot center out reaching trajectories for 4param batch sims
def plotCenterOutTraj4param(simdatadir, trajStartArg, trajStopArg):
    #######################
    ##### obtain target locations
    #########################
    targetsAng = calculateTargetsCenterOut()
    #targetsAng=[[deg2rad(90),deg2rad(70)],[deg2rad(90),deg2rad(100)],[deg2rad(100),deg2rad(80)],[deg2rad(100),deg2rad(70)],[deg2rad(95),deg2rad(95)]]

    #targetsAng=[[deg2rad(90),deg2rad(100)],[deg2rad(100),deg2rad(80)]] # only targets 14 and 15

    # convert to cartesian position
    varmLen=[0.4634 - 0.173, 0.7169 - 0.4634]
    varmCenter = angles2pos(0.62, 1.53, varmLen[0], varmLen[1])
    targetsPos = zeros((8,2))
    for i in range(len(targetsAng)):
        targetsPos[i]=angles2pos(targetsAng[i][0], targetsAng[i][1], varmLen[0], varmLen[1])
    
    ###########################
    # read and plot trajectory data (t,x,y from nqa) 
    ########################
    
    wseedvals =[120456,398115, 534031, 796321, 895199]
    iseedvals = [1235, 2837, 3955, 4506, 6789]

    # set param range by hand
    param1_range = arange(200,400,50) # EMNoiseRate train
    param2_range= arange(0, 8, 2) # EMNoiseMuscleGroup
    param3_range = arange(500,2000,500) # EMMuscleNoiseChangeDT
    param4_range = arange(8,40,8) # exploreTot
    param5_range = arange(4) # only four targets
    wseed_range = arange(2)
    inseed_range = arange(2)

    # read from param file
    with open('%s/params'% (simdatadir)) as f:
        param1_range, param2_range, param3_range, param4_range, param5_range, wseed_range, inseed_range = pickle.load(f)

    # Visualization options
    figsize =[1000,700] # Figure size in pixels
    targetColor = 'green';#array([(1,0.4,0) , (0,0.2,0.8)]) # Define excitatory and inhibitory colors -- orange and turquoise
    targetLine = 2 #
    targetSize = 30
    targetMarker = 'x'
    #colorlist = ['black','darkgrey','blue', 'cyan', 'green', 'brown', 'red', 'magenta', 'orange','yellow']
    colorlist = ['blue', 'green', 'red', 'magenta', 'darkgrey']
    trajLine = 1
    trajSize = 1
    trajMarker = '.'    
    trajStart = trajStartArg#2 # sample number to start from (each sample = 10 ms)
    trajStop = trajStopArg#400 # sample number to finish on (max 400=4sec)
        
    # create subplots for each of the param values tested
    ion()
    for param1 in param1_range:
        for param2 in param2_range:
            fig = figure(figsize=(figsize[0]/100,figsize[1]/100),dpi=100)
            fig.canvas.set_window_title('param1 = '+str(param1)+', param2 = '+str(param2))
            #fig.suptitle('param1 = '+str(param1)+', param2 = '+str(param2), fontsize=8)
            fig.subplots_adjust(left=0.02) # Less space on left
            fig.subplots_adjust(right=0.93) # Less space on right
            fig.subplots_adjust(bottom=0.08) # Less space on bottom
            subplotsy = len(param3_range)
            subplotsx = len(param4_range)
            maxSubplots = subplotsx*subplotsy
            paramSubplots =  [None] * min(len(param3_range)*len(param4_range), maxSubplots)
            border = 0.1
            for i in range(len(param3_range)*len(param4_range)):
                paramSubplots[i] = subplot(subplotsy, subplotsx, i+1)
                setp(paramSubplots[i].get_xticklabels(), visible=False) # hide x and y ticks
                setp(paramSubplots[i].get_yticklabels(), visible=False)
                paramSubplots[i].set_xlim([targetsPos[1][0]-border, targetsPos[0][0]+border]) # set x-y lims
                paramSubplots[i].set_ylim([targetsPos[3][1]-border, targetsPos[2][1]+border])
                
            # plot data for each of the 4 starting conditions in same color; different color for each target
            # Loop over param3 values
            h('objref nqaload')
            iparam3 = -1
            for param3 in param3_range:
                iparam3 = iparam3 + 1
                iparam4 = -1
                # Loop over param3 values
                for param4 in param4_range:
                    iparam4 = iparam4 + 1
                    iparam5 = -1
                    paramSubplots[iparam3*len(param4_range)+iparam4].set_title('param3 = '+str(param3)+', param4 = '+str(param4),fontsize=8)
                    # Loop over param5 values (target)
                    for param5 in param5_range:
                        iparam5 = iparam5 + 1
                        iwseed = -1
                        # plot target location
                        
                        paramSubplots[iparam3*len(param4_range)+iparam4].scatter(targetsPos[iparam5][0], targetsPos[iparam5][1],  c=colorlist[iparam5], marker=targetMarker, linewidth=targetLine, s=targetSize)
                        # Loop over wiring seed...
                        for wseed in array(wseedvals)[wseed_range]:
                            iwseed = iwseed + 1
                            iiseed = -1
                            # Loop over input seed...
                            for iseed in array(iseedvals)[inseed_range]:
                                iiseed = iiseed + 1
                                skipValue=0
                                # set filename and check if file exists
                                outfilestem = '%s/p1-%d_p2-%d_p3-%d_p4-%d_p5-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, param3, param4, param5, iseed, wseed)
                                if os.path.isfile(outfilestem):
                                    print "loading "+str(outfilestem)
                                else:
                                    outfilestem = '%s/p1-%.2f_p2-%.2f_p3-%.2f_p4-%.2f_p5-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, param3, param4, param5, iseed, wseed)
                                    if os.path.isfile(outfilestem):
                                        print "loading "+str(outfilestem)
                                    else: 
                                        outfilestem = '%s/p1-%.1f_p2-%d_p3-%.3f_p4-%d_p5-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, param3, param4, param5, iseed, wseed)
                                        if os.path.isfile(outfilestem):
                                            print "loading "+str(outfilestem)
                                        else:
                                            skipValue=1
                                    
                                if skipValue==0:
                                    # get errors from nqs
                                    h('nqaload = new NQS("%s")'%outfilestem)
                                    print("extracting trajectory data...")
                                    t=array(h.nqaload.getcol("t"))
                                    x=array(h.nqaload.getcol("x"))
                                    y=array(h.nqaload.getcol("y"))
                                    #print t,x,y
                            
                                    paramSubplots[iparam3*len(param4_range)+iparam4].scatter(x[trajStart:trajStop], y[trajStart:trajStop],  edgecolor=colorlist[iparam5], marker=trajMarker, linewidth=trajLine, s=trajSize) 
                
            # set tight layout and save figure
            tight_layout()
            fig.savefig('gif/%s_p1-%.2f_p2-%.2f_traj.png' % (simdatadir[5:], param1, param2)) # Save PNG files for movie
    close('all')        
        

    
# save spiking data and arm trajectory to matlab file


# plot center out reaching trajectories
def plotCenterOutAng4param(simdatadir, trajStartArg, trajStopArg):
    #######################
    ##### obtain target locations
    #########################
    targetsAng = calculateTargetsCenterOut()
    targetsAng=[[deg2rad(90),deg2rad(70)],[deg2rad(90),deg2rad(100)],[deg2rad(100),deg2rad(80)],[deg2rad(100),deg2rad(70)],[deg2rad(95),deg2rad(95)]]

    targetsAng=[[deg2rad(90),deg2rad(100)],[deg2rad(100),deg2rad(80)]] # only targets 14 and 15

    # convert to cartesian position
    varmLen=[0.4634 - 0.173, 0.7169 - 0.4634]
    varmCenter = angles2pos(0.62, 1.53, varmLen[0], varmLen[1])
    targetsPos = zeros((8,2))
    for i in range(len(targetsAng)):
        targetsPos[i]=angles2pos(targetsAng[i][0], targetsAng[i][1], varmLen[0], varmLen[1])
    
    ###########################
    # read and plot trajectory data (t,x,y from nqa) 
    ########################
    
    wseedvals =[120456,398115, 534031, 796321, 895199]
    iseedvals = [1235, 2837, 3955, 4506, 6789]

    # set param range by hand
    param1_range = arange(200,400,50) # EMNoiseRate train
    param2_range= arange(0, 8, 2) # EMNoiseMuscleGroup
    param3_range = arange(500,2000,500) # EMMuscleNoiseChangeDT
    param4_range = arange(8,40,8) # exploreTot
    param5_range = arange(4) # only four targets
    wseed_range = arange(2)
    inseed_range = arange(2)

    # read from param file
    with open('%s/params'% (simdatadir)) as f:
        param1_range, param2_range, param3_range, param4_range, param5_range, wseed_range, inseed_range = pickle.load(f)

    param5_range=arange(14,15,1)
    
    # Visualization options
    figsize =[1000,700] # Figure size in pixels
    targetColor = 'green';#array([(1,0.4,0) , (0,0.2,0.8)]) # Define excitatory and inhibitory colors -- orange and turquoise
    targetLine = 2 #
    targetSize = 30
    targetMarker = 'x'
    #colorlist = ['black','darkgrey','blue', 'cyan', 'green', 'brown', 'red', 'magenta', 'orange','yellow']
    colorlist = ['blue', 'green', 'red', 'magenta', 'darkgrey']
    trajLine = 1
    trajSize = 1
    trajSize2 = 1
    trajMarker = '.'
    trajMarker2 = '*'    
    trajStart = trajStartArg#2 # sample number to start from (each sample = 10 ms)
    trajStop = trajStopArg#400 # sample number to finish on (max 400=4sec)
        
    # create subplots for each of the param values tested
    ion()
    for param1 in param1_range:
        for param2 in param2_range:
            fig = figure(figsize=(figsize[0]/100,figsize[1]/100),dpi=100)
            fig.canvas.set_window_title('param1 = '+str(param1)+', param2 = '+str(param2))
            #fig.suptitle('param1 = '+str(param1)+', param2 = '+str(param2), fontsize=8)
            fig.subplots_adjust(left=0.02) # Less space on left
            fig.subplots_adjust(right=0.93) # Less space on right
            fig.subplots_adjust(bottom=0.08) # Less space on bottom
            subplotsy = len(param3_range)
            subplotsx = len(param4_range)
            maxSubplots = subplotsx*subplotsy
            paramSubplots =  [None] * min(len(param3_range)*len(param4_range), maxSubplots)
            border = 0.1
            for i in range(len(param3_range)*len(param4_range)):
                paramSubplots[i] = subplot(subplotsy, subplotsx, i+1)
                setp(paramSubplots[i].get_xticklabels(), visible=False) # hide x and y ticks
                setp(paramSubplots[i].get_yticklabels(), visible=False)
                #paramSubplots[i].set_xlim([targetsPos[1][0]-border, targetsPos[0][0]+border]) # set x-y lims
                #paramSubplots[i].set_ylim([targetsPos[3][1]-border, targetsPos[2][1]+border])
                
            # plot data for each of the 4 starting conditions in same color; different color for each target
            # Loop over param3 values
            h('objref nqaload')
            iparam3 = -1
            for param3 in param3_range:
                iparam3 = iparam3 + 1
                iparam4 = -1
                # Loop over param3 values
                for param4 in param4_range:
                    iparam4 = iparam4 + 1
                    iparam5 = -1
                    paramSubplots[iparam3*len(param4_range)+iparam4].set_title('param3 = '+str(param3)+', param4 = '+str(param4),fontsize=8)
                    # Loop over param5 values (target)
                    for param5 in param5_range:
                        iparam5 = iparam5 + 1
                        iwseed = -1
                        # plot target location
                        
                        #paramSubplots[iparam3*len(param4_range)+iparam4].scatter(targetsPos[iparam5][0], targetsPos[iparam5][1],  c=colorlist[iparam5], marker=targetMarker, linewidth=targetLine, s=targetSize)
                        # Loop over wiring seed...
                        for wseed in array(wseedvals)[wseed_range]:
                            iwseed = iwseed + 1
                            iiseed = -1
                            # Loop over input seed...
                            for iseed in array(iseedvals)[inseed_range]:
                                iiseed = iiseed + 1
                                skipValue=0
                                # set filename and check if file exists
                                outfilestem = '%s/p1-%d_p2-%d_p3-%d_p4-%d_p5-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, param3, param4, param5, iseed, wseed)
                                if os.path.isfile(outfilestem):
                                    print "loading "+str(outfilestem)
                                else:
                                    outfilestem = '%s/p1-%.2f_p2-%.2f_p3-%.2f_p4-%.2f_p5-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, param3, param4, param5, iseed, wseed)
                                    if os.path.isfile(outfilestem):
                                        print "loading "+str(outfilestem)
                                    else: 
                                        outfilestem = '%s/p1-%.1f_p2-%d_p3-%.3f_p4-%d_p5-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, param3, param4, param5, iseed, wseed)
                                        if os.path.isfile(outfilestem):
                                            print "loading "+str(outfilestem)
                                        else:
                                            skipValue=1
                                    
                                if skipValue==0:
                                    # get errors from nqs
                                    h('nqaload = new NQS("%s")'%outfilestem)
                                    print("extracting trajectory data...")
                                    t=array(h.nqaload.getcol("t"))
                                    x=array(h.nqaload.getcol("x"))
                                    y=array(h.nqaload.getcol("y"))
                                    shang=array(h.nqaload.getcol("ang0"))
                                    elang=array(h.nqaload.getcol("ang1"))
                                
                            
                                    #paramSubplots[iparam3*len(param4_range)+iparam4].scatter(x[trajStart:trajStop], y[trajStart:trajStop],  edgecolor=colorlist[iparam5], marker=trajMarker, linewidth=trajLine, s=trajSize)  
                                    # plot shoulder ang
                                    paramSubplots[iparam3*len(param4_range)+iparam4].scatter(t[trajStart:trajStop], shang[trajStart:trajStop],  edgecolor=colorlist[(iwseed*2)+iiseed], marker=trajMarker, linewidth=trajLine, s=trajSize)       
                                    # plot elbow ang
                                    paramSubplots[iparam3*len(param4_range)+iparam4].scatter(t[trajStart:trajStop], elang[trajStart:trajStop],  edgecolor=colorlist[(iwseed*2)+iiseed], marker=trajMarker2, linewidth=trajLine, s=trajSize2)   
                                    # plot shoulder+elbow targets
                                    #paramSubplots[iparam3*len(param4_range)+iparam4].plot([t[trajStart], t[trajStop]], [targetsAng[0][0], targetsAng[0][0]], color='black', linestyle='..')
                                    #paramSubplots[iparam3*len(param4_range)+iparam4].plot([t[trajStart], t[trajStop]], [targetsAng[0][1], targetsAng[0][1]], color='darkgrey', linestyle='..')

                                    paramSubplots[iparam3*len(param4_range)+iparam4].scatter(t[trajStart:trajStop], [targetsAng[0][0]]*len(t[trajStart:trajStop]), edgecolor='black', marker=trajMarker, linewidth=trajLine, s=trajSize)
                                    paramSubplots[iparam3*len(param4_range)+iparam4].scatter(t[trajStart:trajStop], [targetsAng[0][1]]*len(t[trajStart:trajStop]), edgecolor='darkgrey', marker=trajMarker, linewidth=trajLine, s=trajSize)



                
            # set tight layout and save figure
            tight_layout()
            fig.savefig('gif/%s_p1-%.2f_p2-%.2f_traj.png' % (simdatadir[5:], param1, param2)) # Save PNG files for movie
    close('all')        
        
# save spiking data and arm trajectory to matlab file


# plot center out reaching trajectories

def errorCenterOutTraj4param(simdatadir, trajStartArg, trajStopArg, accuracyFactor=0, deviationFactor=0):
    #######################
    ##### obtain target locations
    #########################
    targetsAng = calculateTargetsCenterOut()
    # convert to cartesian position
    varmLen=[0.4634 - 0.173, 0.7169 - 0.4634]
    varmCenter = angles2pos(0.62, 1.53, varmLen[0], varmLen[1])
    targetsPos = zeros((8,2))
    for i in range(len(targetsAng)):
        targetsPos[i]=angles2pos(targetsAng[i][0], targetsAng[i][1], varmLen[0], varmLen[1])
    
    ###########################
    # read and plot trajectory data (t,x,y from nqa) 
    ########################
    
    wseedvals =[120456,398115, 534031, 796321, 895199]
    iseedvals = [1235, 2837, 3955, 4506, 6789]
    
    param1_range = arange(200,400,50) # EMNoiseRate train
    param2_range= arange(0, 8, 2) # EMNoiseMuscleGroup
    param3_range = arange(500,2000,500) # EMMuscleNoiseChangeDT
    param4_range = arange(8,40,8) # exploreTot
    param5_range = arange(4) # only four targets
    wseed_range = arange(2)
    inseed_range = arange(2)

    with open('%s/params'% (simdatadir)) as f:
        param1_range, param2_range, param3_range, param4_range, param5_range, wseed_range, inseed_range = pickle.load(f)

    # Visualization options
    figsize =[700,700] # Figure size in pixels
    targetColor = 'green';#array([(1,0.4,0) , (0,0.2,0.8)]) # Define excitatory and inhibitory colors -- orange and turquoise
    targetLine = 2 #
    targetSize = 30
    targetMarker = 'x'
    #colorlist = ['black','darkgrey','blue', 'cyan', 'green', 'brown', 'red', 'magenta', 'orange','yellow']
    colorlist = ['blue', 'green', 'red', 'magenta']
    trajLine = 1
    trajSize = 1
    trajMarker = '.'    
    trajStart = trajStartArg#2 # sample number to start from (each sample = 10 ms)
    trajStop = trajStopArg#400 # sample number to finish on (max 400=4sec)

    fig = figure(figsize=(figsize[0]/100,figsize[1]/100),dpi=100)
    #fig.canvas.set_window_title('param1 = '+str(param1)+', param2 = '+str(param2))
    #fig.suptitle('param1 = '+str(param1)+', param2 = '+str(param2), fontsize=8)
    fig.subplots_adjust(left=0.02) # Less space on left
    fig.subplots_adjust(right=0.93) # Less space on right
    fig.subplots_adjust(bottom=0.08) # Less space on bottom
    subplotsy = len(param1_range)
    subplotsx = len(param2_range)
    maxSubplots = subplotsx*subplotsy
    paramSubplots =  [None] * maxSubplots #min(len(param2_range)*len(param2_range), maxSubplots)
    border = 0.1
    for i in range(len(param1_range)*len(param2_range)):
        paramSubplots[i] = subplot(subplotsy, subplotsx, i+1)
        
    # create subplots for each of the param values tested
    errorAvg = zeros((len(param1_range),len(param2_range),len(param3_range),len(param4_range)))
    ion()
    iparam1 =-1
    for param1 in param1_range:
        iparam1 = iparam1 + 1
        iparam2 = -1
        for param2 in param2_range:
            iparam2 = iparam2 + 1
            # plot data for each of the 4 starting conditions in same color; different color for each target
            # Loop over param3 values
            
            accuracy = zeros((len(param3_range),len(param4_range),len(param5_range),len(wseed_range),len(inseed_range)))
            deviation = zeros((len(param3_range),len(param4_range),len(param5_range),len(wseed_range),len(inseed_range)))

            accuracy_mean = zeros((len(param3_range),len(param4_range),len(param5_range)))
            deviation_mean = zeros((len(param3_range),len(param4_range),len(param5_range)))

            accuracy_max = zeros((len(param3_range),len(param4_range)))
            deviation_max = zeros((len(param3_range),len(param4_range)))

            
            h('objref nqaload')
            iparam3 = -1
            for param3 in param3_range:
                iparam3 = iparam3 + 1
                iparam4 = -1
                paramSubplots[iparam1*len(param2_range)+iparam2].set_title('param1 = '+str(param1)+', param2 = '+str(param2),fontsize=8)
                # Loop over param3 values
                for param4 in param4_range:
                    iparam4 = iparam4 + 1
                    iparam5 = -1
                    # Loop over param5 values (target)
                    for param5 in param5_range:
                        iparam5 = iparam5 + 1
                        iwseed = -1
                        # Loop over wiring seed...
                        for wseed in array(wseedvals)[wseed_range]:
                            iwseed = iwseed + 1
                            iiseed = -1
                            # Loop over input seed...
                            for iseed in array(iseedvals)[inseed_range]:
                                iiseed = iiseed + 1
                                skipValue=0
                                # set filename and check if file exists
                                outfilestem = '%s/p1-%d_p2-%d_p3-%d_p4-%d_p5-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, param3, param4, param5, iseed, wseed)
                                if os.path.isfile(outfilestem):
                                    print "loading "+str(outfilestem)
                                else:
                                    outfilestem = '%s/p1-%.2f_p2-%.2f_p3-%.2f_p4-%.2f_p5-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, param3, param4, param5, iseed, wseed)
                                    if os.path.isfile(outfilestem):
                                        print "loading "+str(outfilestem)
                                    else: 
                                        outfilestem = '%s/p1-%.1f_p2-%d_p3-%.3f_p4-%d_p5-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, param3, param4, param5, iseed, wseed)
                                        if os.path.isfile(outfilestem):
                                            print "loading "+str(outfilestem)
                                        else:
                                            skipValue=1
                                    
                                if skipValue==0:
                                    # get errors from nqs
                                    h('nqaload = new NQS("%s")'%outfilestem)
                                    print("extracting trajectory data...")
                                    t=array(h.nqaload.getcol("t"))
                                    x=array(h.nqaload.getcol("x"))
                                    y=array(h.nqaload.getcol("y"))
                                    #print t,x,y

                                    accuracy[iparam3,iparam4, iparam5, iwseed, iiseed] = trajAccuracy(x[trajStart:trajStop], y[trajStart:trajStop], targetsPos[param5], varmLen, [int(0.1*(trajStop-trajStart-1)),trajStop-trajStart-1])
                                    deviation[iparam3,iparam4, iparam5, iwseed, iiseed] = trajDeviation(x[trajStart:trajStop], y[trajStart:trajStop], varmCenter, param5, targetsPos[0][0]) 

                        # calculate mean over seeds
                        accuracy_mean[iparam3,iparam4, iparam5]= mean(accuracy[iparam3,iparam4,iparam5,:,:])
                        deviation_mean[iparam3,iparam4, iparam5]= mean(deviation[iparam3,iparam4,iparam5,:,:])

                    # calculate max over targets
                    accuracy_max[iparam3,iparam4]= max(accuracy_mean[iparam3,iparam4,:])
                    deviation_max[iparam3,iparam4]= max(deviation_mean[iparam3,iparam4,:])

                    # calcualate error as max of acc and dev for each param3 and param4
                    #error = accuracyFactor*accuracy+deviationFactor*deviation

                    # store 
                    #errorAvg[iparam1, iparam2, iparam3,iparam4] = mean(error[iparam3,iparam4,:,:,:])
                    errorAvg[iparam1, iparam2, iparam3,iparam4] = max(accuracy_max[iparam3,iparam4], deviation_max[iparam3,iparam4])
                                                        
    #print errorAvg

    iparam1 =-1
    for param1 in param1_range:
        iparam1 = iparam1 + 1
        iparam2 = -1
        for param2 in param2_range:
            iparam2 = iparam2 + 1

            # plot error results
            #imsubplot=paramSubplots[iparam1*len(param2_range)+iparam2].imshow(errorAvg, vmin=0, vmax=0.04, origin='upper', interpolation='none')
            imsubplot=paramSubplots[iparam1*len(param2_range)+iparam2].pcolor(errorAvg[iparam1,iparam2], vmin=nanmin(errorAvg), vmax=nanmax(errorAvg))
            paramSubplots[iparam1*len(param2_range)+iparam2].invert_yaxis()

            paramSubplots[iparam1*len(param2_range)+iparam2].set_xticklabels(param4_range)
            paramSubplots[iparam1*len(param2_range)+iparam2].set_xticks(arange(0.5, len(param4_range), 1))
            paramSubplots[iparam1*len(param2_range)+iparam2].set_yticklabels(param3_range)
            paramSubplots[iparam1*len(param2_range)+iparam2].set_yticks(arange(0.5, len(param3_range), 1))
            paramSubplots[iparam1*len(param2_range)+iparam2].tick_params(labelsize=8)


            #scatter(x[trajStart:trajStop], y[trajStart:trajStop],  edgecolor=colorlist[iparam5], marker=trajMarker, linewidth=trajLine, s=trajSize) 
                
    # set tight layout and save figure
    cfraction = 1
    cpad = 1
    cshrink = 1
    cfontsize = 8
    #c=colorbar(paramSubplots[iparam1*len(param2_range)+iparam2, ax=rasterPsubplot, fraction=cfraction, pad=cpad, shrink=cshrink)
    tight_layout()
    c=colorbar(imsubplot,ax=paramSubplots[iparam1*len(param2_range)+iparam2])
    c.ax.tick_params(labelsize=cfontsize) 
    fig.savefig('gif/%s_error.png' % (simdatadir[5:])) # Save PNG files for movie
    
# evaluation function over 4 tragets -- note: not adapted to multiple seeds
def errorCenterOutTraj(simdatafile, trajStart=2, trajStop=200, accFactor=1, devFactor=0):
    
    simdatafiles = []
    simdatafiles.append(simdatafile.replace("target_3","target_0"))
    simdatafiles.append(simdatafile.replace("target_3","target_1"))
    simdatafiles.append(simdatafile.replace("target_3","target_2"))
    simdatafiles.append(simdatafile)

    #######################
    ##### obtain target locations
    #########################
    targetsAng = calculateTargetsCenterOut()
    # convert to cartesian position
    varmLen=[0.4634 - 0.173, 0.7169 - 0.4634]
    varmCenter = angles2pos(0.62, 1.53, varmLen[0], varmLen[1])
    targetsPos = zeros((8,2))
    for i in range(len(targetsAng)):
        targetsPos[i]=angles2pos(targetsAng[i][0], targetsAng[i][1], varmLen[0], varmLen[1])
    
    ###########################
    # read and plot trajectory data (t,x,y from nqa) 
    ########################
    
    accuracy = zeros((len(simdatafiles)))
    deviation = zeros((len(simdatafiles)))

    # create subplots for each of the param values tested
    ifile =-1
    for filename in simdatafiles:
        ifile = ifile + 1
            
        h('objref nqaload')
        full_filename = '%s_test-nqa.nqs' % (filename)
        skipValue=0
        if os.path.isfile(full_filename):
            print "loading "+str(full_filename)
        else:
            skipValue=1
            
        if skipValue==0:
            # get errors from nqs
            h('nqaload = new NQS("%s")'%full_filename)
            print("extracting trajectory data...")
            t=array(h.nqaload.getcol("t"))
            x=array(h.nqaload.getcol("x"))
            y=array(h.nqaload.getcol("y"))
            #print t,x,y

            accuracy[ifile] = trajAccuracy(x[trajStart:trajStop], y[trajStart:trajStop], targetsPos[ifile], varmLen, [int(0.01*(trajStop-trajStart-1)),trajStop-trajStart-1], 'sum')
            deviation[ifile] = trajDeviation(x[trajStart:trajStop], y[trajStart:trajStop], varmCenter, ifile, targetsPos[0][0]) 

    # calculate max over targets
    print "accuracy:"
    print accuracy
    print "deviation:"
    print deviation
    accuracy_max= accFactor*max(accuracy)
    deviation_max= devFactor*max(deviation)

    error = max(accuracy_max, deviation_max)
    #saveFile=simdatafile[0:simdatafile.find("_iter")]
    saveFile=simdatafile[0:simdatafile.find("_target")]
    with open('%s_errortmp'% (saveFile), 'w') as f:
        pickle.dump(error, f)

    return error

def errorCenterOutTrajTmp(simdatafile, trajStart=2, trajStop=200):
    simdatafiles = []
    simdatafiles.append(simdatafile.replace("target_3","target_0"))
    simdatafiles.append(simdatafile.replace("target_3","target_1"))
    simdatafiles.append(simdatafile.replace("target_3","target_2"))
    simdatafiles.append(simdatafile)

    #######################
    ##### obtain target locations
    #########################
    targetsAng = calculateTargetsCenterOut()
    # convert to cartesian position
    varmLen=[0.4634 - 0.173, 0.7169 - 0.4634]
    varmCenter = angles2pos(0.62, 1.53, varmLen[0], varmLen[1])
    targetsPos = zeros((8,2))
    for i in range(len(targetsAng)):
        targetsPos[i]=angles2pos(targetsAng[i][0], targetsAng[i][1], varmLen[0], varmLen[1])
    
    ###########################
    # read and plot trajectory data (t,x,y from nqa) 
    ########################
    
    accuracy = zeros((len(simdatafiles)))
    deviation = zeros((len(simdatafiles)))

    # create subplots for each of the param values tested
    ifile =-1
    for filename in simdatafiles:
        ifile = ifile + 1
            
        h('objref nqaload')
        full_filename = '%s_test-nqa.nqs' % (filename)
        skipValue=0
        if os.path.isfile(full_filename):
            print "loading "+str(full_filename)
        else:
            skipValue=1
            
        if skipValue==0:
            # get errors from nqs
            h('nqaload = new NQS("%s")'%full_filename)
            print("extracting trajectory data...")
            t=array(h.nqaload.getcol("t"))
            x=array(h.nqaload.getcol("x"))
            y=array(h.nqaload.getcol("y"))
            #print t,x,y

            accuracy[ifile] = trajAccuracy(x[trajStart:trajStop], y[trajStart:trajStop], targetsPos[ifile], varmLen, [int(0.1*(trajStop-trajStart-1)),trajStop-trajStart-1])
            deviation[ifile] = trajDeviation(x[trajStart:trajStop], y[trajStart:trajStop], varmCenter, ifile, targetsPos[0][0]) 

    # calculate max over targets
    print "accuracy:"
    print accuracy
    print "deviation:"
    print deviation
    accuracy_max= max(accuracy)
    deviation_max= max(deviation)

    error = max(accuracy_max, deviation_max)
    saveFile=simdatafile[0:simdatafile.find("_iter")]
    #with open('%s_errortmp'% (saveFile), 'w') as f:
    #    pickle.dump(error, f)

    return error
    
    
# save spiking data and arm trajectory to matlab file
def save2Matlab(simdatadir, param1Arg, t1Arg, t2Arg):
    ###########################
    # read and save spike and trajectory data (t,x,y from nqa) 
    ########################
    wseedvals =[120456]#, 398115]#, 534031, 796321, 895199]
    iseedvals =     [1235, 2837, 3955, 4506, 6789, 1236, 2838, 3956, 4507, 6790, 1237, 2839, 3957, 4508, 6791, 1238, 2840, 3958, 4509, 6792]# [1235, 2837]#, 3955, 4506, 6789]
    
    # parameter values
    param2_range = [0,1]#[0,1,2,3,4,5,6,7]

            
    # plot data for each of the 4 starting conditions in same color; different color for each target
    # Loop over param1 values
    h('objref nqaload')
    h('objref spikesload')
    param1=param1Arg
    iparam2 = -1
    # Loop over param2 values
    for param2 in param2_range:
        iparam2 = iparam2 + 1
        iwseed = -1
        # Loop over wiring seed...
        for wseed in wseedvals:
            iwseed = iwseed + 1
            iiseed = -1
            # Loop over input seed...
            for iseed in iseedvals:
                iiseed = iiseed + 1
                # set filename
                #outfilestem = '"%s/p1-%.2f_p2-%d_i-%d_w-%d_test-nqa.nqs"' % (simdatadir, param1, param2, iseed, wseed)
                outfilestem = '"%s/p1-%d_p2-%d_i-%d_w-%d_test-nqa.nqs"' % (simdatadir, param1, param2, iseed, wseed)
                #outfilestem = '"%s/target-%d_i-%d_w-%d_test-nqa.nqs"' % (simdatadir, param2, iseed, wseed)
                outfilestem_format = '%s/p1-%d_p2-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param1, param2, iseed, wseed)
                
                # get fields from nqs
                if os.path.isfile(outfilestem_format):
                    try:
                        h('nqaload = new NQS(%s)'%outfilestem)
                        h('nqaload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                        t=array(h.nqaload.getcol("t"))
                        xHand = array(h.nqaload.getcol("x"))
                        yHand = array(h.nqaload.getcol("y"))
                        xElbow = array(h.nqaload.getcol("ex"))
                        yElbow = array(h.nqaload.getcol("ey"))
                        shAng = array(h.nqaload.getcol("ang0"))
                        elAng = array(h.nqaload.getcol("ang1"))
                        shExtMusLength = array(h.nqaload.getcol("ML0"))
                        shFlexMusLength =  array(h.nqaload.getcol("ML1"))
                        elExtMusLength = array(h.nqaload.getcol("ML2"))
                        elFlexMusLength = array(h.nqaload.getcol("ML3"))
                        
                        #outfilestem = '"%s/p1-%.2f_p2-%d_i-%d_w-%d_test-spk.nqs"' % (simdatadir, param1, param2, iseed, wseed)
                        outfilestem = '"%s/p1-%d_p2-%d_i-%d_w-%d_test-spk.nqs"' % (simdatadir, param1, param2, iseed, wseed)

                        h('spikesload = new NQS(%s)' % outfilestem)
                        h('spikesload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                        tspike=array(h.spikesload.getcol("t"))
                        cellid=array(h.spikesload.getcol("id"))
                        celltype=array(h.spikesload.getcol("type"))
                        muscleid=array(h.spikesload.getcol("mid"))
                        
                        #save to matlab
                        #scipy.io.savemat(('bmm_%s_p1-%.2f_p2-%d_i-%d_w-%d_test_spk.mat'%simdatadir, param1, param2, iseed, wseed), \
                        #mdict={'t': t, 'cellId': 'cellId, 'celltype': celltype, 'muscleid': muscleid})    

                        scipy.io.savemat(('%s_target%d_i%d_w%d_spk.mat'%(simdatadir, iparam2, iiseed, iwseed)), \
                        mdict={'tspike': tspike, 'cellid': cellid, 'celltype': celltype, 'muscleid': muscleid})    
                                    
                        scipy.io.savemat(('%s_target%d_i%d_w%d_arm.mat'%(simdatadir, iparam2, iiseed, iwseed)), \
                        mdict={'t': t, 'xhand': xHand, 'yHand': yHand, 'xElbow': xElbow,  'yElbow': yElbow, 'shAng': shAng, 'elAng': elAng, 'shExtMusLength': shExtMusLength, 'shFlexMusLength': shFlexMusLength,  'elExtMusLength': elExtMusLength, 'elFlexMusLength': elFlexMusLength })    
                    except:
                        pass

# save spiking data and arm trajectory to matlab file
def save2Matlab_iseeds(simdatadir, t1Arg=20, t2Arg=1000, simtype='test'):
    ###########################
    # read and save spike and trajectory data (t,x,y from nqa) 
    ########################
    #wseedvals =[120456]#, 398115]#, 534031, 796321, 895199]
    #iseedvals =     [1235, 2837, 3955, 4506, 6789, 1236, 2838, 3956, 4507, 6790, 1237, 2839, 3957, 4508, 6791, 1238, 2840, 3958, 4509, 6792]# [1235, 2837]#, 3955, 4506, 6789]
    wseedvals =[120456, 398115, 534031, 796321, 895199]
    iseedvals = arange(1235,  1235+(17*100), 17)
    
    # parameter values
    param2_range = [0,1,2,3]#,4,5,6,7]

            
    # plot data for each of the 4 starting conditions in same color; different color for each target
    # Loop over param1 values
    h('objref nqaload')
    h('objref spikesload')
    iparam2 = -1
    # Loop over param2 values
    for param2 in param2_range:
        iparam2 = iparam2 + 1
        iwseed = -1
        # Loop over wiring seed...
        for wseed in wseedvals:
            iwseed = iwseed + 1
            iiseed = -1
            # Loop over input seed...
            for iseed in iseedvals:
                iiseed = iiseed + 1
                # set filename
                outfilestem = '"%s/target-%d_i-%d_w-%d_%s-nqa.nqs"' % (simdatadir, param2, iseed, wseed, simtype)
                outfilestem_format = '%s/target-%d_i-%d_w-%d_%s-nqa.nqs' % (simdatadir, param2, iseed, wseed, simtype)

                # get fields from nqs
                if os.path.isfile(outfilestem_format):
                    try:
                        # get fields from nqs
                        h('nqaload = new NQS(%s)'%outfilestem)
                        h('nqaload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                        t=array(h.nqaload.getcol("t"))
                        xHand = array(h.nqaload.getcol("x"))
                        yHand = array(h.nqaload.getcol("y"))
                        xElbow = array(h.nqaload.getcol("ex"))
                        yElbow = array(h.nqaload.getcol("ey"))
                        shAng = array(h.nqaload.getcol("ang0"))
                        elAng = array(h.nqaload.getcol("ang1"))
                        shExtMusLength = array(h.nqaload.getcol("ML0"))
                        shFlexMusLength =  array(h.nqaload.getcol("ML1"))
                        elExtMusLength = array(h.nqaload.getcol("ML2"))
                        elFlexMusLength = array(h.nqaload.getcol("ML3"))
                        
                        outfilestem = '"%s/target-%d_i-%d_w-%d_%s-spk.nqs"' % (simdatadir, param2, iseed, wseed, simtype)
                                              
                        h('spikesload = new NQS(%s)' % outfilestem)
                        h('spikesload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                        tspike=array(h.spikesload.getcol("t"))
                        cellid=array(h.spikesload.getcol("id"))
                        celltype=array(h.spikesload.getcol("type"))
                        muscleid=array(h.spikesload.getcol("mid"))
                        
                        #save to matlab
                       
                        scipy.io.savemat(('%s/target%d_i%d_w%d_spk.mat'%(simdatadir, iparam2, iiseed, iwseed)), \
                        mdict={'tspike': tspike, 'cellid': cellid, 'celltype': celltype, 'muscleid': muscleid})    
                                    
                        scipy.io.savemat(('%s/target%d_i%d_w%d_arm.mat'%(simdatadir, iparam2, iiseed, iwseed)), \
                        mdict={'t': t, 'xhand': xHand, 'yHand': yHand, 'xElbow': xElbow,  'yElbow': yElbow, 'shAng': shAng, 'elAng': elAng, 'shExtMusLength': shExtMusLength, 'shFlexMusLength': shFlexMusLength,  'elExtMusLength': elExtMusLength, 'elFlexMusLength': elFlexMusLength })    
                    except:
                        pass


# save spiking data and arm trajectory to matlab file
def save2Matlab_traintimes(simdatadir, t1Arg, t2Arg):
    ###########################
    # read and save spike and trajectory data (t,x,y from nqa) 
    ########################
    #wseedvals =[120456]#, 398115]#, 534031, 796321, 895199]
    #iseedvals =     [1235, 2837, 3955, 4506, 6789, 1236, 2838, 3956, 4507, 6790, 1237, 2839, 3957, 4508, 6791, 1238, 2840, 3958, 4509, 6792]# [1235, 2837]#, 3955, 4506, 6789]
    wseedvals =[120456]#, 398115]#, 534031, 796321, 895199]
    iseedvals = arange(1235,  1235+(17*100), 17)
    
    # parameter values
    param2_range = [0,1,2,3]#[0,1,2,3,4,5,6,7]
    tphase1_range = [0, 2, 4, 6, 8, 10]  # range of training epochs for phase 1
    tphase2_range = [0, 10, 20, 30, 40, 50, 60] # range of training epochs for phase 2
            
    # plot data for each of the 4 starting conditions in same color; different color for each target
    # Loop over param1 values
    h('objref nqaload')
    h('objref spikesload')
    iparam2 = -1
    # Loop over param2 values
    for param2 in param2_range:
        iparam2 = iparam2 + 1
        for tphase1 in tphase1_range:
            # loop over epochs for training phase 2
            for tphase2 in tphase2_range:
                iwseed = -1
                # Loop over wiring seed...
                for wseed in wseedvals:
                    iwseed = iwseed + 1
                    iiseed = -1
                    # Loop over input seed...
                    for iseed in iseedvals:
                        iiseed = iiseed + 1
                        # set filename
                        outfilestem = '"%s/target-%d_tp1-%d_tp2-%d_i-%d_w-%d_test-nqa.nqs"' % (simdatadir, param2, tphase1, tphase2, iseed, wseed)
                        outfilestem_format = '%s/target-%d_tp1-%d_tp2-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param2, tphase1, tphase2, iseed, wseed)
                        
                        print
                        # get fields from nqs
                        if os.path.isfile(outfilestem_format):
                            try:
                                # get fields from nqs
                                h('nqaload = new NQS(%s)'%outfilestem)
                                h('nqaload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                                t=array(h.nqaload.getcol("t"))
                                xHand = array(h.nqaload.getcol("x"))
                                yHand = array(h.nqaload.getcol("y"))
                                xElbow = array(h.nqaload.getcol("ex"))
                                yElbow = array(h.nqaload.getcol("ey"))
                                shAng = array(h.nqaload.getcol("ang0"))
                                elAng = array(h.nqaload.getcol("ang1"))
                                shExtMusLength = array(h.nqaload.getcol("ML0"))
                                shFlexMusLength =  array(h.nqaload.getcol("ML1"))
                                elExtMusLength = array(h.nqaload.getcol("ML2"))
                                elFlexMusLength = array(h.nqaload.getcol("ML3"))
                                
                                outfilestem = '"%s/target-%d_tp1-%d_tp2-%d_i-%d_w-%d_test-spk.nqs"' % (simdatadir, param2, tphase1, tphase2, iseed, wseed)

                                h('spikesload = new NQS(%s)' % outfilestem)
                                h('spikesload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                                tspike=array(h.spikesload.getcol("t"))
                                cellid=array(h.spikesload.getcol("id"))
                                celltype=array(h.spikesload.getcol("type"))
                                muscleid=array(h.spikesload.getcol("mid"))
                                
                                #save to matlab
                               
                                scipy.io.savemat(('%s/target%d_tp1%d_tp2%d_i%d_w%d_spk.mat'%(simdatadir, iparam2, tphase1, tphase2, iiseed, iwseed)), \
                                mdict={'tspike': tspike, 'cellid': cellid, 'celltype': celltype, 'muscleid': muscleid})    
                                            
                                scipy.io.savemat(('%s/target%d_tp1%d_tp2%d_i%d_w%d_arm.mat'%(simdatadir, iparam2, tphase1, tphase2, iiseed, iwseed)), \
                                mdict={'t': t, 'xhand': xHand, 'yHand': yHand, 'xElbow': xElbow,  'yElbow': yElbow, 'shAng': shAng, 'elAng': elAng, 'shExtMusLength': shExtMusLength, 'shFlexMusLength': shFlexMusLength,  'elExtMusLength': elExtMusLength, 'elFlexMusLength': elFlexMusLength })    
                            except:
                                pass



                    
def save2MatlabTrain(simdatadir, param1Arg, t1Arg, t2Arg):
    ###########################
    # read and save spike and trajectory data (t,x,y from nqa) 
    ########################
    wseedvals =[120456]#, 398115]#, 534031, 796321, 895199]
    iseedvals = [1235]#, 2837]#, 3955, 4506, 6789]
    
    # parameter values
    param2_range = [0,1,2,3,4,5,6,7]

            
    # plot data for each of the 4 starting conditions in same color; different color for each target
    # Loop over param1 values
    h('objref nqaload')
    h('objref spikesload')
    param1=param1Arg
    iparam2 = -1
    # Loop over param2 values
    for param2 in param2_range:
        iparam2 = iparam2 + 1
        iwseed = -1
        # Loop over wiring seed...
        for wseed in wseedvals:
            iwseed = iwseed + 1
            iiseed = -1
            # Loop over input seed...
            for iseed in iseedvals:
                iiseed = iiseed + 1
                # set filename
                #outfilestem = '"%s/p1-%.2f_p2-%d_i-%d_w-%d_train-nqa.nqs"' % (simdatadir, param1, param2, iseed, wseed)
                outfilestem = '"%s/p1-%d_p2-%d_i-%d_w-%d_train-nqa.nqs"' % (simdatadir, param1, param2, iseed, wseed)

                # get fields from nqs
                h('nqaload = new NQS(%s)'%outfilestem)
                h('nqaload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                t=array(h.nqaload.getcol("t"))
                xHand = array(h.nqaload.getcol("x"))
                yHand = array(h.nqaload.getcol("y"))
                xElbow = array(h.nqaload.getcol("ex"))
                yElbow = array(h.nqaload.getcol("ey"))
                shAng = array(h.nqaload.getcol("ang0"))
                elAng = array(h.nqaload.getcol("ang1"))
                shExtMusLength = array(h.nqaload.getcol("ML0"))
                shFlexMusLength =  array(h.nqaload.getcol("ML1"))
                elExtMusLength = array(h.nqaload.getcol("ML2"))
                elFlexMusLength = array(h.nqaload.getcol("ML3"))
                
                #outfilestem = '"%s/p1-%.2f_p2-%d_i-%d_w-%d_test-spk.nqs"' % (simdatadir, param1, param2, iseed, wseed)
                outfilestem = '"%s/p1-%d_p2-%d_i-%d_w-%d_train-spk.nqs"' % (simdatadir, param1, param2, iseed, wseed)

                h('spikesload = new NQS(%s)' % outfilestem)
                h('spikesload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                tspike=array(h.spikesload.getcol("t"))
                cellid=array(h.spikesload.getcol("id"))
                celltype=array(h.spikesload.getcol("type"))
                muscleid=array(h.spikesload.getcol("mid"))
                
                #save to matlab
                #scipy.io.savemat(('bmm_%s_p1-%.2f_p2-%d_i-%d_w-%d_test_spk.mat'%simdatadir, param1, param2, iseed, wseed), \
                #mdict={'t': t, 'cellId': 'cellId, 'celltype': celltype, 'muscleid': muscleid})    

                scipy.io.savemat(('%s_target%d_i%d_w%d_spk.mat'%(simdatadir, iparam2, iiseed, iwseed)), \
                mdict={'tspike': tspike, 'cellid': cellid, 'celltype': celltype, 'muscleid': muscleid})    
                            
                scipy.io.savemat(('%s_target%d_i%d_w%d_arm.mat'%(simdatadir, iparam2, iiseed, iwseed)), \
                mdict={'t': t, 'xhand': xHand, 'yHand': yHand, 'xElbow': xElbow,  'yElbow': yElbow, 'shAng': shAng, 'elAng': elAng, 'shExtMusLength': shExtMusLength, 'shFlexMusLength': shFlexMusLength,  'elExtMusLength': elExtMusLength, 'elFlexMusLength': elFlexMusLength })    

def save2MatlabMistPerturb(simdatadir, t1Arg, t2Arg):
    ###########################
    # read and save spike and trajectory data (t,x,y from nqa) 
    ########################
    wseedvals =[120456, 398115, 534031, 796321, 895199]
    iseedvals = [1235, 2837, 3955, 4506, 6789]

    param1_range = arange(0,192,1) # mistCell
    param2_range= arange(200, 600, 200) # mistStart
    param3_range = arange(200,201) # mistDuration
    param4_range = arange(250,501,250) # mistRate
    param5_range = arange(1) # only four targets
    wseed_range = arange(1)
    inseed_range = arange(1)

    param1_range = arange(0,192)#192,1) # mistCell
    param2_range= arange(0,1)# 600, 200) # mistStart
    param3_range = arange(1,2) # mistDuration
    param4_range = arange(1,2)#(501,250) # mistRate
    param5_range = arange(1) # only four targets
    wseed_range = arange(1)
    inseed_range = arange(1)

    #with open('%s/params'% (simdatadir)) as f:
    #    param1_range, param2_range, param3_range, param4_range, param5_range, wseed_range, inseed_range = pickle.load(f)

    param1_range = arange(20,192,1) # mistCell
        
    # plot data for each of the 4 starting conditions in same color; different color for each target
    # Loop over param1 values
    h('objref nqaload')
    h('objref spikesload')
    # Loop over param1 vals
    for param1 in param1_range:
        # Loop over param2 vals
        for param2 in param2_range:
            # Loop over param3 vals
            for param3 in param3_range:
                # Loop over param4 vals
                for param4 in param4_range:
                    # Loop over wiring seed...
                    for wseed in wseed_range:
                        # Loop over input seed...
                        for iseed in inseed_range:
                            # set filename
                            #outfilestem = '"%s/p1-%d_p2-%d_i-%d_w-%d_train-nqa.nqs"' % (simdatadir, param1, param2, iseed, wseed)
                            outfilestem = '%s/cell-%d_start-%d_dur-%d_rate-%d_test-nqa.nqs' % (simdatadir, param1, param2, param3, param4)
                            
                            # get fields from nqs
                            if os.path.isfile(outfilestem):
                                outfilestem = '"%s/cell-%d_start-%d_dur-%d_rate-%d_test-nqa.nqs"' % (simdatadir, param1, param2, param3, param4)
                                h('nqaload = new NQS(%s)'%outfilestem)
                                h('nqaload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                                t=array(h.nqaload.getcol("t"))
                                xHand = array(h.nqaload.getcol("x"))
                                yHand = array(h.nqaload.getcol("y"))
                                xElbow = array(h.nqaload.getcol("ex"))
                                yElbow = array(h.nqaload.getcol("ey"))
                                shAng = array(h.nqaload.getcol("ang0"))
                                elAng = array(h.nqaload.getcol("ang1"))
                                shExtMusLength = array(h.nqaload.getcol("ML0"))
                                shFlexMusLength =  array(h.nqaload.getcol("ML1"))
                                elExtMusLength = array(h.nqaload.getcol("ML2"))
                                elFlexMusLength = array(h.nqaload.getcol("ML3"))

                                #outfilestem = '"%s/p1-%.2f_p2-%d_i-%d_w-%d_test-spk.nqs"' % (simdatadir, param1, param2, iseed, wseed)
                                outfilestem = '"%s/cell-%d_start-%d_dur-%d_rate-%d_test-spk.nqs"' % (simdatadir, param1, param2, param3, param4)

                                h('spikesload = new NQS(%s)' % outfilestem)
                                h('spikesload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                                tspike=array(h.spikesload.getcol("t"))
                                cellid=array(h.spikesload.getcol("id"))
                                celltype=array(h.spikesload.getcol("type"))
                                muscleid=array(h.spikesload.getcol("mid"))

                                #save to matlab
                                #scipy.io.savemat(('bmm_%s_p1-%.2f_p2-%d_i-%d_w-%d_test_spk.mat'%simdatadir, param1, param2, iseed, wseed), \
                                #mdict={'t': t, 'cellId': 'cellId, 'celltype': celltype, 'muscleid': muscleid})    

                                scipy.io.savemat(('%s/cell-%d_start-%d_dur-%d_rate-%d_spk.mat'%(simdatadir, param1, param2, param3, param4)), \
                                mdict={'tspike': tspike, 'cellid': cellid, 'celltype': celltype, 'muscleid': muscleid})    

                                scipy.io.savemat(('%s/cell-%d_start-%d_dur-%d_rate-%d_arm.mat'%(simdatadir, param1, param2, param3, param4)), \
                                mdict={'t': t, 'xhand': xHand, 'yHand': yHand, 'xElbow': xElbow,  'yElbow': yElbow, 'shAng': shAng, 'elAng': elAng, 'shExtMusLength': shExtMusLength, 'shFlexMusLength': shFlexMusLength,  'elExtMusLength': elExtMusLength, 'elFlexMusLength': elFlexMusLength })    

def save2MatlabMistProbing(simdatadir, t1Arg, t2Arg):
    target_range = [1,3] #arange(4) # only four targets
    param1_range = [0,1] # kill cells or synapses
    param2_range = [0,5,10] # kill 5% or 10%
    param3_range = arange(0,99)#arange(0,191,1) # mistCell
    param4_range = [0]#[400] # mistStart
    param5_range = [0]#[200] # mistDuration
    param6_range = [0]#[250] # mistRate

    param1_range = [0,1] #kill cells or synapses
    param2_range = [5,10] # kill 5% or 10%
    param3_range = arange(0,191)#arange(0,191,1) # mistCell
    param4_range = [100]#[400] # mistStart
    param5_range = [200]#[200] # mistDuration
    param6_range = [250]#[250] # mistRate

    h('objref nqaload')
    h('objref spikesload')

    # Loop over target vals
    for target in target_range:
        # Loop over param1 vals
        for param1 in param1_range:
            # Loop over param2 vals
            for param2 in param2_range:
                # Loop over param3 vals
                for param3 in param3_range:
                    # Loop over param4 vals
                    for param4 in param4_range:
                        # Loop over param5 vals
                        for param5 in param5_range:
                            # Loop over param6 vals
                            for param6 in param6_range:

                                # set filename
                                outfilestem = '%s/target-%d_ptype-%d_pperc-%d_cell-%d_start-%d_dur-%d_rate-%d_test-nqa.nqs' % \
                                (simdatadir, target, param1, param2, param3, param4, param5, param6)
                                
                                # get fields from nqs
                                if os.path.isfile(outfilestem):
                                    outfilestem = '"%s/target-%d_ptype-%d_pperc-%d_cell-%d_start-%d_dur-%d_rate-%d_test-nqa.nqs"' % \
                                (simdatadir, target, param1, param2, param3, param4, param5, param6)
                                    h('nqaload = new NQS(%s)'%outfilestem)
                                    h('nqaload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                                    t=array(h.nqaload.getcol("t"))
                                    xHand = array(h.nqaload.getcol("x"))
                                    yHand = array(h.nqaload.getcol("y"))
                                    xElbow = array(h.nqaload.getcol("ex"))
                                    yElbow = array(h.nqaload.getcol("ey"))
                                    shAng = array(h.nqaload.getcol("ang0"))
                                    elAng = array(h.nqaload.getcol("ang1"))
                                    shExtMusLength = array(h.nqaload.getcol("ML0"))
                                    shFlexMusLength =  array(h.nqaload.getcol("ML1"))
                                    elExtMusLength = array(h.nqaload.getcol("ML2"))
                                    elFlexMusLength = array(h.nqaload.getcol("ML3"))

                                    #outfilestem = '"%s/p1-%.2f_p2-%d_i-%d_w-%d_test-spk.nqs"' % (simdatadir, param1, param2, iseed, wseed)
                                    outfilestem = '"%s/target-%d_ptype-%d_pperc-%d_cell-%d_start-%d_dur-%d_rate-%d_test-spk.nqs"' % \
                                (simdatadir, target, param1, param2, param3, param4, param5, param6)

                                    h('spikesload = new NQS(%s)' % outfilestem)
                                    h('spikesload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                                    tspike=array(h.spikesload.getcol("t"))
                                    cellid=array(h.spikesload.getcol("id"))
                                    celltype=array(h.spikesload.getcol("type"))
                                    muscleid=array(h.spikesload.getcol("mid"))

                                    #save to matlab
                                    #scipy.io.savemat(('bmm_%s_p1-%.2f_p2-%d_i-%d_w-%d_test_spk.mat'%simdatadir, param1, param2, iseed, wseed), \
                                    #mdict={'t': t, 'cellId': 'cellId, 'celltype': celltype, 'muscleid': muscleid})    

                                    scipy.io.savemat(('%s/target-%d_ptype-%d_pperc-%d_cell-%d_start-%d_dur-%d_rate-%d_spk.mat' % (simdatadir, target, param1, param2, param3, param4, param5, param6)), \
                                    mdict={'tspike': tspike, 'cellid': cellid, 'celltype': celltype, 'muscleid': muscleid})    

                                    scipy.io.savemat(('%s/target-%d_ptype-%d_pperc-%d_cell-%d_start-%d_dur-%d_rate-%d_arm.mat' % (simdatadir, target, param1, param2, param3, param4, param5, param6)), \
                                    mdict={'t': t, 'xhand': xHand, 'yHand': yHand, 'xElbow': xElbow,  'yElbow': yElbow, 'shAng': shAng, 'elAng': elAng, 'shExtMusLength': shExtMusLength, 'shFlexMusLength': shFlexMusLength,  'elExtMusLength': elExtMusLength, 'elFlexMusLength': elFlexMusLength })    

def save2MatlabMistProbingMulti(simdatadir, t1Arg, t2Arg):
    target_range = [1,3] #arange(4) # only four targets
    param1_range = [0,1] # kill cells or synapses
    param2_range = [0,5,10] # kill 5% or 10%
    param3_range = arange(0,99)#arange(0,191,1) # mistCell
    param4_range = [0]#[400] # mistStart
    param5_range = [0]#[200] # mistDuration
    param6_range = [0]#[250] # mistRate

    param1_range = [0,1] #kill cells or synapses
    param2_range = [5,10] # kill 5% or 10%
    param3_range = arange(0,191)#arange(0,191,1) # mistCell
    param4_range = [100]#[400] # mistStart
    param5_range = [200]#[200] # mistDuration
    param6_range = [250]#[250] # mistRate

    h('objref nqaload')
    h('objref spikesload')

    # Loop over target vals
    for target in target_range:
        # Loop over param1 vals
        for param1 in param1_range:
            # Loop over param2 vals
            for param2 in param2_range:
                # Loop over param3 vals
                for param3 in param3_range:
                    # Loop over param4 vals
                    for param4 in param4_range:
                        # Loop over param5 vals
                        for param5 in param5_range:
                            # Loop over param6 vals
                            for param6 in param6_range:

                                # set filename
                                outfilestem = '%s/target-%d_ptype-%d_pperc-%d_cell-%d_start-%d_dur-%d_rate-%d_multi_test-nqa.nqs' % \
                                (simdatadir, target, param1, param2, param3, param4, param5, param6)
                                
                                # get fields from nqs
                                if os.path.isfile(outfilestem):
                                    outfilestem = '"%s/target-%d_ptype-%d_pperc-%d_cell-%d_start-%d_dur-%d_rate-%d_multi_test-nqa.nqs"' % \
                                (simdatadir, target, param1, param2, param3, param4, param5, param6)
                                    h('nqaload = new NQS(%s)'%outfilestem)
                                    h('nqaload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                                    t=array(h.nqaload.getcol("t"))
                                    xHand = array(h.nqaload.getcol("x"))
                                    yHand = array(h.nqaload.getcol("y"))
                                    xElbow = array(h.nqaload.getcol("ex"))
                                    yElbow = array(h.nqaload.getcol("ey"))
                                    shAng = array(h.nqaload.getcol("ang0"))
                                    elAng = array(h.nqaload.getcol("ang1"))
                                    shExtMusLength = array(h.nqaload.getcol("ML0"))
                                    shFlexMusLength =  array(h.nqaload.getcol("ML1"))
                                    elExtMusLength = array(h.nqaload.getcol("ML2"))
                                    elFlexMusLength = array(h.nqaload.getcol("ML3"))

                                    #outfilestem = '"%s/p1-%.2f_p2-%d_i-%d_w-%d_test-spk.nqs"' % (simdatadir, param1, param2, iseed, wseed)
                                    outfilestem = '"%s/target-%d_ptype-%d_pperc-%d_cell-%d_start-%d_dur-%d_rate-%d_multi_test-spk.nqs"' % \
                                (simdatadir, target, param1, param2, param3, param4, param5, param6)

                                    h('spikesload = new NQS(%s)' % outfilestem)
                                    h('spikesload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                                    tspike=array(h.spikesload.getcol("t"))
                                    cellid=array(h.spikesload.getcol("id"))
                                    celltype=array(h.spikesload.getcol("type"))
                                    muscleid=array(h.spikesload.getcol("mid"))

                                    #save to matlab
                                    #scipy.io.savemat(('bmm_%s_p1-%.2f_p2-%d_i-%d_w-%d_test_spk.mat'%simdatadir, param1, param2, iseed, wseed), \
                                    #mdict={'t': t, 'cellId': 'cellId, 'celltype': celltype, 'muscleid': muscleid})    

                                    scipy.io.savemat(('%s/target-%d_ptype-%d_pperc-%d_cell-%d_start-%d_dur-%d_rate-%d_multi_spk.mat' % (simdatadir, target, param1, param2, param3, param4, param5, param6)), \
                                    mdict={'tspike': tspike, 'cellid': cellid, 'celltype': celltype, 'muscleid': muscleid})    

                                    scipy.io.savemat(('%s/target-%d_ptype-%d_pperc-%d_cell-%d_start-%d_dur-%d_rate-%d_multi_arm.mat' % (simdatadir, target, param1, param2, param3, param4, param5, param6)), \
                                    mdict={'t': t, 'xhand': xHand, 'yHand': yHand, 'xElbow': xElbow,  'yElbow': yElbow, 'shAng': shAng, 'elAng': elAng, 'shExtMusLength': shExtMusLength, 'shFlexMusLength': shFlexMusLength,  'elExtMusLength': elExtMusLength, 'elFlexMusLength': elFlexMusLength })    
                 
def save2MatlabMistRepair(simdatadir, t1Arg, t2Arg):
    target_range = [1,3] #arange(4) # only four targets
    param1_range = [0,1] # kill cells or synapses
    param2_range = [0,5,10] # kill 5% or 10%
    param3_range = ['perturb', 'repair']

    h('objref nqaload')
    h('objref spikesload')

    # Loop over target vals
    for target in target_range:
        # Loop over param1 vals
        for param1 in param1_range:
            # Loop over param2 vals
            for param2 in param2_range:
                # Loop over param3 vals
                for param3 in param3_range:

                    # set filename
                    outfilestem = '%s/target-%d_ptype-%d_pperc-%d_%s_test-nqa.nqs' % \
                    (simdatadir, target, param1, param2, param3)
                    
                    # get fields from nqs
                    if os.path.isfile(outfilestem):
                        outfilestem = '"%s/target-%d_ptype-%d_pperc-%d_%s_test-nqa.nqs"' % \
                    (simdatadir, target, param1, param2, param3)
                        h('nqaload = new NQS(%s)'%outfilestem)
                        h('nqaload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                        t=array(h.nqaload.getcol("t"))
                        xHand = array(h.nqaload.getcol("x"))
                        yHand = array(h.nqaload.getcol("y"))
                        xElbow = array(h.nqaload.getcol("ex"))
                        yElbow = array(h.nqaload.getcol("ey"))
                        shAng = array(h.nqaload.getcol("ang0"))
                        elAng = array(h.nqaload.getcol("ang1"))
                        shExtMusLength = array(h.nqaload.getcol("ML0"))
                        shFlexMusLength =  array(h.nqaload.getcol("ML1"))
                        elExtMusLength = array(h.nqaload.getcol("ML2"))
                        elFlexMusLength = array(h.nqaload.getcol("ML3"))

                        #outfilestem = '"%s/p1-%.2f_p2-%d_i-%d_w-%d_test-spk.nqs"' % (simdatadir, param1, param2, iseed, wseed)
                        outfilestem = '"%s/target-%d_ptype-%d_pperc-%d_%s_test-spk.nqs"' % \
                    (simdatadir, target, param1, param2, param3)

                        h('spikesload = new NQS(%s)' % outfilestem)
                        h('spikesload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
                        tspike=array(h.spikesload.getcol("t"))
                        cellid=array(h.spikesload.getcol("id"))
                        celltype=array(h.spikesload.getcol("type"))
                        muscleid=array(h.spikesload.getcol("mid"))

                        #save to matlab
                        #scipy.io.savemat(('bmm_%s_p1-%.2f_p2-%d_i-%d_w-%d_test_spk.mat'%simdatadir, param1, param2, iseed, wseed), \
                        #mdict={'t': t, 'cellId': 'cellId, 'celltype': celltype, 'muscleid': muscleid})    

                        scipy.io.savemat(('%s/target-%d_ptype-%d_pperc-%d_%s_spk.mat' % (simdatadir, target, param1, param2, param3)), \
                        mdict={'tspike': tspike, 'cellid': cellid, 'celltype': celltype, 'muscleid': muscleid})    

                        scipy.io.savemat(('%s/target-%d_ptype-%d_pperc-%d_%s_arm.mat' % (simdatadir, target, param1, param2, param3)), \
                        mdict={'t': t, 'xhand': xHand, 'yHand': yHand, 'xElbow': xElbow,  'yElbow': yElbow, 'shAng': shAng, 'elAng': elAng, 'shExtMusLength': shExtMusLength, 'shFlexMusLength': shFlexMusLength,  'elExtMusLength': elExtMusLength, 'elFlexMusLength': elFlexMusLength })    


def save2MatlabMistSingle(simdatadir, t1Arg, t2Arg):
    ###########################
    # read and save spike and trajectory data (t,x,y from nqa) 
    ########################
    
    
    # plot data for each of the 4 starting conditions in same color; different color for each target
    # Loop over param1 values
    h('objref nqaload')
    h('objref spikesload')
    
    # set filename
    #outfilestem = '"%s/p1-%d_p2-%d_i-%d_w-%d_train-nqa.nqs"' % (simdatadir, param1, param2, iseed, wseed)
    outfilestem = '%s_test-nqa.nqs' % (simdatadir)
                            
    # get fields from nqs
    if os.path.isfile(outfilestem):
        outfilestem = '"%s_test-nqa.nqs"' % (simdatadir)
        h('nqaload = new NQS(%s)'%outfilestem)
        h('nqaload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
        t=array(h.nqaload.getcol("t"))
        xHand = array(h.nqaload.getcol("x"))
        yHand = array(h.nqaload.getcol("y"))
        xElbow = array(h.nqaload.getcol("ex"))
        yElbow = array(h.nqaload.getcol("ey"))
        shAng = array(h.nqaload.getcol("ang0"))
        elAng = array(h.nqaload.getcol("ang1"))
        shExtMusLength = array(h.nqaload.getcol("ML0"))
        shFlexMusLength =  array(h.nqaload.getcol("ML1"))
        elExtMusLength = array(h.nqaload.getcol("ML2"))
        elFlexMusLength = array(h.nqaload.getcol("ML3"))

        #outfilestem = '"%s/p1-%.2f_p2-%d_i-%d_w-%d_test-spk.nqs"' % (simdatadir, param1, param2, iseed, wseed)
        outfilestem = '"%s_test-spk.nqs"' % (simdatadir)

        h('spikesload = new NQS(%s)' % outfilestem)
        h('spikesload.select("t","[]", %d, %d)'%(t1Arg,t2Arg))
        tspike=array(h.spikesload.getcol("t"))
        cellid=array(h.spikesload.getcol("id"))
        celltype=array(h.spikesload.getcol("type"))
        muscleid=array(h.spikesload.getcol("mid"))

        #save to matlab
        #scipy.io.savemat(('bmm_%s_p1-%.2f_p2-%d_i-%d_w-%d_test_spk.mat'%simdatadir, param1, param2, iseed, wseed), \
        #mdict={'t': t, 'cellId': 'cellId, 'celltype': celltype, 'muscleid': muscleid})    

        scipy.io.savemat(('%s_spk.mat'%(simdatadir)), \
    mdict={'tspike': tspike, 'cellid': cellid, 'celltype': celltype, 'muscleid': muscleid})    

        scipy.io.savemat(('%s_arm.mat'%(simdatadir)), \
    mdict={'t': t, 'xhand': xHand, 'yHand': yHand, 'xElbow': xElbow,  'yElbow': yElbow, 'shAng': shAng, 'elAng': elAng, 'shExtMusLength': shExtMusLength, 'shFlexMusLength': shFlexMusLength,  'elExtMusLength': elExtMusLength, 'elFlexMusLength': elFlexMusLength })    


# plot center out reaching trajectory for single target
def plotNCM14CenterOutTraj(simdatadir, trajStartArg, trajStopArg):

    #######################
    ##### obtain target locations
    #########################
    targetsAng = calculateTargetsCenterOut()
    # convert to cartesian position
    varmLen=[0.4634 - 0.173, 0.7169 - 0.4634]
    varmCenter = angles2pos(0.62, 1.53, varmLen[0], varmLen[1])
    elbowCenter = [varmLen[0] * cos(0.62), varmLen[0] * sin(0.62)]
    targetsPos = zeros((8,2))
    for i in range(len(targetsAng)):
        targetsPos[i]=angles2pos(targetsAng[i][0], targetsAng[i][1], varmLen[0], varmLen[1])
    print targetsPos
    print varmCenter
    
    
    ###########################
    # read and plot trajectory data (t,x,y from nqa) 
    ########################
    wseedvals =[120456]#, 398115]#, 534031, 796321, 895199]
    iseedvals = arange(1235,  1235+(17*100), 17)#, 2837, 3955, 4506, 6789, 1236, 2838, 3956, 4507, 6790, 1237, 2839, 3957, 4508, 6791, 1238, 2840, 3958, 4509, 6792]
#[1235]#, 2837]#, 3955, 4506, 6789]
    
    # parameter values
    param1_range = arange(1,2,1)#  arange(0.8,1.24,0.04)#arange(200,375,25)#arange(20,180,20)##arange(10,110,10)# [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]##arange(0.8,1.24,0.04)# arange(1,9)#arange(50,550,50)# arange(10,110,10)## #[0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]#[10,20,30,40,50,60,70,80,90,100]#[50, 100,250,500,750,1000]#arange(20,180,20)##
    param2_range = [0,1,2,3] # [0,1,2,3,4,5,6,7]

    # Visualization options
    figsize =[800,800]# [1300,500] # Figure size in pixels
    targetColor = 'green';#array([(1,0.4,0) , (0,0.2,0.8)]) # Define excitatory and inhibitory colors -- orange and turquoise
    targetLine = 2*2 #??
    targetSize = 30 *2
    targetMarker = 'x'
    colorlist = ['black','blue', 'red', 'green', 'brown', 'cyan','darkgrey', 'magenta', 'orange','yellow']
    trajLine = 1
    trajSize = 1
    trajMarker = '.'    
    trajStart = trajStartArg#2 # sample number to start from (each sample = 10 ms)
    trajStop = trajStopArg#400 # sample number to finish on (max 400=4sec)
        
    # create subplots for each of the param values tested
    ion()
    fig = figure(figsize=(figsize[0]/100,figsize[1]/100),dpi=100)
    fig.subplots_adjust(left=0.02) # Less space on left
    fig.subplots_adjust(right=0.93) # Less space on right
    fig.subplots_adjust(bottom=0.08) # Less space on bottom
    subplotsx = 1#5
    subplotsy = 1#2
    maxSubplots = subplotsx*subplotsy
    param1Subplots =  [None] * min(len(param1_range), maxSubplots)
    border = 0.1
    for i in range(min(len(param1_range), maxSubplots)):
        param1Subplots[i] = subplot(subplotsy, subplotsx, i+1)
        #setp(param1Subplots[i].get_xticklabels(), visible=False) # hide x and y ticks
        #setp(param1Subplots[i].get_yticklabels(), visible=False)
        param1Subplots[i].set_xlim([targetsPos[1][0]-border*1.5, targetsPos[0][0]+border]) # set x-y lims
        param1Subplots[i].set_ylim([targetsPos[3][1]-border*2.5, targetsPos[2][1]+border])
    
    # plot data for each of the 4 starting conditions in same color; different color for each target
    # Loop over param1 values
    h('objref nqaload')
    iparam1=0
    param1Subplots[iparam1].set_title('Trajectories for multiple input seeds')
    # Loop over param2 values
    iparam2=-1
    for param2 in param2_range:
        iparam2 = iparam2 + 1
        iwseed = -1

        if param2==0:
            simdatadir="data/14feb07_iseeds7"
            trajStop=200
        elif param2==1:
            simdatadir="data/14mar10_iseeds8"
            trajStop=200
        else:
            simdatadir="data/14mar24_iseeds6"
            trajStop=trajStopArg
        # plot target location
        param1Subplots[iparam1].scatter(targetsPos[iparam2][0], targetsPos[iparam2][1],  c=colorlist[iparam2], marker=targetMarker, linewidth=targetLine, s=targetSize)
        # Loop over wiring seed...
        for wseed in wseedvals:
            iwseed = iwseed + 1
            iiseed = -1
            # Loop over input seed...
            for iseed in iseedvals:
                iiseed = iiseed + 1
                # set filename
                #outfilestem = '"%s/p1-%.2f_p2-%d_i-%d_w-%d_train-nqa.nqs"' % (simdatadir, param1, param2, iseed, wseed)
                #outfilestem = '"%s/p1-%d_p2-%d_i-%d_w-%d_test-nqa.nqs"' % (simdatadir, param1, param2, iseed, wseed)
                outfilestem = '"%s/target-%d_i-%d_w-%d_test-nqa.nqs"' % (simdatadir, param2, iseed, wseed)
                filename = '%s/target-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, param2, iseed, wseed)
                if os.path.isfile(filename):
                    print outfilestem
                    # get errors from nqs
                    h('nqaload = new NQS(%s)'%outfilestem)
                    t=array(h.nqaload.getcol("t"))
                    x=array(h.nqaload.getcol("x"))
                    y=array(h.nqaload.getcol("y"))
                    param1Subplots[iparam1].scatter(x[trajStart:trajStop], y[trajStart:trajStop],  edgecolor=colorlist[iparam2], marker=trajMarker, linewidth=trajLine, s=trajSize) 
                
    # draw arm
    plot([0,elbowCenter[0]], [0, elbowCenter[1]],'k-',lw=4) #elbow
    plot([elbowCenter[0], varmCenter[0]], [elbowCenter[1], varmCenter[1]],'k-',lw=4)
    tight_layout()
    fig.savefig('gif/%s_traj_%d.png' % (simdatadir[5:], trajStop))

# save EMG data to matlab
def save2MatlabEMGs(simfolder):
    for file in os.listdir(simfolder): 
        if file.endswith("muscles.p"):
            with open(simfolder+'/'+file) as f:
                [musExcSeq, musActSeq, musForcesSeq]=pickle.load(f)
            #print [musExcSeq, musActSeq, musForcesSeq]
            print "Saving %s .mat" % (file)
            scipy.io.savemat(('%s/%s.mat'%(simfolder,file[0:-2])), \
            mdict={'musExcSeq': musExcSeq, 'musActSeq': musActSeq, 'musForcesSeq': musForcesSeq})    

# generate pdf figure of fitness evolution
def figureFitEvol(outfilestem):    
    with open('data/%s/fitevol'% (outfilestem), 'r') as f:
        [stat_gens, stat_avgfits, stat_worstfits, stat_bestfits] = pickle.load(f) 
        
        fontsiz=14
        genMax = 100;
        figure()        
        plot(stat_gens[0:genMax], stat_avgfits[0:genMax], 'b-')
        plot(stat_gens[0:genMax], stat_worstfits[0:genMax], 'r--')
        plot(stat_gens[0:genMax], stat_bestfits[0:genMax], 'g--')
        xlabel('Generation number',fontsize=fontsiz)
        ylabel('Fitness',fontsize=fontsiz)
        savefig("fitevol.pdf",format='pdf') 

# find best fitness solutions and generate data for analysis
def bestFitness(folder,islands=10, top=50, runsims=1, iseeds=1, wseeds=5, calculateWeights=0):  
    #folder='14dec18_evol'
    #islands=10
    #top=50
    timeout = 500
    wseedvals = [120456, 398115, 534031, 796321, 895199]
    wseedvals = wseedvals[0:wseeds]
    iseedvals = arange(1235,  1235+(17*100), 17)
    iseedvals = iseedvals[0:iseeds]
        
    print "loading data..."
    # load data from fitness evaluations
    dataFrom = 'fitness'  # 'individuals'
    ind_gens_isl, ind_cands_isl, ind_fits_isl, ind_cs_isl, stat_gens_isl, \
    stat_worstfits_isl, stat_bestfits_isl, stat_avgfits_isl, stat_stdfits_isl, \
    fits_sort_isl, gens_sort_isl, cands_sort_isl, params_sort_isl \
    = analyse_funcs.loadData(folder,islands,dataFrom)
    
    print "calculating best solutions..."
    # combine data from all islands
    all_fits = all_gens = all_cands = all_params = all_islands = []
    for isl in fits_sort_isl: 
        all_fits = all_fits + isl  # get fits from all islands
        all_islands = all_islands + [fits_sort_isl.index(isl)] * len(isl) # create vector with island
    for isl in gens_sort_isl: 
        all_gens = all_gens + isl # get gens from all islands
    for isl in cands_sort_isl: 
        all_cands = all_cands + isl # get gens from all islands
    for isl in params_sort_isl: 
        all_params = all_params + isl # get params from all islands
    
    # unique params and sort fits 
    all_params, unique_indices = analyse_funcs.uniqueList(all_params) 
    all_fits, all_gens, all_cands, all_islands = zip(*[(all_fits[i],all_gens[i],all_cands[i],all_islands[i]) for i in unique_indices])
    sort_indices = np.argsort(all_fits) # sort fitness
    all_fits, all_gens, all_cands, all_params, all_islands  = zip(*[(all_fits[i],all_gens[i],all_cands[i],all_params[i],all_islands[i]) for i in sort_indices])

    if runsims: 
        print "running sims ..."
        # for each of the top solutions run sim to get data
        if iseeds==1:
            for i in range(top):
                # call runpbatchpbs to run sims saving muscle, weight and LFP data  
                command = 'python runbatchpbs_iseeds_16params.py %s_best data/%s_island_%s/gen_%s_cand_%s %d 0 -1' % (folder,folder,all_islands[i], all_gens[i], all_cands[i], iseeds)
                print command   
                os.system(command)  
        elif iseeds>1:
            i = top
            # call runpbatchpbs to run sims saving muscle, weight and LFP data  
            command = 'python runbatchpbs_iseeds_16params.py %s_best data/%s_island_%s/gen_%s_cand_%s %d 0 -1' % (folder,folder,all_islands[i], all_gens[i], all_cands[i], iseeds)
            print command   
            os.system(command) 
        
        print 'Sleeping for 10 mins to wait for results...'
        time.sleep(100)
        
    # for each of the top solutions run sim to get data
    if top > 1 and iseeds == 1:
        total_jobs = top
        converted = [None]*total_jobs
        num_iters = 0
        jobs_completed=0
        while jobs_completed < total_jobs:
            print str(num_iters)+" iterations, "+str(jobs_completed)+" / "+str(total_jobs)+" jobs completed"
            unfinished = [i for i, x in enumerate(converted) if x is None]
            print "unfinished:" + str(unfinished)
            for i in unfinished:
                # check if file exists
                subfolder = 'island_%s_gen_%s_cand_%s' % (all_islands[i], all_gens[i], all_cands[i])
                simdatadir = 'data/%s/%s' % (folder+'_best', subfolder)
                for iseed in iseedvals:         
                    for wseed in wseedvals:
                        try:
                            for itarget in range(4):
                                filename = '%s/target-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, itarget, iseed, wseed)    
                                with open(filename) as f:
                                    print 'Reading file: ' + f.name
                            save2Matlab_iseeds(simdatadir, 20, 1000) 
                            save2MatlabEMGs(simdatadir)
                            if calculateWeights:                        
                                popMuscles(simdatadir)   
                            converted[i] = 1
                            jobs_completed+=1
                        except:
                            #print 'Not found: '+filename
                            pass
            num_iters += 1
            if num_iters >= timeout: 
                print "max iterations reached without completing all jobs"
                for j in unfinished:
                    converted[j] = 2
                    jobs_completed += 1
            time.sleep(30)
    # for each of the iseeds -- NEEDS FIXING, SOMETIMES NOT ALL .MAT FILES ARE GENERATED
    elif top >= 1 and iseeds > 1:
        total_jobs = iseeds
        converted = [None]*total_jobs
        num_iters = 0
        jobs_completed=0
        while jobs_completed < total_jobs:
            print str(num_iters)+" iterations, "+str(jobs_completed)+" / "+str(total_jobs)+" jobs completed"
            unfinished = [i for i, x in enumerate(converted) if x is None]
            print "unfinished:" + str(unfinished)
            isol = top # top solution only
            # check if file exists
            subfolder = 'island_%s_gen_%s_cand_%s' % (all_islands[isol], all_gens[isol], all_cands[isol])
            simdatadir = 'data/%s/%s' % (folder+'_best', subfolder)
            for iseed,iseedval in enumerate(iseedvals):         
                for wseed in wseedvals:
                    try:
                        for itarget in range(4):
                            filename = '%s/target-%d_i-%d_w-%d_test-nqa.nqs' % (simdatadir, itarget, iseedval, wseed)    
                            with open(filename) as f:
                                print 'Reading file: ' + f.name  
                        converted[iseed] = 1
                        jobs_completed+=1
                    except:
                        #print 'Not found: '+filename
                        pass
            num_iters += 1
            if num_iters >= timeout: 
                print "max iterations reached without completing all jobs"
                for j in unfinished:
                    converted[j] = 2
                    jobs_completed += 1
            time.sleep(30)
        # after all iseeds have finished - convert all together
        save2Matlab_iseeds(simdatadir, 20, 1000) 
        save2MatlabEMGs(simdatadir)
        if calculateWeights:                        
            popMuscles(simdatadir) 
    else:
        print "Have to choose either top>1 and iseeds=1 or top=1 and iseeds>1"


# Script code (run always)
#


# Load default parameters and initialize the network.
# COMMENT OUT WHEN RUNNING SIM.PY !!!
#h.xopen("/usr/site/nrniv/simctrl/hoc/setup.hoc")
#h.xopen("/usr/site/nrniv/simctrl/hoc/nrnoc.hoc")

#h.load_file("syncode.hoc")
#h.load_file("decnqs.hoc")
#h.load_file("analysis.hoc")

#bestFitness('14dec18_evol')
