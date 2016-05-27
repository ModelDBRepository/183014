# armGraphs.py -- Python code for the following purposes:
# (modified from testArm.py)
# *Read joint angles, muscle exc, muscle activation and force from .pnt files
# *Plot for each muscle: exc, act, force, muscle lengths (3 fibres+avg) 
# *Plot joint angles and trajectory over time
# Last update: 6/17/13 (salvadord)

import sys	#for exit
import struct
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.text as text
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pylab import figure, show
from numpy import * 
import csv
#import os


###############################
# Global constant parameters
###############################

#secLength = 6.01 # length of simulation in seconds (first 10 ms used to set start pos)
#msecInterval = 10.0 # interval at which packets are sent in miliseconds
#n = int(secLength*1000/msecInterval) # calculate number of samples
#toSamples = 1/msecInterval*1000 
numMuscles = 4 # number of muscles = shFlex (PECT), shExt (DELT), elFlex (BIC), elExt (TRIC)
#muscleNames =  ["Shoulder ext (post Deltoid+Infraspinatus)", "Shoulder flex (Pectoralis+ant Deltoid)", "Elbow ext (Triceps)", "Elbow flex (Biceps+Brachialis)"]
muscleNames =  ["Shoulder ext", "Shoulder flex", "Elbow ext", "Elbow flex"]
numMusBranches = 18 # number of muscle branches 
numJoints = 2 # number of joints (DOFs) 
armLen = (0.5*0.55, 0.5*(0.3+0.06+0.095)) # shoulder to elbow (check if ms = wam length !) #elbow to center of hand = elbow to wrist + wrist length + hand length/2 (check if ms = wam legth)
useJointPos = 0 # use joint positions vs joint angles for 2d arm animation
showBranches = 0 # include muscle branches in graphs
verbose = 0 # whether to show output on screen 


###############################
# Function definition
###############################

	
# function to read the result .pnt file containing muscle excitation and force	
def readPntFiles(msmFolder, pntFile, secLength, msecInterval):
	
	n = int(secLength*1000/msecInterval) # calculate number of samples

	###################################
	# read joint position
	
	# open pnt file and read data
	fileName = msmFolder+"SUNY_arm_2DOFs_horizon_static_coordinate_status.pnt"
	try:
		#f = open(fileName, "r" )
		jointData = []	
		with open(fileName, "r") as f:
			#next(f)
			for line in f:
				print line
				jointData.append([float(i) for i in line.split()])
		jointData = array(jointData).transpose()
		jointData = jointData[:, len(jointData[0])-n:] # make number of rows equal to n (packets received) 
		
		# create arrays to store read data (joints include shoulder, elbow and wrist * 3 coords (xyz))
		jointPosSeq = zeros(((numJoints+1)*3, n)) 
		
		# assign activation and force to output arrays
		# file has format: time,ground_thorax_xyz,sternoclavicular_xyz,acromioclavicular_xyz,shoulder_xyz,elbow_xyz,radioulnar_xyz,radius_hand_xyz
		
		#jointPosSeq[0:6,:] = jointData[10:16,:]
		#jointPosSeq[6:9,:] = jointData[19:22,:]

		jointPosSeq[0,:] = jointData[10,:]
		jointPosSeq[1,:] = jointData[12,:]
		jointPosSeq[2,:] = jointData[13,:]
		jointPosSeq[3,:] = jointData[15,:]
		jointPosSeq[4,:] = jointData[19,:]
		jointPosSeq[5,:] = jointData[21,:]
	except:
		jointPosSeq=[]
		if verbose: 
			print "coordinate pnt file not available"

	########################################################
	# Read muscle activation and force
	
	# open pnt file and read data
	#fileName = msmFolder+"SUNY_arm_2DOFs_horizon_static_muscle_status.pnt"
	fileName = pntFile
	#f = open(fileName, "r" )
	muscleData = []	
	with open(fileName, "r") as f:
		#next(f)
		for line in f:
			muscleData.append([float(i) for i in line.split()])
	muscleData = array(muscleData).transpose() # transpose
	muscleData = muscleData[:, len(muscleData[0])-n:] # make number of rows equal to n (packets received) 
	
	# create arrays to store read data
	musExcSeq = zeros((numMusBranches, n))
	musActSeq = zeros((numMusBranches, n))
	musForcesSeq = zeros((numMusBranches, n))
	
	# assign activation and force to output arrays
	# file has format: time  DELT1_excitation  DELT1_activation  DELT1_force  DELT2_excitation  DELT2_activation  DELT2_force  ...
	musExcSeq = muscleData[1::3 , :]
	musActSeq = muscleData[2::3 , :]
	musForcesSeq = muscleData[3::3 , :]	
	
	return jointPosSeq, musExcSeq, musActSeq, musForcesSeq

# 	Plot for each muscle: exc, act, force, muscle lengths (3 fibres+avg), joint angles and trajectory over time
def plotGraphs(jointPosSeq, jointAnglesSeq, musLengthsSeq, musExcSeq, musActSeq, musForcesSeq, t1, t2, msecInterval, armAnimation, saveGraphs, saveName):
	# graph parameters
	toDegrees = 360/(2*pi)
	toSamples = 1/msecInterval*1000 
	legFont = 11
	linWidth = 3
	color1 = "red"
	color2 = "blue"
	color3 = "green"
	color4 = "purple"
	line1 = "-"
	line2 = ":"
	gridOn = 1
	
	# plot with 6 subplots for joint angles, trajectory and each of muscles
	fig1 = figure()
		
	# time variables
	T = arange(t1, t2, msecInterval/1000.0);
	t1Samples = t1*toSamples
	t2Samples = t2*toSamples
	
	if armAnimation:
		speedFactor = 1 # speed of animation
		
		#create figure and axes
		figAnim = figure()
		axAnim = figAnim.add_subplot(111)
		axAnim.grid(gridOn);
		# set axes size to fit arm
		axAnim.set_xlim(-armLen[0]-armLen[1]-0.1, armLen[0]+armLen[1]+0.1)
		axAnim.set_ylim(-armLen[0]-armLen[1]-0.1, armLen[0]+armLen[1]+0.1)
		
		###########################
		# 2D arm movement animation
		armImages=[]
		for t in arange(t1Samples,t2Samples):
			# Update the arm position based on jointPosSeq
			if useJointPos:
				# use x = z (pnt file) ; y = x (pnt file)
				shoulderPosx = jointPosSeq[1, t]
				shoulderPosy = jointPosSeq[0, t]
				elbowPosx = jointPosSeq[3, t]
				elbowPosy = jointPosSeq[2, t]
				wristPosx = jointPosSeq[5, t]
				wristPosy = jointPosSeq[4, t]
				
				# update jointAnglesSeq based on pos !!!
				
			# Update the arm position based on jointAnglesSeq	
			else:
				armAng = jointAnglesSeq[:,t]
				shoulderPosx = 0
				shoulderPosy = 0
				elbowPosx = armLen[0] * cos(armAng[0]) # end of elbow
				elbowPosy = armLen[0] * sin(armAng[0])
				wristPosx = elbowPosx + armLen[1] * cos(+armAng[0]+armAng[1]) # wrist=arm position
				wristPosy = elbowPosy + armLen[1] * sin(+armAng[0]+armAng[1])
			
			# create
			armLine1 = lines.Line2D([0, elbowPosx-shoulderPosx], [0, elbowPosy-shoulderPosy], color=color1, linestyle=line1, linewidth=linWidth)
			armLine2 = lines.Line2D([elbowPosx-shoulderPosx, wristPosx-shoulderPosx], [elbowPosy-shoulderPosy, wristPosy-shoulderPosy], color=color2, linestyle=line1, linewidth=linWidth)
			axAnim.add_line(armLine1)
			axAnim.add_line(armLine2)
			#label = plt.legend(armLine1, str(t/toSamples) )
			label = text.Text(x=0, y=0.5, text="time = "+str(t/toSamples), weight='bold' )
			axAnim.add_artist(label)
			armImages.append([armLine1, armLine2, label])
		
		# add blank frames
		blankFrames = int(1*toSamples)
		for t in range(blankFrames):
			# Update the arm position
			armLine1 = lines.Line2D([0, 0], [0, 0], color=color1, linestyle=line1, linewidth=linWidth)
			armLine2 = lines.Line2D([0,0 ], [0,0], color=color2, linestyle=line1, linewidth=linWidth)
			axAnim.add_line(armLine1)
			axAnim.add_line(armLine2)			
			armImages.append([armLine1, armLine2])
		
		# generate animation 
		armAnim = animation.ArtistAnimation(figAnim, armImages, interval=msecInterval/speedFactor, repeat_delay=500, blit=True)
	
	###########################
	# Plot joint angles vs time
	ax = fig1.add_subplot(321)
	#T = arange(0, secLength, msecInterval/1000.0);
	
	T=T[:len(jointAnglesSeq[0,t1Samples:t2Samples])]
	ax.plot(T,jointAnglesSeq[0,t1Samples:t2Samples]*toDegrees,color=color1,linestyle=line1, linewidth=linWidth, label="shoulder")
	ax.plot(T,jointAnglesSeq[1,t1Samples:t2Samples]*toDegrees,color=color2,linestyle=line1, linewidth=linWidth, label="elbow")
      
	ax.set_ylabel('angle (deg)', fontsize = legFont)
	ax.set_xlabel('time (sec)', fontsize = legFont)
	ax.set_title('Joint angles')
	ax.legend(loc='upper center', bbox_to_anchor=(1.0, 1.0),  borderaxespad=0., prop={'size':legFont})
	#ax.set_xlim([t1, t2])
    #ax.set_ylim(bmmYlims_sh)
	ax.grid(gridOn)
	
	############################
	# Plot x-y pos vs time
	ax = fig1.add_subplot(322)
	
	# calculate x and y trajectories based on angles
	if useJointPos:
		xTraj = jointPosSeq[4, t1Samples:t2Samples]
		yTraj = jointPosSeq[5, t1Samples:t2Samples]
	else:
		xTraj = armLen[0]*cos(jointAnglesSeq[0,t1Samples:t2Samples])+armLen[1]*cos(jointAnglesSeq[1,t1Samples:t2Samples])
		yTraj = armLen[0]*sin(jointAnglesSeq[0,t1Samples:t2Samples])+armLen[0]*sin(jointAnglesSeq[1,t1Samples:t2Samples])
	
	#ax.plot(xTraj, yTraj,color=color2,linestyle=line1, linewidth=linWidth)
	ax.plot(T, xTraj,color=color1,linestyle=line1, linewidth=linWidth, label="x")
	ax.plot(T, yTraj,color=color2,linestyle=line1, linewidth=linWidth, label="y")
	
	ax.set_ylabel('position (m)', fontsize = legFont)
	#ax.set_xlabel('x position (m)', fontsize = legFont)
	ax.set_xlabel('time (sec)', fontsize = legFont)
	ax.set_title('X-Y trajectory')
	ax.legend(loc='upper center', bbox_to_anchor=(1.0, 1.0),  borderaxespad=0., prop={'size':legFont})
	#ax.set_xlim([t1, t2])
    #ax.set_ylim(bmmYlims_sh)
	ax.grid(gridOn)
	
	############################
	# Plot excitation, activation and force for each muscle
	
	# calculate normalized force (activation and length already normalized)
	#musActSeqNorm = musActSeq/musActSeq[:,t1Samples:t2Samples].max()
	musForcesSeqNorm = musForcesSeq/musForcesSeq[:,t1Samples:t2Samples].max()
	#musLengthsSeqNorm = musLengthsSeq/musLengthsSeq[:,t1Samples:t2Samples].max()
	
	# Note arrangement of muscle branches in data arrays:
	#DELT1(0)  DELT2(1) DELT3(2) Infraspinatus(3) Latissimus_dorsi_1(4) Latissimus_dorsi_2(5) Latissimus_dorsi_3(6) 
	#Teres_minor(7) PECM1(8) PECM2(9) PECM3(10) Coracobrachialis(11) TRIlong(12) TRIlat(13) TRImed(14) BIClong(15) BICshort(16) BRA(17)  
	# is different from muscle groups:
	# Sh ext = DELT3, Infraspinatus, Latissimus_dorsi_1-3, Teres_minor
	# Sh flex = PECM1, PECM2, PECM3, DELT1, Coracobrachialis
	# El ext = TRIlong, TRIlat, TRImed
	# El flex = BIClong, BICshort, BRA
	shext=[2,3,4,5,6,7]
	shflex=[0,8,9,10,11]
	elext=[12,13,14]
	elflex=[15,16,17]
	musList=[shext, shflex,elext,elflex]
		
	for iMus in range(numMuscles):
		ax = fig1.add_subplot(3,2,iMus+3)
		# set number of muscle branches - assume each node has 3 branches (treat the Brachialis as a branch of Biceps=elbow flexor)
		#iBranches = 3
		
		### Excitation and Activation ####
		# equivalent for all branches of same muscle group
		offset = 2  # use offset 3 because only DELT3 is used (order of muscle branches doesn't correspond muscle groups!)
		ax.plot(T, musExcSeq[musList[iMus][offset],t1Samples:t2Samples],color=color1,linestyle=line1, linewidth=linWidth, label="excitation")
		ax.plot(T, musActSeq[musList[iMus][offset],t1Samples:t2Samples],color=color2,linestyle=line1, linewidth=linWidth, label="activation")
		
		# for show branches plot individual branches and mean value for force and length
		if showBranches:
			### Force and Length ###
			for iBranch in range(len(musList[iMus])):
				ax.plot(T, musForcesSeqNorm[musList[iMus][iBranch],t1Samples:t2Samples],color=color3,linestyle=line2, linewidth=linWidth-1)
				ax.plot(T, musLengthsSeq[musList[iMus][iBranch],t1Samples:t2Samples],color=color4,linestyle=line2, linewidth=linWidth-1)
			ax.plot(T, musForcesSeqNorm[musList[iMus],t1Samples:t2Samples].mean(axis=0),color=color3,linestyle=line1, linewidth=linWidth, label="force (mean)")
			ax.plot(T, musLengthsSeq[musList[iMus],t1Samples:t2Samples].mean(axis=0),color=color4,linestyle=line1, linewidth=linWidth, label="length (mean)")
		# for NOT show branches show mean value for force and single value for length
		else:
			### Force ###
			# For shoulder extensor group show only posterior Deltoid, branch 3 (DELT3) or Infraspinatus (INFSP) 
			# branch 2 (DELT2 = lateral deltoid) also available but currently not included in shoulder extensor group
			if iMus == 0:
				offset = 2 # DELT3
				#offset = 12 # INFSP 
				ax.plot(T, musForcesSeqNorm[musList[iMus][offset],t1Samples:t2Samples],color=color3,linestyle=line1, linewidth=linWidth, label="force")
			# For rest of muscles use mean value of all branches
			else:
				offset=0
				ax.plot(T, musForcesSeqNorm[musList[iMus],t1Samples:t2Samples].mean(axis=0),color=color3,linestyle=line1, linewidth=linWidth, label="force")
			
			### Length ####
			# show length only of one muscle indicated by the index 'offset' - DELT3, PECM1, BIClong, TRIlong
			maxLength = 0.20 
			ax.plot(T, musLengthsSeq[musList[iMus][offset],t1Samples:t2Samples]/maxLength,color=color4,linestyle=line1, linewidth=linWidth, label="length")
	
		# show branche label
		if (showBranches):
			ax.plot(-1, -1,color=color3,linestyle=line2, linewidth=linWidth, label="force (branches)")
			ax.plot(-1, -1,color=color4,linestyle=line2, linewidth=linWidth, label="length (branches)")
		
		# axis properties
		ax.set_ylabel('normalized value', fontsize = legFont)
		ax.set_ylim([0,1])
		ax.set_xlim([t1,t2])
		ax.set_xlabel('time (sec)', fontsize = legFont)
		ax.set_title(muscleNames[iMus])
		if iMus==3:
			ax.legend(loc='upper center', bbox_to_anchor=(-0.2, 1.8),  borderaxespad=0., prop={'size':legFont})
		ax.grid(gridOn)
	
	# show graphs
	fig1.tight_layout()
	show()
	
	# save graphs using startPos and pattern in filename
	if saveGraphs:
		saveFolder = 'gif/'
		fig1.savefig(saveFolder+saveName, bbox_inches=0)

		#if armAnimation:
			#armAnim.save('test.mp4')
			#armAnim.save(saveFolder+saveName+'.mp4',writer = writer)
	

# run single test (udp transfer, read files, plot graphs)	
def readAndPlot(jointAnglesSeq, musLengthsSeq, msmFolder, armAnimation, saveGraphs, saveName, timeRange, msecInterval):
	# Sim parameters
	#armAnimation = 1 #  show 2D arm animation
	#saveGraphs = 1 # save graph and animation 
	
	# define time interval to display
	#timeInterval = [0.1, 30]
	
	# Send muscle excitations to MSM and receive joint angles and muscle lengths
	#jointAnglesSeq, musLengthsSeq = sendAndReceiveMsmData(initJointAngles, musExcSeq, readSimFromFile)
	
	# Read data from .pnt files
	jointPosSeq,musExcSeq, musActSeq, musForcesSeq = readPntFiles(msmFolder, timeRange[1], msecInterval)
	
	# Plot results (last 2 arguments = initial and end times in seconds)
	plotGraphs(jointPosSeq, jointAnglesSeq, musLengthsSeq, musExcSeq, musActSeq, musForcesSeq, timeRange[0], timeRange[1], msecInterval, armAnimation, saveGraphs, saveName)

##############################
# Main script
##############################
'''
jointAnglesSeq = zeros((numJoints, n))
musLengthsSeq = zeros((numMusBranches, n))
armAnimation = 1 #  show 2D arm animation
saveGraphs = 1 # save graph and animation 
timeInterval = [0.1, 30]
saveName='temp'
msmFolder = "/home/salvadord/Documents/ISB/Models_linux/msarm/source/test/"

readAndPlot(jointAnglesSeq, musLengthsSeq, msmFolder, armAnimation, saveGraphs, saveName, timeInterval)
'''

