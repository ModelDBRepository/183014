# arminterface_pipe.py -- Python code for interfacing the sim with 
# the musculoskeletal arm using pipes (code simplified from arminterface.py)
# 
# Last update: 8/29/14 (salvadordura@gmail.com)

from socket import *  
import select	# for socket reading
import struct	# to pack messages for socket comms
from time import *	
import numpy
import pickle
import subprocess	# to tun the muscskel arm code
import os  # to kill subprocess
import signal  # to kill subprocess
import xml.etree.ElementTree as ET
import atexit
from cStringIO import StringIO
import random
import os.path
from math import radians

# Verbosity level (for debugging).
verbose = 0

# Save timestamps and joint angles to file
saveDataExchanged = 0

# Save arm data: excitation, activation, muscle forces
saveDataMuscles = 0


# Flag to choose whether the msm feeds back joint angles, muscle lengths or both
# can take values 'angles', 'lengths' or 'both'
msmFeedback = 'both'

# Flag to choose whether to send joint position ('pos'); joint velocity ('vel') to msm; or joint velocity calculated from joint positions ('vel2')
wamForwardType = 'pos' 

# MSM returns length of all branches per muscle - decide which to use as proprioceptive info for BMM (0 = mean); one value per muscle
muscleLengthBranch = [3,1,1,1]

# Flag to run MSM from Python 
msmRun = 1
#msmFolder = "/home/salvadord/Documents/ISB/Models_linux/msarm/source/test/"
msmFolder = "msarm/" #update to use os.getcwd()

# .osim file with arm model description 
# will be copied with timestamp in 'osimFile' to enable multiple instances running simultaneously
osimOriginal = msmFolder + "SUNY_arm_2DOFs_horizon.osim"
osimFile = osimOriginal
	
# Flag to show MSM animation
msmAnim  = 0 

# Flag to plot MSM graphs
msmGraphs = 0

#if msmGraphs or saveDataMuscles:
import armGraphs 	# to plot muscskel arm graphs

# Use a simple dummy arm model instead of musculoskeletal arm
dummyArm = 1


class PipeReader(object):
    def __init__(self, fd):
        self._fd = fd
        self._buf = ''

    def fileno(self):
        return self._fd

    def readlines(self):
        data = os.read(self._fd, 4096)
        return data

#
# Initialize and setup the sockets, data format and other variables
#
def setup(secLength, msecInterval, shInit, elInit, targetx, targety, damping, shExtGain, shFlexGain, elExtGain, elFlexGain):
	# input variables
	global verbose
	global savedDataSent
	global savedDataReceived
	
	# output variables
	global msmCommand
	global msmPipe
	global jointAnglesSeq
	global musLengthsSeq
	global packetID
	global jointAngleID
	global muscleLengthID
	global time1
	global xmlFile
	global osimFile
	global msmSockMaxIter
	global proc_stdout
	global armReady
	global inputbuffer
	global pntFile
	global jointAngles
	global anglesReceived
	global lengthsReceived

	# Packet ID numbers (0 is first packet, with sim start time)
	packetID = -1
	jointAngleID = -1
	muscleLengthID = -1

	# initialize timeInterval
	time1 = time()

	# Variable to store the joint angles (and velocities) or muscle lengths received from the msm, stored globally so can be accessed by NEURON using python function
	anglesReceived = [0, 0, 0, 0]
	lengthsReceived = [0, 0, 0, 0]

	# dummy arm
	if dummyArm:
		
		# set initial joint angles
		jointAngles = [shInit, elInit]

			
		# save data		
		if saveDataExchanged:
			savedDataSent = []
			savedDataReceived = []

	# musculoskeletal arm
	else:
		
		# if plot MSM graphs initialize required arrays (numJoints, numMusBranches and n declared in armGraphs.py)
		if msmGraphs:
			# Create arrays to store received data
			n = int(secLength*1000/msecInterval) # calculate number of samples
			jointAnglesSeq = numpy.zeros((armGraphs.numJoints, n))
			musLengthsSeq = numpy.zeros((armGraphs.numMusBranches, n))

		
		# Run MSM simulation asynchronously (i.e. Python goes on while MSM is running)
		if msmRun:
			# Set paths to run MSM from Python
			msmCommand = msmFolder+"Run" # runs MSM sim
			if msmAnim:
				xmlOriginal = msmFolder+"SUNY_arm_horizon_fwd_10ms.xml" # xml file with sim parametersd - with 3D visualization
			else:
				xmlOriginal = msmFolder+"SUNY_arm_horizon_fwd_no_visual_10ms.xml" # xml file with sim parameters - no visualization
				
			# make copy of .xml and .osim to enable running multiple isntances simultaneously
			while 1:
				rand=int(random.random()*100000)
				xmlTemp = "xml_temp_"+str(rand)+".xml"
				xmlFile = msmFolder+xmlTemp
				if not os.path.isfile(xmlFile): # if file doesn't exist, exit loop; otherwise try different filename
					break

			osimTemp = "osim_temp_"+str(rand)+".osim"
			osimFile = msmFolder+osimTemp
			cp_xml_str='cp %s %s' % (xmlOriginal, xmlFile)
			cp_osim_str = 'cp %s %s' % (osimOriginal, osimFile)
			os.system(cp_xml_str)
			os.system(cp_osim_str)

			# generate name of temporary pnt file to store muscle data
			pntTemp = "muscleData_temp_"+str(rand)+".pnt"
			pntFile = msmFolder+pntTemp

			# set temporary .pnt file name in temporary .xml file
			msmSetPntFileName(xmlFile, pntFile)
			
			# set temporary .osim file name in temporary .xml file
			msmSetOsimFileName(xmlFile, osimTemp)
			
			# set msm duration = sim duration via XML
			msmSetDurationXML(secLength, xmlFile)
			
			# set initial joint angles via XML
			msmSetJointAnglesXML(shInit,elInit)
			
			# set damping
			msmSetDampingXML(damping)

			# set max isometric force
			shMuscleList = ['DELT1', 'DELT2', 'DELT3', 'Infraspinatus', 'Latissimus_dorsi_1', 'Latissimus_dorsi_2', 'Latissimus_dorsi_3','Teres_minor', 'PECM1', 'PECM2', 'PECM3', 'Coracobrachialis'] # only shoulder muscles
			shMuscleOriginalValues = [shFlexGain*1142.60, shFlexGain*1142.60, shExtGain*259.88, shExtGain*1210.84, shExtGain*389.10, shExtGain*389.10, 281.66, shExtGain*354.25, 364.41, 515.41, 390.55, 242.46] 

			elMuscleList =  ['TRIlong', 'TRIlat', 'TRImed', 'BIClong', 'BICshort', 'BRA'] # only elbow muscles
			elMuscleOriginalValues = [elExtGain*798.52, elExtGain*624.30, elExtGain*624.30, elFlexGain*624.30, elFlexGain*435.56, elFlexGain*987.26]

			for i in range(len(shMuscleList)):
				msmSetMaxIsometricForceXML(shMuscleList[i], shMuscleOriginalValues[i]*3.0) #*3

			for i in range(len(elMuscleList)):
				msmSetMaxIsometricForceXML(elMuscleList[i], elMuscleOriginalValues[i]*1.0)
			
			# set target location
			msmSetTargetPositionXML(targetx,targety)		
			
			# run MSM and create pipe to read output
			msmPipe = subprocess.Popen([msmCommand, xmlFile],  preexec_fn=os.setsid, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
			proc_stdout = PipeReader(msmPipe.stdout.fileno())
			
			# initialize timeInterval
			time1 = time()

			# Flag to ensure first pacekt is sent once the virtual arm executable is ready
			armReady = 0

			# Buffer to store data during pipe communication 
			inputbuffer = StringIO()
			
			# save data		
			if saveDataExchanged:
				savedDataSent = []
				savedDataReceived = []

def sendAndReceiveDataPackets(simtime, data1, data2, data3, data4):
	# input variables
	global packetID
	global muscleLengthID
	global jointAngleID
	global verbose
	global anglesReceived   
	global savedDataSent
	global savedDataReceived
	global jointAnglesSeq
	global musLengthsSeq
	global muscleLengthBranch
	global wamForwardType
	global packetID
	global jointAngleID
	global muscleLengthID
	global time1
	global msmReady
	global proc_stdout
	global armReady
	global inputbuffer
	global jointAngles
	global anglesReceived
	global lengthsReceived

	# concatenate input arguments into a list
	data = [data1, data2,data3,data4]
	
	# Increase packet ID.
	packetID += 1.0


	# dummy arm
	if dummyArm:

		shmin = radians(-10) # shoulder
  		shmax = radians(110)
  		elmin = radians(0)  # elbow
  		elmax = radians(140)
  
  		L = [0.03, 0.23]  # min and max muscle length value

  		scaleFactor = 0.04
		angleChange = [scaleFactor*(data2-data1), scaleFactor*(data4-data3)]  # shflex - shext 
		jointAngles = [x + y for x, y in zip(jointAngles, angleChange)] # update joint angles
		anglesReceived = [min(max(jointAngles[0], shmin), shmax), min(max(jointAngles[1], elmin), elmax)]
		#print anglesReceived

		# calculate normalized muscle lengths as a function of joint angles
		extL = [(anglesReceived[0] - shmin) / (shmax-shmin), (anglesReceived[1] - elmin) / (elmax-elmin)]
		flexL = [(1-extL[0]), (1-extL[1])] 
		lengthsReceived = [extL[0] * (L[1]-L[0]) + L[0], flexL[0] * (L[1]-L[0]) + L[0], extL[1]* (L[1]-L[0]) + L[0], flexL[1]* (L[1]-L[0]) + L[0]]  
		#print lengthsReceived

	# musculoskeletal arm
	else: 

		#####################
		# Send packets 
		#####################
		
		# Send packets to MSM   

		while not armReady: # Ensure virtual arm is ready to receive
			ready, _, _ = select.select([proc_stdout], [], [], 0.0)
			if not ready:
				continue
			lines = ready[0].readlines()
			if verbose:
				print lines
			if 'READY TO RUN' in lines:
				armReady=1 
		
		musclesExcSend = data
			
		try:
			msmPipe.stdin.write(str(musclesExcSend)[1:-1])
			msmPipe.stdin.write("\n")
			if verbose: 
				print "\nWriting to MSM pipe: packetID=%f"%(packetID)
				print(str(musclesExcSend))
		except:
			print "timeout while sending packet to msarm"
			packetID = -2 # stop sending/receiving packets
				
		#####################
		# Receive packets 
		#####################
			
		# Receive packets from MSM   
		dataReceived = []
		dataReceived2 = []
		dataReceivedPacked = []
		dataReceivedPacked2 = [] 
		numJoints = 2
		numMuscles = 18

		while len(dataReceived) == 0 or len(dataReceived2) == 0:
			buffline = inputbuffer.readline()  # read next line in buffer
			if buffline == '':   # if buffer empty, read more lines from pipe
				ready, _, _ = select.select([proc_stdout], [], [], 1.0)
				if ready:
					inputbuffer = StringIO(ready[0].readlines())
					buffline = inputbuffer.readline()  
			if 'Totoal' in buffline: # if end of input from virtual arm, stop
				break
			tmp = [float(x) for x in buffline.split()]  # split line into space delimited floats
			if verbose:
				print "read from msm pipe: " + str(len(buffline))
				print tmp
				
			if len(tmp) == numJoints:
				dataReceived = tmp
			elif len(tmp) == numMuscles:
				dataReceived2 = tmp

		# store data in array required to plot msm graphs (armGraphs.py)
		if msmGraphs:		
			muscleLengthID += 1
			musLengthsSeq[:, muscleLengthID] = numpy.array(dataReceived2)
		
		# print received data
		if verbose:
			print "Received packet %f from MSM: (%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)" % \
		(packetID, dataReceived2[0],dataReceived2[1],dataReceived2[2],dataReceived2[3],dataReceived2[4],dataReceived2[5],dataReceived2[6],dataReceived2[7],dataReceived2[8],dataReceived2[9],dataReceived2[10])

		# store the mean value of the length of branches of each muscle in a new variable				
		#DELT1(0)  DELT2(1) DELT3(2) Infraspinatus(3) Latissimus_dorsi_1(4) Latissimus_dorsi_2(5) Latissimus_dorsi_3(6) 
		#Teres_minor(7) PECM1(8) PECM2(9) PECM3(10) Coracobrachialis(11) TRIlong(12) TRIlat(13) TRImed(14) BIClong(15) BICshort(16) BRA(17)  

		# Deltoid muscle length = mean(DELT1  DELT2 DELT3)
		if muscleLengthBranch[0] == 0:
			lengthsReceived[0] = numpy.mean(dataReceived2[0:3]) 
		# Deltoid muscle length = length of branch number 'muscleLengthBranch'
		else:
			lengthsReceived[0] = dataReceived2[muscleLengthBranch[0]-1]
		
		# Pectoralis muscle length = mean(PECM1 PECM2 PECM3)
		if muscleLengthBranch[1] == 0:
			lengthsReceived[1] = numpy.mean(dataReceived2[8:11]) 
		# Pectoralis muscle length = length of branch number 'muscleLengthBranch'
		else:
			lengthsReceived[1] = dataReceived2[muscleLengthBranch[1]+7]		

		# Triceps muscle length = mean(TRIlong TRIlat TRImed)
		if muscleLengthBranch[2] == 0:
			lengthsReceived[2] = numpy.mean(dataReceived2[12:15]) 
		# Triceps muscle length = length of branch number 'muscleLengthBranch'
		else:
			lengthsReceived[2] = dataReceived2[muscleLengthBranch[2]+11]

		# Biceps/Brachialis muscle length = mean(BIClong BICshort BRA)		
		if muscleLengthBranch[3] == 0:
			lengthsReceived[3] = numpy.mean(dataReceived2[15:18]) 	
		# Biceps/Brachialis muscle length = length of branch number 'muscleLengthBranch'
		else:
			lengthsReceived[3] = dataReceived2[muscleLengthBranch[3]+14]

		# save received data
		if saveDataExchanged:
			savedDataReceived.append([simtime, getCurrTime(), lengthsReceived[0], lengthsReceived[1], lengthsReceived[2], lengthsReceived[3]])
		
		######################
		# Receive joint angles	 
		
		# store data in array required to plot msm graphs (armGraphs.py)
		if msmGraphs:	
			jointAngleID += 1	
			jointAnglesSeq[:, jointAngleID] = numpy.array(dataReceived[0:2]) 
		
		# print received data
		if verbose:
			print "Received packet %f from MSM: (%.3f,%.3f)" % (packetID, dataReceived[0],dataReceived[1])

		# if receiving joint angles, store the received shoulder and elbow angles in a new variable
		anglesReceived = dataReceived

		# save received data
		if saveDataExchanged:
			savedDataReceived.append([simtime, getCurrTime(), anglesReceived[0], anglesReceived[1]])
			

# Function to close sockets, save data and plot graphs
def closeSavePlot(secLength, msecInterval, filestem=''):
	global saveDataMuscles
	global savedDataReceived
	global savedDataSent
	global jointAnglesSeq
	global musLengthsSeq
	global msmPipe
	global msmRun
	global xmlFile
	global osimFile
	global pntFile

	# dummy arm
	if dummyArm:
		pass

	# musculoskeletal arm
	else:
		# delete temporal copy of xml and osim files
		rm_str = 'rm %s %s' % (xmlFile, osimFile)
		os.system(rm_str)
			
		# close msm pipe	
		if msmRun:
			os.killpg(msmPipe.pid, signal.SIGTERM)
		
		if verbose:
			print "Sockets closed"

		# save data exchanged via udp to file
		if saveDataExchanged:
			f = open( 'data_BMM.txt', 'w' )
			for item in savedDataSent:
				f.write("%s\n" % item)
			for item in savedDataReceived:
				f.write("%s\n" % item)
			f.close()
			if verbose:
				print 'Data saved to file'

		# save muscle data to file
		if saveDataMuscles:	
			msmFolder = '' # data files saved locally
	#		# Read data from .pnt files
			jointPosSeq,musExcSeq, musActSeq, musForcesSeq = armGraphs.readPntFiles(msmFolder, pntFile, secLength, msecInterval)
			with open("%s-muscles.p"%(filestem),'w') as f:
				pickle.dump([musExcSeq, musActSeq, musForcesSeq], f)

			
		# plot graphs
		if msmGraphs:
			saveName = 'msmGraphs.png'
			saveGraphs = saveDataExchanged #set func argument to save graphs
			armAnimation = 0# msmGraphs #set func argument to show arm animation
			timeRange = [0.1, secLength] 
			msmFolder = '' # data files saved locally
			armGraphs.readAndPlot(jointAnglesSeq, musLengthsSeq, msmFolder, armAnimation, saveGraphs, saveName, timeRange, msecInterval)

		# delete temporal pnt file
		rm_str = 'rm %s' % (pntFile)
		os.system(rm_str)

# Time function
def getCurrTime():
	return time()*1000

#
# Functions to return data received via udp (called from NEURON)
#
def getAngleReceived(joint):
	global anglesReceived
	return anglesReceived[int(joint)]

def getLengthReceived(muscle):
	global lengthsReceived
	return lengthsReceived[int(muscle)]

def getPacketLoss():
	if packetID==-1:
		return 1
	else:
		return 0

# Function to set joint angles by modifying the XML .osim file- also runs the MSM program in pipe
def msmSetJointAnglesXML(shAng,elAng):
	global simtime
	global msmCommand
	
	# read .osim file
	tree = ET.parse(osimFile)
	root = tree.getroot()
	
	# set shoulder angle initial value
	for shoulder_flex_tag in root.iter('Coordinate'):
		if shoulder_flex_tag.get('name') == 'arm_flex':
			break
	
	init_shoulder_tag = shoulder_flex_tag.find('initial_value')
	init_shoulder_tag.text = str(shAng);
	
	# set elbow angle initial value
	for elbow_flex_tag in root.iter('Coordinate'):
		if elbow_flex_tag.get('name') == 'elbow_flex':
			break
	
	init_elbow_tag = elbow_flex_tag.find('initial_value')
	init_elbow_tag.text = str(elAng);
	
	# save xml file
	tree.write(osimFile)
	
def msmSetOsimFileName(xmlFile, osimTemp):
	# read .osim file

	tree = ET.parse(xmlFile)
	root = tree.getroot()
	
	# set osim file name
	for tmp in root.iter('OsimFile'):
		tmp.set('name', osimTemp) 
	
	# save xml file
	tree.write(xmlFile)

def msmSetPntFileName(xmlFile, pntFile):
	# read .osim file

	tree = ET.parse(xmlFile)
	root = tree.getroot()
	
	# set osim file name
	for tmp in root.iter('PntOutput'):
		tmp.set('name', pntFile)
		break # exit after first appearance 
	
	# save xml file
	tree.write(xmlFile)

# set duration of arm sim in .xml file
def msmSetDurationXML(secLength, xmlFile):
	# read .osim file
	tree = ET.parse(xmlFile)
	root = tree.getroot()
	
	# set end time
	endTime = root.find('EndTime')
	endTime.text = str(secLength)
	
	# save xml file
	tree.write(xmlFile)
	
def msmSetDampingXML(damping):
	# read .osim file
	tree = ET.parse(osimFile)
	root = tree.getroot()
	
	# set damping value
	for damping_tag in root.iter('damping'):
		damping_tag.text = str(damping)
	
	# save xml file
	tree.write(osimFile)
	
def msmSetTargetPositionXML(targetx, targety):
	# read .osim file
	tree = ET.parse(osimFile)
	root = tree.getroot()

	for Body_tag in root.iter('Body'):
		if Body_tag.get('name') == 'ground':
			break
	transform_tag = Body_tag.find('VisibleObject')
	transform_tag = transform_tag.find('GeometrySet')
	transform_tag = transform_tag.find('objects')
	transform_tag = transform_tag.find('DisplayGeometry')	
	transform_tag = transform_tag.find('transform')
	
	transform_tag.text = str(0)+' '+str(0)+' '+str(0)+' '+str(-targetx-0.06)+' '+str(-0.05)+' '+str(-targety+0.09) # y postive = x-axis right
	
	# save xml file
	tree.write(osimFile)
		
	
def msmSetMaxIsometricForceXML(muscleName, value):
	# read .osim file
	tree = ET.parse(osimFile)
	root = tree.getroot()
	
	# set shoulder angle initial value
	for muscle_tag in root.iter('Schutte1993Muscle'):
		if muscle_tag.get('name') == muscleName:
			break
	
	muscleForce_tag = muscle_tag.find('max_isometric_force')
	#print muscleForce_tag.text
	muscleForce_tag.text = str(value);
	
	# save xml file
	tree.write(osimFile)

def saveEMG():
	global saveDataMuscles
	saveDataMuscles = 1


