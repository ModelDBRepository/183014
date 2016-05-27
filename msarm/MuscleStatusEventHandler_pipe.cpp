#include "MuscleStatusEventHandler.h"

#include <Multibody/MultibodyDyna.h> 

#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <fcntl.h>      /* To change socket to nonblocking mode */
#include <arpa/inet.h>  /* For inet_pton() */

FILE_STATIC_CALLHACK(MuscleStatusEventHandler);

namespace mf_mbd
{
	const int numMuscles = 18;
	double muscleLengths[numMuscles] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	bool verboseSend2 = 0;
	
	InitFactoryStaticMembersMacro(MuscleStatusEventHandler, PeriodicEventHandler);

	//class MultibodySystem;
	MuscleStatusEventHandler::MuscleStatusEventHandler()
		:PeriodicEventHandler()
	{
	}

	MuscleStatusEventHandler::MuscleStatusEventHandler(MultibodySystem& system)
		:PeriodicEventHandler()
	{
		setMultibodySystem(system);
	}

	MuscleStatusEventHandler::MuscleStatusEventHandler(MultibodySystem& system, Real interval)
		:PeriodicEventHandler(interval)
	{
		setMultibodySystem(system);
	}

	MuscleStatusEventHandler::~MuscleStatusEventHandler()
	{
		if(_out.is_open()) _out.close();
	}

	bool MuscleStatusEventHandler::handle(Real currTime)
	{
		//Real currTime = getMultibodySystem()->getMultibodyDyna()->getCurrTime();
		if(!needHandle(currTime)) 
			return false;
		PeriodicEventHandler::_setHandledTime(currTime);

		std::vector<double> data;
		data.reserve(_muscles.size() * _varNames.size());
		//data.reserve(_muscles.size() * _varNames.size() + 1);		
		//data.push_back(currTime); //put the time to the first column;

		for(int m = 0; m < _muscles.size(); ++m) {
			LOAMuscle::Ref muscle = _muscles[m];

			for(int i = 0; i < _varNames.size(); ++i) {
				std::string name = _varNames[i];
				if(name == "excitation") {
					data.push_back(muscle->getExcitation());
				}
				else if(name == "activation") {
					data.push_back(muscle->getActivation());
				}
				else if(name == "activationDeriv") {
					data.push_back(muscle->getActivationDeriv());
				}
				else if(name == "force") { //total or tendon force
					data.push_back(muscle->getForce());
				}
				else if(name == "stress") {
					data.push_back(muscle->getStress());
				}
				else if(name == "speed") {
					data.push_back(muscle->getSpeed());
				}
				else if(name == "activeFiberForce") { 
					data.push_back(muscle->getActiveFiberForce());
				}
				else if(name == "passiveFiberForce") { 
					data.push_back(muscle->getPassiveFiberForce());
				}
				else if(name == "normalizedFiberLength") { 
					data.push_back(muscle->getNormalizedFiberLength());
				}
				else if(name == "RelativeMaxContraction") {
					data.push_back(muscle->getForce()/muscle->getMaxIsometricForce());
				}			
				else if(name == "fiberLengthDeriv") {
					data.push_back(muscle->getFiberLengthDeriv());
				}
				else if(name == "isometricFiberForce") {
					data.push_back(muscle->computeFiberIsometricForce(muscle->getActivation(),muscle->getFiberLength()));
				}			
				else if(name == "capacity") {
					data.push_back(muscle->getCapacity());
				}			
				//else if(name = "momentArms") {
				//	const std::map<ArticulatedJoint::Ref,VecN>& getMomentArms() {return _momArms;} 
				//	data.push_back(muscle->getMomentArms());
				//}
				else{
					CHK_ERR(false, "Can not find muscle variable with name " + name);
					continue;
				}
			}
			
			// store all muscle normalized lengths in a vector that can be sent via udp
			// stored in the following order: DELT1  DELT2 DELT3 Infraspinatus Latissimus_dorsi_1 Latissimus_dorsi_2 Latissimus_dorsi_3 Teres_minor PECM1 PECM2 PECM3 Coracobrachialis TRIlong TRIlat TRImed BIClong BICshort BRA 
			//muscleLengths[m] = muscle->getNormalizedFiberLength();
			//muscleLengths[m] = muscle->getOptimalFiberLength(); // 
			muscleLengths[m] = muscle->getFiberLength();
		}

		// send packets
		for(int m = 0; m < numMuscles; ++m) {
			std::cout << muscleLengths[m] << "  ";
		}
		std::cout << std::endl;
		
		if (verboseSend2) {
			printf("\nSent muscle lengths to stdout\n");
		}
		return true;
	}

	void MuscleStatusEventHandler::initBeforeRun()
	{
		EventHandler::initBeforeRun();
	}

	void MuscleStatusEventHandler::readFromXML(DOMNode* node)
	{
		XMLDOM::DOMElement* tmpNode = NULL;
		
		/*
		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"PortSend2");
		CHK_ERR(tmpNode, "Can not find socket port");
		portSend2 =  XMLDOM::getValueAsType<int>(tmpNode);
		*/
		
		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"Interval");
		double interval = XMLDOM::getValueAsType<double>(tmpNode);
		this->setEventInterval(interval);
	
		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"LOAMuscleForceSubsystem");
		CHK_ERR(tmpNode, "Can not find LOAMuscleForceSubsystem node");
		std::string sysName = XMLDOM::getAttribute(tmpNode, "name");

		ForceSubsystem* fsys = getMultibodySystem()->getForceSubsystem(sysName);
		LOAMuscleForceSubsystem* mfsys = dynamic_cast<LOAMuscleForceSubsystem*>(fsys);
		std::string err_msg = "The ForceSubsystem " + sysName + " must be a LOAMuscleForceSubsystem";
		CHK_ERR(mfsys, err_msg);

		_msclSys = mfsys;

		bool needAll = false;
		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"MuscleNames");
		CHK_ERR(tmpNode, "Can not find MuscleNames node");
		std::string all = XMLDOM::getAttribute(tmpNode,"all");
		if(!all.empty()) {
			boost::to_lower(all);
			if(all == "true") {
				needAll = true;
			}
		}

		if(!needAll) { //read all muscle names
			std::string str = XMLDOM::getTextAsStdStringAndTrim(tmpNode);
			mf_utils::Tokenizer tokens(str);
			for(int i = 0; i < tokens.size(); ++i) {
				std::string name = tokens[i];
				//muscle* msl = _msclSys->getMuscle(name);
				LOAMuscle* msl = dynamic_cast<LOAMuscle*>(_msclSys->getForceMatt(name));
				if(!msl) CHK_ERR(tmpNode, "Can not find muscle with name " + name);
				_muscles.push_back(msl);
			}
		}
		else { //all the muscles
			//_muscles = _msclSys->getMuscles();
			_muscles.resize(_msclSys->getNumForceMatts());
			for(int i = 0; i < _muscles.size(); ++i) _muscles[i] = _msclSys->getForceMattTrueType(i);
		}

		needAll = false;
		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"MuscleVars");
		CHK_ERR(tmpNode, "Can not find MuscleVars node");
		all = XMLDOM::getAttribute(tmpNode,"all");
		if(!all.empty()) {
			boost::to_lower(all);
			if(all == "true") {
				needAll = true;
			}
		}

		if(!needAll) { //read all muscle names
			std::string str = XMLDOM::getTextAsStdStringAndTrim(tmpNode);
			mf_utils::Tokenizer tokens(str);
			for(int i = 0; i < tokens.size(); ++i) {
				std::string name = tokens[i];
				if(name == "excitation") {
				}
				else if(name == "activation") {
				}
				else if(name == "activationDeriv") {
				}
				else if(name == "force") { //total or tendon force
				}
				else if(name == "stress") {
				}
				else if(name == "speed") {
				}
				else if(name == "activeFiberForce") { 
				}
				else if(name == "passiveFiberForce") { 
				}
				else if(name == "normalizedFiberLength") { 
				}
				else if(name == "RelativeMaxContraction") { 
				}			
				else if(name == "fiberLengthDeriv") { 
				}
				else if(name == "isometricFiberForce") { 
				}
				else if(name == "capacity")
				{
				}
				//else if(name = "momentArms") { 
				//}
				else{
					continue;
				}

				_varNames.push_back(name);
			}
		}
		else { //all the vars
			_varNames.push_back("excitation");
			_varNames.push_back("activation");
			_varNames.push_back("activationDeriv");
			_varNames.push_back("force");
			_varNames.push_back("stress");
			_varNames.push_back("speed");
			_varNames.push_back("activeFiberForce");
			_varNames.push_back("passiveFiberForce");
			_varNames.push_back("normalizedFiberLength");
			_varNames.push_back("RelativeMaxContraction");
			_varNames.push_back("fiberLengthDeriv");
			_varNames.push_back("isometricFiberForce");
			_varNames.push_back("capacity");
		}

		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"PntOutput");
		if(tmpNode) {
			_outFileName = XMLDOM::getAttribute(tmpNode,"name");
		}

		if(_outFileName.empty()) {
			_outFileName = _msclSys->getName() + "_status.pnt";
		}

		_out.open(_outFileName.c_str());
		if(!_out.is_open()) CL_ERR("Can not open file " + _outFileName);

	}

} //end namespace 
