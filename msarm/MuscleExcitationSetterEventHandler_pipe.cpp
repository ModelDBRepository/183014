#include "MuscleExcitationSetterEventHandler.h"

#include <Multibody/MultibodyDyna.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <fcntl.h>      /* To change socket to nonblocking mode */
#include <arpa/inet.h>  /* For inet_pton() */
#include <string> 
#include <boost/algorithm/string.hpp>
			



FILE_STATIC_CALLHACK(MuscleExcitationSetterEventHandler);

namespace mf_mbd
{
	double musclesExc[4] = {0.0,0.0,0.0,0.0};
	static const int SIZE_OF_MSG_IN = 4 * sizeof(double);
	bool verboseReceive = 0;
	
	InitFactoryStaticMembersMacro(MuscleExcitationSetterEventHandler, PeriodicEventHandler);

	//class MultibodySystem;
	MuscleExcitationSetterEventHandler::MuscleExcitationSetterEventHandler()
		:PeriodicEventHandler()
	{
	}

	MuscleExcitationSetterEventHandler::MuscleExcitationSetterEventHandler(MultibodySystem& system)
		:PeriodicEventHandler()
	{
		setMultibodySystem(system);
	}

	MuscleExcitationSetterEventHandler::MuscleExcitationSetterEventHandler(MultibodySystem& system, Real interval)
		:PeriodicEventHandler(interval)
	{
		setMultibodySystem(system);
	}

	MuscleExcitationSetterEventHandler::~MuscleExcitationSetterEventHandler()
	{
	}

	bool MuscleExcitationSetterEventHandler::handle(Real currTime)
	{
		
		//Real currTime = getMultibodySystem()->getMultibodyDyna()->getCurrTime();
		if(!needHandle(currTime)) 
			return false;
		PeriodicEventHandler::_setHandledTime(currTime);

		string input;
		std::getline(std::cin, input);	
		std::string::size_type sz;     // alias of size_t
		std::vector<std::string> strs;
		
		if (input.size() > 0) {
			boost::split(strs, input, boost::is_any_of("\t "));
			
			for (int i = 0; i < strs.size(); i++) {
				musclesExc[i] = double(std::atof(strs[i].c_str()));
			}
		} else {
			for (int i = 0; i < strs.size(); i++) {
				musclesExc[i] = 0.0;
			}
		}
		

		if (verboseReceive) {
			printf("DELT_excs=%f, PECM_exc=%f, TRI_exc=%f, BIC_exc=%f ", musclesExc[0], musclesExc[1],musclesExc[2], musclesExc[3]);
		}
		
		// Set muscle excitations in arm model - all branches use same input
		for(int m = 0; m < _muscles.size(); ++m) {
			LOAMuscle::Ref muscle = _muscles[m];
			std::string name = muscle->getName();

			if(name.substr(0,3) == "BIC" || name.substr(0,3) == "BRA" ) {
			//if(name.substr(0,3) == "BRA" ) {
				muscle->setExcitation(musclesExc[3]);
			}
			else if(name.substr(0,3) == "TRI") {
			//else if(name.substr(0,6) == "TRIlon") {
				muscle->setExcitation(musclesExc[2]);
			}
			else if(name.substr(0,4) == "PECM" || name.substr(0,5) == "DELT1" || name.substr(0,6) == "Coraco" ) {
			//else if(name.substr(0,4) == "PECM" || name.substr(0,5) == "DELT1") {
			//else if(name.substr(0,4) == "PECM") {
			//else if(name.substr(0,4) == "PECM" || name.substr(0,5) == "DELT2") {	
			//else if(name.substr(0,5) == "PECM1") {
			//else if(name.substr(0,5) == "PECM2" || name.substr(0,5) == "PECM3") {
				muscle->setExcitation(musclesExc[1]);
			}
			else if(name.substr(0,5) == "DELT3" || name.substr(0,5) == "Infra" || name.substr(0,5) == "Latis" || name.substr(0,5) == "Teres") { // just branch 3 of deltoid + infraspinatus
			//else if(name.substr(0,4) == "DELT") { // all branches of deltoid
			//else if(name.substr(0,5) == "DELT3") { // branch 3 (posterior) of deltoid 
			//else if(name.substr(0,5) == "DELT3" || name.substr(0,5) == "DELT2") { // just branches 2 and 3 of deltoid; also need to add infraspinatus
			//lse if(name.substr(0,5) == "DELT3" || name.substr(0,5) == "INFSP") { // just branch 3 of deltoid + infraspinatus
			//else if(name.substr(0,5) == "DELT3"  || name.substr(0,5) == "DELT2" || name.substr(0,5) == "INFSP") { // just branches 2+3 of deltoid + infraspinatus
				muscle->setExcitation(musclesExc[0]);
			}
			else {//do nothing
			}
		}


		return true;
	}

	void MuscleExcitationSetterEventHandler::initBeforeRun()
	{
		EventHandler::initBeforeRun();

		// declare temporal variables
		int flags, err;
		
	}

	void MuscleExcitationSetterEventHandler::readFromXML(DOMNode* node)
	{
		XMLDOM::DOMElement* tmpNode = NULL;
		
		/*
		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"PortReceive");
		CHK_ERR(tmpNode, "Can not find socket port");
		portReceive = XMLDOM::getValueAsType<int>(tmpNode);
		*/
		
		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"Interval");
		double interval = XMLDOM::getValueAsType<double>(tmpNode);
		this->setEventInterval(interval);
	
		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"LOAMuscleForceSubsystem");
		CHK_ERR(tmpNode, "Can not find LOAMuscleForceSubsystem node");
		std::string sysName = XMLDOM::getAttribute(tmpNode, "name");

		ForceSubsystem* fsys = getMultibodySystem()->getForceSubsystem(sysName);
		LOAMuscleForceSubsystem* mfsys = dynamic_cast<LOAMuscleForceSubsystem*>(fsys);
		std::string err_msg = "The ForceSubsystem " + sysName + " must be a MuscleForceSubsystem";
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
				//LOAMuscle* msl = _msclSys->getMuscle(name);
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

		// Read also articulated body and coordinates to use in special messages such as set joint angles
		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"Body");
		CHK_ERR(tmpNode, "Can not find Body node");
		std::string bodyname = XMLDOM::getAttribute(tmpNode, "name");

		if(bodyname.empty()) {
			CHK_ERR(tmpNode, "Can not find Body name");
		}

		Body* body = getMultibodySystem()->getMattSubsystem()->getBody(bodyname);
		if(!body) {
			CHK_ERR(tmpNode, "Can not find Body with given name");
		}

		_artBody = dynamic_cast<ReducedCoordArtBody*>(body);
		if(!_artBody) {
			CHK_ERR(tmpNode, "The found body is not a reduced coordiante articulated body");
		}

		needAll = false;
		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"CoordinateNames");
		CHK_ERR(tmpNode, "Can not find CoordinateNames node");
		all = XMLDOM::getAttribute(tmpNode,"all");
		if(!all.empty()) {
			boost::to_lower(all);
			if(all == "true") {
				needAll = true;
			}
		}

		if(needAll) { //read all muscle names
			_coords = _artBody->getCoords();
		}
		else {
			std::vector<Coordinate::Ref>& cods = _artBody->getCoords();
			std::string str = XMLDOM::getTextAsStdStringAndTrim(tmpNode);

			mf_utils::Tokenizer tokens(str);
			for(int i = 0; i < tokens.size(); ++i) {
				std::string name = tokens[i];

				for(int n = 0; n < cods.size(); ++n) {
					if(cods[n]->getName() == name) {
						_coords.push_back(cods[n]);
						break;
					}
				}
			}

		}


	}

} //end namespace 
