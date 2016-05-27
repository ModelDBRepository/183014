#include "CoordinateOutputEventHandler.h"

#include <Multibody/MultibodyDyna.h> 

#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <fcntl.h>      /* To change socket to nonblocking mode */
#include <arpa/inet.h>  /* For inet_pton() */

FILE_STATIC_CALLHACK(CoordinateOutputEventHandler);

namespace mf_mbd
{
	bool verboseSend = 0;
	//double shoulderAngInput;
	//double elbowAngInput;
	
	InitFactoryStaticMembersMacro(CoordinateOutputEventHandler, PeriodicEventHandler);

	//class MultibodySystem;
	CoordinateOutputEventHandler::CoordinateOutputEventHandler()
		:PeriodicEventHandler()
	{
	}

	CoordinateOutputEventHandler::CoordinateOutputEventHandler(MultibodySystem& system)
		:PeriodicEventHandler()
	{
		setMultibodySystem(system);
	}

	CoordinateOutputEventHandler::CoordinateOutputEventHandler(MultibodySystem& system, Real interval)
		:PeriodicEventHandler(interval)
	{
		setMultibodySystem(system);
	}

	CoordinateOutputEventHandler::~CoordinateOutputEventHandler()
	{
		if(_out.is_open()) _out.close();
	}

	bool CoordinateOutputEventHandler::handle(Real currTime)
	{
		if(!needHandle(currTime)) 
			return false;
		PeriodicEventHandler::_setHandledTime(currTime);

		//std::cout << currTime << " ";

		for(int n = 0; n < _coords.size(); ++n) {
			Coordinate& cd = *_coords[n];
			std::cout << " " <<  cd.getQ();// << "\t" << cd.getQd() << "\t" << cd.getQdd(); //velocity and acceleration of coordinate
		}
		
		std::cout << std::endl;
		
		// send packets
		if (verboseSend) {
			printf("\nSent joint angles to stdout\n");
		}
		
		return true;
	}
	
	void CoordinateOutputEventHandler::initBeforeRun()
	{
		EventHandler::initBeforeRun();

		if(!_out.is_open()) {
			_out.open(_outFileName.c_str());
			if(!_out.is_open()) CL_ERR("Can not open file " + _outFileName);
		}

	}

	void CoordinateOutputEventHandler::readFromXML(DOMNode* node)
	{
		XMLDOM::DOMElement* tmpNode = NULL;
		
		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"Interval");
		double interval = XMLDOM::getValueAsType<double>(tmpNode);
		this->setEventInterval(interval);
		
		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"PntOutput");
		if(tmpNode) {
			_outFileName = XMLDOM::getAttribute(tmpNode,"name");
		}

		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"Body");
		CHK_ERR(tmpNode, "Can not find Body node");
		std::string bodyname = XMLDOM::getAttribute(tmpNode, "name");

		if(bodyname.empty()) {
			CHK_ERR(tmpNode, "Can not find Body name");
		}

		Body* body = getMultibodySystem()->getMattSubsystem()->getBody(bodyname);
        if(!body) {
                    CL_ERR("Can not find Body with given name " + bodyname);
             }
 
        _artBody = dynamic_cast<ReducedCoordArtBody*>(body);
        if(!_artBody) {
                    CL_ERR("The found body " + bodyname + " is not a reduced coordiante articulated body ");
             }		

		bool needAll = false;
		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"CoordinateNames");
		CHK_ERR(tmpNode, "Can not find CoordinateNames node");
		std::string all = XMLDOM::getAttribute(tmpNode,"all");
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

		tmpNode = XMLDOM::getFirstChildElementByTagName(node,"PntOutput");
		if(tmpNode) {
			_outFileName = XMLDOM::getAttribute(tmpNode,"name");
		}

		if(_outFileName.empty()) {
			_outFileName = _artBody->getName() + "_coordates_status.pnt";
		}

		_out.open(_outFileName.c_str());
		if(!_out.is_open()) CL_ERR("Can not open file " + _outFileName);

	}

} //end namespace 
