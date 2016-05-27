#include <Math/MathHeaders.h>
#include <Multibody/MultibodyHeaders.h>
#include <BodyController/ControllerHeaders.h>
#include <OsgVisualization/OsgVizHeaders.h>

#include <Utils/Notify.h>
#include <FileIO/AtbTestBuilder.h>
#include <FileIO/ForceTestBuilder.h>

#include <iostream>
#include <iomanip>
#include "LSODAIntegrator2.h"

#include <FileIO/ControlledMbdSimulation.h>


#ifdef STATIC_CALLHACK
	Extern_Static_Call_Hack(CoordinateOutputEventHandler);
	//Extern_Static_Call_Hack(MuscleStatusEventHandler);
	Extern_Static_Call_Hack(MuscleExcitationSetterEventHandler);
#endif

int testFwdDynXML(int argc, char** argv)
{
	using namespace mf_mbd;

	const Real PI = 3.141592653589793238462643;

	try {

		if(argc == 1){
			throw std::runtime_error("Please specify the input file name");
		}

		std::string filename(argv[1]);
		
		ControlledMbdSimulation sim;
		//MbdSimulation sim;

		// read xml file
		sim.readFromFile(filename);

		sim.initBeforeRun();

		std::cout << "READY TO RUN" << std::endl;

		sim.run();

	}
	catch(std::exception& ex) {			//in case there is an error on the sim file, Cobi will stop
		std::cerr << "Error: " << ex.what() << std::endl;
		return 1;
	}	

	return 0;
}

int main(int argc, char** argv){

	/*Turns out that the object files containing the static initializers 
	 were not included by the linker because nothing referenced any functions in them. 
	 Therefore we need the line below for now, how to solve this????
	 Use Library Dependency Inputs set to yes fixed this problem
	*/

	try{
		//SLog::getLog().setOutputFile("log.txt");
		mf_mbd::IntegratorCreator::getFactoryReg()->regist("LSODAIntegrator2", &Math::User::LSODAIntegrator2<VecN>::createInstance);
		testFwdDynXML(argc,argv);
	}
	catch(std::exception& ex) {			//in case there is an error on the sim file, Cobi will stop
		std::cerr << "Error: " << ex.what() << std::endl;
		return 1;
	}	

	return 0;
}
