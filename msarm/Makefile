#include mk_rules
.SUFFIXES: .cpp .cc .c

%.o : %.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $< -o $@

%.o : %.cc
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $< -o $@

%.o : %.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

all:
	echo Use mk script to build project

allprograms: Tests  

BSOURCES = main.cpp  \
LSODAIntegrator2.cpp \
CoordinateOutputEventHandler_pipe.cpp \
MuscleExcitationSetterEventHandler_pipe.cpp \
MuscleStatusEventHandler_pipe.cpp

BOBJECTS = $(BSOURCES:.cpp=.o)

Tests : $(BOBJECTS) $(LDLIBS)
	@rm -f $@
	echo "LINK = $(LINK) "
	$(LINK) -o $@ $^ $(LDLIBS) $(LDLIBS) $(DYNAMIC_LINK_FLAG) $(LDFLAGS)

#	$(STRIP) $@


clean:
	rm -f core *.o

realclean:
	rm -f core *.o  dependencies

dep: $(BSOURCES) 
	$(CXX) $(DEPENDFLAG) $(CPPFLAGS) $^ >dependencies

-include dependencies
