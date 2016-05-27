# hocinterface.py -- package of helper functions to allow interface with 
#   various hoc utilities and data structures our models use
#
# Usage:
#   Within the desired place where you want to use the functions do
#     from hocinterface import *
#
# Last update: 12/7/12 (georgec)

# Import pylab here, so it's not required in the functions.
import pylab

# Import neuron here.
from neuron import h

#
# Miscellaneous hoc interface functions
#

# Convert a hoc variable (passed as string) into a numpy array.
def hv2narr (hocvar='vec'):
   exec('%s = h.%s.to_python()' % (hocvar, hocvar))
   exec('%s = pylab.array(%s)' % (hocvar, hocvar))
   exec('x = %s' % hocvar)
   return x

# Convert a numpy array into a hoc variable (passed as string).
def narr2hv (hocvar,narr):
   h('objref %s' % hocvar)
   h('%s = new Vector()' % hocvar)
   h('objref tmp')
   h.tmp = narr
   h('{%s.from_python(tmp)}' % hocvar)

# Get the CTYP number from the string (e.g. 'E2').
def get_ctyp_num (ctyp_str):
   ctyp_ind = -1
   for ii in range(int(h.CTYPi)):
      if (h.CTYP.o(ii).s == ctyp_str):
         ctyp_ind = ii
   return ctyp_ind

# Get the CTYP string from the CTYP number.
def get_ctyp_str (ctyp_num):
   return h.CTYP.o(ctyp_num).s

# Get the STYP number from the string (e.g. 'AM2').
def get_styp_num (styp_str):
   styp_ind = -1
   for ii in range(int(h.STYPi)):
      if (h.STYP.o(ii).s == styp_str):
         styp_ind = ii
   return styp_ind

# Get the STYP string from the STYP number.
def get_styp_str (styp_num):
   return h.STYP.o(styp_num).s

#
# NQS interface functions
#

# Do a gethdrs() for an NQS table.
def shownqshdr (nqsvar='col[0].cellsnq'):
   print nqsvar
   h('%s.gethdrs()' % nqsvar)
 
# Do a pr(numrows) for an NQS table.
def nqspr (nqsvar='col[0].cellsnq', numrows=10):
   print nqsvar    
   h('%s.pr(%d)' % (nqsvar, numrows))
   
# Convert a hoc NQS column into a numpy array.
def nqscol2narr (nqsvar='col[0].cellsnq', colstr='col'):
   h('objref tmpv')
   h('tmpv = new Vector()')
   h('tmpv = %s.getcol("%s")' % (nqsvar, colstr))
   tmpv = h.tmpv.to_python()
   tmpv = pylab.array(tmpv)
   return tmpv
     
#
# grvec functions
#

# Load saved grvec simulation info.
# NOTE: Both the file and its dot-prefixed equivalent need to be present for gvnew() to succeed.
def ldgrvec (grvec_fname):
   h.gvnew(grvec_fname)

# Look at the grvec printlist for the current sim or a saved grvec file.
def lookgrveclist (gvcobjnum=0): 
   if (gvcobjnum == 0):
      gvcstr = 'current simulation'
   else:
      gvcstr = h.panobjl.o(gvcobjnum).filename
   print 'GRVEC List #%d (%s)' % (gvcobjnum, gvcstr)
   print '--------------------------------------------'
   if (gvcobjnum == 0):
      for ii in range(int(h.printlist.count())):
         prstr = '%d %s %d ' % (ii, h.printlist.o(ii).name, \
            h.printlist.o(ii).vec.size())
         print prstr,
         if (h.printlist.o(ii).tvec == None):
            print '(vec only)'
         else:
            print '\n',
   else:
      for ii in range(int(h.panobjl.o(gvcobjnum).printlist.count())):
         print '%d %s %d' % (ii, h.panobjl.o(gvcobjnum).printlist.o(ii).name, \
            h.panobjl.o(gvcobjnum).printlist.o(ii).size)

# Get tvec and vec (in numpy form) from grvec (gvcobjnum=0 means current 
# sim; >0 means saved grvec file)
def getgrvecdat (gvcobjnum=0, vecname='C0_X0_Y0_SPKS'):
   found_ind = -1
   for ii in range(int(h.panobjl.o(gvcobjnum).printlist.count())):
       if (h.panobjl.o(gvcobjnum).printlist.o(ii).name == vecname):
          found_ind = ii
   if (found_ind == -1):
      print 'No such array is on the printlist.'
      return None, None
   elif (gvcobjnum == 0):
      vec = h.printlist.o(found_ind).vec.to_python()    
      tvec = h.printlist.o(found_ind).tvec
      if (tvec == None):
         tvec = pylab.linspace(0,len(vec)-1,len(vec))
      else:
         tvec = pylab.array(tvec.to_python())
   else:
      h('goodread = panobjl.o(%d).rv_readvec(%d,tvec,vec)' % (gvcobjnum, found_ind))
      if (not h.goodread):
         h('panobjl.o(%d).rv_readvec(%d,vec)' % (gvcobjnum, found_ind))
         vec = h.vec.to_python()
         tvec = pylab.linspace(0,len(vec)-1,len(vec))
      else:         
         tvec = pylab.array(h.tvec.to_python())
         vec = h.vec.to_python()
   vec = pylab.array(vec)
   return tvec, vec

# Plot tvec and vec (in numpy form) from grvec (gvcobjnum=0 means current 
# sim; >0 means saved grvec file)
def plotgrvecdat (gvcobjnum=0, vecname='C0_X0_Y0_SPKS'):
   tvec,vec = getgrvecdat(gvcobjnum, vecname)
   if (tvec != None):
      pylab.plot(tvec,vec)
