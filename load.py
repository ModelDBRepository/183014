from pylab import *
from scipy import ndimage
from scipy.interpolate import interp1d
import sys,os,numpy,scipy,subprocess
from neuron import h
from math import ceil
h.load_file("stdrun.hoc") # creates cvode object
from vector import *
from nqs import *
import tables # for reading matlab file format (HDF5)
from mtspec import * # python version of multitaper power spectrum estimation
from spectrogram import *
from scipy.stats.stats import pearsonr
from filter import lowpass
from multiprocessing import Pool

# frequency band ranges for monkeys
dbands = {}
dbands['delta'] = [0.5,2.5]
dbands['theta'] = [5,9]
dbands['alpha'] = [9,14]
dbands['beta'] = [15,25]
dbands['logamma'] = [25,55]
dbands['higamma'] = [65,115]

# channel/layer info - applies to all recordings?
def makeDLayers ():
  dlyrL,dlyrR={},{}
  dlyrL['supra'] = [4,5,6,7,8,9]
  dlyrL['gran'] = [12,13,14]
  dlyrL['infra'] = [16,17,18,19]
  dlyrR['supra'] = [5,6,7,8]
  dlyrR['gran'] = [10,11]
  dlyrR['infra'] = [13,14,15,16]
  for D in [dlyrL,dlyrR]:
    lk = D.keys()
    for k in lk:
      for c in D[k]:
        D[c] = k
  return dlyrL,dlyrR

dlyrL,dlyrR = makeDLayers()

# if recording is from the left side
def leftname (fname):
  f = None
  if fname.count("/"):
    f = fname.split("/")[1]
  else:
    f = fname
  f = f.split("-")
  if int(f[0]) == 1: return True
  return False

# matching pair of data-files
def namepair (f1,f2):
  if f1.count("spont") != f2.count("spont"): return False
  if f1.count("1-") == f2.count("1-"): return False
  f1sub,f2sub = f1.split("-")[1],f2.split("-")[1]
  if f1sub[0:2] != f2sub[0:2]: return False
  num1,num2=int(f1sub[2:len(f1sub)]),int(f2sub[2:len(f2sub)])
  if abs( num1 - num2 ) != 1: return False
  lf1,lf2=leftname(f1),leftname(f2)
  if lf1 and num1 > num2: return False # left side should have lower number
  if lf2 and num2 > num1: return False # left side should have lower number
  return True

# make nqs with file information
def makenqf ():
  nqf = NQS("fname","spont","left","pidx"); nqf.strdec("fname")
  lfiles = []; fl = os.listdir("data")
  for f in fl:
    if f.endswith(".mat"):
      lfiles.append("data/"+f)
      nqf.append("data/"+f,f.count("spont"),leftname(f),0)
  nqf.getcol("pidx").fill(-1) # invalid pair index
  pidx = 0 # pair index
  for i in xrange(int(nqf.v[0].size())):
    s1 = nqf.get("fname",i).s
    for j in xrange(i+1,int(nqf.v[0].size()),1):
      s2 = nqf.get("fname",j).s
      if namepair(s1,s2):
        print s1, " and " , s2, " are pair " , pidx
        nqf.getcol("pidx").x[i] = nqf.getcol("pidx").x[j] = pidx
        pidx += 1
        break
  return nqf,lfiles

# return recording pair data with: sampr,dt,datL,ttL,datR,ttR , using file info in nqf
def loadpair (nqf,pidx):
  datL,datR = None,None
  if pidx == -1: return None,None,None,None,None,None
  if nqf.select(-1,"pidx",pidx) != 2.0:
    print "couldn't find valid pair with id " , pidx
    return None,None,None,None,None,None
  if nqf.select(-1,"pidx",pidx,"left",1) == 1.0:
    fleft = nqf.get("fname",nqf.ind.x[0]).s
    print "loading " , fleft
    sampr,datL,dt,ttL = rdmat(fleft)
  else:
    print "couldn't find left element for pair " , pidx
    return None,None,None,None,None,None
  if nqf.select(-1,"pidx",pidx,"left",0) == 1.0:
    fright = nqf.get("fname",nqf.ind.x[0]).s
    print "loading " , fright
    sampr,datR,dt,ttR = rdmat(fright)
  else:
    print "couldn't find right element for pair " , pidx
    return None,None,None,None,None,None
  return sampr,dt,datL,ttL,datR,ttR    

nqf,lfiles = makenqf()

ion() # interactive mode for pylab

# read the matlab .mat file and return the sampling rate and electrophys data
def rdmat (fname):  
  fp = tables.openFile(fname) # open the .mat / HDF5 formatted data
  sampr = fp.root.adrate[0][0] # sampling rate
  dt = 1 / sampr # time-step in seconds
  dat = fp.getNode("/cnt") # cnt record stores the electrophys data
  dat = dat.read() # read it into memory
  fp.close()
  tmax = ( len(dat) - 1.0 ) * dt 
  tt = numpy.linspace(0,tmax,len(dat)) # time in seconds
  return sampr,dat,dt,tt

# convert data from rdmat into NQS
def mat2nq (dat,tt):
  nq = NQS(len(dat[0]) + 1)
  nq.v[0].from_python(tt)
  nq.s[0].s = "t"
  for i in xrange(len(dat[0])): nq.v[i+1].from_python(dat[:,i])
  return nq

#
def plotlfpspec (lfp,tt,mint=-1,maxt=-1):
  figure(); subplot(2,1,1);
  plotspectrogram(lfp,rate=sampr,maxfreq=100)
  if mint > -1 and maxt > -1: xlim((mint,maxt));
  subplot(2,1,2); plot(tt,lfp,linewidth=1.0);
  if mint > -1 and maxt > -1: xlim((mint,maxt));

#
def plotspecs (dat,tt,sampr,which):
  figure();
  nrows = len(which)
  j = 0
  for i in which:
    subplot(nrows,1,j);
    plotspectrogram(dat[:,i],rate=sampr,maxfreq=100)
    j += 1

# get correlation matrix between all pairs of columns
def cormat (mat):
  rv = numpy.zeros( (len(mat[0]),len(mat[0])) )
  for i in xrange(len(mat[0])):
    rv[i][i]=1.0
    for j in xrange(i+1,len(mat[0]),1):
      rv[i][j] = rv[j][i] = pearsonr(mat[:,i],mat[:,j])[0]
  return rv

# get euclidean distance
def dist (x,y):
  return numpy.sqrt(numpy.sum((x-y)**2))

# get distance matrix between all pairs of columns
def distmat (mat):
  rv = numpy.zeros( (len(mat[0]),len(mat[0])) )
  for i in xrange(len(mat[0])):
    rv[i][i]=1.0
    for j in xrange(i+1,len(mat[0]),1):
      rv[i][j] = rv[j][i] = dist(mat[:,i],mat[:,j])
  return rv

# from /usr/site/nrniv/local/python/spectrogram.py - modified here
def getspec (tseries,rate=20000,window=1,maxfreq=125,tsmooth=0,fsmooth=0,winf=numpy.hanning,logit=False):
  from pylab import size, array, zeros, fft, convolve, r_    
  # Handle input arguments
  if maxfreq==0 or maxfreq>rate/2: maxfreq=rate/2 # Set maximum frequency if none entered or if too big
  npts=size(tseries,0) # How long the data are
  maxtime=npts/rate # Maximum time    
  ts = tseries - tseries.mean() # Remove mean
  print 'Calculating spectra...'
  nwind=int(maxtime/window) # Number of windows
  lwind=int(window*rate) # Length of each window
  spectarr=zeros((lwind/2,nwind))
  if winf is None:
    for i in xrange(nwind):
      tstart=lwind*i # Initial time point
      tfinish=lwind*(i+1) # Final timepoint
      thists=ts[tstart:tfinish] # Pull out the part of the time series to make this spectrum
      spectarr[:,i]=abs(fft(thists))[0:lwind/2]
  else:
    winh = winf(lwind)
    for i in xrange(nwind):
      tstart=lwind*i # Initial time point
      tfinish=lwind*(i+1) # Final timepoint
      thists=ts[tstart:tfinish] # Pull out the part of the time series to make this spectrum
      tmp=winh*thists
      spectarr[:,i]=abs(fft(tmp))[0:lwind/2]
  if fsmooth > 0 or tsmooth > 0: smooth2D(spectarr,tsmooth,fsmooth) # Perform smoothing
  # Calculate time and frequency limits
  finalfreq=int(window*maxfreq)
  F=r_[0:finalfreq]/float(window)
  T=r_[0:nwind]*window
  if logit:
    return F,T,10*numpy.log10(spectarr[0:finalfreq,:])
  else:
    return F,T,spectarr[0:finalfreq,:]

# get CSD and associated specgrams (uses getspec). dat is a list of LFPs , eg from loadpair
def getCSDspec (dat,sampr,window=1,maxfreq=125,logit=True,tsmooth=0,fsmooth=0):
  CSD = getCSD(dat,sampr)
  lsp = []   # get specgrams
  F,T=None,None
  for i in xrange(CSD.shape[0]):
    F,T,sp = getspec(CSD[i,:],rate=sampr,window=window,maxfreq=maxfreq,tsmooth=tsmooth,fsmooth=fsmooth,logit=logit)
    lsp.append(sp)
  return CSD,F,T,lsp

# get spec from all channels in dat
def getALLspec (dat,sampr,window=1,maxfreq=125,logit=True,tsmooth=0,fsmooth=0):
  lF,lT,lspec = [],[],[]
  nchan = dat.shape[1]
  for cdx in xrange(nchan):
    print cdx
    F,T,spec = getspec(dat[:,cdx],rate=sampr,window=window,logit=logit,tsmooth=tsmooth,fsmooth=fsmooth)
    lF.append(F); lT.append(T); lspec.append(spec);
  return lF,lT,lspec

#
def smooth2D (arr,xsmooth,ysmooth):
  xblur=array([0.25,0.5,0.25])
  yblur=xblur
  nr = arr.shape[0]
  nc = arr.shape[1]
  if ysmooth > 0:
    print 'Smoothing y...'
    for i in xrange(nc): # Smooth in frequency
      for j in xrange(ysmooth): 
        arr[:,i]=convolve(arr[:,i],yblur,'same')
  if xsmooth > 0:
    print 'Smoothing x...'
    for i in xrange(nr): # Smooth in time
      for j in xrange(xsmooth): 
        arr[i,:]=convolve(arr[i,:],xblur,'same')

# from /usr/site/nrniv/local/python/spectrogram.py - modified here
def getmtspecg (ts,rate=20000,window=1,maxfreq=125,tsmooth=0,fsmooth=0):
    from pylab import size, array, zeros, fft, convolve, r_    
    # Handle input arguments
    if maxfreq==0 or maxfreq>rate/2: maxfreq=rate/2 # Set maximum frequency if none entered or if too big
    F = None
    npts=size(ts,0) # How long the data are
    maxtime=npts/rate # Maximum time    
    ts = ts - ts.mean()
    # Calculating spectra
    print 'Calculating spectra...'
    nwind=int(maxtime/window) # Number of windows
    lwind=int(window*rate) # Length of each window
    spectarr=zeros((lwind/2,nwind))
    for i in xrange(nwind):
      tstart=lwind*i # Initial time point
      tfinish=lwind*(i+1) # Final timepoint
      thists=ts[tstart:tfinish] # Pull out the part of the time series to make this spectrum
      p,w = mtspec(thists,1.0/sampr,4)
      spectarr[:,i] = p[0:lwind/2]
      F = w      
    if fsmooth > 0 or tsmooth > 0: smooth2D(spectarr,tsmooth,fsmooth) # Perform smoothing
    # Calculate time and frequency limits
    finalfreq=int(window*maxfreq)
    F = F[0:finalfreq]/float(window)
    T=r_[0:nwind]*window
    return F,T,spectarr[0:finalfreq,:]

#
def plotspec (T,F,S,vc=[]):
  if len(vc) == 0: vc = [amin(S), amax(S)]
  figure();
  imshow(S,extent=(0,amax(T),0,amax(F)),origin='lower',interpolation='None',aspect='auto',vmin=vc[0],vmax=vc[1]);
  colorbar(); xlabel('Time (s)'); ylabel('Frequency (Hz)');

#
def slicenoise (arr,F,minF=58,maxF=62):
  sidx,eidx = -1,-1
  for i in xrange(len(arr)):
    if F[i] >= minF and sidx == -1:
      sidx = i
    if F[i] >= maxF and eidx == -1:
      eidx = i
  return numpy.append(arr[0:sidx+1], arr[eidx+1:len(arr)])

#
def slicenoisebycol (arr2D,F,minF=58,maxF=62):
  aout = []
  for i in xrange(arr2D.shape[1]):
    aout.append(slicenoise(arr2D[:,i],F,minF,maxF))
  tmp = numpy.zeros( (len(aout[0]), len(aout) ) )
  for i in xrange(len(aout)):
    tmp[:,i] = aout[i]
  return tmp

#
def keepF (arr,F,minF=25,maxF=55):
  sidx,eidx = -1,-1
  for i in xrange(len(arr)):
    if F[i] >= minF and sidx == -1:
      sidx = i
    if F[i] >= maxF and eidx == -1:
      eidx = i
  return numpy.array(arr[sidx:eidx+1])

#
def keepFbycol (arr2D,F,minF=25,maxF=55):
  aout = []
  for i in xrange(arr2D.shape[1]):
    aout.append(keepF(arr2D[:,i],F,minF,maxF))
  tmp = numpy.zeros( (len(aout[0]), len(aout) ) )
  for i in xrange(len(aout)):
    tmp[:,i] = aout[i]
  return tmp

# integrated power time-series -- gets power in range of minF,maxF frequencies
def powinrange (lspec,F,minF,maxF):
  nchan = len(lspec)
  lpow = numpy.zeros( (nchan,lspec[0].shape[1]) )
  F1idx,F2idx=-1,-1
  for i in xrange(len(F)):
    if minF <= F[i] and F1idx == -1: F1idx = i
    if maxF <= F[i] and F2idx == -1: F2idx = i
  # print F1idx,F[F1idx],F2idx,F[F2idx]
  rng = F2idx-F1idx+1
  for i in xrange(nchan): # channels
    for j in xrange(lspec[i].shape[1]): # time
      lpow[i][j] = numpy.sum(lspec[i][F1idx:F2idx+1,j])/rng
  return lpow

# get a list of lists
def LList (nrow):
  ll = [ [] for i in xrange(nrow) ]
  return ll

# get minima,maxima in integrated power time-series
def getpowlocalMinMax (lpow,lth = []):
  lpowMAX,lpowMIN = zeros(lpow.shape),zeros(lpow.shape)
  nchan = lpow.shape[0]
  maxx,maxy,minx,miny=LList(nchan),LList(nchan),LList(nchan),LList(nchan)
  if len(lth) == 0: # dynamic threshold
    for i in xrange(nchan): lth.append(median(lpow[i,:]))
  for i in xrange(nchan):
    for j in xrange(1,lpow.shape[1]-1,1):
      if lpow[i][j] > lth[i] and lpow[i][j] > lpow[i][j-1] and lpow[i][j] > lpow[i][j+1]:
        lpowMAX[i][j]=1
        maxx[i].append(j)
        maxy[i].append(lpow[i][j])
      if lpow[i][j] <= lth[i] and lpow[i][j] < lpow[i][j-1] and lpow[i][j] < lpow[i][j+1]:
        lpowMIN[i][j]=1
        minx[i].append(j)
        miny[i].append(lpow[i][j])
  return lpowMIN,lpowMAX,minx,miny,maxx,maxy

# get indices splitting the data into low/high power for the band
def splitBYBandPow (ddat,minF,maxF):
  llpow,llpowMIN,llpowMAX = [],[],[]
  lminx,lminy,lmaxx,lmaxy = [],[],[],[]
  for fn in ddat.keys():
    print fn
    F = ddat[fn]['F']
    lspec = ddat[fn]['nplsp']
    lpow=powinrange(lspec,F,minF,maxF); llpow.append(lpow)
    lpowMIN,lpowMAX,minx,miny,maxx,maxy=getpowlocalMinMax(lpow)
    llpowMIN.append(lpowMIN); llpowMAX.append(lpowMAX); lminx.append(minx);
    lminy.append(miny); lmaxx.append(maxx); lmaxy.append(maxy);
  dout={}
  dout['llpow']=llpow;
  dout['llpowMIN']=llpowMIN; dout['llpowMAX']=llpowMAX;
  dout['lminx']=lminx; dout['lminy']=lminy;
  dout['lmaxx']=lmaxx; dout['lmaxy']=lmaxy;
  return dout

# draws the CSD (or LFP) split by high/low power in a particular band. requires
# the lminx,lmaxx arrays from splitBYBandPow, and ddat (from rdspecgbatch)
def drawsplitbyBand (ddat,lminx,lmaxx,useCSD=True,ltit=['OFF','ON'],xls=(0,125),yls=(33,50)):
  csm=cm.ScalarMappable(cmap=cm.winter_r); csm.set_clim((0,1));
  avgON,avgOFF,cntON,cntOFF=[],[],[],[]; ii = 0; maxChan=19; minChan=1;
  chanSub,chanAdd=1,2; ylab='CSDSpec';
  if not useCSD: 
    chanSub=0; chanAdd=1; ylab='LFPSpec'; maxChan=23; minChan=0;
  for chan in xrange(minChan,maxChan,1):
    fn=ddat.keys()[0]; F=ddat[fn]['F'];
    print chan
    avgON.append(zeros((1,len(F)))); avgOFF.append(zeros((1,len(F))))
    cntON.append(0); cntOFF.append(0); fdx=0
    for fn in ddat.keys():
      ldat = ddat[fn]['nplsp']
      for i in xrange(chan-chanSub,chan+chanAdd,1):
        for j in lminx[fdx][i]:
          avgOFF[-1] += numpy.array(ldat[i][:,j]); cntOFF[-1] += 1
        for j in lmaxx[fdx][i]:
          avgON[-1] += numpy.array(ldat[i][:,j]); cntON[-1] += 1
      fdx += 1
    if cntON[-1]>0: avgON[-1] /= cntON[-1];
    if cntOFF[-1]>0: avgOFF[-1] /= cntOFF[-1];
    subplot(1,2,1); plot(F,avgOFF[-1].T,color=csm.to_rgba(float(chan)/(maxChan)),linewidth=1)
    xlabel('Frequency (Hz)'); ylabel(ylab); xlim(xls); ylim(yls); title(ltit[0]); grid(True)
    subplot(1,2,2); plot(F,avgON[-1].T,color=csm.to_rgba(float(chan)/(maxChan)),linewidth=1)
    xlabel('Frequency (Hz)'); ylabel(ylab); xlim(xls); ylim(yls); title(ltit[1]); grid(True)
  

# lowpass filter the items in lfps. lfps is a list or numpy array of LFPs arranged spatially by column
def getlowpass (lfps,sampr,maxf=200):
  datlow = []
  for i in xrange(len(lfps[0])): datlow.append(lowpass(lfps[:,i],maxf,df=sampr,zerophase=True))
  datlow = numpy.array(datlow)
  return datlow

# get CSD - first do a lowpass filter. lfps is a list or numpy array of LFPs arranged spatially by column
def getCSD (lfps,sampr,maxf=200):
  datlow = getlowpass(lfps,sampr,maxf)
  CSD = -numpy.diff(datlow,n=2,axis=0) # now each row is an electrode -- CSD along electrodes
  return CSD             

# downsamp - moving average downsampling
def downsamp (vec,winsz):
  sz = int(vec.size())
  i = 0
  k = 0
  vtmp = Vector(sz / winsz + 1)
  while i < sz:
    j = i + winsz - 1
    if j >= sz: j = sz - 1
    if j > i:            
      vtmp.x[k] = vec.mean(i,j)
    else:
      vtmp.x[k] = vec.x[i]
    k += 1
    i += winsz
  return vtmp

# downsamples the list of python lists using a moving average (using winsz samples)
def downsamplpy (lpy, winsz):
  vec,lout=Vector(),[]
  for py in lpy:
    vec.from_python(py)
    lout.append(downsamp(vec,winsz))
    lout[-1] = numpy.array(lout[-1].to_python())
  return lout


# gets population correlation vectors. lfps is column vectors. samples are rows.
def getpcorr (lfps, winsz):
  idx,jdx,n,sz = 0,0,len(lfps[0]),len(lfps)
  pcorr = numpy.zeros((int(math.ceil(sz/winsz)+1),n*(n-1)/2))
  for sidx in xrange(0,sz,winsz):
    if idx % 10 == 0: print idx
    eidx = sidx + winsz
    if eidx >= sz: eidx = sz - 1
    jdx = 0
    for i in xrange(len(lfps[0])):
      v1 = lfps[sidx:eidx,i]
      for j in xrange(i+1,len(lfps[0]),1):
        v2 = lfps[sidx:eidx,j]
        pcorr[idx][jdx] = pearsonr(v1,v2)[0]
        jdx += 1
    idx += 1
  return pcorr

#
def getpco (pcorr):
  sz = len(pcorr)
  pco = numpy.zeros((sz,sz))
  for i in xrange(sz):
    pco[i][i]=1.0
    for j in xrange(sz):
      pco[i][j] = pco[j][i] = pearsonr(pcorr[i,:],pcorr[j,:])[0]
  return pco

# cut out the individual blobs via thresholding and component labeling
def blobcut (im,thresh):
  mask = im > thresh
  labelim, nlabels = ndimage.label(mask)
  return labelim, nlabels

# get an nqs with sample entropy entries - lts is a list of time-series
#  sampenM = epoch size, sampenR = error tolerance, sampenN = normalize, sampenSec = seconds to use
#  slideR = whether to use a sliding tolerance
def getnqsampen (lts,sampr,scale=1,sampenM=2,sampenR=0.2,sampenN=0,sampenSec=1,slideR=1):
  if h.INSTALLED_sampen == 0.0: h.install_sampen()
  nq = NQS("t","sampen","chid")
  vec,vs,vt,vch = Vector(), Vector(),Vector(),Vector()
  sampenWinsz = sampenSec * sampr # size in samples
  if sampenWinsz < 100 and sampenSec > 0:
    print "getnqsampen WARNING: sampenWinsz was : ", sampenWinsz, " set to 100."
    sampenWinsz = 100
    sampenSec = sampenWinsz / sampr # reset sampenSec
    nq.clear( (len(lts[0]) / sampenWinsz + 1) * len(lts) )
  else:
    nq.clear(len(lts))
  chid = 0 # channel ID
  for ts in lts:
    vec.from_python(ts)
    if sampenSec > 0:
      print "chid : " , chid, " of " , len(lts)
      vs.resize( vec.size() / sampenWinsz + 1); vs.fill(0)
      vec.vsampenvst(sampenM,sampenR,sampenN,sampenWinsz,vs,slideR)
      if vt.size() < 1:
        vt.indgen(0,vs.size()-1,1); vt.mul(sampenSec); vt.add(sampenSec / 2.0)
        vch.resize(vt.size());
      vch.fill(chid)
      nq.v[0].append(vt); nq.v[1].append(vs); nq.v[2].append(vch)
    else: # single value for the time-series on the channel
      nq.append(0,vec.vsampen(sampenM,sampenR,sampenN),chid)
    chid += 1
  return nq

# calculates/saves sampen from the mat file (fname)
def savenqsampen (fname,ldsz=[200],csd=False,scale=1,sampenM=2,sampenR=0.2,sampenN=0,sampenSec=1,slideR=1):
  print ' ... ' + fname + ' ... '
  sampr,dat,dt,tt=None,None,None,None
  try:
    sampr,dat,dt,tt = rdmat(fname)
  except:
    print 'could not open ' , fname
    return False
  print dat.shape
  maxf=300; datlow=getlowpass(dat,sampr,maxf); # lowpass filter the data
  del dat
  dat = datlow # reassign dat to lowpass filtered data
  if csd:
    CSD,F,T,lsp=getCSDspec(dat,sampr,window=1,maxfreq=maxf,logit=True)
    del dat
    if len(ldsz) > 0:
      for dsz in ldsz:
        V = downsamplpy(CSD,dsz)
        nq = getnqsampen(V,sampr/dsz,scale,sampenM,sampenR,sampenN,sampenSec,slideR)
        nq.sv("/u/samn/plspont/data/sampen/"+fname.split("/")[1]+"_CSD_dsz_"+str(dsz)+"_sampen.nqs")
        del V
        nqsdel(nq)
    else:
      nq = getnqsampen(CSD,sampr/dsz,scale,sampenM,sampenR,sampenN,sampenSec,slideR)
      nq.sv("/u/samn/plspont/data/sampen/"+fname.split("/")[1]+"_CSD_sampen.nqs")      
    del CSD,F,T,lsp
  else:
    lts = dat # transpose
    print 'lts.shape = ', lts.shape
    if len(ldsz) > 0:
      for dsz in ldsz:
        print 'dsz : ', dsz
        V = downsamplpy(lts,dsz)
        nq = getnqsampen(V,sampr/dsz,scale,sampenM,sampenR,sampenN,sampenSec,slideR)
        nq.sv("/u/samn/plspont/data/sampen/"+fname.split("/")[1]+"_dsz_"+str(dsz)+"_sampen.nqs")
        nqsdel(nq)
        del V
    else:
      nq = getnqsampen(lts,sampr,scale,sampenM,sampenR,sampenN,sampenSec,slideR)
      nq.sv("/u/samn/plspont/data/sampen/"+fname.split("/")[1]+"_sampen.nqs")
      print 'nq.gethdrs():'
      nq.gethdrs()
  del tt
  return True

# calculates/saves spectrogram from the mat file (fn)
def savespecg (fn,csd=False,rate=20e3,window=1,maxfreq=125,tsmooth=0,fsmooth=0,logit=True):
  print ' ... ' + fn + ' ... '
  sampr,dat,dt,tt=None,None,None,None
  try:
    sampr,dat,dt,tt = rdmat(fn)
  except:
    print 'could not open ' , fn
    return False
  print dat.shape
  fname = "/u/samn/plspont/data/specg/"+fn.split("/")[1]
  fname += "_window_"+str(window)+"_maxfreq_"+str(maxfreq)
  if csd:
    fname += "_CSD_specg.npz"
    CSD,F,T,lsp = getCSDspec(dat,sampr,window=window,maxfreq=maxfreq,logit=logit)
    nplsp = numpy.array(lsp)
    numpy.savez(fname,F=F,T=T,nplsp=lsp)
    del CSD,F,T,lsp,nplsp
  else:
    F,T,lsp=None,None,[]
    dat = dat.T
    for ts in dat:
      F,T,sp = getspec(ts,rate=sampr,window=window,logit=logit)
      lsp.append(sp)
    fname += "_specg.npz"
    nplsp = numpy.array(lsp)
    numpy.savez(fname,F=F,T=T,nplsp=lsp)
    del F,T,lsp,nplsp

# run mtspecg on files in lf (list of file paths)
def specgbatch (lf,csd=False,exbbn=True,rate=20e3,window=1,maxfreq=125,tsmooth=0,fsmooth=0):
  for fn in lf:
    if exbbn and fn.count("spont") < 1: continue
    savespecg(fn,csd,rate=rate,window=window,maxfreq=maxfreq,tsmooth=tsmooth,fsmooth=fsmooth)

#
def rdspecgbatch (lf,csd=False,exbbn=True,window=1,maxfreq=125):
  ddat = {}
  for fn in lf:
    if exbbn and fn.count("spont") < 1: continue
    fdat = '/u/samn/plspont/data/specg/'+fn.split('/')[1]
    fdat += "_window_"+str(window)+"_maxfreq_"+str(maxfreq)
    if csd: fdat += '_CSD'
    fdat += '_specg.npz'
    try:
      ddat[fn] = numpy.load(fdat)
    except:
      print 'could not load ' , fdat
  return ddat


# run sampen on files in lf (list of file paths)
def sampenbatch (lf,nproc=10,ldsz=[200],csd=False,exbbn=True,\
                 scale=1,sampenM=2,sampenR=0.2,sampenN=0,sampenSec=1,slideR=1):
  #pool = Pool(processes=nproc)
  #args = ((fn,dsz,csd,scale,sampenM,sampenR,sampenN,sampenSec,slideR) for fn in lf)
  #print 'args : ' , args
  #pool.map_async(savenqsampen,args)
  #pool.close(); pool.join()  
  for fn in lf:
    if exbbn and fn.count("spont") < 1: continue
    savenqsampen(fn,ldsz,csd,scale,sampenM,sampenR,sampenN,sampenSec,slideR)

# lf is list of files, exbbn == exclude broadband noise files
def rdsampenbatch (lf,dsz=200,csd=False,exbbn=True):
  dnq = {}
  for fn in lf:
    if exbbn and fn.count("spont") < 1: continue
    fnq = '/u/samn/plspont/data/sampen/'+fn.split('/')[1]
    if csd: fnq += '_CSD'
    if dsz > 0: fnq += '_dsz_'+str(dsz)
    fnq += '_sampen.nqs'
    try:
      dnq[fn]=NQS(fnq)
      if dnq[fn].m[0] < 2:
        dnq.pop(fn,None) # get rid of it
    except:
      print 'could not open ' , fnq
  return dnq


# sampr,datR,dt,ttR = rdmat(lfiles[0]) # right hemisphere - spontaneous
# sampr,datL,dt,ttL = rdmat(lfiles[1]) # left hemisphere - spontaneous

# sampr,dt,datL,ttL,datR,ttR = loadpair(nqf,18)

dlfp=rdspecgbatch(lfiles,csd=False,exbbn=True,window=1,maxfreq=125)
dcsd=rdspecgbatch(lfiles,csd=True,exbbn=True,window=1,maxfreq=125)

