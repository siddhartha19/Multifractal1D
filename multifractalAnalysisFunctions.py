''' -------- Functions for multifractal analysis of a 1D measure -------
-------------------- by Siddhartha Mukherjee ---------------------------
------------- ICTS-TIFR, Bangalore, 22nd Sept 2023 -----------------'''

import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline

def calcMeasure1D( field, boxCoarsest, nx ):
	epsVals = [2**i for i in np.arange(boxCoarsest+1)]
	Zvals = []
	print("Calculating Measure")
	for eps in epsVals:
		zv = []
		for x in np.arange(0,nx,eps):
			zv.append( np.sum(field[ x:x+eps ]) )
		Zvals.append( np.array(zv) )
	return Zvals, epsVals

def calcMoments1D( Zvals, qvals, boxCoarsest ):
	epsVals = [2**i for i in np.arange(boxCoarsest+1)]
	neps = len(epsVals); nqs = len(qvals)
	print("Calculating Moments")
	Zqs = np.zeros((neps,nqs))
	for indxe, eps in enumerate(epsVals):
		for indxq, q in enumerate(qvals):
			Zqs[indxe,indxq] = np.log(np.sum(Zvals[indxe]**q))/(q-1.0)
	return Zqs

def calcGeneralizedDimensions1D( Zqs, qvals, boxCoarsest ):
	epsVals = [2**i for i in np.arange(boxCoarsest+1)]
	epsLog = np.log(np.array(epsVals))
	neps = len(epsVals); nqs = len(qvals)
	DqVals = np.zeros((nqs))
	print("Calculating Dq and Dh")
	startID = 0
	endID = 10
	for indxq, q in enumerate(qvals):
		z = np.polyfit( epsLog[startID:endID], Zqs[startID:endID,indxq], 1 )
		DqVals[indxq] = z[0]
	return DqVals, epsLog

def calcFalphaSplineFit1D( Dqs, qvals, qvalsInterp, dim, nqEnds=35, nqMid=50 ):
	if qvalsInterp==0:
		# nqEnds = 15
		# nqMid = 40
		qinterm = 15
		qvalsInterp = np.array( [i for i in np.linspace(-51,-qinterm,nqEnds)] + [i for i in np.linspace(-qinterm+0.25,qinterm-0.25,nqMid,endpoint=False)] + [i for i in np.linspace(qinterm,51,nqEnds)] )
	
	nalpha = len(qvalsInterp)
	alpha = np.zeros( nalpha )
	falpha = np.zeros( nalpha )
	print("Calculating Multifractal Spectrum using CubicSpline interpolation")
	dqs = Dqs[:]
	splFullFit = CubicSpline(qvals, dqs)
	dqsInterp = splFullFit(qvalsInterp)
	alphaFunc = (qvalsInterp-1.0)*( dqsInterp - dim + 1.0 )
	alphaFuncFit = CubicSpline(qvalsInterp, alphaFunc)
	alphaFuncDeriv = alphaFuncFit.derivative()
	alphasInterp = alphaFuncDeriv(qvalsInterp)
	falphaInterp = alphasInterp*qvalsInterp - (qvalsInterp-1.0)*(dqsInterp-dim+1.0) + dim-1.0
	alpha = alphasInterp
	falpha = falphaInterp
	return alpha, falpha





