''' ------- Sample code for multifractal analysis of a 1D measure ------
-------------------- by Siddhartha Mukherjee ---------------------------
------------- ICTS-TIFR, Bangalore, 22nd Sept 2023 ----------------- '''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from multifractalAnalysisFunctions import *

mpl.rc('lines', linewidth=2.0, markersize=8.0)
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['legend.numpoints'] = 1
mpl.rcParams.update({'font.size': 22})
plt.rcParams["font.family"] = "serif"

boxCoarsest = 15
nx 			= 2**boxCoarsest
# field 		= np.random.random(nx)

##-- Range of moment values for calculating the partition function scaling
#-- Picking an "Odd" 41 values so that the linspace skips q=1, for which the general analysis will fail. D_q for q=1 an be obtained by using the L'Hopital rule instead, we shall use simple interpolation

qmax  		= 41.0
qvals 		= np.linspace(-qmax,qmax,41) 
print(qvals)

def cascade(nb, fr):
	field = [1.0]
	for i in np.arange(nb):
		f_ = []
		for f in field:
			# frac = np.random.random()
			# frac = 0.4 if np.random.random() < 0.1 else 0.15
			frac = np.random.uniform(0.5-fr,0.5+fr)
			f_.append( f*frac )
			f_.append( f*(1.0-frac) )
		field = f_
	print(len(field))
	return np.array(field)

''' ------------ Create measure using random curdling -------------
field = cascade( Number of curdling steps, "degree of curdling" )
Degree of curdling, when set to low values like 0.01 will lead to 
very little multifractality. While values around (max) 0.5 leads 
to extreme variation within the measure, possibly collapsing the 
multifractal analysis due to very small and large values.
------------------------------------------------------------------'''
fr 		   = 0.25
field 	   = cascade( boxCoarsest, fr )
print(field.shape)
#-- Dividing measure by the mean value helps in the analysis, by alleviating the "smallest" measure values, which can then [numerically] survive being raised to very large moments
field 	  /= np.mean(field)

plt.figure(1)
plt.plot( field )
plt.tight_layout()
plt.yscale('log')
# plt.show()

##----------------------------------------
##-- Multifractal functions
nboxes 				= boxCoarsest
Zvals, epsVals  	= calcMeasure1D( field, nboxes, nx )
Zqs  				= calcMoments1D( Zvals, qvals, nboxes )

'''----------------------------------------
----- Check measure moments scaling -------

You can use this section to check how well Z_q scales with r,
for some values of q
'''

epsVals = [2**i for i in np.arange(nboxes+1)]
epsLog 	= np.log(np.array(epsVals))
indxq 	= 10

plt.figure(4)
print(indxq, qvals[indxq])
for indxq in [0, 10, 20, 30, 40]:
	plt.plot(epsLog, Zqs[:,indxq] , 'o', label=r'$%d$' % qvals[indxq])

plt.xlabel(r'${\rm log}(r)$')
plt.ylabel(r'${\rm log}(Z^{1/(q-1)})$')
plt.tight_layout()
plt.legend(fontsize=16)
# plt.show()

'''--------------------------------------------
--------Calculate generalized dimensions ------

Here, we shall use linear fits to the loglog plot of Z_q vs r. See the function calcGeneralizedDimensions1D where we specify the range of r over which to calculate the linear fit. The results will be sensitive to the choice of this range [startID, endID], and one must check where a linear scaling is obtained for their dataset. '''

DqVals, epsLog  	= calcGeneralizedDimensions1D( Zqs, qvals, nboxes )

for d in DqVals:
	print(d)

'''-------------------------------------------
-------Calculate Singularity Spectrum --------

Here we calculate the singualrity spectrum. The D_q VS q and f_alpha vs alpha are related via a Legendre transform. We calculate this numerically, using spline fits to calculate additional alpha values corresponding to the roughly -15 < q < 15 region, which forms the tails of the f_alpha distribution. It is important to check whether the tails have converged.

'''

alpha, falpha  		= calcFalphaSplineFit1D( DqVals, qvals, 0, 1 )

plt.figure(2)
plt.plot( qvals, DqVals, 'ok' )
plt.xlabel(r'$q$')
plt.ylabel(r'$D_q$')
# plt.ylim([0.5, 1.75])
plt.tight_layout()

plt.figure(3)
plt.plot( alpha, falpha, 'or' )
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$f_\alpha$')
plt.xlim([0, 2.5])
plt.tight_layout()

plt.show()