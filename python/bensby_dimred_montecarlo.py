print(__doc__)

from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter

import numpy as np

from astroML.decorators import pickle_results
from astroML.density_estimation import XDGMM
from astroML.plotting.tools import draw_ellipse
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=10, usetex=True)

from astropy.io import fits as pyfits
import scipy
from sklearn import manifold, datasets

# The dataset
plot     = True
age      = False
kin      = False
feh      = True
mc       = 50 # needs to be an integer >=1. If ==1, then no MC magic will happen

# READ BENSBY DATA
hdu = pyfits.open(
    '/home/friedel/Astro/Spectro/Bensby/Bensby2014_SN_survey.fits',
    names=True)
data=hdu[1].data
data=data[ (data['nO1']>0) * (data['nNa1']>0) * (data['nMg1']>0) * (data['nAl1']>0) *
           (data['nSi1']>0) * (data['nCa1']>0) * (data['nTi1']>0) * (data['nCr1']>0) *
           (data['nNi1']>0) * (data['nZn1']>0) * (data['nY2']>0) * (data['nBa2']>0) *
           (data['nFe1']>0) ]

n_points = len(data)
n_components = 2
verr = 10. * np.ones(len(data))
X        = np.c_[data['Fe_H'],data['O_Fe'],data['Na_Fe'],data['Mg_Fe'],
                 data['Al_Fe'],data['Si_Fe'],data['Ca_Fe'],data['Ti_Fe'],
                 data['Cr_Fe'],data['Ni_Fe'],data['Zn_Fe'],data['Y_Fe'],
                 data['Ba_Fe'],data['Age']]
Xerr1    = np.c_[data['e_Fe_H'],data['e_O_Fe'],data['e_Na_Fe'],data['e_Mg_Fe'],
                 data['e_Al_Fe'],data['e_Si_Fe'],data['e_Ca_Fe'],data['e_Ti_Fe'],
                 data['e_Cr_Fe'],data['e_Ni_Fe'],data['e_Zn_Fe'],data['e_Y_Fe'],
                 data['e_Ba_Fe'],0.5*(data['B_Age1']-data['b_Age'])]
Xerr     = np.mean( Xerr1, axis=0)

if mc > 1:
    Y        = np.zeros(( mc*len(data), 14 ))
    for ii in np.arange(mc):
        Y[ii::mc, :] = scipy.random.normal(loc=X, scale=Xerr1, size=X.shape)
    X = Y

if not age:
    X = X[:,:-1]; Xerr = Xerr[:-1]
if kin:
    X    = np.append(X, np.vstack((data['Ulsr'],data['Vlsr'],data['Wlsr'])).T, axis=1)
    Xerr = np.append(Xerr, np.array([10,10,10]).T, axis=0)
if not feh:
    X = X[:,1:]; Xerr = Xerr[1:]
    
Xnorm    = (X/Xerr[np.newaxis,:])
#Xgood    = (np.sum(data['ELEMFLAG'][:, [0,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,
#                                18,19]], axis=1)== 0) #* (data['NA_FE']>-0.8) 
#X        = X[Xgood]
print len(X), " good ones"

colors   = [data['Fe_H'], data['Ti_Fe'], data['Fe_H']+data['O_Fe'],
            data['Mg_Fe']-(data['Fe_H']+data['O_Fe']),
            data['Mg_Ti'],data['Ni_Ti'],data['Al_Fe']-data['Mg_Fe'],
            data['Ba_Fe']-(data['Fe_H']+data['O_Fe']),
            data['Zn_Fe']-(data['Fe_H']+data['O_Fe']),
            data['Y_Fe']-data['Ba_Fe'],data['Y_Fe']-data['Zn_Fe'],data['Ni_Fe'],
            data['Teff'],data['logg'], data['xi'],np.log10(data['td_d']),
            data['Age'],data['Ulsr'],data['Vlsr'],data['Wlsr']]
titles   = [r'$\rm [Fe/H]$', r'$\rm [Ti/Fe]$', r'$\rm [O/H]$', r'$\rm [Mg/O]$',
          r'$\rm [Mg/Ti]$', r'$\rm [Ni/Ti]$', r'$\rm [Al/Mg]$', r'$\rm [Ba/O]$',
          r'$\rm [Zn/O]$', r'$\rm [Y/Ba]$', r'$\rm [Y/Zn]$', r'$\rm [Ni/Fe]$',
          r'$T_{\rm eff}$', r'$\log g$', r'$\xi$', r'$TD/D$',
          r'$\tau$', r'$U$', r'$V$', r'$W$']

for p in [15, 30, 40, 50, 75, 100, 120, 150]:#[
    for i in [0]:
        t0 = time()
        # Use the recommendations of Linderman & Steinerberger 2017
        # for the learning rate and the exaggeration parameter
        tsne = manifold.TSNE(n_components=n_components, init='pca',
                             random_state=i, perplexity=p,
                             learning_rate=1, early_exaggeration=len(X)/10)
        Y = tsne.fit_transform(Xnorm)
        t1 = time()
        print "p=", p, " , random state ", i
        print("t-SNE: %.2g sec" % (t1 - t0))

        f = plt.figure(figsize=(12, 12))
        plt.suptitle("t-SNE manifold learning for the Bensby+2014 sample", fontsize=14)
        import matplotlib.gridspec as gridspec
        gs0 = gridspec.GridSpec(5, 4)
        gs0.update(left=0.08, bottom=0.08, right=0.92, top=0.92,
                   wspace=0.05, hspace=0.12)

        for ii in range(5):
            for jj in range(4):
                ax = plt.Subplot(f, gs0[ii, jj])
                f.add_subplot(ax)
                scat = plt.scatter(Y[:, 0], Y[:, 1],
                                   c=np.repeat(colors[4*ii+jj],mc),
                                   s=15./mc, cmap=plt.cm.jet, lw=0)
                if jj!=0:
                    ax.yaxis.set_major_formatter(NullFormatter())
                if ii==0:
                    ax2 = ax.twiny()
                    ax2.set_xlim(ax.get_xlim())
                ax.xaxis.set_major_formatter(NullFormatter())
                #ax.yaxis.set_major_formatter(NullFormatter())
                divider = make_axes_locatable(ax)
                cax = divider.new_vertical(size="5%", pad=0.07, pack_start=True)
                f.add_axes(cax)
                plt.title(titles[4*ii+jj])
                f.colorbar(scat, cax=cax, orientation="horizontal")
                #plt.axis('tight')

        rec = np.rec.fromarrays((np.repeat(data['HIP'],mc), Y[:, 0], Y[:, 1]),
                                dtype=[('ID','|S18'),('X_tsne','float'),
                                       ('Y_tsne','float')])
        add = ""
        if age or kin or mc>1 or not feh:
            add = add + "_with"
            if age:
                add = add + "age"
            if kin:
                add = add + "kin"
            if not feh:
                add = add + "nofeh"
            if mc>1:
                add = add + "mc" + str(mc)
                
        np.savetxt("../tsne_results/bensby2014_tsne_results"+add+str(p)+
                           "_rand"+str(i)+".csv", rec, delimiter=',',
               fmt = ('%s, %.5e, %.5e'))
        plt.savefig("../im/Bensby2014_tsne_plots"+add+str(p)+
                           "_rand"+str(i)+".png", dpi=200)
        
        plt.clf()
