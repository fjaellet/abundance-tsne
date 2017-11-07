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
from sklearn import manifold, datasets
import scipy

# Next line to silence pyflakes. This import is needed.
#Axes3D

# The dataset
plot     = True
age      = False
kin      = False
feh      = True
teffcut  = True
mc       = 50 # needs to be an integer >=1. If ==1, then no MC magic will happen

# READ DelGADO-MENA DATA
hdu = pyfits.open(
    '/home/friedel/Astro/Spectro/HARPS/DelgadoMena2017.fits',
    names=True)
data=hdu[1].data
if teffcut:
    data=data[ (data['Teff']>5300) * (data['Teff']<6000) * \
               (data['logg_hip']>3) * (data['logg_hip']<5) ]
data=data[ (data['nCu']>0) * (data['nZn']>0) * (data['nSr']>0) * (data['nY']>0) *
           (data['nZrII']>0) * (data['nBa']>0) * (data['nCe']>0) * (data['errAl']<1) *
           (data['nMg']>0) * (data['nSi']>0) * (data['nCa']>0) * (data['nTiI']>0) *
           np.isfinite(data['meanage']) ] 
#if age:
#    data = data[ np.isfinite(data['meanage']) ]

n_points = len(data)
n_components = 2
verr = 10. * np.ones(len(data))
X        = np.c_[data['feh'],data['CuFe'],data['ZnFe'],data['SrFe'],
                 data['YFe'],data['ZrIIFe'],data['BaFe'],data['CeFe'],
                 data['AlFe'],data['MgFe'],data['SiFe'],data['CaFe'],
                 data['TiIFe'],data['meanage']]
Xerr1    = np.c_[data['erfeh'],data['errCu'],data['errZn'],data['errSr'],
                 data['errY'],data['errZrII'],data['errBa'],data['errCe'],
                 data['errAl'],data['errMg'],data['errSi'],data['errCa'],
                 data['errTiI'],data['agestd']]
Xerr     = np.mean( Xerr1, axis=0)

if mc > 1:
    Y        = np.zeros(( mc*len(data), 14 ))
    for ii in np.arange(mc):
        # Take care of the 0.0 uncertainties: forced minimum to 0.02 
        Y[ii::mc, :] = scipy.random.normal(loc=X, size=X.shape, 
                                           scale=np.maximum(Xerr1, 0.02*np.ones(Xerr1.shape)))                                      
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

colors   = [data['feh'], data['TiIFe'], data['CaFe'], data['MgFe'],
            data['MgFe']-data['TiIFe'],data['CuFe'],data['AlFe']-data['MgFe'],
            data['BaFe'], data['ZnFe'], 
            data['YFe']-data['BaFe'],data['YFe']-data['ZnFe'],data['CeFe'],
            data['Teff'],data['logg'], data['vt_1'],np.log10(data['S/N']),
            data['meanage'],data['Ulsr'],data['Vlsr'],data['Wlsr']]
titles   = [r'$\rm [Fe/H]$', r'$\rm [Ti/Fe]$', r'$\rm [Ca/Fe]$', r'$\rm [Mg/Fe]$',
          r'$\rm [Mg/Ti]$', r'$\rm [Cu/Fe]$', r'$\rm [Al/Mg]$', r'$\rm [Ba/Fe]$',
          r'$\rm [Zn/Fe]$', r'$\rm [Y/Ba]$', r'$\rm [Y/Zn]$', r'$\rm [Ce/Fe]$',
          r'$T_{\rm eff}$', r'$\log g$', r'$\xi$', r'lg $S/N$',
          r'$\tau$', r'$U$', r'$V$', r'$W$']

for p in [5, 15, 30, 40, 50, 75, 100, 120, 150]:#[
    for i in [0]:
        t0 = time()
        tsne = manifold.TSNE(n_components=n_components, init='pca',
                             random_state=i, perplexity=p,
                             learning_rate=1, early_exaggeration=len(X)/10)
        Y = tsne.fit_transform(Xnorm)
        t1 = time()
        print "p=", p, " , random state ", i
        print("t-SNE: %.2g sec" % (t1 - t0))

        f = plt.figure(figsize=(12, 12))
        plt.suptitle("t-SNE manifold learning for the HARPS sample (Delgado-Mena et al. 2017)", fontsize=14)
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

        rec = np.rec.fromarrays((np.repeat(data['Star'],mc), Y[:, 0], Y[:, 1]),
                                dtype=[('ID','|S18'),('X_tsne','float'),
                                       ('Y_tsne','float')])
        add = ""
        if age or kin or mc>1 or teffcut or not feh:
            add = add + "_with"
            if age:
                add = add + "age"
            if kin:
                add = add + "kin"
            if not feh:
                add = add + "nofeh"
            if teffcut:
                add = add + "teffcut"
            if mc>1:
                add = add + "mc" + str(mc)
        print add        
        np.savetxt("../tsne_results/harps_tsne_results"+add+str(p)+
                           "_rand"+str(i)+".csv", rec, delimiter=',',
               fmt = ('%s, %.5e, %.5e'))
        plt.savefig("../im/HARPS_tsne_plots"+add+str(p)+
                           "_rand"+str(i)+".png", dpi=200)
        
        plt.clf()
