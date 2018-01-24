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
import open_data

teffcut  = True
age      = False
kin      = False
feh      = True
p        = 50
i        = 0
mc       = 10 # needs to be an integer >=1. If ==1, then no MC magic will happen

add = ""
if age or kin or mc>1 or not feh:
    add = add + "_with"
    if teffcut:
        add = add + "teffcut"
    if age:
        add = add + "age"
    if kin:
        add = add + "kin"
    if not feh:
        add = add + "nofeh"
    if mc>1:
        add = add + "mc" + str(mc)
print add        

# The dataset
tsne =  np.genfromtxt("../tsne_results/bensby2014_tsne_results"+add+str(p)+
                           "_rand"+str(i)+".csv", delimiter=',',
                      dtype=[('Name', "|S14"), ('X', float), ('Y', float)])

means=  np.zeros((len(tsne)/mc, 6))
for ii in np.arange(means.shape[0]):
    means[ii,:3] = np.percentile(tsne["X"][mc*ii:mc*(ii+1)], [16,50,84])
    means[ii,3:] = np.percentile(tsne["Y"][mc*ii:mc*(ii+1)], [16,50,84])

# READ DelGADO-MENA DATA
ben = open_data.bensby(teffcut=teffcut)
data=ben.data
#if teffcut:
#    data=data[ (data['Teff']>5300) * (data['Teff']<6000) * \
#               (data['logg_hip']>3) * (data['logg_hip']<5) ]
data=data[ (data['nO1']>0) * (data['nNa1']>0) * (data['nMg1']>0) * (data['nAl1']>0) *
           (data['nSi1']>0) * (data['nCa1']>0) * (data['nTi1']>0) * (data['nCr1']>0) *
           (data['nNi1']>0) * (data['nZn1']>0) * (data['nY2']>0) * (data['nBa2']>0) *
           (data['nFe1']>0) ]

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
        scat = plt.scatter(tsne['X'], tsne['Y'],
                           c=np.repeat(colors[4*ii+jj],mc),
                           s=30./mc, cmap=plt.cm.jet, lw=0, alpha=0.1)
        """scat = plt.errorbar(means[:,1], means[:,4],
                            xerr=[abs(means[:,1]-means[:,0]),
                                  abs(means[:,2]-means[:,1]) ],
                            yerr=[abs(means[:,4]-means[:,3]),
                                  abs(means[:,5]-means[:,4]) ],
                           c="grey", zorder=0, ms=0, lw=0)"""
        scat = plt.scatter(means[:,1], means[:,4],
                           c=colors[4*ii+jj],
                           s=15, cmap=plt.cm.jet, lw=0.01)
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

rec = np.rec.fromarrays((data['HIP'], means[:,1], means[:,4]),
                        dtype=[('ID','|S18'),('X_tsne','float'),
                               ('Y_tsne','float')])
np.savetxt("../tsne_results/bensby2014_tsne_results"+add+str(p)+
                   "_rand"+str(i)+"_montecarloaverage.csv", rec, delimiter=',',
       fmt = ('%s, %.5e, %.5e'))
plt.savefig("../im/Bensby2014_tsne_plots"+add+str(p)+
                   "_montecarloaverage.png", dpi=200)
        
