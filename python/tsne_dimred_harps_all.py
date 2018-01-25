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

import sample
from astropy.io import fits as pyfits
from sklearn import manifold, datasets

p = [15, 30, 40, 50, 75]
# The datasets

f = plt.figure(figsize=(7, 2*len(p)))
#plt.suptitle("t-SNE manifold learning in chemical-abundance space", fontsize=14)
import matplotlib.gridspec as gridspec
gs0 = gridspec.GridSpec(len(p), 4)
gs0.update(left=0.08, bottom=0.08, right=0.92, top=0.96,
           wspace=0.05, hspace=0.12)

plotsubsets = True

if plotsubsets:
    hdu = pyfits.open(
            '/home/friedel/Astro/Spectro/HARPS/DelgadoMena2017.fits',
            names=True)
    refdata=hdu[1].data
    subsets = ["thin", "thick1", "thick2", "thick3",
               "mpthin", "smr",
               "t1trans", "debris", "highAlMg",
               "t3trans", "highTioutlier","lowalphaoutlier"]
    sym = ["o", "v", "^", ">", "s", "*", "<", "D", "p", "8", "P", "X"]
    al  = [.4, .6, .8, .8, .75, 1,1,1,1,1,1,1]
    lw  = [0,0,.5,.5, .5, .5, .5, .5, .5, .5, .5, .5]
    size= [6,9,9,9,12,20,18,18,18,18,22,25]
    col = ["k", "r", "orange", "gold", "g", "orange",
          "brown", "yellow", "royalblue", "hotpink", 
          "lime", "black"]
for ii in range(len(p)):
    print ii
    # READ DATA
    bensb0 = np.genfromtxt("specialsets/harps_tsne_results_withnofeh"+str(p[ii])+
                        "_rand0.csv", delimiter=',')
    bensb1 = np.genfromtxt("specialsets/harps_tsne_results_withnofehteffcut"+str(p[ii])+
                        "_rand0.csv", delimiter=',')
    bensb2 = np.genfromtxt("specialsets/harps_tsne_results_withteffcut"+str(p[ii])+
                        "_rand0.csv", delimiter=',')
    bensb3 = np.genfromtxt("specialsets/harps_tsne_results_withageteffcut"+str(p[ii])+
                        "_rand0.csv", delimiter=',')
    data = [bensb0, bensb1, bensb2, bensb3]
    title= [r"No $T_{\rm eff}$ cut, only [X/Fe]",
            "Only [X/Fe]", "[X/Fe] + [Fe/H]", "Abundances + age"]
    #best = [1, 5, 4]
    for jj in range(4):
        ax = plt.Subplot(f, gs0[ii, jj])
        f.add_subplot(ax)
        if jj == 2 and ii == 2:
            ax.set_axis_bgcolor('yellow')
        if jj == 0:
            data[jj][:, 1] = -data[jj][:, 1]
        if plotsubsets:
            for kk in np.arange(len(subsets)):
                mask = (np.char.rstrip(refdata["tSNE_class"],' ') == subsets[kk])
                scat = plt.scatter(data[jj][mask, 1], data[jj][mask, 2],
                                   s=size[kk], lw=lw[kk], edgecolors="k",
                                   c=col[kk], alpha=al[kk], marker=sym[kk])
        else:
            scat = plt.scatter(data[jj][:, 1], data[jj][:, 2], c='k', marker="o",
                           lw=0, s=6, alpha=.5)
        if jj!=0:
            ax.yaxis.set_major_formatter(NullFormatter())
        if jj==3:
            ax.text(1.05, 0.5, r"$p=$" + str(p[ii]), transform=ax.transAxes)
        if ii==0:
            #ax2 = ax.twiny()
            #ax2.set_xlim(ax.get_xlim())
            ax.xaxis.set_major_formatter(NullFormatter())
            plt.title(title[jj])
        elif ii==len(p)-1:
            pass
        else:
            ax.xaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

if plotsubsets:
    plt.savefig("../tSNE/harps-tSNE_perplexitytest_withsubsets.png", dpi=200)
else:
    plt.savefig("../tSNE/harps-tSNE_perplexitytest.png", dpi=200)
plt.show()
