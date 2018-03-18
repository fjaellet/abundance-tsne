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
setup_text_plots(fontsize=13, usetex=True)

import open_data
from astropy.io import fits as pyfits
from sklearn import manifold, datasets

p = [15, 30, 40, 50, 75]
# The datasets
sets = "teffcut"

f = plt.figure(figsize=(7, 2*len(p)))
plt.suptitle("Sensitivity of abundance-space t-SNE to input parameters",
             fontsize=20)
import matplotlib.gridspec as gridspec
gs0 = gridspec.GridSpec(len(p), 4)
gs0.update(left=0.03, bottom=0.03, right=0.92, top=0.9,
           wspace=0.05, hspace=0.05)

plotsubsets = True

if plotsubsets:
    t     = open_data.harps()
    refdata  = t.data
    t.get_tsne_subsets( sets = sets )

for ii in range(len(p)):
    print ii
    # READ DATA
    #bensb0 = np.genfromtxt("../tsne_results/harps_tsne_results"+str(p[ii])+
    #                    "_rand0.csv", delimiter=',')
    bensb1 = np.genfromtxt("../tsne_results/harps_tsne_results_withnofehteffcut"+str(p[ii])+
                        "_rand0.csv", delimiter=',')
    bensb2 = np.genfromtxt("../tsne_results/harps_tsne_results_withteffcut"+str(p[ii])+
                        "_rand0.csv", delimiter=',')
    bensb3 = np.genfromtxt("../tsne_results/harps_tsne_results_withageteffcut"+str(p[ii])+
                        "_rand0.csv", delimiter=',')
    bensb4 = np.genfromtxt("../tsne_results/harps_tsne_results_withageteffcut"+str(p[ii])+
                        "_rand0.csv", delimiter=',')
    data = [bensb1, bensb2, bensb3, bensb4]
    title= ["Only [X/Fe]", "[X/Fe] + [Fe/H]", "Abunds.+age",
            "Abunds.+age+kin."]
    #best = [1, 5, 4]
    for jj in range(4):
        ax = plt.Subplot(f, gs0[ii, jj])
        f.add_subplot(ax)
        if jj == 1 and ii == 2:
            ax.set_facecolor('yellow')
        if jj == 0 and ii == 4:
            data[jj][:, 1] = -data[jj][:, 1]
        if plotsubsets:
            for kk in np.arange(len(t.subsets)):
                mask = (t.classcol == t.subsets[kk])
                scat = plt.scatter(data[jj][mask, 1], data[jj][mask, 2],
                                   s=t.size[kk], lw=t.lw[kk], edgecolors="k",
                                   c=t.col[kk], alpha=t.al[kk], marker=t.sym[kk])
        else:
            scat = plt.scatter(data[jj][:, 1], data[jj][:, 2], c='k', marker="o",
                           lw=0, s=6, alpha=.5)
        #if jj!=0:
        ax.yaxis.set_major_formatter(NullFormatter())
        if jj==3:
            ax.text(1.02, 0.5, r"$p=$" + str(p[ii]), fontsize=14,
                    transform=ax.transAxes)
        if ii==0:
            #ax2 = ax.twiny()
            #ax2.set_xlim(ax.get_xlim())
            ax.xaxis.set_major_formatter(NullFormatter())
            plt.title(title[jj])
        #elif ii==len(p)-1:
        #    pass
        else:
            ax.xaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

if plotsubsets:
    plt.savefig("../im/harps-tSNE_perplexitytest_withsubsets.png", dpi=200)
else:
    plt.savefig("../im/harps-tSNE_perplexitytest.png", dpi=200)
plt.show()
