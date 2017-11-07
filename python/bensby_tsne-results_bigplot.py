"""
Multi-dim. extreme deconvolution of the local APOGEE_TGAS (d<1 kpc) stars
-----------------------------

An example of extreme deconvolution showing a simulated two-dimensional
distribution of points, where the positions are subject to errors. The top two
panels show the distributions with small (left) and large (right) errors. The
bottom panels show the densities derived from the noisy sample (top-right
panel) using extreme deconvolution; the resulting distribution closely matches
that shown in the top-left panel.
"""
# Author: F. Anders, but mostly really Jake VanderPlas
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
import numpy as np
from matplotlib import pyplot as plt

from astroML.decorators import pickle_results
from astroML.density_estimation import XDGMM
from astroML.plotting.tools import draw_ellipse

import sample
from astropy.io import fits as pyfits
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=10, usetex=True)

    
hdu = pyfits.open(
    '/home/friedel/Astro/Spectro/Bensby/Bensby2014_with_tSNE.fits',
    names=True)
data=hdu[1].data

labels = [r'$\rm \tau \ [Gyr]$', r'$\rm [Fe/H]$', r'$\rm [Ti/Fe]$',
          r'$\rm [Y/Ba]$', r'$\rm [Mg/O]$', r'$\rm [Al/Mg]$']

xx  = [data['Age'], data['Fe_H'], data['Ti_Fe'], data['Y_Fe']-data['Ba_Fe'],
          data['Mg_Fe']-data['O_Fe'], data['Al_Fe']-data['Mg_Fe']]
#exlist = [x1e**2., x2e**2., x3e**2., x4e**2., x5e**2.,
#          x6e**2., x7e**2., x8e**2., x9e**2., x10e**2., x11e**2., x12e**2.]

subsets = ["thin", "thick1", "thick2", "metalpoorthin",
           "highY/Zn", "smr",
           "supersolarlowY/Zn",
           "lowalphahighTihighY/Ba", "debris", "supersolarhighalpha",
           "transition"]
names   = ["Thin disc", "Thick Disc I", "Thick Disc II", "Metal-poor \n thin disc",
           "High-[Ba/Zn]", "SMR", "Super-solar \n high-Zn",
           r"High-[Ti/$\alpha$]", "Satellite \n debris",
           "Super-solar \n high-[Y/Ba]", "Thin/Thick-II \n transition"]
Xcoords = [-30, 12,  -7, 15, -41, -42, -24, -11, 24, -39, -10]
Ycoords = [5  ,-25,  -16, 17,  23,   -3, -19, 15,-13, -24, -8]
fsize   = [20 , 16,  16, 16,  13,  13, 11, 11, 11, 11, 11]
sym = ["o", "v", "^", "s", "p", "*", "8", "h", "D", "d", "H"]
al  = [.2, .4, .75, .75, 1,1,1,1,1,1,1]
lw  = [0,0,.5,.5, 1,1,1,1,1,1,1]
size= [6,9,9,12,15,22,18,18,18,18,19]
col = ["k", "r", "gold", "g", "b",
      "orange", "cyan", "lime", "yellow", "m",
      "hotpink", "peru", "cornflowerblue"]
#------------------------------------------------------------
# Plot the results
import matplotlib.gridspec as gridspec

#################
# t-SNE plot apart
#################
g   = plt.figure(figsize=(7, 9))
gs0 = gridspec.GridSpec(1, 1)
gs0.update(left=0.1, bottom=0.05, right=0.98, top=0.7)
ax  = plt.Subplot(g, gs0[0, 0])
g.add_subplot(ax)
for kk in np.arange(len(subsets)):
    mask = (np.char.rstrip(data["tSNE_class"],' ') == subsets[kk])
    ax.scatter(data["X_tsne"][mask], data["Y_tsne"][mask],
               s=4*size[kk], lw=lw[kk], edgecolors="k",
               c=col[kk], alpha=al[kk],
               marker=sym[kk])
    # Annotate population names
    ax.text(Xcoords[kk], Ycoords[kk], names[kk], fontsize=1.25*fsize[kk])
ax.set_xlabel("t-SNE X dimension", fontsize=13)
ax.set_ylabel("t-SNE Y dimension", fontsize=13)
gs = gridspec.GridSpec(1, 3)
gs.update(left=0.1, bottom=0.77, right=0.98, top=0.98,
           wspace=0.44, hspace=0.05)
exinds = [ [1,2], [4,5], [2,3] ]

for jj in range(3):
    ax = plt.Subplot(g, gs[0, jj])
    g.add_subplot(ax)
    for kk in np.arange(len(subsets)):
        mask = (np.char.rstrip(data["tSNE_class"],' ') == subsets[kk])
        ax.scatter(xx[exinds[jj][0]][mask], xx[exinds[jj][1]][mask],
                   s=size[kk], lw=lw[kk], edgecolors="k",
                   c=col[kk], alpha=al[kk],
                   marker=sym[kk])
    ax.set_xlabel(labels[exinds[jj][0]], fontsize=12)
    ax.set_ylabel(labels[exinds[jj][1]], fontsize=12)
plt.savefig("../bensby2014_tsne-plot.png", dpi=200)  

#------------------------------------------------------------
# Plot the results
f = plt.figure(figsize=(12, 12))

#################
# t-SNE plot
#################
gs1 = gridspec.GridSpec(1, 1)
gs1.update(left=0.64, bottom=0.64, right=0.98, top=0.98)
ax = plt.Subplot(f, gs1[0, 0])
f.add_subplot(ax)
for kk in np.arange(len(subsets)):
    mask = (np.char.rstrip(data["tSNE_class"],' ') == subsets[kk])
    ax.scatter(data["X_tsne"][mask], data["Y_tsne"][mask],
               s=size[kk], lw=lw[kk], edgecolors="k",
               c=col[kk], alpha=al[kk],
               marker=sym[kk])
    # Annotate population names
    ax.text(Xcoords[kk], Ycoords[kk], names[kk], fontsize=fsize[kk])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

#################
# UVW plots
#################
uvw = [data['Ulsr'], data['Vlsr'], data['Wlsr']]
lab = [r"$U\ \rm{[km/s]}$", r"$V\ \rm{[km/s]}$", r"$W\ \rm{[km/s]}$"]
gs2 = gridspec.GridSpec(2, 2)
gs2.update(left=0.63, bottom=0.25, right=0.98, top=0.62,
           wspace=0.05, hspace=0.05)
for ii in range(2):
    for jj in range(2):
        if ii <= jj:
            ax = plt.Subplot(f, gs2[ii, jj])
            f.add_subplot(ax)
            for kk in np.arange(len(subsets)):
                mask = (np.char.rstrip(data["tSNE_class"],' ') == subsets[kk])
                ax.scatter(uvw[1-jj][mask], uvw[2-ii][mask],
                           s=size[kk], lw=lw[kk], edgecolors="k",
                           c=col[kk], alpha=al[kk],
                           marker=sym[kk])
            if ii==jj:
                ax.set_xlabel(lab[1-ii], fontsize=13)
                ax.set_ylabel(lab[2-ii], fontsize=13)
            else:
                ax.xaxis.set_major_formatter(plt.NullFormatter())
                ax.yaxis.set_major_formatter(plt.NullFormatter())
        else:
            pass

#################
# eRz plots
#################
uvw = [data['e'], 0.5*(data['Rmin']+data['Rmax']), data['zmax']]
lab = [r"$e$", r"$R_{\rm mean}\ \rm{[kpc]}$",
       r"$|Z_{\rm max}|\ \rm{[kpc]}$"]
gs2 = gridspec.GridSpec(2, 2)
gs2.update(left=0.28, bottom=0.63, right=0.62, top=0.98,
           wspace=0.05, hspace=0.05)
for ii in range(2):
    for jj in range(2):
        if ii <= jj:
            ax = plt.Subplot(f, gs2[ii, jj])
            f.add_subplot(ax)
            for kk in np.arange(len(subsets)):
                mask = (np.char.rstrip(data["tSNE_class"],' ') == subsets[kk])
                ax.scatter(uvw[1-jj][mask], uvw[2-ii][mask],
                           s=size[kk], lw=lw[kk], edgecolors="k",
                           c=col[kk], alpha=al[kk],
                           marker=sym[kk])
            if ii==0:
                ax.set_yscale("log", nonposy='clip')
            if jj==1:
                ax.set_xscale("log", nonposx='clip')
            if ii==jj:
                ax.set_xlabel(lab[1-ii], fontsize=13)
                ax.set_ylabel(lab[2-ii], fontsize=13)
                if ii==0:
                    ax.set_xlim([4,13])
                else:
                    ax.set_ylim([4,13])
            else:
                ax.xaxis.set_major_formatter(plt.NullFormatter())
                ax.yaxis.set_major_formatter(plt.NullFormatter())

        else:
            pass


#################
# Triangle plot
#################
n_dim = 5
gs0 = gridspec.GridSpec(n_dim, n_dim)
gs0.update(left=0.06, bottom=0.05, right=0.89, top=0.89,
           wspace=0.05, hspace=0.05)


for ii in range(n_dim):
    for jj in range(n_dim):
        if ii >= jj:
            ax = plt.Subplot(f, gs0[ii, jj])
            f.add_subplot(ax)
            for kk in np.arange(len(subsets)):
                mask = (np.char.rstrip(data["tSNE_class"],' ') == subsets[kk])
                ax.scatter(xx[jj][mask], xx[ii+1][mask],
                           s=size[kk], lw=lw[kk], edgecolors="k",
                           c=col[kk], alpha=al[kk],
                           marker=sym[kk])
            #ax.set_ylim(1.01 * np.min(xx[ii]), 1.01 * np.max(xx[ii]))
            #ax.set_xlim(1.01 * np.min(xx[jj]), 1.01 * np.max(xx[jj]))
            """elif ii==jj:
            # Make data histogram!
            ax.set_ylim(0, 0.16)
            ax.set_xlim(1.01 * np.min(X[:,jj]), 1.01 * np.max(X[:,jj]))
            mask = (np.sqrt(exlist[ii]) < np.max(abs(xlist[ii])) )
            weights = np.ones_like(X[:, ii][mask])/len(X[:, ii][mask])
            ax.hist(X[:, ii][mask], weights=weights, bins=50,
                    histtype='stepfilled', alpha=0.4)
            xx = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)"""
            # Beautify axes
            if ii == n_dim - 1:
                ax.set_xlabel(labels[jj], fontsize=13)
            else:
                ax.xaxis.set_major_formatter(plt.NullFormatter())
            if jj == 0:
                ax.set_ylabel(labels[ii+1], fontsize=13)
            else:
                ax.yaxis.set_major_formatter(plt.NullFormatter())
        else:
            pass

plt.savefig("../bensby2014_tsne-summary-plot.png", dpi=200)  
