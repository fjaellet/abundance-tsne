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
import matplotlib

from astroML.decorators import pickle_results
from astroML.density_estimation import XDGMM
from astroML.plotting.tools import draw_ellipse

import sample
from astropy.io import fits as pyfits
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=15, usetex=True)

    
hdu = pyfits.open(
    '/home/friedel/Astro/Spectro/Bensby/Bensby2014_with_tSNE.fits',
    names=True)
data=hdu[1].data

Z=[ 8, 11, 12, 13, 14, 20, 22, 24, 26, 28, 30, 39, 56]
Znames = ["[O/H]", "Na", "Mg", "Al", "Si",
          "Ca", "Ti", "Cr", "Fe", "Ni", "Zn", "Y", "Ba"]

xx  = np.concatenate([data['Fe_H']+data['O_Fe'],  data['Na_Fe']-data['O_Fe'],
       data['Mg_Fe']-data['O_Fe'], data['Al_Fe']-data['O_Fe'],
       data['Si_Fe']-data['O_Fe'], data['Ca_Fe']-data['O_Fe'],
       data['Ti_Fe']-data['O_Fe'], data['Cr_Fe']-data['O_Fe'],
       -data['O_Fe'], data['Ni_Fe']-data['O_Fe'],
       data['Zn_Fe']-data['O_Fe'], data['Y_Fe']-data['O_Fe'],
       data['Ba_Fe']-data['O_Fe']        ])
o   = data['Fe_H']+data['O_Fe']
xx=xx.reshape( (len(Z), len(o)) )

subsets = ["thin", "thick1", "thick2", "metalpoorthin",
           "highY/Zn", "smr",
           "supersolarlowY/Zn",
           "lowalphahighTihighY/Ba", "debris", "supersolarhighalpha",
           "transition"]
names   = ["Thin disc", "Thick Disc I", "Thick Disc II", "Metal-poor \n thin disc",
           "Low-[Zn/Y]", "SMR", "Super-solar \n high-[Zn/Y]",
           r"High-[Ti/$\alpha$]", "Satellite \n debris",
           "Super-solar \n high-[Y/Ba]", "Thin/Thick-II \n transition"]

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
f = plt.figure(figsize=(12, 8))
#import matplotlib.gridspec as gridspec
#gs1 = gridspec.GridSpec(len(subsets), 1)
#gs1.update(left=0.08, bottom=0.08, right=0.98, top=0.98)

ax = f.add_subplot(111) #plt.Subplot(f, gs1[kk, 0])
#for ii in np.arange(len(data)):
#    plt.plot(Z, xx[:,ii].ravel(), '-', c='grey',
#    alpha=0.05, ms=0)

#f.add_subplot(ax)
for kk in np.arange(len(subsets)):
    mask = np.where(np.char.rstrip(data["tSNE_class"],' ') == subsets[kk])[0]
    violins = ax.violinplot([xx[jj,mask] for jj in np.arange(len(Z))],
                            np.log10(Z),
                            sym[kk]+'-', widths=0.035, showextrema=False)
    for pc in violins['bodies']:
        pc.set_facecolor(col[kk])
        pc.set_alpha(.2)
    plt.plot(np.log10(Z), np.median(xx[:,mask], axis=1), sym[kk]+'-', c=col[kk],
             alpha=1, ms=12, lw=1, mec="k")

ax.axis([np.log10(Z[0])-.05, np.log10(Z[-1])+0.05, -.9, .6])
ax.set_ylabel(r"[X/O] abundance",fontsize=20)

# Annotate population names
#ax.text(Xcoords[kk], Ycoords[kk], names[kk], fontsize=fsize[kk])
ax.set_xlabel(r"Proton number $Z$",fontsize=20)
ax.set_xticks(np.log10(Z))
ax.set_xticklabels([str(zz) for zz in Z])

ax1=ax.twiny()
ax1.set_xlabel(r"Element",fontsize=20)
ax1.axis([np.log10(Z[0])-.05, np.log10(Z[-1])+0.05, -.9, .6])
ax1.set_xticks(np.log10(Z))
ax1.set_xticklabels(Znames)
ax1.get_xaxis().set_tick_params(which='minor', size=0)
ax1.get_xaxis().set_tick_params(which='minor', width=0)
#ax.text(7.2,0.7,"[O/H]",fontsize=12)
#ax.set_xticks(Z)
#ax.set_xticklabels(Znames)
    
plt.savefig("../bensby2014_tsne_abundances-relto-O.png", dpi=200)  
