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
    '/home/friedel/Astro/Spectro/HARPS/DelgadoMena2017.fits',
    names=True)
data=hdu[1].data

"""Z=[ 6, 8, 12, 13, 14, 20, 22, 26, 29, 30, 38, 39, 40, 56, 58, 60, 63]
Znames = ["C", "O", "Mg", "Al", "Si", "Ca", "Ti", "[Fe/H]",
          "Cu", "Zn", "Sr", "Y", "Zr", "Ba", "Ce", "Nd", "Eu"]
xx  = np.concatenate([data['[C/H]_SA17']-data['feh'],
                      data['[O/H]_6158_BdL15']-data['feh'],
                      data['MgFe'], data['AlFe'], data['SiFe'],
                      data['CaFe'], data['TiIFe'], data['feh'],
                      data['CuFe'], data['ZnFe'], data['SrFe'],
                      data['YFe'],  data['ZrIIFe'], data['BaFe'],
                      data['CeFe'], data['NdFe'], data['EuFe']       ])"""
Z=[ 12, 13, 14, 20, 22, 26, 29, 30, 38, 39, 40, 56, 58, 60, 63]

Znames = ["Mg", "Al", "Si", "Ca", "Ti", "",
          "Cu", "", "Sr", "", "Zr", "Ba", "", "Nd", ""]
Znames2= ["", "", "", "", "", "[Fe/H]",
          "", "Zn", "", "Y", "", "", "Ce", "", "Eu"]
# Polishing:
data['NdFe'][ data['errNd'] > .5 ] = np.nan
data['EuFe'][ data['errEu'] > .8 ] = np.nan
xx  = np.concatenate([data['MgFe'], data['AlFe'], data['SiFe'],
                      data['CaFe'], data['TiIFe'], data['feh'],
                      data['CuFe'], data['ZnFe'], data['SrFe'],
                      data['YFe'],  data['ZrIIFe'], data['BaFe'],
                      data['CeFe'], data['NdFe'], data['EuFe']       ])
fe   = data['feh']
xx=xx.reshape( (len(Z), len(fe)) )

subsets = ["thin", "thick1", "thick2", "thick3",
           "mpthin", "smr",
           "t1trans", "debris", "highAlMg",
           "t3trans", "highTioutlier","lowalphaoutlier"]

fsize   = [20 , 16,  16, 16,  13,  13, 11, 11, 11, 11, 11]
sym = ["o", "v", "^", ">", "s", "*", "<", "D", "p", "8", "P", "X"]
al  = [.4, .6, .8, .8, .75, 1,1,1,1,1,1,1]
lw  = [0,0,.5,.5, .5, .5, .5, .5, .5, .5, .5, .5]
size= [6,9,9,9,12,20,18,18,18,18,22,25]
col = ["k", "r", "orange", "gold", "g", "orange",
      "brown", "yellow", "royalblue", "hotpink", 
      "lime", "black"]
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
    if kk < 10:
        violins = ax.violinplot([xx[jj,mask] for jj in np.arange(len(Z))],
                                np.log10(Z),
                                sym[kk]+'-', widths=0.035, showextrema=False)
        for pc in violins['bodies']:
            pc.set_facecolor(col[kk])
            pc.set_alpha(.2)
        plt.plot(np.log10(Z), np.nanmedian(xx[:,mask], axis=1), sym[kk]+'-', c=col[kk],
                 alpha=1, ms=12, lw=1, mec="k")
    else:
        plt.plot(np.log10(Z), xx[:,mask], sym[kk]+'-', c=col[kk],
                 alpha=1, ms=12, lw=1, mec="k")
        

ax.axis([np.log10(Z[0])-.05, np.log10(Z[-1])+0.05, -.9, .9])
ax.set_ylabel(r"[X/Fe] abundance",fontsize=20)

# Annotate population names
#ax.text(Xcoords[kk], Ycoords[kk], names[kk], fontsize=fsize[kk])
ax.text(np.log10(26), -1.07, r"Proton number $Z$", 
        horizontalalignment='center',fontsize=20)
ax.set_xticks(np.log10(Z))
ax.set_xticklabels([]) #str(zz) for zz in Z
for ii in np.arange(len(Znames2)):
    ax.text(np.log10(Z[ii]), .83, Znames2[ii], horizontalalignment='center')
for ii in np.arange(len(Znames2)):
    if Znames2[ii]!="" and ii > 6:
        ax.text(np.log10(Z[ii]), -.87, str(Z[ii]), horizontalalignment='center')
    else:
        ax.text(np.log10(Z[ii]), -.97, str(Z[ii]), horizontalalignment='center')
        

ax1=ax.twiny()
ax1.set_xlabel(r"Element",fontsize=20)
ax1.axis([np.log10(Z[0])-.05, np.log10(Z[-1])+0.05, -.9, .9])
ax1.set_xticks(np.log10(Z))
ax1.set_xticklabels(Znames)
ax1.get_xaxis().set_tick_params(which='minor', size=0)
ax1.get_xaxis().set_tick_params(which='minor', width=0)
#ax.text(7.2,0.7,"[O/H]",fontsize=12)
#ax.set_xticks(Z)
#ax.set_xticklabels(Znames)
    
plt.savefig("../harps_tsne_abundances-relto-Fe.png", dpi=200)  
