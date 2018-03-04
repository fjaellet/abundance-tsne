# Author: F. Anders
# License: BSD

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from astroML.decorators import pickle_results
from astroML.density_estimation import XDGMM
from astroML.plotting.tools import draw_ellipse

import open_data
from astropy.io import fits as pyfits
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=18, usetex=True)

    
sets = "teffcut"                # either 'mc', 'errlim' or 'plain'
what = {"thick":[0,1,2,3,4], "thin":[0,5,7,9],
        "strange": [0,10,11,12,13,14,16,17,18,19]}
note = {"thick":r"High-$[\alpha$/Fe] populations",
        "thin":r"Low-$[\alpha$/Fe] populations",
        "strange":r"Peculiar stars"}

inds = "strange"
  
t     = open_data.harps()
data  = t.data
t.get_tsne_subsets( sets = sets )


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


#------------------------------------------------------------
# Plot the results
f = plt.figure(figsize=(12, 8))

ax = f.add_subplot(111)

for kk in what[inds]:
    mask = np.where(t.classcol == t.subsets[kk])[0]
    if kk < 10:
        violins = ax.violinplot([xx[jj,mask] for jj in np.arange(len(Z))],
                                np.log10(Z),
                                t.sym[kk]+'-', widths=0.035, showextrema=False)
        for pc in violins['bodies']:
            pc.set_facecolor(t.col[kk])
            pc.set_alpha(.2)
        plt.plot(np.log10(Z), np.nanmedian(xx[:,mask], axis=1), t.sym[kk]+'-',
                 c=t.col[kk], alpha=1, ms=12, lw=1, mec="k")
    else:
        plt.plot(np.log10(Z), xx[:,mask], t.sym[kk]+'-', c=t.col[kk],
                 alpha=1, ms=12, lw=1, mec="k")
        

ax.axis([np.log10(Z[0])-.05, np.log10(Z[-1])+0.05, -.9, .9])
ax.set_ylabel(r"[X/Fe] abundance",fontsize=25)
ax.text(np.log10(13),.6, note[inds], horizontalalignment='left',fontsize=25)

ax.set_xticks(np.log10(Z))
ax.set_xticklabels([]) #str(zz) for zz in Z
if inds=="strange":
    ax.text(np.log10(26), -1.07, r"Proton number $Z$", 
        horizontalalignment='center',fontsize=25)
    for ii in np.arange(len(Znames2)):
        if Znames2[ii]!="" and ii > 6:
            ax.text(np.log10(Z[ii]), -.87, str(Z[ii]), horizontalalignment='center')
        else:
            ax.text(np.log10(Z[ii]), -.97, str(Z[ii]), horizontalalignment='center')
        
ax1=ax.twiny()
ax1.axis([np.log10(Z[0])-.05, np.log10(Z[-1])+0.05, -.9, .9])
ax1.set_xticks(np.log10(Z))
if inds=="thin":
    for ii in np.arange(len(Znames2)):
        ax.text(np.log10(Z[ii]), .83, Znames2[ii], horizontalalignment='center')
    ax1.set_xlabel(r"Element",fontsize=25)
    ax1.set_xticklabels(Znames)
else:
    ax1.set_xticklabels([])
ax1.get_xaxis().set_tick_params(which='minor', size=0)
ax1.get_xaxis().set_tick_params(which='minor', width=0)
    
plt.savefig("../im/harps_tsne_abundances-relto-Fe_" + inds + ".png", dpi=200)  
