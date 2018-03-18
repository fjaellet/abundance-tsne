print(__doc__)

from time import time

import matplotlib.pyplot as plt
from matplotlib import colors as mpcols
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter

import numpy as np

from astroML.decorators import pickle_results
from astroML.density_estimation import XDGMM
from astroML.plotting.tools import draw_ellipse
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=12, usetex=True)

from astropy.io import fits as pyfits
from sklearn import manifold, datasets
import scipy
import open_data

# READ DelGADO-MENA DATA
harps = open_data.harps(teffcut=True, ages=True, abunds=True)
data  = harps.data
colors   = [data['feh'], data['TiIFe'], data['CaFe'], data['MgFe'],
            data['MgFe']-data['TiIFe'],data['CuFe'],data['AlFe']-data['MgFe'],
            data['BaFe'], data['ZnFe'], 
            data['YFe']-data['BaFe'],data['YFe']-data['ZnFe'],data['CeFe'],
            data['Teff'],data['logg'], data['vt_1'],np.log10(data['S/N']),
            data['age50'],data['vXg'],data['vYg'],data['vZg']]
titles   = [r'$\rm [Fe/H]$', r'$\rm [Ti/Fe]$', r'$\rm [Ca/Fe]$', r'$\rm [Mg/Fe]$',
          r'$\rm [Mg/Ti]$', r'$\rm [Cu/Fe]$', r'$\rm [Al/Mg]$', r'$\rm [Ba/Fe]$',
          r'$\rm [Zn/Fe]$', r'$\rm [Y/Ba]$', r'$\rm [Y/Zn]$', r'$\rm [Ce/Fe]$',
          r'$T_{\rm eff}$ [K]', r'$\log g$', r'$\xi$ [km/s]', r'lg $S/N$',
          r'$\tau$ [Gyr]', r'$v_X$ [km/s]', r'$v_Y$ [km/s]', r'$v_Z$ [km/s]']


f = plt.figure(figsize=(12, 12))
plt.suptitle("t-SNE manifold learning for the HARPS sample (Delgado-Mena et al. 2017)", fontsize=14)
import matplotlib.gridspec as gridspec
gs0 = gridspec.GridSpec(5, 4)
gs0.update(left=0.08, bottom=0.05, right=0.92, top=0.92,
           wspace=0.05, hspace=0.14)

for ii in range(5):
    for jj in range(4):
        print jj
        ax = plt.Subplot(f, gs0[ii, jj])
        f.add_subplot(ax)
        scat = plt.scatter(data['X_tsne_teffcut40'], data['Y_tsne_teffcut40'], 
                           c=colors[4*ii+jj], norm=None, lw=0., s=30,
                           cmap=plt.cm.viridis, alpha=0.6)
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
        #plt.title(titles[4*ii+jj])
        ax.text(.98, .7, titles[4*ii+jj], horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes, fontsize=15)
        f.colorbar(scat, cax=cax, orientation="horizontal")
plt.axis('tight')

plt.savefig("../im/HARPS_tsne_plots_40_nice.png", dpi=200)
        
