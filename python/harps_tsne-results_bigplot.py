"""
Produces HARPS t-SNE plots
"""
# Author: F. Anders
# License: BSD

import numpy as np
from matplotlib import pyplot as plt

from astroML.decorators import pickle_results
from astroML.density_estimation import XDGMM
from astroML.plotting.tools import draw_ellipse

from astropy.io import fits as pyfits
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=10, usetex=True)

sets = "errlim" # either 'mc', 'errlim' or 'plain'
  
hdu = pyfits.open(
    '/home/friedel/Astro/Spectro/HARPS/DelgadoMena2017.fits',
    names=True)
data=hdu[1].data

labels = [r'$\rm \tau \ [Gyr]$', r'$\rm [Fe/H]$', r'$\rm [Ti/Fe]$',
          r'$\rm [Y/Mg]$', r'$\rm [Zr/Fe]$', r'$\rm [Al/Mg]$']

xx  = [data['meanage'], data['feh'], data['TiIFe'], data['YFe']-data['MgFe'],
          data['ZrIIFe'], data['AlFe']-data['MgFe']]
xerr = [data['agestd'], data['erfeh'], data['errTiI'],
        np.sqrt(data['errY']**2.+data['errMg']**2.), data['errZrII'],
        np.sqrt(data['errAl']**2.+data['errMg']**2.) ]

if sets=="mc":
    Xt = data["X_tsne_teffcut40_mc"]
    Yt = data["Y_tsne_teffcut40_mc"]
    classcol= np.char.rstrip(data["tSNE_class_mc"],' ')
    subsets = ["thin", "thick1", "thick2", "thick3", "thick4",
               "mpthin", "smr", "t4trans",
               "debris1", "debris2", "debris3", "debris4", "debris5?", 
               "t2trans1", "t2trans2", "highTi","thicklow"]
    names   = ["Thin Disc", "Thick Disc I", "Thick Disc II", "Thick Disc III",
               "Thick Disc IV", "Metal-poor \n Thin Disc", "SMR", "Transition",
               "", "", "Satellite \n debris", "", "", r"TII/III", "", 
               r"Extreme-Ti star", r"Lowest-[Fe/H] star"]
    Xcoords = [-25, 15, 4.5, -12,  18, -31, 22, 26,-22.5, -14, -2, -25]
    Ycoords = [5.5 ,-6,  -2, -4,   6,  0,   1.5, -.5, -7, -2, -6, 14]
    fsize   = [20 , 16,  12, 12,  15,  13, 11, 11, 11, 11, 11, 11]
    sym = ["o", "v", "^", ">", "<", "s", "*", "<", "D", "h", "d", "H", "v", "p", "8", "H", "p"]
    al  = [.6, .8, .8, .8, .8, .6, 1,1,1,1,1,1,1,1,1,1,1,1]
    lw  = [0,.5,.5,.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5]
    size= [9,12,12,12,12,15,18,18,18,18,18,18,18,18,18,18,18]
    col = ["k", "m", "hotpink", "crimson", "r", "g", "orange", "gold",
          "yellow", "yellow", "yellow", "yellow", "green", "royalblue", "royalblue",
           "lime", "m"]
elif sets=="errlim":
    Xt = data["X_tsne_teffcut40_errlim_mc"]
    Yt = data["Y_tsne_teffcut40_errlim_mc"]
    classcol= np.char.rstrip(data["tSNE_class_errlim_mc"],' ')
    subsets = ["thin", "thick1", "thick2", "thick3", "thick4",
               "mpthin", "youngthin", "smr", "t4trans",
               "debris1", "debris2", "debris3", "debris4", "debris5?", 
               "t2trans1", "t2trans2", "highTi","t2trans3", "smr2", "lowMg"]
    names   = ["Thin Disc", "Thick Disc I", "Thick Disc II", "Thick Disc III",
               "Thick Disc IV", "Metal-poor \n Thin Disc", "Young", "SMR", "Transition",
               "", "", "Satellite \n debris", "", "", r"TII/III", "", 
               r"Extreme-Ti star", r""]
    Xcoords = [-25, 15, 4.5, -12,  18, -31, 22, 26,-22.5, -14, -2, -25]
    Ycoords = [5.5 ,-6,  -2, -4,   6,  0,   1.5, -.5, -7, -2, -6, 14]
    fsize   = [20 , 16,  12, 12,  15,  13, 11, 11, 11, 11, 11, 11]
    sym = ["o", "v", "^", ">", "<", "s", "o", "*", "<", "D", "h", "d", "H", "v",
           "p", "8", "H", "p", "*", "s"]
    al  = [.6, .8, .8, .8, .8, .6, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    lw  = [0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5]
    size= [9,12,12,12,12,15,18,22,18,18,18,18,18,18,22,22,22,22,28,20]
    col = ["k", "m", "hotpink", "crimson", "r", "g", "lightgrey", "orange", "gold",
          "yellow", "yellow", "yellow", "yellow", "green", "royalblue", "royalblue",
           "lime", "green", "gold", "k"]
elif sets=="plain":
    Xt = data["X_tsne_teffcut40"]
    Yt = data["Y_tsne_teffcut40"]
    classcol= np.char.rstrip(data["tSNE_class"],' ')
    subsets = ["thin", "thick1", "thick2", "thick3",
               "mpthin", "smr",
               "t1trans", "debris", "highAlMg",
               "t3trans", "highTioutlier","lowalphaoutlier"]
    names   = ["Thin disc", "Thick Disc I", "Thick Disc II", "Thick Disc III",
               "Metal-poor \n thin disc", "SMR", "Transition I",
               "Satellite \n debris", r"High-[Al/Mg]", "Transition III",
               r"Extreme-Ti star", r"Low-[Mg/Fe] star"]
    Xcoords = [-25, 15, 4.5, -12,  18, -31, 22, 26,-22.5, -14, -2, -25]
    Ycoords = [5.5 ,-6,  -2, -4,   6,  0,   1.5, -.5, -7, -2, -6, 14]
    fsize   = [20 , 16,  12, 12,  15,  13, 11, 11, 11, 11, 11, 11]
    sym = ["o", "v", "^", ">", "s", "*", "<", "D", "p", "8", "H", "h"]
    al  = [.6, .6, .8, .8, .75, 1,1,1,1,1,1,1]
    lw  = [0,0,.5,.5, .5, .5, .5, .5, .5, .5, .5, .5]
    size= [6,9,9,9,12,20,18,18,18,18,22,25]
    col = ["k", "r", "orange", "gold", "g", "orange",
          "brown", "yellow", "royalblue", "hotpink", 
          "lime", "black"]
else:
    raise ValueError("No valid 'subsets' name set")

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
if sets != "plain":
    # plot MC realisations
    mcres =  np.genfromtxt("../tsne_results/harps_tsne_results_withteffcutmc5040_rand0.csv", delimiter=',',
                      dtype=[('Name', "|S14"), ('X', float), ('Y', float)])
    ax.scatter(mcres["X"], mcres["Y"], s=4, lw=0, c="grey", alpha=0.15)
for kk in np.arange(len(subsets)):
    mask = (classcol == subsets[kk])
    if subsets[kk] == "debris1" and sets=="errlim":
        ax.plot(Xt[mask], 9.7, marker=r"$\uparrow$", zorder=0, ms=20)
        ax.scatter(Xt[mask], 9.5, s=4*size[kk], lw=lw[kk], edgecolors="k",
                   c=col[kk], alpha=al[kk], marker=sym[kk])
    else:
        ax.scatter(Xt[mask], Yt[mask], s=4*size[kk], lw=lw[kk], edgecolors="k",
               c=col[kk], alpha=al[kk], marker=sym[kk])
    # Annotate population names
    #ax.text(Xcoords[kk], Ycoords[kk], names[kk], fontsize=1.25*fsize[kk])
ax.set_xlabel("t-SNE X dimension", fontsize=13)
ax.set_ylabel("t-SNE Y dimension", fontsize=13)
#ax.set_yscale("symlog")
ax.axis([-13, 25, -8, 10.2])
#ax.text(-10, 9, r"What happens when we impose $\sigma_{\rm [Fe/H]}, \sigma_{\rm [X/Fe]}> 0.03$", fontsize=15)
gs = gridspec.GridSpec(1, 3)
gs.update(left=0.1, bottom=0.77, right=0.98, top=0.98,
           wspace=0.44, hspace=0.05)
exinds = [ [1,2], [4,5], [2,3] ]

for jj in range(3):
    ax = plt.Subplot(g, gs[0, jj])
    g.add_subplot(ax)
    for kk in np.arange(len(subsets)):
        mask = (classcol == subsets[kk])
        ax.errorbar(xx[exinds[jj][0]][mask], xx[exinds[jj][1]][mask],
                    xerr=xerr[exinds[jj][0]][mask], yerr=xerr[exinds[jj][1]][mask],
                   ms=0, mec="k", capthick=0, elinewidth=1,
                   mfc=col[kk], alpha=al[kk]/3., ecolor=col[kk], lw=0,
                   marker=sym[kk], zorder=0)
        ax.scatter(xx[exinds[jj][0]][mask], xx[exinds[jj][1]][mask],
                   s=size[kk], lw=lw[kk], edgecolors="k",
                   c=col[kk], alpha=al[kk],
                   marker=sym[kk])
    ax.set_xlabel(labels[exinds[jj][0]], fontsize=12)
    ax.set_ylabel(labels[exinds[jj][1]], fontsize=12)
plt.savefig("../im/harps_tsne-plot_test-errlim.png", dpi=200)  

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
    mask = (classcol == subsets[kk])
    ax.scatter(Xt[mask], Yt[mask],
               s=size[kk], lw=lw[kk], edgecolors="k",
               c=col[kk], alpha=al[kk],
               marker=sym[kk])
    # Annotate population names
    #ax.text(Xcoords[kk], Ycoords[kk], names[kk], fontsize=.92*fsize[kk])
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
                mask = (classcol == subsets[kk])
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
uvw = [data['JR_st_m'], data['JPhi_st_m'], data['JZ_st_m']]
lab = [r"$J_R$", r"$J_{\Phi}$", r"$J_Z$"]
gs2 = gridspec.GridSpec(2, 2)
gs2.update(left=0.28, bottom=0.63, right=0.62, top=0.98,
           wspace=0.05, hspace=0.05)
for ii in range(2):
    for jj in range(2):
        if ii <= jj:
            ax = plt.Subplot(f, gs2[ii, jj])
            f.add_subplot(ax)
            for kk in np.arange(len(subsets)):
                mask = (classcol == subsets[kk])
                mask = mask * (data['JZ_st_m'] > 0)
                ax.scatter(uvw[1-jj][mask], uvw[2-ii][mask],
                           s=size[kk], lw=lw[kk], edgecolors="k",
                           c=col[kk], alpha=al[kk],
                           marker=sym[kk])
            if not ii==1:
                ax.set_yscale("log", nonposy='clip')
                ax.set_ylim([0.000001, .5])
            if not jj==0:
                ax.set_xscale("log", nonposx='clip')
                ax.set_xlim([0.000001, .5])
            if ii==jj:
                ax.set_xlabel(lab[1-ii], fontsize=13)
                ax.set_ylabel(lab[2-ii], fontsize=13)
                """if ii==0:
                    ax.set_xlim([4,13])
                else:
                    ax.set_ylim([4,13])"""
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
                mask = (classcol == subsets[kk])
                ax.scatter(xx[jj][mask], xx[ii+1][mask],
                           s=size[kk], lw=lw[kk], edgecolors="k",
                           c=col[kk], alpha=al[kk],
                           marker=sym[kk])
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

plt.savefig("../im/harps_tsne-summary-plot_" + sets + ".png", dpi=200)  
