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

import open_data

sets = "teffcut"                # either 'mc', 'errlim' or 'plain'
  
t     = open_data.harps()
data  = t.data
t.get_tsne_subsets( sets = sets )

labels = [r'$\rm \tau \ [Gyr]$', r'$\rm [Fe/H]$', r'$\rm [Ti/Fe]$', r'$\rm [Y/Mg]$', 
          r'$\rm [Zr/Fe]$', r'$\rm [Al/Mg]$', r'$\rm [Mg/Fe]$', r'$U$ [km/s]',
          r'$V$ [km/s]', r'$W$ [km/s]', r'$T_{\rm eff}$ [K]', r'$\log g_{\rm HIP}$',
          r'$R_{\rm Gal}$ [kpc]', r'$Z_{\rm Gal}$ [kpc]', r'$X_{\rm Gal}$ [kpc]', r'$Y_{\rm Gal}$ [kpc]',
          r'$\rm [Ba/Fe]$', r'$\rm [Y/Al]$', r'$\rm [Nd/Fe]$', r'$\rm [Sr/Ba]$',
          r'$\rm [Ce/Fe]$', r'$\rm [Zr/Ba]$', r'$\rm [Zn/Fe]$', r'$\rm [Eu/Fe]$',
          r'$\rm [Cu/Fe]$', r'$\rm [O/H]$', r'$\rm [C/Fe]$', r'$\rm [C/O]$',
          r'$\rm [Si/Fe]$'
          ]

xx  = [data['meanage'], data['feh'],             # 0
       data['TiIFe'], data['YFe']-data['MgFe'],
       data['ZrIIFe'], data['AlFe']-data['MgFe'],
       data['MgFe'], data['Ulsr'],
       data['Vlsr'], data['Wlsr'], 
       data['Teff'], data['logg_hip'],           # 10
       data['Rg'], data['Zg'], 
       data['Xg'], data['Yg'], 
       data['BaFe'], data['YFe']-data['AlFe'],
       data['NdFe'], data['SrFe']-data['BaFe'],
       data['CeFe'], data['ZrIIFe']-data['BaFe'],  # 20
       data['ZnFe'], data['EuFe'],
       data['CuFe'], data['[O/H]_6158_BdL15'],
       data['[C/H]_SA17']-data['feh'],data['[C/H]_SA17']-data['[O/H]_6158_BdL15'],
       data['SiFe'] ]

xerr = [data['agestd'], data['erfeh'], data['errTiI'],
        np.sqrt(data['errY']**2.+data['errMg']**2.), data['errZrII'],
        np.sqrt(data['errAl']**2.+data['errMg']**2.),
        data['errMg'], 9.8 * np.ones(len(data)),
        9.8 * np.ones(len(data)), 9.8 * np.ones(len(data)), 
        data['erTeff'], data['erlogg_hip'],           # 10
        data['Rg_sig'], data['Zg_sig'], 
        data['Xg_sig'], data['Yg_sig'], 
        data['errBa'], np.sqrt(data['errY']**2.+data['errAl']**2.),
        data['errNd'], np.sqrt(data['errSr']**2.+data['errBa']**2.),
        data['errCe'], np.sqrt(data['errZrII']**2.+data['errBa']**2.),  # 20
        data['errZn'], data['errEu'],
        data['errCu'], data['e_[O/H]_6158_BdL15'],
        np.sqrt(data['e_[C/H]_SA17']**2.+data['erfeh']**2.), np.sqrt(data['e_[C/H]_SA17']**2.+data['e_[O/H]_6158_BdL15']**2.),
        data['errSi']
        ]

# age plots

exinds = [ [0,1], [0,6], [0,2], [0,22],
           [0,3], [0,17],[0,22],[0,16] ]
limits = [ None,  None,  None,  None, 
           None,  None,  None,  None]

#------------------------------------------------------------
# Plot the results
import matplotlib.gridspec as gridspec

g   = plt.figure(figsize=(12, 7))

gs = gridspec.GridSpec(2, 4)
gs.update(left=0.06, bottom=0.07, right=0.98, top=0.98,
           wspace=0.3, hspace=0.2)

for jj in range(8):
    print jj
    ax = plt.Subplot(g, gs[jj/4, jj%4])
    if exinds[jj] != None:
        g.add_subplot(ax)
        for kk in np.arange(len(t.subsets)):
            mask = (t.classcol == t.subsets[kk]) * (xerr[exinds[jj][1]] < 9.9)
            ax.errorbar(xx[exinds[jj][0]][mask], xx[exinds[jj][1]][mask],
                        xerr=xerr[exinds[jj][0]][mask], yerr=xerr[exinds[jj][1]][mask],
                       ms=0, mec="k", capthick=0, elinewidth=1,
                       mfc=t.col[kk], alpha=t.al[kk]/4., ecolor=t.col[kk], lw=0,
                       marker=t.sym[kk], zorder=0)
            ax.scatter(xx[exinds[jj][0]][mask], xx[exinds[jj][1]][mask],
                       s=t.size[kk], lw=t.lw[kk], edgecolors="k",
                       c=t.col[kk], alpha=t.al[kk],
                       marker=t.sym[kk])
        ax.set_xlabel(labels[exinds[jj][0]], fontsize=12)
        ax.set_ylabel(labels[exinds[jj][1]], fontsize=12)
        if limits[jj] != None:
            ax.axis(limits[jj])
        ax.locator_params(tight=True, nbins=4)

plt.savefig("../im/harps_tsne-age-abundsplot_"+sets+".png", dpi=200)  


# t-SNE + abundances plot

exinds = [ [1,6], [1,28], [1,2], [1,5],
           [1,22], None,  None,  None, 
           [1,24], None,  None,  None, 
           [1,3], None,  None,  None, 
           [1,16],[1,20],[1,21],[1,19] ]
limits = [ None,  None,  None,  None, 
           None,  None,  None,  None, 
           None,  None,  None,  None, 
           None,  None,  None,  None, 
           None,  None,  None,  None ]

#------------------------------------------------------------
# Plot the results
import matplotlib.gridspec as gridspec

g   = plt.figure(figsize=(12, 14))

#################
# abundance plots around
#################
gs = gridspec.GridSpec(5, 4)
gs.update(left=0.06, bottom=0.06, right=0.98, top=0.98,
           wspace=0.4, hspace=0.33)

for jj in range(20):
    print jj
    ax = plt.Subplot(g, gs[jj/4, jj%4])
    if exinds[jj] != None:
        g.add_subplot(ax)
        for kk in np.arange(len(t.subsets)):
            mask = (t.classcol == t.subsets[kk]) * (xerr[exinds[jj][1]] < 9.9)
            ax.errorbar(xx[exinds[jj][0]][mask], xx[exinds[jj][1]][mask],
                        xerr=xerr[exinds[jj][0]][mask], yerr=xerr[exinds[jj][1]][mask],
                       ms=0, mec="k", capthick=0, elinewidth=1,
                       mfc=t.col[kk], alpha=t.al[kk]/4., ecolor=t.col[kk], lw=0,
                       marker=t.sym[kk], zorder=0)
            ax.scatter(xx[exinds[jj][0]][mask], xx[exinds[jj][1]][mask],
                       s=t.size[kk], lw=t.lw[kk], edgecolors="k",
                       c=t.col[kk], alpha=t.al[kk],
                       marker=t.sym[kk])
        ax.set_xlabel(labels[exinds[jj][0]], fontsize=12)
        ax.set_ylabel(labels[exinds[jj][1]], fontsize=12)
        if limits[jj] != None:
            ax.axis(limits[jj])
        ax.locator_params(tight=True, nbins=4)

#################
# t-SNE plot in the center
#################
gs0 = gridspec.GridSpec(1, 1)
gs0.update(left=0.315, bottom=0.275, right=0.98, top=0.78)
ax  = plt.Subplot(g, gs0[0, 0])
g.add_subplot(ax)
if sets not in ["teffcut", "plain"]:
    # plot MC realisations
    mcres =  np.genfromtxt("../tsne_results/harps_tsne_results_withteffcutmc5040_rand0.csv", delimiter=',',
                      dtype=[('Name', "|S14"), ('X', float), ('Y', float)])
    ax.scatter(mcres["X"], mcres["Y"], s=4, lw=0, c="grey", alpha=0.15)
for kk in np.arange(len(t.subsets)):
    mask = (t.classcol == t.subsets[kk])
    if t.subsets[kk] == "debris1" and sets=="errlim":
        ax.plot(t.Xt[mask], 9.7, marker=r"$\uparrow$", zorder=0, ms=20)
        ax.scatter(t.Xt[mask], 9.5, s=6*t.size[kk], lw=t.lw[kk], edgecolors="k",
                   c=t.col[kk], alpha=t.al[kk], marker=t.sym[kk])
    else:
        ax.scatter(t.Xt[mask], t.Yt[mask], s=6*t.size[kk], lw=t.lw[kk], edgecolors="k",
               c=t.col[kk], alpha=t.al[kk], marker=t.sym[kk])
    # Annotate population names
    #if kk < len(t.names):
    #    ax.text(t.Xcoords[kk], t.Ycoords[kk], t.names[kk], fontsize=1.25*t.fsize[kk])
ax.set_xlabel("t-SNE X dimension", fontsize=18)
ax.set_ylabel("t-SNE Y dimension", fontsize=18)
#ax.axis([-13, 25, -8, 10.2])

plt.savefig("../im/harps_tsne-abundsplot_"+sets+".png", dpi=200)  


"""
labels = [r'$\rm \tau \ [Gyr]$', r'$\rm [Fe/H]$', r'$\rm [Ti/Fe]$', r'$\rm [Y/Mg]$', 
          r'$\rm [Zr/Fe]$', r'$\rm [Al/Mg]$', r'$\rm [Mg/Fe]$', r'$U$ [km/s]',
          r'$V$ [km/s]', r'$W$ [km/s]', r'$T_{\rm eff}$ [K]', r'$\log g_{\rm HIP}$',
          r'$R_{\rm Gal}$ [kpc]', r'$Z_{\rm Gal}$ [kpc]', r'$X_{\rm Gal}$ [kpc]', r'$Y_{\rm Gal}$ [kpc]',
          r'$\rm [Ba/Fe]$', r'$\rm [Y/Al]$', r'$\rm [Nd/Fe]$', r'$\rm [Sr/Ba]$',
          r'$\rm [Ce/Fe]$', r'$\rm [Zr/Ba]$', r'$\rm [Zn/Fe]$', r'$\rm [Eu/Fe]$',
          r'$\rm [Cu/Fe]$', r'$\rm [O/H]$', r'$\rm [C/Fe]$', r'$\rm [C/O]$'
          ]

xx  = [data['meanage'], data['feh'],             # 0
       data['TiIFe'], data['YFe']-data['MgFe'],
       data['ZrIIFe'], data['AlFe']-data['MgFe'],
       data['MgFe'], data['Ulsr'],
       data['Vlsr'], data['Wlsr'], 
       data['Teff'], data['logg_hip'],           # 10
       data['Rg'], data['Zg'], 
       data['Xg'], data['Yg'], 
       data['BaFe'], data['YFe']-data['AlFe'],
       data['NdFe'], data['SrFe']-data['BaFe'],
       data['CeFe'], data['ZrIIFe']-data['BaFe'],  # 20
       data['ZnFe'], data['EuFe'],
       data['CuFe'], data['[O/H]_6158_BdL15'],
       data['[C/H]_SA17']-data['feh'], data['[C/H]_SA17']-data['[O/H]_6158_BdL15'] ]

xerr = [data['agestd'], data['erfeh'], data['errTiI'],
        np.sqrt(data['errY']**2.+data['errMg']**2.), data['errZrII'],
        np.sqrt(data['errAl']**2.+data['errMg']**2.),
        data['errMg'], 9.8 * np.ones(len(data)),
        9.8 * np.ones(len(data)), 9.8 * np.ones(len(data)), 
        data['erTeff'], data['erlogg_hip'],           # 10
        data['Rg_sig'], data['Zg_sig'], 
        data['Xg_sig'], data['Yg_sig'], 
        data['errBa'], np.sqrt(data['errY']**2.+data['errAl']**2.),
        data['errNd'], np.sqrt(data['errSr']**2.+data['errBa']**2.),
        data['errCe'], np.sqrt(data['errZrII']**2.+data['errBa']**2.),  # 20
        data['errZn'], data['errEu'],
        data['errCu'], data['e_[O/H]_6158_BdL15'],
        np.sqrt(data['e_[C/H]_SA17']**2.+data['erfeh']**2.), np.sqrt(data['e_[C/H]_SA17']**2.+data['e_[O/H]_6158_BdL15']**2.) 
        ]

exinds = [ [0,2], [1,2], [1,6], [8,9], [7,9],
           [0,1], None,  None,  None,  [7,8],
           [0,3], None,  None,  None, [14,15],
           [0,5], None,  None,  None, [12,13],
           [0,16],[1,24],[1,22],[1,23],[10,11],
           [0,27],[25,27],[1,18],[1,20],[1,19] ]
limits = [ None,  None,  None,  None,  None,
           None,  None,  None,  None,  None,
           None,  None,  None,  None,  [8.15, 8.4, -.1, .05],
           None,  None,  None,  None,  [8.15, 8.4, -.06, .15],
           None,  None,  None,  None,  [6030,5270, 5, 3],
           None,  None,  None,  None,  None ]
controlpanels = [3,4,9,14,19,24,25,26]
controlpanels2= [0,5,10,15,20,23,27]

#------------------------------------------------------------
# Plot the results
import matplotlib.gridspec as gridspec

g   = plt.figure(figsize=(12, 14))

#################
# abundance plots around
#################
gs = gridspec.GridSpec(6, 5)
gs.update(left=0.05, bottom=0.06, right=0.98, top=0.98,
           wspace=0.4, hspace=0.33)

for jj in range(30):
    print jj
    ax = plt.Subplot(g, gs[jj/5, jj%5])
    if exinds[jj] != None:
        g.add_subplot(ax)
        if jj in controlpanels:
            ax.set_axis_bgcolor('aliceblue')
        elif jj in controlpanels2:
            ax.set_axis_bgcolor('cornsilk')
        for kk in np.arange(len(t.subsets)):
            mask = (t.classcol == t.subsets[kk]) * (xerr[exinds[jj][1]] < 9.9)
            ax.errorbar(xx[exinds[jj][0]][mask], xx[exinds[jj][1]][mask],
                        xerr=xerr[exinds[jj][0]][mask], yerr=xerr[exinds[jj][1]][mask],
                       ms=0, mec="k", capthick=0, elinewidth=1,
                       mfc=t.col[kk], alpha=t.al[kk]/4., ecolor=t.col[kk], lw=0,
                       marker=t.sym[kk], zorder=0)
            ax.scatter(xx[exinds[jj][0]][mask], xx[exinds[jj][1]][mask],
                       s=t.size[kk], lw=t.lw[kk], edgecolors="k",
                       c=t.col[kk], alpha=t.al[kk],
                       marker=t.sym[kk])
        ax.set_xlabel(labels[exinds[jj][0]], fontsize=12)
        ax.set_ylabel(labels[exinds[jj][1]], fontsize=12)
        if limits[jj] != None:
            ax.axis(limits[jj])
        ax.locator_params(tight=True, nbins=4)

#################
# t-SNE plot in the center
#################
gs0 = gridspec.GridSpec(1, 1)
gs0.update(left=0.25, bottom=0.4, right=0.78, top=0.81)
ax  = plt.Subplot(g, gs0[0, 0])
g.add_subplot(ax)
if sets not in ["teffcut", "plain"]:
    # plot MC realisations
    mcres =  np.genfromtxt("../tsne_results/harps_tsne_results_withteffcutmc5040_rand0.csv", delimiter=',',
                      dtype=[('Name', "|S14"), ('X', float), ('Y', float)])
    ax.scatter(mcres["X"], mcres["Y"], s=4, lw=0, c="grey", alpha=0.15)
for kk in np.arange(len(t.subsets)):
    mask = (t.classcol == t.subsets[kk])
    if t.subsets[kk] == "debris1" and sets=="errlim":
        ax.plot(t.Xt[mask], 9.7, marker=r"$\uparrow$", zorder=0, ms=20)
        ax.scatter(t.Xt[mask], 9.5, s=4*t.size[kk], lw=t.lw[kk], edgecolors="k",
                   c=t.col[kk], alpha=t.al[kk], marker=t.sym[kk])
    else:
        ax.scatter(t.Xt[mask], t.Yt[mask], s=4*t.size[kk], lw=t.lw[kk], edgecolors="k",
               c=t.col[kk], alpha=t.al[kk], marker=t.sym[kk])
    # Annotate population names
    #if kk < len(t.names):
    #    ax.text(t.Xcoords[kk], t.Ycoords[kk], t.names[kk], fontsize=1.25*t.fsize[kk])
ax.set_xlabel("t-SNE X dimension", fontsize=13)
ax.set_ylabel("t-SNE Y dimension", fontsize=13)
#ax.axis([-13, 25, -8, 10.2])

plt.savefig("../im/harps_tsne-bigplot_"+sets+".png", dpi=200)  



gs0 = gridspec.GridSpec(1, 1)
gs0.update(left=0.1, bottom=0.05, right=0.98, top=0.7)
ax  = plt.Subplot(g, gs0[0, 0])
g.add_subplot(ax)

gs = gridspec.GridSpec(1, 3)
gs.update(left=0.1, bottom=0.77, right=0.98, top=0.98,
           wspace=0.44, hspace=0.05)
exinds = [ [1,2], [4,5], [2,3] ]
for jj in range(3):
    ax = plt.Subplot(g, gs[0, jj])
    g.add_subplot(ax)
    for kk in np.arange(len(t.subsets)):
        mask = (t.classcol == t.subsets[kk])
        ax.errorbar(xx[exinds[jj][0]][mask], xx[exinds[jj][1]][mask],
                    xerr=xerr[exinds[jj][0]][mask], yerr=xerr[exinds[jj][1]][mask],
                   ms=0, mec="k", capthick=0, elinewidth=1,
                   mfc=t.col[kk], alpha=t.al[kk]/3., ecolor=t.col[kk], lw=0,
                   marker=t.sym[kk], zorder=0)
        ax.scatter(xx[exinds[jj][0]][mask], xx[exinds[jj][1]][mask],
                   s=t.size[kk], lw=t.lw[kk], edgecolors="k",
                   c=t.col[kk], alpha=t.al[kk],
                   marker=t.sym[kk])
    ax.set_xlabel(labels[exinds[jj][0]], fontsize=12)
    ax.set_ylabel(labels[exinds[jj][1]], fontsize=12)

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
for kk in np.arange(len(t.subsets)):
    mask = (t.classcol == t.subsets[kk])
    ax.scatter(t.Xt[mask], t.Yt[mask],
               s=t.size[kk], lw=t.lw[kk], edgecolors="k",
               c=t.col[kk], alpha=t.al[kk],
               marker=t.sym[kk])
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
            for kk in np.arange(len(t.subsets)):
                mask = (t.classcol == t.subsets[kk])
                ax.scatter(uvw[1-jj][mask], uvw[2-ii][mask],
                           s=t.size[kk], lw=t.lw[kk], edgecolors="k",
                           c=t.col[kk], alpha=t.al[kk],
                           marker=t.sym[kk])
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
            for kk in np.arange(len(t.subsets)):
                mask = (t.classcol == t.subsets[kk])
                mask = mask * (data['JZ_st_m'] > 0)
                ax.scatter(uvw[1-jj][mask], uvw[2-ii][mask],
                           s=t.size[kk], lw=t.lw[kk], edgecolors="k",
                           c=t.col[kk], alpha=t.al[kk],
                           marker=t.sym[kk])
            if not ii==1:
                ax.set_yscale("log", nonposy='clip')
                ax.set_ylim([0.000001, .5])
            if not jj==0:
                ax.set_xscale("log", nonposx='clip')
                ax.set_xlim([0.000001, .5])
            if ii==jj:
                ax.set_xlabel(lab[1-ii], fontsize=13)
                ax.set_ylabel(lab[2-ii], fontsize=13)
                #if ii==0:
                #    ax.set_xlim([4,13])
                #else:
                #    ax.set_ylim([4,13])
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
            for kk in np.arange(len(t.subsets)):
                mask = (t.classcol == t.subsets[kk])
                ax.scatter(xx[jj][mask], xx[ii+1][mask],
                           s=t.size[kk], lw=t.lw[kk], edgecolors="k",
                           c=t.col[kk], alpha=t.al[kk],
                           marker=t.sym[kk])
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
"""
