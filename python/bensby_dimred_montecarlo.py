print(__doc__)

from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter
import matplotlib.gridspec as gridspec
        
import numpy as np

from astroML.decorators import pickle_results
from astroML.density_estimation import XDGMM
from astroML.plotting.tools import draw_ellipse
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=10, usetex=True)

import scipy
import open_data
from sklearn import manifold, datasets

# The dataset
age      = False
kin      = False
feh      = True
teffcut  = True
mc       = 1

# READ BENSBY DATA
ben = open_data.bensby(teffcut=teffcut)
ben.get_ndimspace(age=age, kin=kin, feh=feh, mc=mc)
n_components = 2

data = ben.data

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

for p in [5, 15, 30, 40, 50, 75, 100, 120, 150]:#[
    for i in [0]:
        t0 = time()
        # Use the recommendations of Linderman & Steinerberger 2017
        # for the learning rate and the exaggeration parameter
        tsne = manifold.TSNE(n_components=n_components, init='pca',
                             random_state=i, perplexity=p,
                             learning_rate=1, early_exaggeration=len(data)/10)
        Y = tsne.fit_transform(ben.Xnorm)
        t1 = time()
        print "p=", p, " , random state ", i
        print("t-SNE: %.2g sec" % (t1 - t0))

        f = plt.figure(figsize=(12, 12))
        plt.suptitle("t-SNE manifold learning for the Bensby+2014 sample", fontsize=14)
        gs0 = gridspec.GridSpec(5, 4)
        gs0.update(left=0.08, bottom=0.08, right=0.92, top=0.92,
                   wspace=0.05, hspace=0.12)

        for ii in range(5):
            for jj in range(4):
                ax = plt.Subplot(f, gs0[ii, jj])
                f.add_subplot(ax)
                scat = plt.scatter(Y[:, 0], Y[:, 1],
                                   c=np.repeat(colors[4*ii+jj],mc),
                                   s=15./mc, cmap=plt.cm.jet, lw=0)
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

        rec = np.rec.fromarrays((np.repeat(data['HIP'],mc), Y[:, 0], Y[:, 1]),
                                dtype=[('ID','|S18'),('X_tsne','float'),
                                       ('Y_tsne','float')])
        add = ""
        if age or teffcut or kin or mc>1 or not feh:
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
        np.savetxt("../tsne_results/bensby2014_tsne_results"+add+str(p)+
                           "_rand"+str(i)+".csv", rec, delimiter=',',
               fmt = ('%s, %.5e, %.5e'))
        plt.savefig("../im/Bensby2014_tsne_plots"+add+str(p)+
                           "_rand"+str(i)+".png", dpi=200)
        
        plt.clf()
