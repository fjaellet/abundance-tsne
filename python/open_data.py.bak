# Author: F. Anders
# License: BSD

"""
Prepares HARPS and Bensby datasets used in the t-SNE project
"""

import numpy as np
from astropy.io import fits as pyfits
import scipy

class harps(object):
    def __init__(self, teffcut=True, abunds=True, ages=True):
        """
        Open the file of Delgado-Mena et al. (2017) and return a recarray
        """
        hdu = pyfits.open('../data/DelgadoMena2017.fits', names=True)
        data = hdu[1].data
        if teffcut:
            data = data[ (data['Teff']>5300) * (data['Teff']<6000) *
                         (data['logg_SH']>3) * (data['logg_SH']<5) ]
        if abunds:
            data = data[ (data['nCu']>0) * (data['nZn']>0) * (data['nSr']>0) *
                         (data['nY']>0) * (data['nZrII']>0) * (data['nBa']>0) *
                         (data['nCe']>0) * (data['errAl']<1) * (data['nMg']>0) *
                         (data['nSi']>0) * (data['nCa']>0) * (data['nTiI']>0) ]
        if ages:                 
            data = data[ np.isfinite(data['meanage']) ]
        self.data = data
        return None 

    def get_tsne_subsets(self, sets = "errlim" ):
        """
        Get the names and indices of the t-sne-defined subsets
        """
        if sets=="teffcut":
            self.Xt = self.data["X_tsne_teffcut40"]
            self.Yt = self.data["Y_tsne_teffcut40"]
            self.classcol= np.char.rstrip(self.data["tsne_class_teffcut40"],' ')
            self.subsets = ["thin", "thick1", "thick2", "thick3", "thick4",
                       "mpthin", "mpthintrans", "smr", "t4trans", "youngthin",
                       "debris1", "debris2", "debris3", "debris4", "debris5", 
                       "smr2", "t2trans1", "highTi","lowMg","highAlMg?"]
            self.names   = ["", "", "", "",
                       "", "", "Transition group", "", "",
                       "Young local disc", "", "", "[s/Fe]-enhanced", "", "", r"", "Debris candidate", 
                       r"Extreme-Ti star", r"Low-[Mg/Fe] star", "High-[Al/Mg] star"]
            self.Xcoords = [10, 11, 4.5, -12,  18, -31, 22, 26,-22.5, -14, -2, -25]
            self.Ycoords = [5.5,.5,  -2, -4,   6,  0,   1.5, -.5, -7, -2, -6, 14]
            self.fsize   = [20 , 16,  12, 12,  15,  13, 11, 11, 11, 11, 11, 11]
            self.sym = ["o", "v", "^", ">", "<", "s", "o", "*", "<", "o",
                        "h", "d", "D", "v", "p", "*", "D", "p", "s", "8"]
            self.al  = [.6, .8, .8, .8, .8, .8, .8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            self.lw  = [0,.5,.5,.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5]
            self.size= [9,12,12,12,12,15,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18]
            self.col = ["k", "m", "hotpink", "crimson", "r",
                        "g", "g", "orange", "gold", "grey",
                        "yellow", "yellow", "yellow", "yellow", "yellow",
                        "gold", "brown", "lime", "k", "royalblue"]
        elif sets=="mc":
            self.Xt = self.data["X_tsne_teffcut40_mc"]
            self.Yt = self.data["Y_tsne_teffcut40_mc"]
            self.classcol= np.char.rstrip(self.data["tSNE_class_mc"],' ')
            self.subsets = ["thin", "thick1", "thick2", "thick3", "thick4",
                       "mpthin", "smr", "t4trans",
                       "debris1", "debris2", "debris3", "debris4", "debris5?", 
                       "t2trans1", "t2trans2", "highTi","thicklow"]
            self.names   = ["Thin Disc", "Thick Disc I", "Thick Disc II", "Inner Disc I",
                       "Inner Disc II", "Outer \n Thin Disc", "SMR", "Inner Disc III",
                       "", "", "Satellite \n debris", "", "", r"TII/III", "", 
                       r"Extreme-Ti star", r"Lowest-[Fe/H] star"]
            self.Xcoords = [-25, 15, 4.5, -12,  18, -31, 22, 26,-22.5, -14, -2, -25]
            self.Ycoords = [5.5 ,-6,  -2, -4,   6,  0,   1.5, -.5, -7, -2, -6, 14]
            self.fsize   = [20 , 16,  12, 12,  15,  13, 11, 11, 11, 11, 11, 11]
            self.sym = ["o", "v", "^", ">", "<", "s", "*", "<", "D", "h", "d", "H", "v", "p", "8", "H", "p"]
            self.al  = [.6, .8, .8, .8, .8, .6, 1,1,1,1,1,1,1,1,1,1,1,1]
            self.lw  = [0,.5,.5,.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5]
            self.size= [9,12,12,12,12,15,18,18,18,18,18,18,18,18,18,18,18]
            self.col = ["k", "m", "hotpink", "crimson", "r", "g", "orange", "gold",
                  "yellow", "yellow", "yellow", "yellow", "green", "royalblue", "royalblue",
                   "lime", "m"]
        elif sets=="errlim":
            self.Xt = self.data["X_tsne_teffcut40_errlim_mc"]
            self.Yt = self.data["Y_tsne_teffcut40_errlim_mc"]
            self.classcol= np.char.rstrip(self.data["tSNE_class_errlim_mc"],' ')
            self.subsets = ["thin", "thick1", "thick2", "thick3", "thick4",
                       "mpthin", "youngthin", "smr", "t4trans",
                       "debris1", "debris2", "debris3", "debris4", "debris5?", 
                       "t2trans1", "t2trans2", "highTi","t2trans3", "smr2", "lowMg"]
            self.names   = ["Thin Disc", "Thick Disc", "Metal-rich Thick Disc",
                       "Metal-poor \n Thin Disc", "Young Local Disc", "SMR",
                       "Transition", "Debris"]
            self.Xcoords = [2,  14, -8, 7, -8, -12, -12.2, 18]
            self.Ycoords = [5.5,-7.3, -6.3,.5, 7.5,  1,  -4.3,  7]
            self.fsize   = [20 ,16, 12,12,  12, 13,  11, 15]
            self.sym = ["o", "v", "^", ">", "<", "s", "o", "*", "<", "D", "h", "d", "H", "v",
                   "p", "8", "H", "p", "*", "s"]
            self.al  = [.6, .8, .8, .8, .8, .6, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            self.lw  = [0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5]
            self.size= [9,12,12,12,12,15,18,22,18,18,18,18,18,18,22,22,22,22,28,20]
            self.col = ["k", "m", "hotpink", "crimson", "r", "g", "lightgrey", "orange", "gold",
                  "yellow", "yellow", "yellow", "yellow", "green", "royalblue", "royalblue",
                   "lime", "green", "gold", "k"]
        elif sets=="plainold":
            self.Xt = self.data["X_tsne_teffcut40"]
            self.Yt = self.data["Y_tsne_teffcut40"]
            self.classcol= np.char.rstrip(self.data["tSNE_class"],' ')
            self.subsets = ["thin", "thick1", "thick2", "thick3",
                       "mpthin", "smr",
                       "t1trans", "debris", "highAlMg",
                       "t3trans", "highTioutlier","lowalphaoutlier"]
            self.names   = ["Thin disc", "Thick Disc I", "Thick Disc II", "Thick Disc III",
                       "Metal-poor \n thin disc", "SMR", "Transition I",
                       "Satellite \n debris", r"High-[Al/Mg]", "Transition III",
                       r"Extreme-Ti star", r"Low-[Mg/Fe] star"]
            self.Xcoords = [-25, 15, 4.5, -12,  18, -31, 22, 26,-22.5, -14, -2, -25]
            self.Ycoords = [5.5 ,-6,  -2, -4,   6,  0,   1.5, -.5, -7, -2, -6, 14]
            self.fsize   = [20 , 16,  12, 12,  15,  13, 11, 11, 11, 11, 11, 11]
            self.sym = ["o", "v", "^", ">", "s", "*", "<", "D", "p", "8", "H", "h"]
            self.al  = [.6, .6, .8, .8, .75, 1,1,1,1,1,1,1]
            self.lw  = [0,0,.5,.5, .5, .5, .5, .5, .5, .5, .5, .5]
            self.size= [6,9,9,9,12,20,18,18,18,18,22,25]
            self.col = ["k", "r", "orange", "gold", "g", "orange",
                  "brown", "yellow", "royalblue", "hotpink", 
                  "lime", "black"]
        else:
            raise ValueError("No valid 'subsets' name set")

        
class bensby(object):
    def __init__(self, teffcut=True, abunds=True):
        """
        Open the file of Bensby et al. (2014) and return a recarray
        """
        hdu = pyfits.open('../data/Bensby2014_with_tSNEandBattistini.fits',
                          names=True)
        self.data = hdu[1].data
        # Mask out stars with non-detections in some of the elements considered
        data = self.data[
                 (self.data['nO1']>0) * (self.data['nNa1']>0) *
                 (self.data['nMg1']>0) * (self.data['nAl1']>0) *
                 (self.data['nSi1']>0) * (self.data['nCa1']>0) *
                 (self.data['nTi1']>0) * (self.data['nCr1']>0) *
                 (self.data['nNi1']>0) * (self.data['nZn1']>0) *
                 (self.data['nY2']>0) * (self.data['nBa2']>0) *
                 (self.data['nFe1']>0) ]
        if teffcut:
            data = data [(data['Teff']>5000) * (data['Teff']<6300)]
        self.data= data
        print len(data), " stars included."
        return None

    def get_ndimspace(self, age=False, kin=False, feh=True, mc=50):
        """
        Cut out missing data and prepare t-SNE input array

        Optional:
            age: Bool  - include age in the analysis
            kin: Bool  - include kinematics (UVW velocities)
            feh: Bool  - include [Fe/H], default: True
            mc:  Int   - if ==1, then no MC magic will happen
        """
        data = self.data
        print len(data), " stars included."
        verr = 10. * np.ones(len(data))
        X        = np.c_[data['Fe_H'],data['O_Fe'],data['Na_Fe'],data['Mg_Fe'],
                         data['Al_Fe'],data['Si_Fe'],data['Ca_Fe'],data['Ti_Fe'],
                         data['Cr_Fe'],data['Ni_Fe'],data['Zn_Fe'],data['Y_Fe'],
                         data['Ba_Fe'],data['Age']]
        Xerr1    = np.c_[data['e_Fe_H'],data['e_O_Fe'],data['e_Na_Fe'],data['e_Mg_Fe'],
                         data['e_Al_Fe'],data['e_Si_Fe'],data['e_Ca_Fe'],data['e_Ti_Fe'],
                         data['e_Cr_Fe'],data['e_Ni_Fe'],data['e_Zn_Fe'],data['e_Y_Fe'],
                         data['e_Ba_Fe'],0.5*(data['B_Age1']-data['b_Age'])]
        Xerr     = np.mean( Xerr1, axis=0)

        if mc > 1:
            Y        = np.zeros(( mc*len(data), 14 ))
            for ii in np.arange(mc):
                Y[ii::mc, :] = scipy.random.normal(loc=X, size=X.shape,
                                                   scale=np.maximum(Xerr1, 0.02*np.ones(Xerr1.shape)))
            X = Y

        if not age:
            X = X[:,:-1]; Xerr = Xerr[:-1]
        if kin:
            X    = np.append(X, np.vstack((data['Ulsr'],data['Vlsr'],data['Wlsr'])).T, axis=1)
            Xerr = np.append(Xerr, np.array([10,10,10]).T, axis=0)
        if not feh:
            X = X[:,1:]; Xerr = Xerr[1:]
            
        self.Xnorm    = (X/Xerr[np.newaxis,:])


    def get_tsne_subsets(self, sets = "errlim" ):
        """
        Get the names and indices of the t-sne-defined subsets
        """
        if sets=="mc":
            self.Xt = self.data["X_tsne_teffcut40_mc"]
            self.Yt = self.data["Y_tsne_teffcut40_mc"]
            self.classcol= np.char.rstrip(self.data["tSNE_class_mc"],' ')
            self.subsets = ["thin", "thick1", "thick2", "thick3", "thick4",
                       "mpthin", "smr", "t4trans",
                       "debris1", "debris2", "debris3", "debris4", "debris5?", 
                       "t2trans1", "t2trans2", "highTi","thicklow"]
            self.names   = ["Thin Disc", "Thick Disc I", "Thick Disc II", "Thick Disc III",
                       "Thick Disc IV", "Metal-poor \n Thin Disc", "SMR", "Transition",
                       "", "", "Satellite \n debris", "", "", r"TII/III", "", 
                       r"Extreme-Ti star", r"Lowest-[Fe/H] star"]
            self.Xcoords = [-25, 15, 4.5, -12,  18, -31, 22, 26,-22.5, -14, -2, -25]
            self.Ycoords = [5.5 ,-6,  -2, -4,   6,  0,   1.5, -.5, -7, -2, -6, 14]
            self.fsize   = [20 , 16,  12, 12,  15,  13, 11, 11, 11, 11, 11, 11]
            self.sym = ["o", "v", "^", ">", "<", "s", "*", "<", "D", "h", "d", "H", "v", "p", "8", "H", "p"]
            self.al  = [.6, .8, .8, .8, .8, .6, 1,1,1,1,1,1,1,1,1,1,1,1]
            self.lw  = [0,.5,.5,.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5]
            self.size= [9,12,12,12,12,15,18,18,18,18,18,18,18,18,18,18,18]
            self.col = ["k", "m", "hotpink", "crimson", "r", "g", "orange", "gold",
                  "yellow", "yellow", "yellow", "yellow", "green", "royalblue", "royalblue",
                   "lime", "m"]
        elif sets=="errlim":
            self.Xt = self.data["X_tsne_teffcut40_errlim_mc"]
            self.Yt = self.data["Y_tsne_teffcut40_errlim_mc"]
            self.classcol= np.char.rstrip(self.data["tSNE_class_errlim_mc"],' ')
            self.subsets = ["thin", "thick1", "thick2", "thick3", "thick4",
                       "mpthin", "youngthin", "smr", "t4trans",
                       "debris1", "debris2", "debris3", "debris4", "debris5?", 
                       "t2trans1", "t2trans2", "highTi","t2trans3", "smr2", "lowMg"]
            self.names   = ["Thin Disc", "Thick Disc", "Metal-rich Thick Disc",
                       "Metal-poor \n Thin Disc", "Young Local Disc", "SMR",
                       "Transition", "Debris"]
            self.Xcoords = [2,  14, -8, 7, -8, -12, -12.2, 18]
            self.Ycoords = [5.5,-7.3, -6.3,.5, 7.5,  1,  -4.3,  7]
            self.fsize   = [20 ,16, 12,12,  12, 13,  11, 15]
            self.sym = ["o", "v", "^", ">", "<", "s", "o", "*", "<", "D", "h", "d", "H", "v",
                   "p", "8", "H", "p", "*", "s"]
            self.al  = [.6, .8, .8, .8, .8, .6, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            self.lw  = [0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5]
            self.size= [9,12,12,12,12,15,18,22,18,18,18,18,18,18,22,22,22,22,28,20]
            self.col = ["k", "m", "hotpink", "crimson", "r", "g", "lightgrey", "orange", "gold",
                  "yellow", "yellow", "yellow", "yellow", "green", "royalblue", "royalblue",
                   "lime", "green", "gold", "k"]
        elif sets=="plain":
            self.Xt = self.data["X_tsne_teffcut40"]
            self.Yt = self.data["Y_tsne_teffcut40"]
            self.classcol= np.char.rstrip(self.data["tSNE_class"],' ')
            self.subsets = ["thin", "thick1", "thick2", "thick3",
                       "mpthin", "smr",
                       "t1trans", "debris", "highAlMg",
                       "t3trans", "highTioutlier","lowalphaoutlier"]
            self.names   = ["Thin disc", "Thick Disc I", "Thick Disc II", "Thick Disc III",
                       "Metal-poor \n thin disc", "SMR", "Transition I",
                       "Satellite \n debris", r"High-[Al/Mg]", "Transition III",
                       r"Extreme-Ti star", r"Low-[Mg/Fe] star"]
            self.Xcoords = [-25, 15, 4.5, -12,  18, -31, 22, 26,-22.5, -14, -2, -25]
            self.Ycoords = [5.5 ,-6,  -2, -4,   6,  0,   1.5, -.5, -7, -2, -6, 14]
            self.fsize   = [20 , 16,  12, 12,  15,  13, 11, 11, 11, 11, 11, 11]
            self.sym = ["o", "v", "^", ">", "s", "*", "<", "D", "p", "8", "H", "h"]
            self.al  = [.6, .6, .8, .8, .75, 1,1,1,1,1,1,1]
            self.lw  = [0,0,.5,.5, .5, .5, .5, .5, .5, .5, .5, .5]
            self.size= [6,9,9,9,12,20,18,18,18,18,22,25]
            self.col = ["k", "r", "orange", "gold", "g", "orange",
                  "brown", "yellow", "royalblue", "hotpink", 
                  "lime", "black"]
        else:
            raise ValueError("No valid 'subsets' name set")

        

