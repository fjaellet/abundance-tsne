"""
Prepares HARPS and Bensby datasets used in the t-SNE project
"""
# Author: F. Anders
# License: BSD

import numpy as np
from astropy.io import fits as pyfits

class harps(object):
    def __init__(self):
        """
        Open the file of Delgado-Mena et al. (2017) and return a recarray
        """
        hdu = pyfits.open('../data/DelgadoMena2017.fits', names=True)
        self.data = hdu[1].data
        return None #self.data

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
                       "Metal-poor \n Thin Disc", "Youngest Thin Disc", "SMR",
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

        

