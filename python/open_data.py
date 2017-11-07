"""
Prepares HARPS and Bensby datasets used in the t-SNE project
"""
# Author: F. Anders
# License: BSD

import numpy as np
from astropy.io import fits as pyfits

class harps(object):
    def __init__():
        """
        Open the file of Delgado-Mena et al. (2017) and return a recarray
        """
        hdu = pyfits.open(
            '/home/friedel/Astro/Spectro/HARPS/DelgadoMena2017.fits',
            names=True)
        self.data = hdu[1].data
        return self.data

    def get_tsne_subsets( mc = True ):
        """
        Get the names and indices of the t-sne-defined subsets
        """
        

