#########################################################################
# corogee.calculate_6d_coords
#
#   calculates relevant transformed coordinates (such as R_gal, Z_gal,
#   V_phi, E, L_z, etc.) along with their uncertainties using Jo Bovy's
#   galpy module for the CoRoGEE dataset
#
#          ----------WORKING BUT STILL UNDER CONSTRUCTION-----------
#
#########################################################################

import os.path
import numpy as np
import sys
import scipy as sc
from astropy.io import fits as pyfits
from galpy.util.bovy_coords import *

# Set structural parameters (new ones as in Piffl et al. 2015)
_Rsun   = 8.3
_Zsun   = 0.014
_vc_sun = 220.
vsun    = [-11.1, 232.24, 7.25]     

# where to read the data and save the results:
readfile= '../data/DelgadoMena2017.fits'
savefile= '../data/DelgadoMena2017_kinematics.fits'

######################################
# READ DATA
######################################

#Object info
data = pyfits.open(readfile)[1].data
data = data[ ( data['dist_FA'] > 0 ) ]

ID       = data['Star']
ra       = data['_RAJ2000']
dec      = data['_DEJ2000']
l        = data['GLON']
b        = data['GLAT']

#APOGEE info
vr       = data['RV_Adi']
e_vr     = 0.1 * np.ones(len(data))

#Corot-PARAM info
d        = data['dist_FA']
e_d      = data['e_dist_FA']

# UCAC-4 info
pmra     = data['pmRA_FA']
e_pmra   = data['e_pmRA_FA']
pmdec    = data['pmDE_FA']
e_pmdec  = data['e_pmDE_FA']
            
#Combined info
l_rad    = l * np.pi/180.
b_rad    = b * np.pi/180.
x        = -( _Rsun - d * np.cos(b_rad) * np.cos(l_rad) )
y        = d * np.cos(b_rad) * np.sin(l_rad)
z        = d * np.sin(b_rad) + _Zsun
Rcur     = np.sqrt( x**2 + y**2 )

############################################
# Now use Bovy's utilities to calculate:
############################################
# (l,b) Proper motions
pmllpmbb=pmrapmdec_to_pmllpmbb(pmra,pmdec,ra,dec,degree=True,epoch=2000.0)
pmll=pmllpmbb[:,0]
pmbb=pmllpmbb[:,1]

# Full 6D heliocentric Cartesian coordinates
xyzuvw=sphergal_to_rectgal(l,b,d,vr,pmll,pmbb,degree=True)
X=xyzuvw[:,0]
Y=xyzuvw[:,1]
Z=xyzuvw[:,2]
U=xyzuvw[:,3]
V=xyzuvw[:,4]
W=xyzuvw[:,5]

# Galactocentric Cartesian coordinates
XYZg=XYZ_to_galcenrect(X,Y,Z,Xsun=_Rsun,Zsun=_Zsun)
Xg=XYZg[:,0]
Yg=XYZg[:,1]
Zg=XYZg[:,2]

# Galactocentric Cartesian coordinates
RPhiZg=rect_to_cyl(Xg,Yg,Zg)
Rg   = Rcur #XYZg[0]               IS THERE A PROBLEM WITH rect_to_cyl ???
Phig = RPhiZg[1]
Zg   = XYZg[:,2]
Phig_deg= Phig*180./(2*np.pi)
print Phig_deg

# Linearised uncertainties
Xg_sig = abs(np.cos(b_rad)*np.cos(l_rad)*e_d)
Yg_sig = abs(np.cos(b_rad)*np.sin(l_rad)*e_d)
Zg_sig = abs(np.sin(b_rad)*e_d)
Rg_sig = ( abs(2*x*np.cos(b_rad)*np.cos(l_rad)*e_d +
               2*y*np.cos(b_rad)*np.sin(l_rad)*e_d)  ) / np.sqrt(x**2+y**2)

# (l,b) Proper motion uncertainties
cov_pmradec=np.zeros((len(ra),2,2))
cov_pmradec[:,0,0]=e_pmra
cov_pmradec[:,1,1]=e_pmdec
cov_pmllbb=cov_pmrapmdec_to_pmllpmbb(cov_pmradec,ra,dec,\
                                     degree=True,epoch=2000.0)
# UVW uncertainties
cov_uvw=cov_dvrpmllbb_to_vxyz(d,e_d,e_vr,pmll,pmbb,cov_pmllbb,l,b,\
                              plx=False,degree=True)

vx_sig= np.sqrt(cov_uvw[:,0,0])
vy_sig= np.sqrt(cov_uvw[:,1,1])
vz_sig= np.sqrt(cov_uvw[:,2,2])

# Transverse velocity + uncertainty
_K=4.7407
pm=np.sqrt(data['pmRA_FA']**2+data['pmDE_FA']**2)  
e_pm=np.sqrt( (data['pmRA_FA'] * data['e_pmRA_FA'])**2
              + (data['pmDE_FA'] * data['e_pmDE_FA'])**2 ) / pm
vt= _K * d * pm
e_vt = _K* np.sqrt( (d*e_pm)**2 + (e_d*pm)**2 )

# Galactocentric Cartesian Velocities - CORRECTED FOR SOLAR MOTION
vxxyzg = vxvyvz_to_galcenrect(U,V,W,vsun=vsun)
vXg, vYg, vZg = vxxyzg[:,0], vxxyzg[:,1], vxxyzg[:,2]

# Galactocentric Cylindrical Velocities - CORRECTED FOR SOLAR MOTION
vRPhiZg = rect_to_cyl_vec(vXg, vYg, vZg,Xg,Yg,Zg, cyl=False)

vRg = vRPhiZg[0]
vPhig = vRPhiZg[1]
vZg = vRPhiZg[2]

#########################################################################
# Cylindrical Velocity Uncertainties
#
#   (this is assuming that Phi is perfectly determined - ideally, we would
#    transform the whole uncertainty covariances from (l,b,d,pmra,pmdec,vr)
#    to (R,Phi,Z, vR, vPhi,Vz)_g --- not implemented in galpy yet.
#########################################################################

vRg_sig = np.sqrt( (sc.cos(Phig)*vx_sig)**2 + (sc.sin(Phig)*vy_sig)**2 )
vPhig_sig = np.sqrt( (sc.sin(Phig)*vx_sig)**2 + (sc.cos(Phig)*vy_sig)**2 )
vZg_sig = vz_sig

#########################################################################
# Specific energy-angular momentum space coordinates (Ruchti et al. 2014)
#        (assuming Bovy's MWPotential2014)
#########################################################################

from galpy.util      import bovy_conversion
from galpy.potential import MWPotential, MWPotential2014
from galpy.potential import evaluatePotentials, evaluateRforces
from galpy.potential import vcirc, dvcircdR

mp= MWPotential2014

# Angular momentum Lz  ( in solar units! )
Lz = (Rg/_Rsun) * (vPhig/_vc_sun)
Lz_sig = np.sqrt( (Rg_sig * vPhig)**2 + (Rg * vPhig_sig)**2 ) / (_vc_sun*_Rsun)

# Vertical Energy in Bovy units
Ez = 0.5*(vZg/_vc_sun)**2 + evaluatePotentials(mp, Rg/_Rsun, Zg/_Rsun)

# Total Energy in Bovy units
E = Ez + 0.5*(vRg/_vc_sun)**2 + 0.5*(vPhig/_vc_sun)**2

###########################################################################
# Ad-hoc (almost-non-parametric) parameters
###########################################################################

# Guiding center radius a la Ivan
Rc = Rg * vPhig / _vc_sun
Rc_sig = np.sqrt( (Rg_sig * vPhig)**2 + (Rg * vPhig_sig)**2 ) / _vc_sun

# Circular Velocity at the guiding center
vcirc_c = _vc_sun  * vcirc(mp, Rc/_Rsun)
# Circular Velocity at the star's position
vcirc_g = _vc_sun  * vcirc(mp, Rg/_Rsun)

###########################################################################
# Simple derived results
###########################################################################

# Ec as in Ruchti et al. (2014)
Ec = evaluatePotentials(mp, Rg/_Rsun,0)# + 0.5*(vcirc_g/_vc_sun)**2
# Guiding center Angular momentum ( in solar units! )
Lc = (Rc/_Rsun) * (vcirc_c/_vc_sun)

###########################################################################
# and now: ACTIONS and their uncertainties.
###########################################################################

'''In order to derive actions, we have to know that the orbits are bound..'''
mask = (E<-0.8)
"""
# Adiabatic Approximation (Binney 2010)
from galpy.actionAngle import actionAngleAdiabatic
aAA= actionAngleAdiabatic(pot=MWPotential2014)

'''Coordinates have to passed to aAA et al. in the form
( R/Rsun, Z/Rsun, Vphi/Vcirc_LSR, Vr/Vcirc_LSR, Vz/Vcirc_LSR, Phi[radians] )'''
print 'Calculating Actions in adiabatic approximation...'

adiabatic_actions_m = aAA(Rg[mask]/_Rsun, Zg[mask]/_Rsun,
                          vPhig[mask]/_vc_sun, vRg[mask]/_vc_sun, vZg[mask]/_vc_sun,
                          Phig[mask] )
JR_adi_m, JPhi_adi_m, JZ_adi_m = -9.9*np.ones(len(Ec)),\
                                 -9.9*np.ones(len(Ec)), -9.9*np.ones(len(Ec))

JR_adi_m[mask] = adiabatic_actions_m[0]
JPhi_adi_m[mask] = adiabatic_actions_m[1]
JZ_adi_m[mask] = adiabatic_actions_m[2]


'''Estimate VERY CONSERVATIVE uncertainties just by varying the input values
in the same direction'''

adiabatic_actions_U = aAA((Rg[mask]+Rg_sig[mask])/_Rsun,
                          (Zg[mask]+Zg_sig[mask])/_Rsun,
                          (vPhig[mask]+vPhig_sig[mask])/_vc_sun,
                          (vRg[mask]+vRg_sig[mask])/_vc_sun,
                          (vZg[mask]+vZg_sig[mask])/_vc_sun,
                          Phig[mask] )
JR_adi_U, JPhi_adi_U, JZ_adi_U = -9.9*np.ones(len(Ec)),\
                                 -9.9*np.ones(len(Ec)), -9.9*np.ones(len(Ec))

JR_adi_U[mask] = adiabatic_actions_U[0]
JPhi_adi_U[mask] = adiabatic_actions_U[1]
JZ_adi_U[mask] = adiabatic_actions_U[2]

adiabatic_actions_L = aAA((Rg[mask]-Rg_sig[mask])/_Rsun,
                          (Zg[mask]-Zg_sig[mask])/_Rsun,
                          (vPhig[mask]-vPhig_sig[mask])/_vc_sun,
                          (vRg[mask]-vRg_sig[mask])/_vc_sun,
                          (vZg[mask]-vZg_sig[mask])/_vc_sun,
                          Phig[mask] )
JR_adi_L, JPhi_adi_L, JZ_adi_L = -9.9*np.ones(len(Ec)),\
                                 -9.9*np.ones(len(Ec)), -9.9*np.ones(len(Ec))

JR_adi_L[mask] = adiabatic_actions_L[0]
JPhi_adi_L[mask] = adiabatic_actions_L[1]
JZ_adi_L[mask] = adiabatic_actions_L[2]
print '...done.'
"""

# Staeckel Approximation (Binney 2012, Sanders 2012)
from galpy.actionAngle import actionAngleStaeckel

aAS= actionAngleStaeckel(pot=MWPotential2014,delta=0.4) #optimal value for MWP14
print 'Calculating Actions in Staeckel approximation...'

staeckel_actions_m = aAS(Rg[mask]/_Rsun, Zg[mask]/_Rsun,
                          vPhig[mask]/_vc_sun, vRg[mask]/_vc_sun, vZg[mask]/_vc_sun,
                          Phig[mask] )
JR_st_m, JPhi_st_m, JZ_st_m = -9.9*np.ones(len(Ec)),\
                                 -9.9*np.ones(len(Ec)), -9.9*np.ones(len(Ec))

JR_st_m[mask] = staeckel_actions_m[0]
JPhi_st_m[mask] = staeckel_actions_m[1]
JZ_st_m[mask] = staeckel_actions_m[2]

staeckel_actions_U = aAS((Rg[mask]+Rg_sig[mask])/_Rsun,
                          (Zg[mask]+Zg_sig[mask])/_Rsun,
                          (vPhig[mask]+vPhig_sig[mask])/_vc_sun,
                          (vRg[mask]+vRg_sig[mask])/_vc_sun,
                          (vZg[mask]+vZg_sig[mask])/_vc_sun,
                          Phig[mask] )
JR_st_U, JPhi_st_U, JZ_st_U = -9.9*np.ones(len(Ec)),\
                                 -9.9*np.ones(len(Ec)), -9.9*np.ones(len(Ec))

JR_st_U[mask] = staeckel_actions_U[0]
JPhi_st_U[mask] = staeckel_actions_U[1]
JZ_st_U[mask] = staeckel_actions_U[2]

staeckel_actions_L = aAS((Rg[mask]-Rg_sig[mask])/_Rsun,
                          (Zg[mask]-Zg_sig[mask])/_Rsun,
                          (vPhig[mask]-vPhig_sig[mask])/_vc_sun,
                          (vRg[mask]-vRg_sig[mask])/_vc_sun,
                          (vZg[mask]-vZg_sig[mask])/_vc_sun,
                          Phig[mask] )
JR_st_L, JPhi_st_L, JZ_st_L = -9.9*np.ones(len(Ec)),\
                                 -9.9*np.ones(len(Ec)), -9.9*np.ones(len(Ec))

JR_st_L[mask] = staeckel_actions_L[0]
JPhi_st_L[mask] = staeckel_actions_L[1]
JZ_st_L[mask] = staeckel_actions_L[2]

print '...done.'


print "Trying to save file..."

c01=pyfits.Column( name='ID', format='12A', array=ID)
#c02=pyfits.Column( name='APOGEE', format='20A', array=APOGEE_ID)
c03=pyfits.Column( name='Xg', format='E', array=Xg)
c04=pyfits.Column( name='Xg_sig', format='E', array=Xg_sig)
c05=pyfits.Column( name='Yg', format='E', array=Yg)
c06=pyfits.Column( name='Yg_sig', format='E', array=Yg_sig)
c07=pyfits.Column( name='Zg', format='E', array=Zg)
c08=pyfits.Column( name='Zg_sig', format='E', array=Zg_sig)
c09=pyfits.Column( name='Rg', format='E', array=Rg)
c10=pyfits.Column( name='Rg_sig', format='E', array=Rg_sig)
c11=pyfits.Column( name='Phig', format='E', array=Phig)
c12=pyfits.Column( name='Phig_deg', format='E', array=Phig_deg)
c13=pyfits.Column( name='vT', format='E', array=vt)
c14=pyfits.Column( name='vT_sig', format='E', array=e_vt)
c15=pyfits.Column( name='vXg', format='E', array=vXg)
c16=pyfits.Column( name='vXg_sig', format='E', array=vx_sig)
c17=pyfits.Column( name='vYg', format='E', array=vYg)
c18=pyfits.Column( name='vYg_sig', format='E', array=vy_sig)
c19=pyfits.Column( name='vZg', format='E', array=vZg)
c20=pyfits.Column( name='vZg_sig', format='E', array=vz_sig)
c21=pyfits.Column( name='vRg', format='E', array=vRg)
c22=pyfits.Column( name='vRg_sig', format='E', array=vRg_sig)
c23=pyfits.Column( name='vPhig', format='E', array=vPhig)
c24=pyfits.Column( name='vPhig_sig', format='E', array=vPhig_sig)
c25=pyfits.Column( name='vcirc_g', format='E', array=vcirc_g)
c26=pyfits.Column( name='vcirc_c', format='E', array=vcirc_c)
c27=pyfits.Column( name='Lz', format='E', array=Lz)
c28=pyfits.Column( name='Lz_sig', format='E', array=Lz_sig)
c29=pyfits.Column( name='Ez', format='E', array=Ez)
c30=pyfits.Column( name='E', format='E', array=E)
c31=pyfits.Column( name='Ec', format='E', array=Ec)
c32=pyfits.Column( name='Rc', format='E', array=Rc)
c33=pyfits.Column( name='Rc_sig', format='E', array=Rc_sig)
#c34=pyfits.Column( name='JR_adi_m', format='E', array=JR_adi_m)
#c35=pyfits.Column( name='JR_adi_L', format='E', array=JR_adi_L)
#c36=pyfits.Column( name='JR_adi_U', format='E', array=JR_adi_U)
#c37=pyfits.Column( name='JPhi_adi_m', format='E', array=JPhi_adi_m)
#c38=pyfits.Column( name='JPhi_adi_L', format='E', array=JPhi_adi_L)
#c39=pyfits.Column( name='JPhi_adi_U', format='E', array=JPhi_adi_U)
#c40=pyfits.Column( name='JZ_adi_m', format='E', array=JZ_adi_m)
#c41=pyfits.Column( name='JZ_adi_L', format='E', array=JZ_adi_L)
#c42=pyfits.Column( name='JZ_adi_U', format='E', array=JZ_adi_U)
c43=pyfits.Column( name='JR_st_m', format='E', array=JR_st_m)
c44=pyfits.Column( name='JR_st_L', format='E', array=JR_st_L)
c45=pyfits.Column( name='JR_st_U', format='E', array=JR_st_U)
c46=pyfits.Column( name='JPhi_st_m', format='E', array=JPhi_st_m)
c47=pyfits.Column( name='JPhi_st_L', format='E', array=JPhi_st_L)
c48=pyfits.Column( name='JPhi_st_U', format='E', array=JPhi_st_U)
c49=pyfits.Column( name='JZ_st_m', format='E', array=JZ_st_m)
c50=pyfits.Column( name='JZ_st_L', format='E', array=JZ_st_L)
c51=pyfits.Column( name='JZ_st_U', format='E', array=JZ_st_U)

columns= pyfits.ColDefs( [c01,c03,c04,c05,c06,c07,c08,c09,c10, \
                          c11,c12,c13,c14,c15,c16,c17,c18,c19,c20, \
                          c21,c22,c23,c24,c25,c26,c27,c28,c29,c30, \
                          c31,c32,c33,#c34,c35,c36,c37,c38,c39,c40,c41,c42,
                          c43,c44,c45,c46,c47,c48,c49,c50, \
                          c51] )
newHDU= pyfits.BinTableHDU.from_columns(columns)
if os.path.exists(savefile):
    print 'output file exists already'
else:
    newHDU.writeto(savefile)
    print '...done'
