# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:18:43 2021

@author: ATeklesadik
"""

import numpy as np
from math import radians,cos,sin,asin,sqrt,degrees

def Generate_Spyderweb_mesh(n_cols,n_rows,tc_radius,lat0):
    """ 
    Generate a polar grid, the spyderweb mesh, which serves as a 2D-representation of the wind and pressure fields
    
    Arguments:
        *n_cols* : Number of columns, representing the number of gridpoints in angular direction ("pie slices")
        *n_rows* : Number of rows, representing the number of gridpoints in radial direction ("radius slices")
        *tc_radius*: Tropical cyclone radius, in km. 
        
    Returns:
        *rlist* : radius from center, in km(in a list)
        *thetalist* : angle, 0 deg = EAST (in a list)
        *xlist* : The corresponding zonal distance (in km) of the (r,theta)-coordinate (in a list)
        *ylist* : The corresponding meridional distance (in km) of the (r,theta)-coordinate (in a list)
    """ 
    
   #Generate a meshgrid of the distance from the center and the angle 
    rlist,thetalist=np.meshgrid(np.linspace(1,tc_radius,n_rows),np.linspace(0,2.*np.pi,n_cols))
    if lat0<0.:
        thetalist=thetalist-radians(180)
        thetalist[thetalist<0]+=radians(360)
    
    #Compute the corresponding size of the vector as a x- and y component
    xlist=rlist*np.cos(thetalist)
    ylist=rlist*np.sin(thetalist)
           
    return rlist,thetalist,xlist,ylist 
#%%
def Generate_Spyderweb_lonlat_coordinates(xlist,ylist,lat1,lon1):
    """
    Generate a grid of longitude/latitude coordinates corresponding to the spyderweb grid.
    
    Arguments:
        *xlist* : The zonal distance (in km) from the eye of the tropical cyclone (in a list)
        *ylist* : The  meridional distance (in km) from the eye of the tropical cyclone (in a list)
        *lat1*  : Latitude position of the eye 
        *lon1*  : Longitude position of the eye
        
    Returns:
      *latlist* : List of latitude points of the spyderweb grid
      *lonlist* : List of longitude points of the spyderweb grid
         
    """
    radius=6371.
    
    latlist=lat1+(xlist/radius)*(180./np.pi)
    lonlist=lon1+(ylist/radius)*(180/np.pi)/cos(lat1*np.pi/180.)
    
    return latlist,lonlist

def Calculate_translation_speed(lon_0,lat_0,lon_1,lat_1,dt):
    """
  
    Calculate translation speed (forward speed) of a tropical cyclone using a haversine function
    
    Arguments:
        *lon0* : Longitude-coordinate of tropical cyclone eye at time step t (in radians)
        *lat0* : Latitude-coordinate of tropical cyclone eye at time step t (in radians)
        *lon1* : Longitude-coordinate of tropical cyclone eye at time step t+1 (in radians)
        *lat1* : Latitude-coordinate of tropical cyclone eye at time step t+1 (in radians)
        *dt*: Size of time step between t and t+1, in seconds
        
    Returns:
        *translation_speed* : Translation speed (forward speed) in m/s
        
    """
    dlon=abs(lon_0-lon_1)
    dlat=abs(lat_0-lat_1)
    a=sin(dlat/2.)**2.+cos(lat_0)*cos(lat_1)*sin(dlon/2.)**2.
    
    c=2.*asin(sqrt(a))                                      #fraction of the radius
    radius=6371000.                                         #radius of the Earth in m 
    distance=c*radius
        
    translation_speed=distance/dt
    
    return translation_speed

def Compute_background_flow(lon0,lat0,lon1,lat1,dt):
    """
    
    Compute the background flow of the tropical cyclone
    
    Arguments:
        *lon0* : Longitude-coordinate of tropical cyclone eye at time step t-1 (ranging between 0-360 deg, in deg)
        *lat0* : Latitude-coordinate of tropical cyclone eye at time step t-1 (in deg)
        *lon1* : Longitude-coordinate of tropical cyclone eye at time step t (ranging between 0-360 deg, in deg)
        *lat1* : Latitude-coordinate of tropical cyclone eye at time step t (in deg)
        *dt*: Size of time step between t and t+1, in seconds
    
    Returns:        
        *bg*  : background flow (=translation speed), in m/s 
        *ubg* : u-component of background flow (zonal), in m/s
        *vbg* : v-component of background flow (meridional), in m/s
                
    """
    #convert lons and lats to radians
    lon_0,lat_0,lon_1,lat_1=map(radians,[lon0,lat0,lon1,lat1])
    
    #Calculate the bearing between the two points. 
    #For details, see http://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/
    
    X=cos(lat_1)*sin(lon_1-lon_0)
    Y=cos(lat_0)*sin(lat_1)-sin(lat_0)*cos(lat_1)*cos(lon_1-lon_0)
    
    bearing=np.arctan2(Y,X)
    
    bearing=degrees(bearing)
    #Convert the projection so that 0 deg = North (instead of East)    
    
    h=450-bearing
    if h>360:
        h=h-360
    
    
    h=radians(h)
    #Finally, compute the background flow (=translation speed), and the u- and v-components of this flow
    bg=Calculate_translation_speed(lon_0,lat_0,lon_1,lat_1,dt)
    ubg=bg*sin(h)
    vbg=bg*cos(h)

    return bg,ubg,vbg

def Holland_model(lat1,P,Usurf,Rmax,r):
    """
    
    Compute the 2D-wind field of a tropical cyclone at a time step using the Holland (1980) model.
    
    Arguments:
        *lat1*  : Latitudinal position of the eye of the tropical cyclone at time step t (in deg)
        *P*     : Minimum central pressure in the eye of the tropical cyclone (in hPa)
        *Usurf* : Maximum surface wind speed, minus the background flow, at time step t (in m/s)
        *Rmax*  : Radius to maximum winds (radius of the eye of the tropical cyclone) at time step t
        *r*     : Radius "slice" from the center (calculated previously) in km.
        
    Returns:
        *Vs*    : Symmetrical surface wind speed at radius r (m/s)
        *Ps*    : Symmetrical pressure at radius r (m/s)
        
    """
    
    Rmax0=Rmax*1000. #convert from km to m
    r=r*1000.       #convert from km to m
    
    Patm=101325.     #ambient pressure (in Pa)
    rho=1.15    
    
    omega=7.292*10.**-5. #in rad/s, rotation rate of the Earth
    f=abs(2.*omega*sin(radians(lat1))) #Coriolis parameter
    
    P=P*100.        #convert from hPa (mb) to Pa
    
    vv=(Usurf+f*Rmax0/2.)**2.-f**2.*Rmax0**2./4.

    if P>=Patm or vv<=0:
      print(P,Usurf,vv)
      P=Patm
      Ps=P
      Vs=0
    
    else:     
      
      B=vv*np.exp(1)*rho/(Patm-P)     
      
      Ps=P+(Patm-P)*np.exp(-(Rmax0/r)**B)
      Vs=sqrt((Rmax0/r)**B*B*(Patm-P)/rho*np.exp(-(Rmax0/r)**B)+r**2.*f**2/4.)-f*r/2.
  
          
    return Vs,Ps

def Inflowangle(r,Rmax,lat0):
    """
    
    Calculate the inflow angle, used for the asymmetry in the wind field.
    
    Arguments:
        *r*    : Distance from the eye (km)
        *Rmax* : Radius to maximum winds (km)
        *lat0*: Latitude
    
    Returns:
        *beta* : Inflow angle at distance r from the eye (in radians)
            
    """
    if r<Rmax:
        beta=10.*(r/Rmax)
    elif r>= Rmax and r<1.2*Rmax:
        beta=10.+75.*((r/Rmax)-1.0)
    else:
        beta=25.
     
    if lat0>0.:
        beta=radians(90-beta)
    else:
        beta=radians(-90+beta)

    return beta
#%%
    