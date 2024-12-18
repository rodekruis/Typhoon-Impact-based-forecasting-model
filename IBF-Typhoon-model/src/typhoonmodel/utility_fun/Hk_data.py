import re
import shutil
import sys
import os
from lxml import etree
from os.path import relpath
from bs4 import BeautifulSoup
import requests
from os import listdir
from os.path import isfile, join
from sys import platform

import pandas as pd
from datetime import datetime ,timedelta
import xarray as xr
import numpy as np


parser="lxml"
def download_hk(Input_folder):
    HKfeed =  BeautifulSoup(requests.get('https://www.weather.gov.hk/wxinfo/currwx/tc_list.xml').content,parser=parser,features="lxml").find_all('tropicalcycloneurl')
    tracks={}
    try:
        for url_link in HKfeed:
            url_link=url_link.text
            data=[]
            HK_track=BeautifulSoup(requests.get(url_link).content,parser=parser,features="lxml").find_all('weatherreport')
            event_name=url_link.split('/')[-1].split('.')[0]
            print(event_name)
            
            for WeatherReport in HK_track:#.find_all('weatherreport'):
                for forecast in WeatherReport.find_all('forecastinformation'):
                    l2=[]
                    for elment in ['index','latitude','longitude','intensity','maximumwind','time']:
                        try:
                            #l2[elment]=[index.text for index in WeatherReport.find_all(elment)]
                            l2.append(forecast.find(elment).text)
                        except: 
                            pass
                    data.append(l2)
            
                       
            df = pd.DataFrame(data,columns=['timestep','latitude','longitude','intensity','maximumwind','time'])#.rename(columns={0:'timestep',1:'lat',2:'lon',3:'forecast_time'}) 
            
            finat_time_stamp=datetime.strptime(df['time'].values[-1], '%Y-%m-%dT%H:%M:%S+00:00')-timedelta(hours=int(df['timestep'].values[-1]))
            forecast_times=[]
            for i in df['timestep'].values:
                forecast_times.append(finat_time_stamp+timedelta(hours=int(i)))

            df['time']=forecast_times
            df['maximumwind']=df['maximumwind'].bfill(axis =0)
            
            lons = [sub.replace('E', "").strip() for sub in df.longitude.values]
            lats = [sub.replace('N', "").strip() for sub in df.latitude.values]
            wind = [sub.replace('km/h', "").strip() for sub in df.maximumwind.values]
            
            df['maximumwind]=wind
            df['longitude]=lons
            df['latitude]=lats
            df.to_csv(os.path.join(Input_folder,f'HK_forecast_{event_name}.csv'),index=True) 

            
            track = xr.Dataset(
                data_vars={
                    'max_sustained_wind': ('time', wind),
                    'environmental_pressure': ('time', np.full(len(df.time.values),1006).tolist()),
                    #'central_pressure': ('time', pd.Series(forcast_df.central_pressure.values).interpolate().tolist()),
                    'lat': ('time', lats),
                    'lon': ('time', lons),
                    #'radius_max_wind': ('time', pd.Series(forcast_df.radius_max_wind.values).interpolate().tolist()),
                    #'max_radius':('time', pd.Series(forcast_df.max_radius.values).interpolate().tolist()),  
                    #'radius_max_wind': ('time', pd.Series(estimate_rmw(forcast_df.radius_max_wind.values, forcast_df.central_pressure.values)).interpolate().tolist()),
                    #'radius_oci':('time', estimate_roci(forcast_df.radius_max_wind.values, forcast_df.central_pressure.values)), 
                    #'time_step':('time', forcast_df.time_step.values),
                    'time_step':('time', np.full(len(df.time.values),1).tolist())
                },
                coords={
                    'time': df.time.values,
                },
                attrs={
                    'max_sustained_wind_unit':'km/h', 
                    #'name': forcast_df.name,
                }
            )
        tracks[event_name]=track
    except:
        pass
        





    