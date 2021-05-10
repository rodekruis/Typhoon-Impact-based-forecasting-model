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
import xml.etree.ElementTree as ET
import lxml.etree as ET2
import pandas as pd
from datetime import datetime
def jtcw_data(Input_folder):
    parser = ET2.XMLParser(recover=True)#
    output=[] 
    index_list=[' WARNING POSITION:',' 12 HRS, VALID AT:',' 24 HRS, VALID AT:',' 36 HRS, VALID AT:',' 48 HRS, VALID AT:',' 72 HRS, VALID AT:',' 96 HRS, VALID AT:',' 120 HRS, VALID AT:']
    index_list_id=[]
    index_list_wd=[]
    
    jtwc_content = BeautifulSoup(requests.get('https://www.metoc.navy.mil/jtwc/rss/jtwc.rss').content,parser=parser,features="lxml")#'html.parser')
    try:    
        for channel in jtwc_content.find_all('channel'):
            for item in channel.find_all('item'):
                for li in item.find_all('li'):
                    for href_li in li.find_all('a',href=True):
                        if href_li.text =='TC Warning Text ':
                            output.append(href_li['href'])        
        output=[event for event in output if event.split('/')[-1][0:2]=='wp']
        jtwc_content = BeautifulSoup(requests.get(output[0]).content,'html.parser')#parser=parser,features="lxml")#'html.parser')
        jtwc_=re.sub(' +', ' ', jtwc_content.text)
        listt=jtwc_.split('\r\n')
        storm_name=[word.split(' ')[4][1:-1] for word in listt if word.startswith('1. TROPICAL STORM')]
        storm_id=[word.split(' ')[3] for word in listt if word.startswith('1. TROPICAL STORM')]
        listt=listt[listt.index(' WARNING POSITION:'):]   

        for i in index_list:            
            if i in listt:
                index_list_id.append(listt[listt.index(i)+1].replace("NEAR ", "").replace("---", ","))            
        for i in listt:
            if (' '.join(i.split()[0:3])=='MAX SUSTAINED WINDS'):
                i_l=i.replace(",", "").split()
                index_list_wd.append(','.join([i_l[-5],i_l[-2]]))              
        jtwc_df = pd.DataFrame(index_list_id)
        sustained_wind=[float(item.split(',')[0]) for item in index_list_wd]
        Gust_wind=[float(item.split(',')[1]) for item in index_list_wd]
        easting=[float(item.split(',')[1].split(' ')[2][:-1]) for item in index_list_id]
        northing=[float(item.split(',')[1].split(' ')[1][:-1]) for item in index_list_id]
        YYYYMM='{0:02d}'.format(date.today().year)+'{0:02d}'.format(date.today().month)
        YYYYMMDDHH=[YYYYMM+item.split(',')[0].replace(" ","")[:-1] for item in index_list_id]
        
        YYYYMMDDHH=[datetime.strptime(item, "%Y%m%d%H%M").strftime("%Y%m%d%H%M") for item in YYYYMMDDHH]
        jtwc_df = pd.DataFrame(list(zip(YYYYMMDDHH,easting,northing,sustained_wind,Gust_wind)),
                  columns =['YYYYMMDDHH','LON','LAT','VMAX','GUST'])
        jtwc_df['STORMNAME']=storm_name*len(sustained_wind)
        #jtwc_.split('\r\n')[2].strip('/')  name of the event 
        jtwc_df.to_csv(os.path.join(Input_folder,'JTCW_%s_%s.csv'%(Input_folder.split('/')[-3],Input_folder.split('/')[-4])),index=True)  
        #jtwc_df.to_csv('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/philipiness/jtwc_df.csv',index=False)
    except:
        pass

#%%

track = xr.Dataset(
        data_vars={
            'max_sustained_wind': ('time', sustained_wind),
           # 'environmental_pressure': ('time', forcast_df.environmental_pressure.values),
            #'central_pressure': ('time', forcast_df.central_pressure.values),
            'lat': ('time', northing),
            'lon': ('time', easting),
            #'radius_max_wind':('time', estimate_rmw(forcast_df.radius_max_wind.values, forcast_df.central_pressure.values)),  
           # 'radius_oci':('time', estimate_roci(forcast_df.radius_max_wind.values, forcast_df.central_pressure.values)), 
            'time_step':('time', forcast_df.time_step.values),
        },
        coords={
            'time': YYYYMMDDHH,
        },
        attrs={
            'max_sustained_wind_unit': 'kn',
            'central_pressure_unit': 'mb',
            'name': storm_name[0],
            'sid': storm_id[0],
            'orig_event_flag': True,#forcast_df.orig_event_flag,
            'data_provider': 'JTCW',
            #'id_no': storm_id[0],
            #'ensemble_number': forcast_df.ensemble_number,
            'is_ensemble':False,
            #'forecast_time': forcast_df.forecast_time,
            'basin': 'wp',
           # 'category': forcast_df.category,
        }
    )
track = track.set_coords(['lat', 'lon'])
    #%%
date.today().month

 '{0:2d}'.format(pi)
 