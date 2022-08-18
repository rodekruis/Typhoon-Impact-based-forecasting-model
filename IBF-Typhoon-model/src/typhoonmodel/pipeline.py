#!/bin/sh

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:01:00 2020

@author: ATeklesadik
"""
import time
import ftplib
import os
import sys
from datetime import datetime, timedelta
from sys import platform
import subprocess
import logging
import traceback
from pathlib import Path
from azure.storage.file import FileService
from azure.storage.file import ContentSettings

import pandas as pd
from pybufrkit.decoder import Decoder
import numpy as np
from geopandas.tools import sjoin
import geopandas as gpd
import click

from climada.hazard import Centroids, TropCyclone,TCTracks
from climada.hazard.tc_tracks_forecast import TCForecast
from typhoonmodel.utility_fun import track_data_clean, Check_for_active_typhoon, Sendemail, \
    ucl_data, plot_intensity, initialize

if platform == "linux" or platform == "linux2": #check if running on linux or windows os
    from typhoonmodel.utility_fun import Rainfall_data
elif platform == "win32":
    from typhoonmodel.utility_fun import Rainfall_data_window as Rainfall_data
from typhoonmodel.utility_fun.forecast_process import Forecast
decoder = Decoder()

initialize.setup_logger()
logger = logging.getLogger(__name__)

@click.command()
@click.option('--path', default='./', help='main directory')
@click.option('--remote_directory', default=None, help='remote directory for ECMWF forecast data') #'20210421120000'
@click.option('--typhoonname', default=None, help='name for active typhoon')
@click.option('--typhoonname', default=None, help='name for active typhoon')
@click.option('--debug', is_flag=True, help='setting for DEBUG option')

def main(path,debug,remote_directory,typhoonname):
    initialize.setup_cartopy()
    start_time = datetime.now()
    landfall_time='NA'
    landfall_location_manucipality='NA'
    EAP_TRIGGERED='no'
    print('---------------------AUTOMATION SCRIPT STARTED---------------------------------')
    print(str(start_time))
    #%% check for active typhoons
    print('---------------------check for active typhoons---------------------------------')
    print(str(start_time))
    remote_dir = remote_directory
    main_path=path
    if debug:
        typhoonname = 'CHANTHU'
        remote_dir = '20210910120000'
        logger.info(f"DEBUGGING piepline for typhoon{typhoonname}")  
    fc = Forecast(main_path,remote_dir,typhoonname, countryCodeISO3='PHP', admin_level=3)
    #fc.data_filenames_list
    #fc.image_filenames_list
    landfall_time='NA'
    landfall_location_manucipality='NA'
    EAP_TRIGGERED='no'
    if not fc.Activetyphoon: #if it is not empty   
        for typhoon_names in fc.Activetyphoon:
            #adm3_pcode,storm_id,is_ensamble,value_count,v_max,name,dis_track_min
            typhoon_wind=fc.typhhon_wind_data[typhoon_names]
            typhoon_wind=typhoon_wind.query('is_ensamble=="False"')['adm3_pcode','v_max']
            #max_06h_rain,max_24h_rain,Mun_Code
            typhoon_rainfall=fc.rainfall_data[typhoon_names]['Mun_Code','max_24h_rain']
            #YYYYMMDDHH,VMAX,LAT,LON,STORMNAME
            typhoon_track=fc.hrs_track_data[typhoon_names]
            landfall_location=fc.landfall_location[typhoon_names]
            if landfall_location:#if dict is not empty         
                landfall_time=list(landfall_location.items())[0][0] #'YYYYMMDDHH'
                landfall_location_manucipality=list(landfall_location.items())[0][1]['adm3_pcode'] #pcode?       
            
            #"","Typhoon_name",">=100k",">=80k",">=70k",">=50k",">=30k","trigger"
            with open (fc.Output_folder+"trigger_"+fc.date_dir+"_"+typhoon_names+".csv") as csv_file1:
                trigger=pd.read_csv(csv_file1)
                if trigger["trigger"].values[0]==1:
                    EAP_TRIGGERED='yes'
            #"","adm3_en","glat","adm3_pcode","adm2_pcode","adm1_pcode","glon","GEN_mun_code","probability_dist50","impact","WEA_dist_track"
            with open (fc.Output_folder+"Average_Impact_"+fc.date_dir+"_"+typhoon_names+".csv") as csv_file2:
                impact=pd.read_csv(csv_file2)
                impact_df=impact["adm3_pcode","probability_dist50","impact","WEA_dist_track"]
 


    print('---------------------AUTOMATION SCRIPT FINISHED---------------------------------')
    print(str(datetime.now()))


#%%#Download rainfall (old pipeline)
#automation_sript(path)
if __name__ == "__main__":
    main()
