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
import json
import zipfile
from climada.hazard import Centroids, TropCyclone,TCTracks
from shapely.geometry import Point, Polygon, MultiPolygon, box

from typhoonmodel.utility_fun import track_data_clean, Check_for_active_typhoon, Sendemail, plot_intensity, initialize
''' 
if platform == "linux" or platform == "linux2": #check if running on linux or windows os
    from typhoonmodel.utility_fun import Rainfall_data
elif platform == "win32":
    from typhoonmodel.utility_fun import Rainfall_data_window as Rainfall_data
'''
from typhoonmodel.utility_fun.forecast_process import Forecast
from typhoonmodel.utility_fun.tc_tracks_forecast import TCForecast
decoder = Decoder()
import requests
from fiona.crs import from_epsg
from climada.util import coordinates  
from typhoonmodel.utility_fun.settings import *
import glob
from typhoonmodel.utility_fun.dynamicDataDb import DatabaseManager 
initialize.setup_logger()
logger = logging.getLogger(__name__)

# Set root logger to log only ERROR and above
logging.basicConfig(level=logging.ERROR)
# Suppress logging from 'requests' and 'urllib3' libraries
#logging.getLogger("requests").setLevel(logging.WARNING)
#logging.getLogger("urllib3").setLevel(logging.WARNING)
def main():
    initialize.setup_cartopy()
    start_time = datetime.now()   
    ############## Defult variables which will be updated if a typhoon is active 
    logger.info('AUTOMATION SCRIPT STARTED')
    logger.info(f'{IBF_API_URL}')
    
    #logger.info(f'simulation started at {start_time}')
    try: 
        for countryCodeISO3 in countryCodes:
            logger.info(f"running piepline for {countryCodeISO3}")  
            admin_level=SETTINGS_SECRET[countryCodeISO3]["admin_level"]
            mock=SETTINGS_SECRET[countryCodeISO3]["mock"]
            mock_nontrigger_typhoon_event=SETTINGS_SECRET[countryCodeISO3]["mock_nontrigger_typhoon_event"]
            mock_trigger_typhoon_event=SETTINGS_SECRET[countryCodeISO3]["mock_trigger_typhoon_event"]
            mock_trigger=SETTINGS_SECRET[countryCodeISO3]["if_mock_trigger"]            
            # Download data 
            #dbm_ = DatabaseManager(countryCodeISO3,admin_level)
            db = DatabaseManager(countryCodeISO3,admin_level)            
            filename='data.zip'
            path = 'typhoon/Gold/ibfdatapipeline/'+ filename
            #admin_area_json1['geometry'] = admin_area_json1.pop('geom')
            DataFile = db.getDataFromDatalake(path)
            if DataFile.status_code >= 400:
                raise ValueError()
            open('./' + filename, 'wb').write(DataFile.content)
            path_to_zip_file='./'+filename
            
            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall('./data') 
            logger.info('finished data download')
            
    

            ###############################################################
            ####  check setting for mock data 
            if mock:
                if mock_trigger:
                    typhoon_names=mock_trigger_typhoon_event
                else:
                    typhoon_names=mock_nontrigger_typhoon_event                
                logger.info(f"mock piepline for typhoon{typhoon_names}")
                
                json_path = mock_data_path  + typhoon_names             
                db.uploadTrackData(json_path)            
                db.uploadTyphoonData(json_path)
                db.sendNotificationTyphoon()

            else:
                #output_folder=Output_folder
                #active_Typhoon_event_list=Active_Typhoon_event_list
                fc = Forecast(countryCodeISO3, admin_level)
                logger.info('_________________finished data processing______________')
                
                if fc.Activetyphoon_landfall: #if it is not empty
                    for typhoon_names in fc.Activetyphoon_landfall.keys():   
                    #for typhoon_names in fc.Activetyphoon:
                        logger.info(f'_________________upload data for {typhoon_names}______________')
                        if fc.Activetyphoon_landfall[typhoon_names] in ['madelandfall','notmadelandfall']:#=='notmadelandfall':
                            # upload data
                            json_path = fc.Output_folder  + typhoon_names  
                            #EAP_TRIGGERED_bool=fc.eap_status_bool[typhoon_names]
                            #EAP_TRIGGERED=fc.eap_status[typhoon_names]    
                            logger.info('__________upload typhoon data_____') 
                            fc.db.uploadTyphoonData(json_path) 
                            logger.info('__________upload track data_____') 
                            fc.db.uploadTrackData(json_path)
                            logger.info('__________upload data to IBF system_____')                                          
                            states=fc.db.postResulToDatalake()
                            forecast_directory=typhoon_names + fc.forecast_time
                            logger.info('__________upload data to datalack 1_____')                            
                            fc.db.postDataToDatalake(datalakefolder=forecast_directory)
                            logger.info('__________upload data to datalack 2_____') 
                            fc.db.postDataToDatalake(datalakefolder=typhoon_names)


                            #fc.db.uploadImage(typhoons=typhoon_names,eventName=typhoon_names)
                            #fc.db.sendNotificationTyphoon() 
                            try:
                                if states==1:
                                    logger.info('posting to skype')
                                    fc.db.postResulToSkype(skypUsername,skypPassword,channel_id)                                    
                            except:
                                pass
                            
                        elif fc.Activetyphoon_landfall[typhoon_names]=='madelandfall_':#'to ignore ':
                            logger.info(f'typhoon{typhoon_names} already made landfall getting data for previous model run')   
                            try:                                                        
                                fc.db.getDataFromDatalake2(datalakefolder=typhoon_names)                           
                                logger.info(f'getting previous model run result from datalake  complete')
                                json_path = fc.Output_folder  + typhoon_names                          
                                fc.db.uploadTrackDataAfterlandfall(json_path)                             
                                fc.db.uploadTyphoonDataAfterlandfall(json_path)
                            except:
                                pass
                            
                        elif fc.Activetyphoon_landfall[typhoon_names]=='Farfromland':   
                            logger.info(f'uploadng data for event far from land ')
                            json_path = fc.Output_folder  + typhoon_names  
                            fc.db.uploadTrackData(json_path)
                            fc.db.uploadTyphoonDataNoLandfall(json_path)
                            
                        #elif len(fc.Activetyphoon_landfall) == 0:
                        elif fc.Activetyphoon_landfall[typhoon_names]=='noEvent':
                            logger.info(f'uploadng data for no active Typhoon ') 
                            df_total_upload=fc.pcode.copy()  #data frame with pcodes 
                            typhoon_names='null'
                            df_total_upload['alert_threshold']=0
                            df_total_upload['affected_population']=0  
                            df_total_upload['houses_affected']=0                     
                            for layer in ["affected_population","houses_affected","alert_threshold"]:
                                exposure_entry=[]
                                # prepare layer
                                logger.info(f"preparing data for {layer}")
                                #exposure_data = {'countryCodeISO3': countrycode}
                                exposure_data = {"countryCodeISO3": "PHL"}
                                exposure_place_codes = []
                                #### change the data frame here to include impact
                                for ix, row in df_total_upload.iterrows():
                                    exposure_entry = {"placeCode": row["adm3_pcode"],
                                                    "amount": row[layer]}
                                    exposure_place_codes.append(exposure_entry)
                                    
                                exposure_data["exposurePlaceCodes"] = exposure_place_codes
                                exposure_data["adminLevel"] = admin_level
                                exposure_data["leadTime"] = "72-hour" #landfall_time_hr
                                exposure_data["dynamicIndicator"] = layer
                                exposure_data["disasterType"] = "typhoon"
                                exposure_data["eventName"] = None                     
                                json_file_path = fc.Output_folder  + f'null_{layer}' + '.json'
                                
                                with open(json_file_path, 'w') as fp:
                                    json.dump(exposure_data, fp)
                                    
                            #upload typhoon data        
                            json_path = fc.Output_folder
                            fc.db.uploadTyphoonData_no_event(json_path)
                            #fc.db.uploadTrackData(json_path)   
                    fc.db.sendNotificationTyphoon()
                else: ##if there is no active typhoon    
                    logger.info('no active Typhoon')
                    df_total_upload=fc.pcode  #data frame with pcodes 
                    typhoon_names='null'
                    df_total_upload['alert_threshold']=0
                    df_total_upload['affected_population']=0    
                    df_total_upload['houses_affected']=0                     
                    for layer in ["affected_population",'houses_affected',"alert_threshold"]:
                        exposure_entry=[]
                        # prepare layer
                        logger.info(f"preparing data for {layer}")
                        #exposure_data = {'countryCodeISO3': countrycode}
                        exposure_data = {"countryCodeISO3": "PHL"}
                        exposure_place_codes = []
                        #### change the data frame here to include impact
                        for ix, row in df_total_upload.iterrows():
                            exposure_entry = {"placeCode": row["adm3_pcode"],
                                              "amount": row[layer]}
                            exposure_place_codes.append(exposure_entry)
                            
                        exposure_data["exposurePlaceCodes"] = exposure_place_codes
                        exposure_data["adminLevel"] = admin_level
                        exposure_data["leadTime"] = "72-hour" #landfall_time_hr
                        exposure_data["dynamicIndicator"] = layer
                        exposure_data["disasterType"] = "typhoon"
                        exposure_data["eventName"] = None                     
                        json_file_path = fc.Output_folder  + f'null_{layer}' + '.json'
                        
                        with open(json_file_path, 'w') as fp:
                            json.dump(exposure_data, fp)
                            
                    #upload typhoon data        
                    json_path = fc.Output_folder
                    fc.db.uploadTyphoonData_no_event(json_path)   
                 
           
    except Exception as e:
        logger.error("Typhoon Data PIPELINE ERROR")
        logger.error(e)
    #elapsedTime = str(time.time() - start_time)
    #logger.info('simulation finished')#str(elapsedTime))
    
 
 
if __name__ == "__main__":
    main()
