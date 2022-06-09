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
import json
from shapely import wkb, wkt

from climada.hazard import Centroids, TropCyclone,TCTracks
from climada.hazard.tc_tracks_forecast import TCForecast
from typhoonmodel.utility_fun import track_data_clean, Check_for_active_typhoon, Sendemail, \
    ucl_data, plot_intensity, initialize,settings
    
from typhoonmodel.utility_fun.dynamicDataDb import DatabaseManager    

if platform == "linux" or platform == "linux2": #check if running on linux or windows os
    from typhoonmodel.utility_fun import Rainfall_data
elif platform == "win32":
    from typhoonmodel.utility_fun import Rainfall_data_window as Rainfall_data
decoder = Decoder()
initialize.setup_logger()
logger = logging.getLogger(__name__)


class Forecast:
    def __init__(self,main_path, remote_dir,typhoonname, countryCodeISO3, admin_level):
        self.TyphoonName = typhoonname
        self.admin_level = admin_level
        #self.db = DatabaseManager(leadTimeLabel, countryCodeISO3,admin_level)
        self.remote_dir = remote_dir
        #self.TriggersFolder = TRIGGER_DATA_FOLDER_TR
        #self.levels = SETTINGS[countryCodeISO3]['levels']        
        #Activetyphoon = Check_for_active_typhoon.check_active_typhoon()
        start_time = datetime.now()
        self.UCL_USERNAME='UCL_USERNAME'
        self.UCL_PASSWORD='UCL_USERNAME'
        self.ECMWF_MAX_TRIES = 3
        self.ECMWF_SLEEP = 30  # s
        self.main_path=main_path
        if not typhoonname:
            Activetyphoon = Check_for_active_typhoon.check_active_typhoon()
            if not Activetyphoon:
                logger.info("No active typhoon in PAR stop pipeline")
                sys.exit()
            logger.info(f"Running on active Typhoon(s) {Activetyphoon}")
        else:
            Activetyphoon = [typhoonname]
        self.Activetyphoon = Activetyphoon
        Alternative_data_point = (start_time - timedelta(hours=24)).strftime("%Y%m%d")
        date_dir = start_time.strftime("%Y%m%d%H")
        Input_folder = os.path.join(main_path, f'forecast/Input/{date_dir}/Input/')
        Output_folder = os.path.join(main_path, f'forecast/Output/{date_dir}/Output/')
        if not os.path.exists(Input_folder):
            os.makedirs(Input_folder)
        if not os.path.exists(Output_folder):
            os.makedirs(Output_folder)
        self.Alternative_data_point =Alternative_data_point 
        self.date_dir = date_dir
        self.Input_folder = Input_folder
        self.Output_folder = Output_folder
        
        #download NOAA rainfall
        try:
            #Rainfall_data_window.download_rainfall_nomads(Input_folder,path,Alternative_data_point)
            Rainfall_data.download_rainfall_nomads(self.Input_folder,self.main_path,self.Alternative_data_point)
            rainfall_error=False
        except:
            traceback.print_exc()
            #logger.warning(f'Rainfall download failed, performing download in R script')
            logger.info(f'Rainfall download failed, performing download in R script')
            rainfall_error=True
        ###### download UCL data
          
        try:
            ucl_data.create_ucl_metadata(self.main_path, 
            self.UCL_USERNAME,self.UCL_PASSWORD)
            ucl_data.process_ucl_data(self.main_path,
            self.Input_folder,self.UCL_USERNAME,self.UCL_PASSWORD)
        except:
            logger.info(f'UCL download failed')        


        ##Create grid points to calculate Winfield
        cent = Centroids()
        cent.set_raster_from_pnt_bounds((118,6,127,19), res=0.05)
        cent.check()
        cent.plot()
        admin=gpd.read_file(os.path.join(self.main_path,"./data-raw/phl_admin3_simpl2.geojson"))
        df = pd.DataFrame(data=cent.coord)
        df["centroid_id"] = "id"+(df.index).astype(str)  
        centroid_idx=df["centroid_id"].values
        ncents = cent.size
        df=df.rename(columns={0: "lat", 1: "lon"})
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
        admin.set_crs(epsg=4326, inplace=True)
        df.set_crs(epsg=4326, inplace=True)

        df_admin = sjoin(df, admin, how="left").dropna()
        self.df_admin=df_admin
        # Sometimes the ECMWF ftp server complains about too many requests
        # This code allows several retries with some sleep time in between
        n_tries = 0
        while True:
            try:
                logger.info("Downloading ECMWF typhoon tracks")
                bufr_files = TCForecast.fetch_bufr_ftp(remote_dir=self.remote_dir)
                fcast = TCForecast()
                fcast.fetch_ecmwf(files=bufr_files)
            except ftplib.all_errors as e:
                n_tries += 1
                if n_tries >= self.ECMWF_MAX_TRIES:
                    logger.error(f' Data downloading from ECMWF failed: {e}, '
                                 f'reached limit of {self.ECMWF_MAX_TRIES} tries, exiting')
                    sys.exit()
                logger.error(f' Data downloading from ECMWF failed: {e}, retrying after {self.ECMWF_SLEEP} s')
                time.sleep(self.ECMWF_SLEEP)
                continue
            break

        #%% filter data downloaded in the above step for active typhoons  in PAR
        # filter tracks with name of current typhoons and drop tracks with only one timestep
        fcast_data = [track_data_clean.track_data_clean(tr) for tr in fcast.data if (tr.time.size>1 and tr.name in Activetyphoon)]  
        self.fcast_data=fcast_data
        
