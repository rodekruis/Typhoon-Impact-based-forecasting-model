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
    ucl_data, plot_intensity, initialize, post_output

if platform == "linux" or platform == "linux2": #check if running on linux or windows os
    from typhoonmodel.utility_fun import Rainfall_data
elif platform == "win32":
    from typhoonmodel.utility_fun import Rainfall_data_window as Rainfall_data

decoder = Decoder()

initialize.setup_logger()
logger = logging.getLogger(__name__)

ECMWF_MAX_TRIES = 3
ECMWF_SLEEP = 30  # s


@click.command()
@click.option('--path', default='./', help='main directory')
@click.option('--remote_directory', default=None, help='remote directory for ECMWF forecast data') #'20210421120000'
@click.option('--typhoonname', default=None, help='name for active typhoon')
@click.option('--debug', is_flag=True, help='setting for DEBUG option')
@click.option('--debug_post', is_flag=True, help='setting for DEBUG-post_outputs() option')
def main(path,debug,debug_post,remote_directory,typhoonname):
    initialize.setup_cartopy()
    start_time = datetime.now()
    print('---------------------AUTOMATION SCRIPT STARTED---------------------------------')
    print(str(start_time))
    #%% check for active typhoons
    print('---------------------check for active typhoons---------------------------------')
    print(str(start_time))
    remote_dir = remote_directory

    if debug_post:
        typhoonname = 'CONSON'
        logger.info(f"DEBUGGING post_output() for typhoon{typhoonname}")
        Activetyphoon = [typhoonname]
        date_dir = start_time.strftime("%Y%m%d%H")
        Output_folder = os.path.join(path, f'forecast/Output/{date_dir}/Output/')
        post_output.post_output(Output_folder, Activetyphoon, debug_post)
        sys.exit()

    if debug:
        typhoonname = 'SURIGAE'
        remote_dir = '20210421120000'
        logger.info(f"DEBUGGING piepline for typhoon{typhoonname}")
        Activetyphoon = [typhoonname]
    else:
        # If passed typhoon name is None or empty string
        if not typhoonname:
            Activetyphoon = Check_for_active_typhoon.check_active_typhoon()
            if not Activetyphoon:
                logger.info("No active typhoon in PAR stop pipeline")
                sys.exit()
            logger.info(f"Running on active Typhoon(s) {Activetyphoon}")
        else:
            Activetyphoon = [typhoonname]
            remote_dir = remote_directory
            logger.info(f"Running on custom Typhoon {Activetyphoon}")

    Alternative_data_point = (start_time - timedelta(hours=24)).strftime("%Y%m%d")

    date_dir = start_time.strftime("%Y%m%d%H")
    Input_folder = os.path.join(path, f'forecast/Input/{date_dir}/Input/')
    Output_folder = os.path.join(path, f'forecast/Output/{date_dir}/Output/')

    if not os.path.exists(Input_folder):
        os.makedirs(Input_folder)
    if not os.path.exists(Output_folder):
        os.makedirs(Output_folder)   
    #download NOAA rainfall
    try:
        #Rainfall_data_window.download_rainfall_nomads(Input_folder,path,Alternative_data_point)
        Rainfall_data.download_rainfall_nomads(Input_folder,path,Alternative_data_point)
        rainfall_error=False
    except:
        traceback.print_exc()
        #logger.warning(f'Rainfall download failed, performing download in R script')
        logger.info(f'Rainfall download failed, performing download in R script')
        rainfall_error=True
    ###### download UCL data
      
    try:
        ucl_data.create_ucl_metadata(path, os.environ['UCL_USERNAME'], os.environ['UCL_PASSWORD'])
        ucl_data.process_ucl_data(path, Input_folder, os.environ['UCL_USERNAME'], os.environ['UCL_PASSWORD'])
    except:
        logger.info(f'UCL download failed')
    #%%
    ##Create grid points to calculate Winfield
    cent = Centroids()
    cent.set_raster_from_pnt_bounds((118,6,127,19), res=0.05)
    #this option is added to make the script scaleable globally To Do
    #cent.set_raster_from_pnt_bounds((LonMin,LatMin,LonMax,LatMax), res=0.05) 
    cent.check()
    cent.plot()
    ####
    admin=gpd.read_file(os.path.join(path,"./data-raw/phl_admin3_simpl2.geojson"))
    df = pd.DataFrame(data=cent.coord)
    df["centroid_id"] = "id"+(df.index).astype(str)  
    centroid_idx=df["centroid_id"].values
    ncents = cent.size
    df=df.rename(columns={0: "lat", 1: "lon"})
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    #df.to_crs({'init': 'epsg:4326'})
    df.crs = {'init': 'epsg:4326'}
    df_admin = sjoin(df, admin, how="left").dropna()
    
    # Sometimes the ECMWF ftp server complains about too many requests
    # This code allows several retries with some sleep time in between
    n_tries = 0
    while True:
        try:
            logger.info("Downloading ECMWF typhoon tracks")
            bufr_files = TCForecast.fetch_bufr_ftp(remote_dir=remote_dir)
            fcast = TCForecast()
            fcast.fetch_ecmwf(files=bufr_files)
        except ftplib.all_errors as e:
            n_tries += 1
            if n_tries >= ECMWF_MAX_TRIES:
                logger.error(f' Data downloading from ECMWF failed: {e}, '
                             f'reached limit of {ECMWF_MAX_TRIES} tries, exiting')
                sys.exit()
            logger.error(f' Data downloading from ECMWF failed: {e}, retrying after {ECMWF_SLEEP} s')
            time.sleep(ECMWF_SLEEP)
            continue
        break

    #%% filter data downloaded in the above step for active typhoons  in PAR
    # filter tracks with name of current typhoons and drop tracks with only one timestep
    fcast.data = [track_data_clean.track_data_clean(tr) for tr in fcast.data if (tr.time.size>1 and tr.name in Activetyphoon)]  
     
    # fcast.data = [tr for tr in fcast.data if tr.name in Activetyphoon]
    # fcast.data = [tr for tr in fcast.data if tr.time.size>1]    
    for typhoons in Activetyphoon:
        #typhoons=Activetyphoon[0]
        logger.info(f'Processing data {typhoons}')
        fname=open(os.path.join(path,'forecast/Input/',"typhoon_info_for_model.csv"),'w')
        fname.write('source,filename,event,time'+'\n')   
        if not rainfall_error:
            line_='Rainfall,'+'%srainfall' % Input_folder +',' +typhoons+','+ date_dir  #StormName #
            fname.write(line_+'\n')

        line_='Output_folder,'+'%s' % Output_folder +',' +typhoons+',' + date_dir  #StormName #
        #line_='Rainfall,'+'%sRainfall/' % Input_folder +','+ typhoons + ',' + date_dir #StormName #
        fname.write(line_+'\n')
    
        #typhoons='SURIGAE'  # to run it manually for any typhoon 
                    # select windspeed for HRS model
                    
        fcast.data=[tr for tr in fcast.data if tr.name==typhoons]
        tr_HRS=[tr for tr in fcast.data if (tr.is_ensemble=='False')]

        if tr_HRS !=[]:
            HRS_SPEED=(tr_HRS[0].max_sustained_wind.values/0.84).tolist()  ############# 0.84 is conversion factor for ECMWF 10MIN TO 1MIN AVERAGE
            dfff=tr_HRS[0].to_dataframe()
            dfff[['VMAX','LAT','LON']]=dfff[['max_sustained_wind','lat','lon']]
            dfff['YYYYMMDDHH']=dfff.index.values
            dfff['YYYYMMDDHH']=dfff['YYYYMMDDHH'].apply(lambda x: x.strftime("%Y%m%d%H%M") )
            dfff['STORMNAME']=typhoons
            dfff[['YYYYMMDDHH','VMAX','LAT','LON','STORMNAME']].to_csv(os.path.join(Input_folder,'ecmwf_hrs_track.csv'), index=False)
            dfff[['YYYYMMDDHH','VMAX','LAT','LON','STORMNAME']].to_csv(os.path.join(Output_folder,'ecmwf_hrs_track.csv'), index=False)
            line_='ecmwf,'+'%secmwf_hrs_track.csv' % Input_folder+ ',' +typhoons+','+ date_dir   #StormName #
            #line_='Rainfall,'+'%sRainfall/' % Input_folder +','+ typhoons + ',' + date_dir #StormName #
            fname.write(line_+'\n')        
            # Adjust track time step
            data_forced=[tr.where(tr.time <= max(tr_HRS[0].time.values),drop=True) for tr in fcast.data]
            # data_forced = [track_data_clean.track_data_force_HRS(tr,HRS_SPEED) for tr in data_forced] # forced with HRS windspeed
           
            #data_forced= [track_data_clean.track_data_clean(tr) for tr in fcast.data] # taking speed of ENS
            # interpolate to 3h steps from the original 6h
            #fcast.equal_timestep(3)
        else:
            len_ar=np.min([ len(var.lat.values) for var in fcast.data ])
            lat_ = np.ma.mean( [ var.lat.values[:len_ar] for var in fcast.data ], axis=0 )
            lon_ = np.ma.mean( [ var.lon.values[:len_ar] for var in fcast.data ], axis=0 )             
            YYYYMMDDHH =pd.date_range(fcast.data[0].time.values[0], periods=len_ar, freq="H")
            vmax_ = np.ma.mean( [ var.max_sustained_wind.values[:len_ar] for var in fcast.data ], axis=0 )
            d = {'YYYYMMDDHH':YYYYMMDDHH,
                 "VMAX":vmax_,
                 "LAT": lat_,
                  "LON": lon_} 
            dfff = pd.DataFrame(d)
            dfff['STORMNAME']=typhoons
            dfff['YYYYMMDDHH']=dfff['YYYYMMDDHH'].apply(lambda x: x.strftime("%Y%m%d%H%M") )
            dfff[['YYYYMMDDHH','VMAX','LAT','LON','STORMNAME']].to_csv(os.path.join(Input_folder,'ecmwf_hrs_track.csv'), index=False)
            line_='ecmwf,'+'%secmwf_hrs_track.csv' % Input_folder+ ',' +typhoons+','+ date_dir   #StormName #
            #line_='Rainfall,'+'%sRainfall/' % Input_folder +','+ typhoons + ',' + date_dir #StormName #
            fname.write(line_+'\n') 
            data_forced=fcast.data
        
        # calculate windfields for each ensamble
        threshold=0            #(threshold to filter dataframe /reduce data )
        df = pd.DataFrame(data=cent.coord)
        df["centroid_id"] = "id"+(df.index).astype(str)  
        centroid_idx=df["centroid_id"].values
        ncents = cent.size
        df=df.rename(columns={0: "lat", 1: "lon"})
        
        #calculate wind field for each ensamble members 
        list_intensity=[]
        distan_track=[]
        for tr in data_forced:
            logger.info(f"Running on ensemble # {tr.ensemble_number} for typhoon {tr.name}")
            track = TCTracks()
            typhoon = TropCyclone()
            track.data=[tr]
            #track.equal_timestep(3)
            tr=track.data[0]
            typhoon.set_from_tracks(track, cent, store_windfields=True)
            # Make intensity plot using the high resolution member
            if tr.is_ensemble == 'False':
                logger.info("High res member: creating intensity plot")
                plot_intensity.plot_inensity(typhoon=typhoon, event=tr.sid, output_dir=Output_folder,
                                             date_dir=date_dir, typhoon_name=tr.name)
            windfield=typhoon.windfields
            nsteps = windfield[0].shape[0]
            centroid_id = np.tile(centroid_idx, nsteps)
            intensity_3d = windfield[0].toarray().reshape(nsteps, ncents, 2)
            intensity = np.linalg.norm(intensity_3d, axis=-1).ravel()
            timesteps = np.repeat(track.data[0].time.values, ncents)
            #timesteps = np.repeat(tr.time.values, ncents)
            timesteps = timesteps.reshape((nsteps, ncents)).ravel()
            inten_tr = pd.DataFrame({
                    'centroid_id': centroid_id,
                    'value': intensity,
                    'timestamp': timesteps,})
            inten_tr = inten_tr[inten_tr.value > threshold]
            inten_tr['storm_id'] = tr.sid
            inten_tr['ens_id'] =tr.sid+'_'+str(tr.ensemble_number)
            inten_tr['name'] = tr.name
            inten_tr = (pd.merge(inten_tr, df_admin, how='outer', on='centroid_id')
                        .dropna()
                        .groupby(['adm3_pcode', 'ens_id'], as_index=False)
                        .agg({"value": ['count', 'max']}))
            inten_tr.columns = [x for x in ['adm3_pcode', 'storm_id', 'value_count', 'v_max']]
            list_intensity.append(inten_tr)
            distan_track1=[]
            for index, row in df.iterrows():
                dist=np.min(np.sqrt(np.square(tr.lat.values-row['lat'])+np.square(tr.lon.values-row['lon'])))
                distan_track1.append(dist*111)
            dist_tr = pd.DataFrame({'centroid_id': centroid_idx,'value': distan_track1})
            dist_tr['storm_id'] = tr.sid
            dist_tr['name'] = tr.name
            dist_tr['ens_id'] =tr.sid+'_'+str(tr.ensemble_number)
            dist_tr = (pd.merge(dist_tr, df_admin, how='outer', on='centroid_id')
                       .dropna()
                       .groupby(['adm3_pcode', 'name', 'ens_id'], as_index=False)
                       .agg({'value': 'min'}))
            dist_tr.columns = [x for x in ['adm3_pcode', 'name', 'storm_id',
                                                 'dis_track_min']]  # join_left_df_.columns.ravel()]
            distan_track.append(dist_tr)
        df_intensity_ = pd.concat(list_intensity)
        distan_track1 = pd.concat(distan_track)

        typhhon_df = pd.merge(df_intensity_, distan_track1,  how='left', on=['adm3_pcode','storm_id']) 
    
        typhhon_df.to_csv(os.path.join(Input_folder,'windfield.csv'), index=False)
    
        line_='windfield,'+'%swindfield.csv' % Input_folder+ ',' +typhoons+','+ date_dir   #StormName #
        #line_='Rainfall,'+'%sRainfall/' % Input_folder +','+ typhoons + ',' + date_dir #StormName #
        fname.write(line_+'\n')
        fname.close()
        
        
        #############################################################
        #### Run IBF model 
        #############################################################
        os.chdir(path)
        
        if platform == "linux" or platform == "linux2": #check if running on linux or windows os
            # linux
            try:
                p = subprocess.check_call(["Rscript", "run_model_V2.R", str(rainfall_error)])
            except subprocess.CalledProcessError as e:
                logger.error(f'failed to excute R sript')
                raise ValueError(str(e))
        elif platform == "win32": #if OS is windows edit the path for Rscript
            try:
                p = subprocess.check_call(["C:/Program Files/R/R-4.1.0/bin/Rscript", "run_model_V2.R", str(rainfall_error)])
            except subprocess.CalledProcessError as e:
                logger.error(f'failed to excute R sript')
                raise ValueError(str(e))
            
        #############################################################
        # send email in case of landfall-typhoon
        #############################################################

        image_filenames = list(Path(Output_folder).glob('*.png'))
        data_filenames = list(Path(Output_folder).glob('*.csv'))

        if image_filenames or data_filenames:
            message_html = """\
            <html>
            <body>
            <h1>IBF model run result </h1>
            <p>Please find attached a map and data with updated model run</p>
            <img src="cid:Impact_Data">
            </body>
            </html>
            """
            Sendemail.sendemail(
                smtp_server=os.environ["SMTP_SERVER"],
                smtp_port=int(os.environ["SMTP_PORT"]),
                email_username=os.environ["EMAIL_LOGIN"],
                email_password=os.environ["EMAIL_PASSWORD"],
                email_subject='Updated impact map for a new Typhoon in PAR',
                from_address=os.environ["EMAIL_FROM"],
                to_address_list=os.environ["EMAIL_TO_LIST"].split(','),
                cc_address_list=os.environ["EMAIL_CC_LIST"].split(','),
                message_html=message_html,
                filename_list=image_filenames + data_filenames
            )
        else:
            raise FileNotFoundError(f'No .png or .csv found in {Output_folder}')
                ##################### upload model output to 510 datalack ##############
        
        file_service = FileService(account_name=os.environ["AZURE_STORAGE_ACCOUNT"],protocol='https', connection_string=os.environ["AZURE_CONNECTING_STRING"])
        file_service.create_share('forecast')
        OutPutFolder=date_dir
        file_service.create_directory('forecast', OutPutFolder) 
        
        for img_file in image_filenames:   
            file_service.create_file_from_path('forecast', OutPutFolder,os.fspath(img_file.parts[-1]),img_file, content_settings=ContentSettings(content_type='image/png'))

        for data_file in data_filenames:
            file_service.create_file_from_path('forecast', OutPutFolder,os.fspath(data_file.parts[-1]),data_file, content_settings=ContentSettings(content_type='text/csv'))
            
        ##################### upload model input(Rainfall+wind intensity) to 510 datalack ############## 
        # To DO
        

    print('---------------------AUTOMATION SCRIPT FINISHED---------------------------------')
    print(str(datetime.now()))


#%%#Download rainfall (old pipeline)
#automation_sript(path)
if __name__ == "__main__":
    main()
