#!/bin/sh

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:01:00 2020

@author: ATeklesadik
"""
import sys
import os
from datetime import datetime, timedelta
from sys import platform
import subprocess
import logging
import traceback

import pandas as pd
from pybufrkit.decoder import Decoder
import numpy as np
from geopandas.tools import sjoin
import geopandas as gpd

# Set up logger
level = logging.INFO
logging.basicConfig(level=level,
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)
# Stop some overly verbose packages
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name != '<module name>':
        logging.getLogger(log_name).setLevel(max(logging.WARNING, level))

decoder = Decoder()
#path='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/'
path='/home/fbf/'

#%%
sys.path.insert(0, path+'lib')
os.chdir(path)

from settings import fTP_LOGIN, fTP_PASSWORD, uCL_USERNAME, uCL_PASSWORD
#from settings import *
#from secrets import *
#from variables import *
 
from climada.hazard import Centroids, TropCyclone,TCTracks
from climada.hazard.tc_tracks import estimate_roci,estimate_rmw
from climada.hazard.tc_tracks_forecast import TCForecast
from utility_fun import track_data_clean,Rainfall_data,Check_for_active_typhoon,Sendemail,ucl_data

#%% check for active typhoons
print('---------------------check for active typhoons---------------------------------')
print(str(datetime.now()))
#### the following code checks if there are active typhoons around the philippines 

Activetyphoon=Check_for_active_typhoon.check_active_typhoon()
#forecast_available_fornew_typhoon= False#'False'

#### if there are no typhoons use SURIGAE data for testing the pipeline 

if Activetyphoon==[]:
  remote_dir='20210421120000' #for downloading test data otherwise set it to None
  Activetyphoon=['SURIGAE']  #name of typhoon for test
else:
  #remote_dir=None #for downloading real time data
  remote_dir=None #'20210518120000' #for downloading test data  Activetyphoon=['SURIGAE']
  
print("currently active typhoon list= %s"%Activetyphoon)


#%% Download Rainfaall

##### create directories for input/output  the new typhoon 


Alternative_data_point=(datetime.strptime(datetime.now().strftime("%Y%m%d%H"), "%Y%m%d%H")-timedelta(hours=24)).strftime("%Y%m%d")
     
Input_folder=os.path.join(path,'forecast/Input/%s/Input/'%(datetime.now().strftime("%Y%m%d%H")))
Output_folder=os.path.join(path,'forecast/Output/%s/Output/'%(datetime.now().strftime("%Y%m%d%H")))

if not os.path.exists(Input_folder):
    os.makedirs(Input_folder)
if not os.path.exists(Output_folder):
    os.makedirs(Output_folder)   


#download NOAA rainfall


try:
    Rainfall_data.download_rainfall_nomads(Input_folder,path,Alternative_data_point)
    rainfall_error=False
except:
    traceback.print_exc()
    logger.warning(f'Rainfall download failed, performing download in R script')
    rainfall_error=True

###### download UCL data
    

try:
    ucl_data.create_ucl_metadata(path,uCL_USERNAME,uCL_PASSWORD)
    ucl_data.process_ucl_data(path,Input_folder,uCL_USERNAME,uCL_PASSWORD)

except:
    pass


#%%

##Create grid points to calculate Winfield

cent = Centroids()
cent.set_raster_from_pnt_bounds((118,6,127,19), res=0.05)
cent.check()
cent.plot()
####

admin=gpd.read_file("./data-raw/phl_admin3_simpl2.shp")
df = pd.DataFrame(data=cent.coord)
df["centroid_id"] = "id"+(df.index).astype(str)  
centroid_idx=df["centroid_id"].values
ncents = cent.size
df=df.rename(columns={0: "lat", 1: "lon"})

df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
#df.to_crs({'init': 'epsg:4326'})
df.crs = {'init': 'epsg:4326'}
df_admin = sjoin(df, admin, how="left")
df_admin=df_admin.dropna()

#%% Download ECMWF forecast for typhoon tracks 

bufr_files = TCForecast.fetch_bufr_ftp(remote_dir=remote_dir)
fcast = TCForecast()
fcast.fetch_ecmwf(files=bufr_files)

#%% filter data downloaded in the above step for active typhoons  in PAR

# filter tracks with name of current typhoons and drop tracks with only one timestep
fcast.data = [tr for tr in fcast.data if tr.name in Activetyphoon]
fcast.data = [tr for tr in fcast.data if tr.time.size>1]

 

def automation_sript(path):
    print('---------------------AUTOMATION SCRIPT STARTED---------------------------------')
    print(str(datetime.now()))   
    for typhoons in Activetyphoon:
        fname=open(os.path.join(path,'forecast/Input/',"typhoon_info_for_model.csv"),'w')
        fname.write('source,filename,event,time'+'\n')            
        line_='Rainfall,'+'%sRainfall' % Input_folder +',' +typhoons+','+ datetime.now().strftime("%Y%m%d%H")  #StormName #
        fname.write(line_+'\n')
        line_='Output_folder,'+'%s' % Output_folder +',' +typhoons+',' + datetime.now().strftime("%Y%m%d%H")  #StormName #
        #line_='Rainfall,'+'%sRainfall/' % Input_folder +','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H") #StormName #
        fname.write(line_+'\n')
    
        #typhoons='SURIGAE'  # to run it manually for any typhoon 
                    # select windspeed for HRS model
                    
        fcast.data=[tr for tr in fcast.data if tr.name==typhoons]
        tr_HRS=[tr for tr in fcast.data if (tr.is_ensemble=='False')]
        HRS_SPEED=(tr_HRS[0].max_sustained_wind.values/0.84).tolist()  ############# 0.84 is conversion factor for ECMWF 10MIN TO 1MIN AVERAGE
        
        dfff=tr_HRS[0].to_dataframe()
        dfff[['VMAX','LAT','LON']]=dfff[['max_sustained_wind','lat','lon']]
        dfff['YYYYMMDDHH']=dfff.index.values
        dfff['YYYYMMDDHH']=dfff['YYYYMMDDHH'].apply(lambda x: x.strftime("%Y%m%d%H%M") )
        dfff['STORMNAME']=typhoons
        dfff[['YYYYMMDDHH','VMAX','LAT','LON','STORMNAME']].to_csv(os.path.join(Input_folder,'ecmwf_hrs_track.csv'), index=False)
        
        
        line_='ecmwf,'+'%secmwf_hrs_track.csv' % Input_folder+ ',' +typhoons+','+ datetime.now().strftime("%Y%m%d%H")   #StormName #
        #line_='Rainfall,'+'%sRainfall/' % Input_folder +','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H") #StormName #
        fname.write(line_+'\n')
        
        # Adjust track time step
        
        data_forced=[tr.where(tr.time <= max(tr_HRS[0].time.values),drop=True) for tr in fcast.data]
         
        data_forced = [track_data_clean.track_data_force_HRS(tr,HRS_SPEED) for tr in data_forced] # forced with HRS windspeed
        
        
      
        data_forced= [track_data_clean.track_data_clean(tr) for tr in data_forced] # taking speed of ENS
        # interpolate to 3h steps from the original 6h
        #fcast.equal_timestep(3)
     
        
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
            print(tr.name)
            track = TCTracks() 
            typhoon = TropCyclone()
            track.data=[tr]
            track.equal_timestep(3)
            tr=track.data[0]
            typhoon.set_from_tracks(track, cent, store_windfields=True)
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
            list_intensity.append(inten_tr)
            distan_track1=[]
            for index, row in df.iterrows():
                dist=np.min(np.sqrt(np.square(tr.lat.values-row['lat'])+np.square(tr.lon.values-row['lon'])))
                distan_track1.append(dist*111)
            dist_tr = pd.DataFrame({'centroid_id': centroid_idx,'value': distan_track1})
            dist_tr['storm_id'] = tr.sid
            dist_tr['name'] = tr.name
            dist_tr['ens_id'] =tr.sid+'_'+str(tr.ensemble_number)
            distan_track.append(dist_tr)                
        df_intensity = pd.concat(list_intensity)
        df_intensity=pd.merge(df_intensity, df_admin, how='outer', on='centroid_id')
        df_intensity=df_intensity.dropna()
        
        df_intensity_=df_intensity.groupby(['adm3_pcode','ens_id'],as_index=False).agg({"value":['count', 'max']}) 
        # rename columns
        df_intensity_.columns = [x for x in ['adm3_pcode','storm_id','value_count','v_max']] 
        distan_track1= pd.concat(distan_track)
        distan_track1=pd.merge(distan_track1, df_admin, how='outer', on='centroid_id')
        distan_track1=distan_track1.dropna()
        
        distan_track1=distan_track1.groupby(['adm3_pcode','name','ens_id'],as_index=False).agg({'value':'min'}) 
        distan_track1.columns = [x for x in ['adm3_pcode','name','storm_id','dis_track_min']]#join_left_df_.columns.ravel()] 
        typhhon_df = pd.merge(df_intensity_, distan_track1,  how='left', on=['adm3_pcode','storm_id']) 
    
        typhhon_df.to_csv(os.path.join(Input_folder,'windfield.csv'), index=False)
    
        line_='windfield,'+'%swindfield.csv' % Input_folder+ ',' +typhoons+','+ datetime.now().strftime("%Y%m%d%H")   #StormName #
        #line_='Rainfall,'+'%sRainfall/' % Input_folder +','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H") #StormName #
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
                raise ValueError(str(e))
        elif platform == "win32": #if OS is windows edit the path for Rscript
            try:
                p = subprocess.check_call(["C:/Program Files/R/R-4.1.0/bin/Rscript", "run_model_V2.R", str(rainfall_error)])
            except subprocess.CalledProcessError as e:
                raise ValueError(str(e))
            
        
        #############################################################
        ### send email in case of landfall-typhoon
        #############################################################

        landfall_typhones=[] #model_output_file_names
        try:
            #fname2=open("forecast/%s_file_names.csv" % typhoons,'r')
            fname2=open(os.path.join(Output_folder,"model_output_file_names.csv"),'r')
            for lines in fname2.readlines():
                print(lines)
                landfall_typhones.append(lines.split(';')[1].strip().strip('"'))
                #landfall_typhones.append(lines.split(';')[1])
#                if (lines.split(' ')[1].split('_')[0]) !='"Nolandfall':
#                    if lines.split(' ')[1] not in landfall_typhones:
#                        landfall_typhones.append(lines.split(' ')[1])
            fname2.close()
        except:
            pass
        
        if not landfall_typhones==[]:
            #image_filename=landfall_typhones[0]
            image_filename=[i for i in landfall_typhones if i.endswith('.png')][0]
            data_filename=[i for i in landfall_typhones if i.endswith('.csv')][0]
            #data_filename=landfall_typhones[1]
            html = """\
            <html>
            <body>
            <h1>IBF model run result </h1>
            <p>Please find below a map and data with updated model run</p>
            <img src="cid:Impact_Data">
            </body>
            </html>
            """
            Sendemail.sendemail(from_addr  = EMAIL_FROM,
                    to_addr_list = EMAIL_LIST,
                    cc_addr_list = CC_LIST,
                    message = message(
                        subject='Updated impact map for a new Typhoon in PAR',
                        html=html,
                        textfile=data_filename,
                        image=image_filename),
                    login  = EMAIL_LOGIN,
                    password= EMAIL_PASSWORD,
                    smtpserver=SMTP_SERVER)


    print('---------------------AUTOMATION SCRIPT FINISHED---------------------------------')
    print(str(datetime.now()))
    


#%%#Download rainfall (old pipeline)
automation_sript(path)
