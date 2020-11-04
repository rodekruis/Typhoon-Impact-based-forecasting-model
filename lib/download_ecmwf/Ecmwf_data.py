import sys
import os
import pandas as pd
import subprocess
import numpy as np
from datetime import datetime
from datetime import timedelta
import re
import zipfile
import geopandas as gpd
from ftplib import FTP
import shutil
from os.path import relpath
from os import listdir
from os.path import isfile, join
from pybufrkit.decoder import Decoder
from pybufrkit.renderer import FlatTextRenderer
from sys import platform

def ecmwf_data_download(Input_folder,filepatern):
    """
    Reads ecmwf forecast data and save to folder
    """    
    if not os.path.exists(os.path.join(Input_folder,'ecmwf/')):
        os.makedirs(os.path.join(Input_folder,'ecmwf/'))
    path_ecmwf=os.path.join(Input_folder,'ecmwf/')
    ftp=FTP("dissemination.ecmwf.int")
    ftp.login("wmo","essential")
    files=ftp.nlst()
    
    ftp.cwd(os.path.join(ftp.pwd(),files[-1]))
    file_list=ftp.nlst()    
    for files in file_list:
        if filepatern in files:
            ftp.retrbinary("RETR " +files,open(os.path.join(path_ecmwf,files),'wb').write)
            
def ecmwf_data_process(Input_folder):
    """
    preprocess ecmwf forecast data downloaded above
    """
    decoder = Decoder()
    path_ecmwf=os.path.join(Input_folder,'ecmwf/')
    path_input=Input_folder #os.path.join(Input_folder,'forecast/Input/')
    ecmwf_files = [f for f in listdir(path_ecmwf) if isfile(join(path_ecmwf, f))]
    for ecmwf_file in ecmwf_files:
        with open(os.path.join(path_ecmwf,ecmwf_file), 'rb') as bin_file:
            bufr_message = decoder.process(bin_file.read())
            text_data = FlatTextRenderer().render(bufr_message)
        f_name='ECMWF_'+ ecmwf_file.split('_')[4]  
        index_list_idd=[]
        list2=text_data.split('\n')
        try:
            for j in range(1,2):
                list3=list2[list2.index('###### subset %s of 52 ######'% j)+1:list2.index('###### subset %s of 52 ######'% str(j+1) )]
                list_hol=[]
                list_hol2=[]
                list_hol3=[]
                for elmen in list3:
                    list_hol.append(elmen[6:13].strip(' '))
                    list_hol2.append(elmen[12:70])
                    list_hol3.append(elmen[-20:].strip(' '))
            ecmwf_df = pd.DataFrame(data={'name_code': list_hol, 'name': list_hol2,'1': list_hol3})
            for j in range(2,52):
                list3=list2[list2.index('###### subset %s of 52 ######'% j)+1:list2.index('###### subset %s of 52 ######'% str(j+1) )]
                list_hol=[]
                for elmen in list3:
                    list_hol.append(elmen[-20:].strip(' '))
                #print(list_hol)
                ecmwf_df[str(j)]=list_hol
            #ecmwf_df.to_csv(os.path.join(path,f_name+'.csv'),index=False)
            #ecmwf_df.to_csv(os.path.join(path,f_name+'.csv'),index=False)
            TIME_PERIOD_OR_DISPLACEMENT=ecmwf_df[ecmwf_df['name_code']=='004024']
            LATITUDE=ecmwf_df[ecmwf_df['name_code']=='005002']
            LONGITUDE=ecmwf_df[ecmwf_df['name_code']=='006002']
            METEOROLOGICAL_ATTRIBUTE_SIGNIFICANCE=ecmwf_df[ecmwf_df['name_code']=='008005']
            YEAR=ecmwf_df[ecmwf_df['name_code']=='004001']['1'].values
            MONTH=ecmwf_df[ecmwf_df['name_code']=='004002']['1'].values
            DAY=ecmwf_df[ecmwf_df['name_code']=='004003']['1'].values
            HOUR=ecmwf_df[ecmwf_df['name_code']=='004004']['1'].values 
            STORMNAME=ecmwf_file.split('_')[8] #ecmwf_df[ecmwf_df['name_code']=='001027']['1'].values 
            
          
            LATITUDE['loction']=METEOROLOGICAL_ATTRIBUTE_SIGNIFICANCE['1'].values
            LONGITUDE['loction']=METEOROLOGICAL_ATTRIBUTE_SIGNIFICANCE['1'].values
            LONGITUDE['time']=np.insert(np.repeat(TIME_PERIOD_OR_DISPLACEMENT['20'].values, 2), [0], [0,0,0])
            LONGITUDE['lon']=LONGITUDE.loc[:, '1':'51'].replace('None', 'NAN').astype('float',errors='ignore').mean(axis=1,skipna = 'TRUE')
            LATITUDE['time']=np.insert(np.repeat(TIME_PERIOD_OR_DISPLACEMENT['20'].values, 2), [0], [0,0,0])
            LATITUDE['lat']=LATITUDE.loc[:, '1':'51'].replace('None', 'NAN').astype('float',errors='ignore').mean(axis=1,skipna = 'TRUE')
            
            WIND=ecmwf_df[ecmwf_df['name_code']=='011012']
            WIND['time']=np.insert(TIME_PERIOD_OR_DISPLACEMENT['20'].values, [0], [0])
            WIND['wind_kmh']=WIND.loc[:, '1':'51'].replace('None', 'NAN').astype('float',errors='ignore').mean(axis=1,skipna = 'TRUE')
            #WIND[['time','wind_kmh']].to_csv(os.path.join(path_input,f_name+'_wind.csv'),index=False)
            #LATITUDE[['time','loction','lat']].to_csv(os.path.join(path_input,f_name+'_latitude.csv'),index=False)
            #LONGITUDE[['time','loction','lon']].to_csv(os.path.join(path_input,f_name+'_longitude.csv'),index=False)
        
            date_object ='%04d%02d%02d%02d'%(int(YEAR[0]),int(MONTH[0]),int(DAY[0]),int(HOUR[0]))
            date_object=datetime.strptime(date_object, "%Y%m%d%H%M")
            
            wind=WIND[['time','wind_kmh']] 
            wind['time']=wind.time.astype(int)
            wind['YYYYMMDDHH']=wind['time'].apply(lambda x: (date_object + timedelta(hours=x)).strftime("%Y%m%d%H%M") )
            
            lon=LONGITUDE[['time','loction','lon']]
            lon['time']=lon.time.astype(int)
            lon['YYYYMMDDHH']=lon['time'].apply(lambda x: (date_object + timedelta(hours=x)).strftime("%Y%m%d%H%M") )
            lon=lon[lon['loction'].isin(['1','4'])]
            lon=lon[~((lon.time == 0) & (lon.loction=='4'))]
            
            lat=LATITUDE[['time','loction','lat']]
            lat['time']=lat.time.astype(int)
            lat['YYYYMMDDHH']=lat['time'].apply(lambda x: (date_object + timedelta(hours=x)).strftime("%Y%m%d%H%M") )
            lat=lat[lat['loction'].isin(['1','4'])]
            lat=lat[~((lat.time == 0) & (lat.loction=='4'))]
            
            df_forecast=lon[['lon','YYYYMMDDHH']].set_index('YYYYMMDDHH').join(lat[['lat','YYYYMMDDHH']].set_index('YYYYMMDDHH'), on='YYYYMMDDHH').join(wind[['wind_kmh','YYYYMMDDHH']].set_index('YYYYMMDDHH'),on='YYYYMMDDHH')
            df_forecast=df_forecast.rename(columns={"lat": "LAT", "lon": "LON", "wind_kmh": "VMAX"})
            df_forecast['STORMNAME']=STORMNAME
            df_forecast.to_csv(os.path.join(Input_folder,'ECMWF_%s_%s.csv'%(Input_folder.split('/')[-3],Input_folder.split('/')[-4])),index=True)  
        except:
            pass         