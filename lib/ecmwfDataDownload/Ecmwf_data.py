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
from io import StringIO
import numpy as np
 
def duplicate(testList, n):
    """
    duplicate list items 
    """  
    x = len(testList)
    new_list = []
    for j in range(x):
        new_list.extend(testList[j] for _ in range(n))
    return new_list

def aggregateF(df,col):
    """
    aggregate panda columns
    """  
    xmin = np.nanmin(pd.to_numeric(df[col]).values)
    xmax = np.nanmax(pd.to_numeric(df[col]).values)
    xmean = np.nanmean(pd.to_numeric(df[col]).values)
    new_list = {'xmin':xmin,'xmax':xmax,'xmean':xmean}
    return new_list


def ecmwf_data_download(Input_folder,filepatern):
    """
    Reads ecmwf forecast data and save to folder
    """    
    if not os.path.exists(os.path.join(Input_folder,'ecmwf/')):
        os.makedirs(os.path.join(Input_folder,'ecmwf/'))
    path_ecmwf=os.path.join(Input_folder,'ecmwf/')
    ftp=FTP("dissemination.ecmwf.int")
    ftp.login("wmo","essential")
    ftp=FTP("dissemination.ecmwf.int")
    ftp.login("wmo","essential")
    sub_folders=ftp.nlst()
    sub_folder = [sub_folder for sub_folder in sub_folders if sub_folder.endswith(("000000", "120000"))]# and link.endswith(('.html', '.xml'))]
    ftp.cwd(os.path.join(ftp.pwd(),sub_folder[-1]))
    file_list=ftp.nlst() 
    for files in file_list:
        if filepatern in files:
            ftp.retrbinary("RETR " +files,open(os.path.join(path_ecmwf,files),'wb').write)
    ftp.quit()
 
def ecmwf_data_process(Input_folder,filepatern):
    """
    preprocess ecmwf forecast data downloaded above
    """
    decoder = Decoder()
    #ecmwf_data_download(Input_folder,filepatern)
    
    path_ecmwf=os.path.join(Input_folder,'ecmwf/')
    decoder = Decoder()
    #1=Storm Centre 4 = Location of the storm in the perturbed analysis
    #5 = Location of the storm in the analysis #3=Location of maximum wind
    ecmwf_files = [f for f in listdir(path_ecmwf) if isfile(join(path_ecmwf, f))]
    #ecmwf_files = [file_name for file_name in ecmwf_files if file_name.startswith('A_JSXX02ECEP')]
    list_df=[]
    for ecmwf_file in ecmwf_files:
        #ecmwf_file=ecmwf_files[0]
        f_name='ECMWF_'+ ecmwf_file.split('_')[1]+'_'+ecmwf_file.split('_')[4]
        model_name=ecmwf_file.split('_')[1][6:10]
        typhoon_name=ecmwf_file.split('_')[-4] 
        with open(os.path.join(path_ecmwf,ecmwf_file), 'rb') as bin_file:
            bufr_message = decoder.process(bin_file.read())
            text_data = FlatTextRenderer().render(bufr_message)
        STORMNAME=ecmwf_file.split('_')[8]
        date_object ='%04d%02d%02d%02d'%(int([line.split()[-1] for line in StringIO(text_data) if line[6:17]=="004001 YEAR" ][0]),
                             int([line.split()[-1] for line in StringIO(text_data) if line[6:18]=="004002 MONTH" ][0]),
                             int([line.split()[-1] for line in StringIO(text_data) if line[6:16]=="004003 DAY" ][0]),
                             int([line.split()[-1] for line in StringIO(text_data) if line[6:17]=="004004 HOUR" ][0]))

        date_object=datetime.strptime(date_object, "%Y%m%d%H%M")

        val_t = [int(line.split()[-1]) for num, line in enumerate(StringIO(text_data), 1) if line[6:40]=="004024 TIME PERIOD OR DISPLACEMENT"]# and link.endswith(('.html', '.xml'))]
        val_wind = [line.split()[-1] for num, line in enumerate(StringIO(text_data), 1) if line[6:12]=="011012" ]#and num > ind_x[0]]# and link.endswith(('.html', '.xml'))]
        val_lat = [line.split()[-1] for num, line in enumerate(StringIO(text_data), 1) if line[6:12]=="005002" ]#and num > ind_x[0]]# and link.endswith(('.html', '.xml'))]
        val_lon = [line.split()[-1] for num, line in enumerate(StringIO(text_data), 1) if line[6:12]=="006002" ]#and num > ind_x[0]]# and link.endswith(('.html', '.xml'))]
        val_ens = [line.split()[-1] for num, line in enumerate(StringIO(text_data), 1) if line[6:12]=="001091" ]#and num > ind_x[0]]# and link.endswith(('.html', '.xml'))]
        val_dis = [line.split()[-1]  for num, line in enumerate(StringIO(text_data), 1) if line[6:12]=="008005" ]#and num > ind_x[0]]# and link.endswith(('.html', '.xml'))]

        if len(val_ens) >1:
            val_t=val_t[0:int(len(val_t)/len(val_ens))]
            val_t.insert(0, 0)
            val_ensamble=duplicate(val_ens, int(len(val_wind)/len(val_ens)))
            val_time=val_t* len(val_ens) #52
        else:
            val_ensamble='NA'
            val_t.insert(0, 0)
            val_time=val_t           
        ecmwf_df=pd.DataFrame({'lon': val_lon,'lat': val_lat,'met_dis': val_dis })
        ecmwf_center=ecmwf_df[ecmwf_df['met_dis']=='1']
        ecmwf_df2=pd.DataFrame({'STORMNAME':STORMNAME,
                                'time':val_time,
                                'lon':ecmwf_center['lon'].values,
                                'lat':ecmwf_center['lat'].values,
                                'windsped':val_wind, 
                                'ens': val_ensamble})
        ecmwf_df2['YYYYMMDDHH']=ecmwf_df2['time'].apply(lambda x: (date_object + timedelta(hours=x)).strftime("%Y%m%d%H%M") )
        dict1=[]
        ecmwf_df2=ecmwf_df2.replace(['None'],np.nan)

        typhoon_df=pd.DataFrame()
        typhoon_df[['YYYYMMDDHH','LAT','LON','VMAX','STORMNAME','ENSAMBLE']]=ecmwf_df2[['YYYYMMDDHH','lat','lon','windsped','STORMNAME','ens']]
        typhoon_df[['LAT','LON','VMAX']] = typhoon_df[['LAT','LON','VMAX']].apply(pd.to_numeric)
        typhoon_df['VMAX'] = typhoon_df['VMAX'].apply(lambda x: x*1.94384449) #convert to knotes
        typhoon_df.to_csv(os.path.join(Input_folder,'ECMWF_%s_%s_%s.csv'%(Input_folder.split('/')[-3],Input_folder.split('/')[-4],model_name)),index=False) 
        for it,group in ecmwf_df2.groupby(['ens','YYYYMMDDHH']):
            dff=pd.DataFrame({'YYYYMMDDHH':it[1],
                              'STORMNAME':Input_folder.split('/')[-4],
                              'LON':aggregateF(df=group,col='lon')['xmean'],
                              'LAT':aggregateF(df=group,col='lat')['xmean'],
                              'Lat_min':aggregateF(df=group,col='lat')['xmin'],
                              'VMAX':aggregateF(df=group,col='windsped')['xmean'] ,
                              'Lat_max': aggregateF(df=group,col='lat')['xmax']},index=[it[1]] )
            dict1.append(dff)
        ecmwf_df3 = pd.concat(dict1)  
        ecmwf_df3['model_name']=model_name
        list_df.append(ecmwf_df3)
        typhoon_fss=ecmwf_df3[['YYYYMMDDHH','LAT','LON','VMAX','STORMNAME']]
        typhoon_fss['VMAX'] = typhoon_fss['VMAX'].apply(lambda x: x*1.94384449) #convert to knotes
        #typhoon_fss.to_csv(os.path.join(Input_folder,'ECMWF_%s_%s_%s.csv'%(Input_folder.split('/')[-3],Input_folder.split('/')[-4],model_name)),index=False) 
    if len(list_df)>1:
        df_forecast = pd.concat(list_df)
    else:
        df_forecast = list_df[0]
    typhoon_fs=pd.DataFrame()  #
    typhoon_fs[['YYYYMMDDHH','LAT','LON','VMAX','STORMNAME','model_name']]=df_forecast[['YYYYMMDDHH','LAT','LON','VMAX','STORMNAME','model_name']]
    typhoon_fs['VMAX'] = typhoon_fs['VMAX'].apply(lambda x: x*1.94384449) #convert to knotes
    #typhoon_fs.to_csv( os.path.join(value,'%s_typhoon.csv' % value.split('/')[-1]))
    #typhoon_fs.to_csv( os.path.join(Input_folder,'UCL_%s_%s.csv' % (Input_folder.split('/')[-3],Input_folder.split('/')[-4])))
    typhoon_fs.to_csv(os.path.join(Input_folder,'ECMWF_%s_%s.csv'%(Input_folder.split('/')[-3],typhoon_name)),index=False) 

