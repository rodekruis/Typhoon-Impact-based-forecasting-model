import sys
import os
import pandas as pd
import subprocess
import numpy as np
from datetime import datetime 
import datetime as dt
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
 
import pybufrkit
decoder = pybufrkit.decoder.Decoder()


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
 
    #ecmwf_data_download(Input_folder,filepatern)
    
    path_ecmwf=os.path.join(Input_folder,'ecmwf/')
    decoder = Decoder()
    #1=Storm Centre 4 = Location of the storm in the perturbed analysis
    #5 = Location of the storm in the analysis #3=Location of maximum wind
    ecmwf_files = [f for f in listdir(path_ecmwf) if isfile(join(path_ecmwf, f))]
    #ecmwf_files = [file_name for file_name in ecmwf_files if file_name.startswith('A_JSXX02ECEP')]
    list_df=[]
    for ecmwf_file in ecmwf_files:
        ecmwf_file=ecmwf_files[0]
        f_name='ECMWF_'+ ecmwf_file.split('_')[1]+'_'+ecmwf_file.split('_')[4]
        model_name=ecmwf_file.split('_')[1][6:10]
        typhoon_name=ecmwf_file.split('_')[-4] 
        with open(os.path.join(path_ecmwf,ecmwf_file), 'rb') as bin_file:
            bufr_message = decoder.process(bin_file.read())
            text_data = FlatTextRenderer().render(bufr_message)
        STORMNAME=typhoon_name #ecmwf_file.split('_')[8]  
        list1=[]
        with StringIO(text_data) as input_data:
            # Skips text before the beginning of the interesting block:
            for line in input_data:
                if line.startswith('<<<<<< section 4 >>>>>>'):  # Or whatever test is needed
                    break
            # Reads text until the end of the block:
            for line in input_data:  # This keeps reading the file
                if line.startswith('<<<<<< section 5 >>>>>>'):
                    break
                list1.append(line)
         
        list_var=["004024","004001","004002","004003","004004","004005","001092","011012","010051","005002","006002","001091",'001092',"008005"]  
        list2=[[int(li.split()[0]),li.split()[1],li.split()[-1]] for li in list1 if li.startswith(" ") and li.split()[1] in list_var]
        
        df = pd.DataFrame(list2,columns=['id','code','Data'])
        
        
        def label_en (row,co):
           if row['code'] == co :
              return int(row['Data'])
           return np.nan
        
        df['model_sgn'] = df.apply (lambda row: label_en(row,co='008005'), axis=1)
        df['ensamble_num'] = df.apply (lambda row: label_en(row,co='001091'), axis=1)
        df['frcst_type'] = df.apply (lambda row: label_en(row,co='001092'), axis=1)
        
        df['frcst_type'] =df['frcst_type'].fillna(method='ffill')
        df['frcst_type'] =df['frcst_type'].fillna(method='bfill')
        
        df['ensamble_num'] =df['ensamble_num'].fillna(method='ffill')
        df['model_sgn'] =df['model_sgn'].fillna(method='ffill')
        df['model_sgn'] =df['model_sgn'].fillna(method='bfill')
        
        df_time = df.query('code in ["004001","004002","004003","004004","004005"]')
        
        date_object ='%04d%02d%02d%02d'%(int(df_time['Data'].to_list()[0]),
                                         int(df_time['Data'].to_list()[1]),
                                         int(df_time['Data'].to_list()[2]),
                                         int(df_time['Data'].to_list()[3]))
        
        date_object=datetime.strptime(date_object, "%Y%m%d%H")
        #(date_object + timedelta(hours=x)).strftime("%Y%m%d%H%M")
                
        df_center = df.query('code in ["010051","005002","006002"] and model_sgn in [1]')
        df_center2 = df.query('code in ["010051","005002","006002"] and model_sgn in [4,5]')
        df_max = df.query('code in ["011012","005002","006002","004024"] and model_sgn in [3]')
         # 1 storm center and 3 maximum wind speed https://vocabulary-manager.eumetsat.int/vocabularies/BUFR/WMO/6/TABLE_CODE_FLAG/008005
        
            
        latc,lonc,pcen,frcst_type,ensambles=[],[],[],[],[]
        for names, group in df_center.groupby("ensamble_num"):
            latc.append(list(group[group.code=="005002"]['Data'].values))
            lonc.append(list(group[group.code=="006002"]['Data'].values))
            pcen.append(list(group[group.code=="010051"]['Data'].values))
            
        lat,lon,vmax,vhr=[],[],[],[]
        for names, group in df_max.groupby("ensamble_num"):
            lat.append(list(group[group.code=="005002"]['Data'].values))
            lon.append(list(group[group.code=="006002"]['Data'].values))
            vmax.append(list(group[group.code=="011012"]['Data'].values))
            vhr.append(list(group[group.code=="004024"]['Data'].values))
            frcst_type.append(list(np.unique(group.frcst_type.values))[0])
            ensambles.append(names)
        
        latc1,lonc1,pcen1=[],[],[]
        for names, group in df_center2.groupby("ensamble_num"):
            latc1.append(list(group[group.code=="005002"]['Data'].values))
            lonc1.append(list(group[group.code=="006002"]['Data'].values))
            pcen1.append(list(group[group.code=="010051"]['Data'].values))
            
        for i in range(len(pcen1)):
            pcen1[i].extend(pcen[i])
         
            
        vhr=['0','6', '12', '18', '24', '30', '36', '42', '48', '54', '60', '66', '72', '78', '84', '90', '96', '102', '108']
        for i in range(len(ensambles)):
            wind=[np.nan if value=='None' else float(value) for value in vmax[i]]
            pre=[np.nan if value=='None' else float(value)/100 for value in pcen1[i]]
            lon_=[np.nan if value=='None' else float(value) for value in lon[i]]
            lat_=[np.nan if value=='None' else float(value) for value in lat[i]]
            lon1_=[np.nan if value=='None' else float(value) for value in lonc[i]]
            lat1_=[np.nan if value=='None' else float(value) for value in latc[i]]
            max_radius=np.sqrt(np.square(np.array(lon_)-np.array(lon1_))+np.square(np.array(lat_)-np.array(lat1_)))*110
            timestamp=[(date_object + timedelta(hours=int(value))).strftime("%Y%m%d%H%M") for value in vhr]
            timestep_int=[int(value)  for value in vhr]
            ['TRUE' if frcst_type[i]==4 else 'False']    
            track = xr.Dataset(
                    data_vars={
                            'max_sustained_wind': ('time', wind),
                            'central_pressure': ('time', pre),
                            'ts_int': ('time', timestep_int),
                            'max_radius': ('time', max_radius),
                            'lat': ('time', lat_),
                            'lon': ('time', lon_),
                            },
                            coords={'time': timestamp,
                                    },
                                    attrs={
                                            'max_sustained_wind_unit': 'm/s',
                                            'central_pressure_unit': 'mb',
                                            'name': typhoon_name,
                                            'sid': 'NA',
                                            'orig_event_flag': False,
                                            'data_provider': 'ECMWF',
                                            'id_no': 'NA',
                                            'ensemble_number': ensambles[i],
                                            'is_ensemble': ['TRUE' if frcst_type[i]==4 else 'False'][0],
                                            'forecast_time': date_object.strftime("%Y%m%d%H%M"),
                                            'basin': 'WP',
                                            'category': 'NA',
                                            })
            track = track.set_coords(['lat', 'lon'])  
    list_df.append(track)
        
        
        
        
        
        
        
#%%        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        date_object ='%04d%02d%02d%02d'%(int([line.split()[-1] for line in StringIO(text_data) if line[6:17].upper=="004001 YEAR" ][0]),
                             int([line.split()[-1] for line in StringIO(text_data) if line[6:18].upper=="004002 MONTH" ][0]),
                             int([line.split()[-1] for line in StringIO(text_data) if line[6:16].upper=="004003 DAY" ][0]),
                             int([line.split()[-1] for line in StringIO(text_data) if line[6:17].upper=="004004 HOUR" ][0]))

        date_object=datetime.strptime(date_object, "%Y%m%d%H%M")
        

        val_t = [int(line.split()[-1]) for num, line in enumerate(StringIO(text_data), 1) if line[6:40].upper=="004024 TIME PERIOD OR DISPLACEMENT"]# and link.endswith(('.html', '.xml'))]
        val_wind = [line.split()[-1] for num, line in enumerate(StringIO(text_data), 1) if line[6:12].upper=="011012" ]#and num > ind_x[0]]# and link.endswith(('.html', '.xml'))]
        val_pre = [line.split()[-1] for num, line in enumerate(StringIO(text_data), 1) if line[6:12].upper=="010051" ]#and num > ind_x[0]]# and link.endswith(('.html', '.xml'))]
        val_lat = [line.split()[-1] for num, line in enumerate(StringIO(text_data), 1) if line[6:12].upper=="005002" ]#and num > ind_x[0]]# and link.endswith(('.html', '.xml'))]
        val_lon = [line.split()[-1] for num, line in enumerate(StringIO(text_data), 1) if line[6:12].upper=="006002" ]#and num > ind_x[0]]# and link.endswith(('.html', '.xml'))]
        val_ens = [line.split()[-1] for num, line in enumerate(StringIO(text_data), 1) if line[6:12].upper=="001091" ]#and num > ind_x[0]]# and link.endswith(('.html', '.xml'))]
        val_dis = [line.split()[-1]  for num, line in enumerate(StringIO(text_data), 1) if line[6:12].upper=="008005" ]#and num > ind_x[0]]# and link.endswith(('.html', '.xml'))]

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
                                'pressure':val_pre, 
                                'ens': val_ensamble})
        ecmwf_df2['YYYYMMDDHH']=ecmwf_df2['time'].apply(lambda x: (date_object + timedelta(hours=x)).strftime("%Y%m%d%H%M") )
        dict1=[]
        ecmwf_df2=ecmwf_df2.replace(['None'],np.nan)

        typhoon_df=pd.DataFrame()
        typhoon_df[['YYYYMMDDHH','LAT','LON','VMAX','PRESSURE','STORMNAME','ENSAMBLE']]=ecmwf_df2[['YYYYMMDDHH','lat','lon','windsped','pressure','STORMNAME','ens']]
        typhoon_df[['LAT','LON','VMAX']] = typhoon_df[['LAT','LON','VMAX']].apply(pd.to_numeric)
        typhoon_df['VMAX'] = typhoon_df['VMAX'].apply(lambda x: x*1.94384449*1.05) #convert to knotes
        typhoon_df.to_csv(os.path.join(Input_folder,'ECMWF_%s_%s_%s.csv'%(Input_folder.split('/')[-3],STORMNAME,model_name)),index=False) 
    #     for it,group in ecmwf_df2.groupby(['ens','YYYYMMDDHH']):
    #         dff=pd.DataFrame({'YYYYMMDDHH':it[1],
    #                           'STORMNAME':STORMNAME,#Input_folder.split('/')[-4],
    #                           'LON':aggregateF(df=group,col='lon')['xmean'],
    #                           'LAT':aggregateF(df=group,col='lat')['xmean'],
    #                           'Lat_min':aggregateF(df=group,col='lat')['xmin'],
    #                           'VMAX':aggregateF(df=group,col='windsped')['xmean'] ,
    #                           'Lat_max': aggregateF(df=group,col='lat')['xmax']},index=[it[1]] )
    #         dict1.append(dff)
    #     ecmwf_df3 = pd.concat(dict1)  
    #     ecmwf_df3['model_name']=model_name
    #     list_df.append(ecmwf_df3)
    #     typhoon_fss=ecmwf_df3[['YYYYMMDDHH','LAT','LON','VMAX','STORMNAME']]
    #     typhoon_fss['VMAX'] = typhoon_fss['VMAX'].apply(lambda x: x*1.94384449) #convert to knotes
    #     typhoon_fss.to_csv(os.path.join(Input_folder,'ECMWF2_%s_%s2.csv'%(Input_folder.split('/')[-3],typhoon_name)),index=False) 
    #     #typhoon_fss.to_csv(os.path.join(Input_folder,'ECMWF_%s_%s_%s.csv'%(Input_folder.split('/')[-3],Input_folder.split('/')[-4],model_name)),index=False) 
    # if len(list_df)>1:
    #     df_forecast = pd.concat(list_df)
    # else:
    #     df_forecast = list_df[0]
    # typhoon_fs=pd.DataFrame()  #
    # typhoon_fs[['YYYYMMDDHH','LAT','LON','VMAX','STORMNAME','model_name']]=df_forecast[['YYYYMMDDHH','LAT','LON','VMAX','STORMNAME','model_name']]
    # typhoon_fs['VMAX'] = typhoon_fs['VMAX'].apply(lambda x: x*1.94384449) #convert to knotes
    # #typhoon_fs.to_csv( os.path.join(value,'%s_typhoon.csv' % value.split('/')[-1]))
    # #typhoon_fs.to_csv( os.path.join(Input_folder,'UCL_%s_%s.csv' % (Input_folder.split('/')[-3],Input_folder.split('/')[-4])))
    # typhoon_fs.to_csv(os.path.join(Input_folder,'ECMWF_%s_%s.csv'%(Input_folder.split('/')[-3],typhoon_name)),index=False) 


#%%
Input_folder='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/temp/'
#date_object ='%04d%02d%02d%02d'%(int(
[line.split()[-1] for line in StringIO(text_data) if line[6:17].upper=="004001 YEAR" ]
path_ecmwf=os.path.join(Input_folder,'ecmwf/')

    #1=Storm Centre 4 = Location of the storm in the perturbed analysis
    #5 = Location of the storm in the analysis #3=Location of maximum wind
ecmwf_files = [f for f in listdir(path_ecmwf) if isfile(join(path_ecmwf, f))]
    #ecmwf_files = [file_name for file_name in ecmwf_files if file_name.startswith('A_JSXX02ECEP')]
list_df=[]
ecmwf_file=ecmwf_files[1]
with open(os.path.join(path_ecmwf,ecmwf_file), 'rb') as bin_file:
    bufr = decoder.process(bin_file.read())
    text_data = FlatTextRenderer().render(bufr)
# Convert the BUFR message to JSON
    #
#%%
    

list1=[]
with StringIO(text_data) as input_data:
    # Skips text before the beginning of the interesting block:
    for line in input_data:
        if line.startswith('<<<<<< section 4 >>>>>>'):  # Or whatever test is needed
            break
    # Reads text until the end of the block:
    for line in input_data:  # This keeps reading the file
        if line.startswith('<<<<<< section 5 >>>>>>'):
            break
        list1.append(line)
 
list_var=["004024","004001","004002","004003","004004","004005","001092","011012","010051","005002","006002","001091",'001092',"008005"]  

list2=[[int(li.split()[0]),li.split()[1],li.split()[-1]] for li in list1 if li.startswith(" ") and li.split()[1] in list_var]

df = pd.DataFrame(list2,columns=['id','code','Data'])


def label_en (row,co):
   if row['code'] == co :
      return int(row['Data'])
   return np.nan

df['model_sgn'] = df.apply (lambda row: label_en(row,co='008005'), axis=1)
df['ensamble_num'] = df.apply (lambda row: label_en(row,co='001091'), axis=1)
df['frcst_type'] = df.apply (lambda row: label_en(row,co='001092'), axis=1)

df['frcst_type'] =df['frcst_type'].fillna(method='ffill')
df['frcst_type'] =df['frcst_type'].fillna(method='bfill')

df['ensamble_num'] =df['ensamble_num'].fillna(method='ffill')
df['model_sgn'] =df['model_sgn'].fillna(method='ffill')
df['model_sgn'] =df['model_sgn'].fillna(method='bfill')

df_time = df.query('code in ["004001","004002","004003","004004","004005"]')

date_object ='%04d%02d%02d%02d'%(int(df_time['Data'].to_list()[0]),
                                 int(df_time['Data'].to_list()[1]),
                                 int(df_time['Data'].to_list()[2]),
                                 int(df_time['Data'].to_list()[3]))

date_object=datetime.strptime(date_object, "%Y%m%d%H")
#(date_object + timedelta(hours=x)).strftime("%Y%m%d%H%M")
        
df_center = df.query('code in ["010051","005002","006002"] and model_sgn in [1]')
df_center2 = df.query('code in ["010051","005002","006002"] and model_sgn in [4,5]')
df_max = df.query('code in ["011012","005002","006002","004024"] and model_sgn in [3]')
 # 1 storm center and 3 maximum wind speed https://vocabulary-manager.eumetsat.int/vocabularies/BUFR/WMO/6/TABLE_CODE_FLAG/008005

    
latc,lonc,pcen,frcst_type,ensambles=[],[],[],[],[]
for names, group in df_center.groupby("ensamble_num"):
    latc.append(list(group[group.code=="005002"]['Data'].values))
    lonc.append(list(group[group.code=="006002"]['Data'].values))
    pcen.append(list(group[group.code=="010051"]['Data'].values))
    
lat,lon,vmax,vhr=[],[],[],[]
for names, group in df_max.groupby("ensamble_num"):
    lat.append(list(group[group.code=="005002"]['Data'].values))
    lon.append(list(group[group.code=="006002"]['Data'].values))
    vmax.append(list(group[group.code=="011012"]['Data'].values))
    vhr.append(list(group[group.code=="004024"]['Data'].values))
    frcst_type.append(list(np.unique(group.frcst_type.values))[0])
    ensambles.append(names)

latc1,lonc1,pcen1=[],[],[]
for names, group in df_center2.groupby("ensamble_num"):
    latc1.append(list(group[group.code=="005002"]['Data'].values))
    lonc1.append(list(group[group.code=="006002"]['Data'].values))
    pcen1.append(list(group[group.code=="010051"]['Data'].values))
    
for i in range(len(pcen1)):
    pcen1[i].extend(pcen[i])
 
    
vhr=['0','6', '12', '18', '24', '30', '36', '42', '48', '54', '60', '66', '72', '78', '84', '90', '96', '102', '108']
for i in range(len(ensambles)):
    wind=[np.nan if value=='None' else float(value) for value in vmax[i]]
    pre=[np.nan if value=='None' else float(value)/100 for value in pcen1[i]]
    lon_=[np.nan if value=='None' else float(value) for value in lon[i]]
    lat_=[np.nan if value=='None' else float(value) for value in lat[i]]
    lon1_=[np.nan if value=='None' else float(value) for value in lonc[i]]
    lat1_=[np.nan if value=='None' else float(value) for value in latc[i]]
    max_radius=np.sqrt(np.square(np.array(lon_)-np.array(lon1_))+np.square(np.array(lat_)-np.array(lat1_)))*110
    timestamp=[(date_object + timedelta(hours=int(value))).strftime("%Y%m%d%H%M") for value in vhr]
    timestep_int=[int(value)  for value in vhr]
    ['TRUE' if frcst_type[i]==4 else 'False']    
    track = xr.Dataset(
            data_vars={
                    'max_sustained_wind': ('time', wind),
                    'central_pressure': ('time', pre),
                    'ts_int': ('time', timestep_int),
                    'max_radius': ('time', max_radius),
                    'lat': ('time', lat_),
                    'lon': ('time', lon_),
                    },
                    coords={'time': timestamp,
                            },
                            attrs={
                                    'max_sustained_wind_unit': 'm/s',
                                    'central_pressure_unit': 'mb',
                                    'name': 'name',
                                    'sid': 'NA',
                                    'orig_event_flag': False,
                                    'data_provider': 'ECMWF',
                                    'id_no': 'NA',
                                    'ensemble_number': ensambles[i],
                                    'is_ensemble': ['TRUE' if frcst_type[i]==4 else 'False'][0],
                                    'forecast_time': date_object.strftime("%Y%m%d%H%M"),
                                    'basin': 'WP',
                                    'category': 'NA',
                                    })
    track = track.set_coords(['lat', 'lon'])  
                #%%
    track = xr.Dataset(
        data_vars={
            'max_sustained_wind': ('time', (1.94384/0.84)*forcast_df.max_sustained_wind.values),
            'environmental_pressure': ('time', forcast_df.environmental_pressure.values),
            'central_pressure': ('time', forcast_df.central_pressure.values),
            'lat': ('time', forcast_df.lat.values),
            'lon': ('time', forcast_df.lon.values),
            'radius_max_wind':('time', estimate_rmw(forcast_df.radius_max_wind.values, forcast_df.central_pressure.values)),  
            'radius_oci':('time', estimate_roci(forcast_df.radius_max_wind.values, forcast_df.central_pressure.values)), 
            'time_step':('time', forcast_df.time_step.values),
        },
        coords={
            'time': forcast_df.time.values,
        },
        attrs={
            'max_sustained_wind_unit': 'kn',
            'central_pressure_unit': 'mb',
            'name': forcast_df.name,
            'sid': forcast_df.sid+str(forcast_df.ensemble_number),
            'orig_event_flag': True,#forcast_df.orig_event_flag,
            'data_provider': forcast_df.data_provider,
            'id_no': forcast_df.id_no,
            'ensemble_number': forcast_df.ensemble_number,
            'is_ensemble':forcast_df.is_ensemble,
            'forecast_time': forcast_df.forecast_time,
            'basin': forcast_df.basin,
            'category': forcast_df.category,
        }
        )
    
    
#%%
track = xr.Dataset(
        data_vars={
            'max_sustained_wind': ('time', (1.94384/0.84)*forcast_df.max_sustained_wind.values),
            'environmental_pressure': ('time', forcast_df.environmental_pressure.values),
            'central_pressure': ('time', forcast_df.central_pressure.values),
            'lat': ('time', forcast_df.lat.values),
            'lon': ('time', forcast_df.lon.values),
            'radius_max_wind':('time', estimate_rmw(forcast_df.radius_max_wind.values, forcast_df.central_pressure.values)),  
            'radius_oci':('time', estimate_roci(forcast_df.radius_max_wind.values, forcast_df.central_pressure.values)), 
            'time_step':('time', forcast_df.time_step.values),
        },
        coords={
            'time': forcast_df.time.values,
        },
        attrs={
            'max_sustained_wind_unit': 'kn',
            'central_pressure_unit': 'mb',
            'name': forcast_df.name,
            'sid': forcast_df.sid+str(forcast_df.ensemble_number),
            'orig_event_flag': True,#forcast_df.orig_event_flag,
            'data_provider': forcast_df.data_provider,
            'id_no': forcast_df.id_no,
            'ensemble_number': forcast_df.ensemble_number,
            'is_ensemble':forcast_df.is_ensemble,
            'forecast_time': forcast_df.forecast_time,
            'basin': forcast_df.basin,
            'category': forcast_df.category,
        }
        )
track = track.set_coords(['lat', 'lon'])    
#%%
for line in StringIO(text_data):
    [line.split()[-1] for line in StringIO(text_data) if line[6:17]=="004001 YEAR" ]
    #print(line)
#%%
from pybufrkit.renderer import FlatJsonRenderer
json_data = FlatJsonRenderer().render(bufr)  


msg = {
    # subset forecast data
    'significance': data_query(bufr, '316082' + '> 008005')}


for index in msg['significance'].subset_indices():
    sig = np.array(msg['significance'].get_values(index), dtype='int')
    print(sig)

#%%
from pybufrkit.decoder import generate_bufr_message
i=1
with open(os.path.join(path_ecmwf,ecmwf_file), 'rb') as ins:
    for bufr in generate_bufr_message(decoder1, ins.read()):
        text_data = FlatTextRenderer().render(bufr)
        f=open('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/temp/file_%s.txt'%i,'w')
        f.write(text_data)
        f.close()
        i=i+1

        
#%%    
    
# setup parsers and querents
npparser = pybufrkit.dataquery.NodePathParser()
data_query = pybufrkit.dataquery.DataQuerent(npparser).query

meparser = pybufrkit.mdquery.MetadataExprParser()
meta_query = pybufrkit.mdquery.MetadataQuerent(meparser).query
fcast_rep='116000'
fcast_rep='316082'

 # query the bufr message
msg = {
    # subset forecast data
    'significance': data_query(bufr, fcast_rep + '> 008005'),
    'latitude': data_query(bufr, fcast_rep + '> 005002'),
    'longitude': data_query(bufr, fcast_rep + '> 006002'),
    'wind_10m': data_query(bufr, fcast_rep + '> 011012'),
    'pressure': data_query(bufr, fcast_rep + '> 010051'),
    'timestamp': data_query(bufr, fcast_rep + '> 004024'),

    # subset metadata
    'wmo_longname': data_query(bufr, '/001027'),
    'storm_id': data_query(bufr, '/001025'),
    'ens_type': data_query(bufr, '/001092'),
    'ens_number': data_query(bufr, '/001091'),
}

timestamp_origin = dt.datetime(
    meta_query(bufr, '%year'), meta_query(bufr, '%month'),
    meta_query(bufr, '%day'), meta_query(bufr, '%hour'),
    meta_query(bufr, '%minute'),
)
timestamp_origin = np.datetime64(timestamp_origin)

id_no = timestamp_origin.item().strftime('%Y%m%d%H') +  str(np.random.randint(1e3, 1e4))

orig_centre = meta_query(bufr, '%originating_centre')
if orig_centre == 98:
    provider = 'ECMWF'
else:
    provider = 'BUFR code ' + str(orig_centre)
for index in msg['significance'].subset_indices():
    name = msg['wmo_longname'].get_values(i)[0].decode().strip()
#%%
for index in msg['significance'].subset_indices():
    name = msg['wmo_longname'].get_values(i)[0].decode().strip()
    sig = np.array(msg['significance'].get_values(index), dtype='int')
    lat = np.array(msg['latitude'].get_values(index), dtype='float')
    lon = np.array(msg['longitude'].get_values(index), dtype='float')
    wnd = np.array(msg['wind_10m'].get_values(index), dtype='float')
    pre = np.array(msg['pressure'].get_values(index), dtype='float')

    sid = msg['storm_id'].get_values(index)[0].decode().strip()

    timestep_int = np.array(msg['timestamp'].get_values(index)).squeeze()
    timestamp = timestamp_origin + timestep_int.astype('timedelta64[h]')
    print(name, sig, lat, lon, wnd, pre,sid)
    
#%%
f=open('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/temp/file4.txt','w')
f.write(text_data)
f.close()
#%%   
    for ecmwf_file in ecmwf_files:
        #ecmwf_file=ecmwf_files[0]
        f_name='ECMWF_'+ ecmwf_file.split('_')[1]+'_'+ecmwf_file.split('_')[4]
        model_name=ecmwf_file.split('_')[1][6:10]
        typhoon_name=ecmwf_file.split('_')[-4] 
        with open(os.path.join(path_ecmwf,ecmwf_file), 'rb') as bin_file:
            bufr_message = decoder.process(bin_file.read())
            text_data = FlatTextRenderer().render(bufr_message)
#path='home/fbf'

#%%
 
        #  decoder = pybufrkit.decoder.Decoder()

        # if hasattr(file, 'read'):
        #     bufr = decoder.process(file.read())
        # elif hasattr(file, 'read_bytes'):
        #     bufr = decoder.process(file.read_bytes())
        # elif os.path.isfile(file):
        #     with open(file, 'rb') as i:
        #         bufr = decoder.process(i.read())
        # else:
        #     raise FileNotFoundError('Check file argument')

        # # setup parsers and querents
        # npparser = pybufrkit.dataquery.NodePathParser()
        # data_query = pybufrkit.dataquery.DataQuerent(npparser).query

        # meparser = pybufrkit.mdquery.MetadataExprParser()
        # meta_query = pybufrkit.mdquery.MetadataQuerent(meparser).query

        # if fcast_rep is None:
        #     fcast_rep = self._find_delayed_replicator(
        #         meta_query(bufr, '%unexpanded_descriptors')
        #     )

        # # query the bufr message
        # msg = {
        #     # subset forecast data
        #     'significance': data_query(bufr, fcast_rep + '> 008005'),
        #     'latitude': data_query(bufr, fcast_rep + '> 005002'),
        #     'longitude': data_query(bufr, fcast_rep + '> 006002'),
        #     'wind_10m': data_query(bufr, fcast_rep + '> 011012'),
        #     'pressure': data_query(bufr, fcast_rep + '> 010051'),
        #     'timestamp': data_query(bufr, fcast_rep + '> 004024'),

        #     # subset metadata
        #     'wmo_longname': data_query(bufr, '/001027'),
        #     'storm_id': data_query(bufr, '/001025'),
        #     'ens_type': data_query(bufr, '/001092'),
        #     'ens_number': data_query(bufr, '/001091'),
        # }

        # timestamp_origin = dt.datetime(
        #     meta_query(bufr, '%year'), meta_query(bufr, '%month'),
        #     meta_query(bufr, '%day'), meta_query(bufr, '%hour'),
        #     meta_query(bufr, '%minute'),
        # )
        # timestamp_origin = np.datetime64(timestamp_origin)

        # if id_no is None:
        #     id_no = timestamp_origin.item().strftime('%Y%m%d%H') + \
        #             str(np.random.randint(1e3, 1e4))

        # orig_centre = meta_query(bufr, '%originating_centre')
        # if orig_centre == 98:
        #     provider = 'ECMWF'
        # else:
        #     provider = 'BUFR code ' + str(orig_centre)

        # for i in msg['significance'].subset_indices():
        #     name = msg['wmo_longname'].get_values(i)[0].decode().strip()
        #     track = self._subset_to_track(
        #         msg, i, provider, timestamp_origin, name, id_no
        #     )
        #     if track is not None:
        #         self.append(track)
        #     else:
        #         LOGGER.debug('Dropping empty track %s, subset %d', name, i)

 