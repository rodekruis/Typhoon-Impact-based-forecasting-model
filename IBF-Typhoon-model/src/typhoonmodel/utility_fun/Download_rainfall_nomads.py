from typhoonmodel.utility_fun.settings import *
import os
from ftplib import FTP
import sys
import os
import re
import zipfile
from pybufrkit.renderer import FlatTextRenderer
from sys import platform
import urllib.request
import requests
from bs4 import BeautifulSoup
from os.path import relpath
import subprocess
from os import listdir
from os.path import isfile, join
import geopandas as gpd
import pandas as pd
from typhoonmodel.utility_fun.settings import *
import rasterio


def download_rainfall_nomads():
    """
    download rainfall 
    
    """

 
    url='https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.%s/'% data_point #datetime.now().strftime("%Y%m%d")
    url2='https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.%s/'% Alternative_data_point #datetime.now().strftime("%Y%m%d")
    
    def listFD(url, ext=''):
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')
        return [url + node.get('href') for node in soup.find_all('a') if node.get('href').split('/')[-2] in ['00','06','12','18']]#.endswith(ext)]
    
    try:        
        base_url=listFD(url, ext='')[-1]
        base_url_hour=base_url+'prcp_bc_gb2/geprcp.t%sz.pgrb2a.0p50.bc_' % base_url.split('/')[-2]
        time_step_list=['06','12','18','24','30','36','42','48','54','60','66','72']
        time_step_list2=['24','30','36','42','48','54','60','66','72']
        rainfall_24=[base_url_hour+'24hf0%s'%t for t in time_step_list2]
        rainfall_06=[base_url_hour+'06hf0%s'%t for t in time_step_list]
        rainfall_24.extend(rainfall_06)
        for rain_file in rainfall_24:
            print(rain_file)
            #output_file= os.path.join(relpath(rainfall_path,path),rain_file.split('/')[-1]+'.grib2')
            output_file= os.path.join(rainfall_path,rain_file.split('/')[-1]+'.grib2')
            batch_ex="wget -O %s %s" %(output_file,rain_file)
            os.chdir(MAIN_DIRECTORY)
            print(batch_ex)
            p = subprocess.call(batch_ex ,cwd=MAIN_DIRECTORY)
    except:
        base_url=listFD(url2, ext='')[-1]
        base_url_hour=base_url+'prcp_bc_gb2/geprcp.t%sz.pgrb2a.0p50.bc_' % base_url.split('/')[-2]
        time_step_list=['06','12','18','24','30','36','42','48','54','60','66','72']
        rainfall_24=[base_url_hour+'24hf0%s'%t for t in time_step_list]
        rainfall_06=[base_url_hour+'06hf0%s'%t for t in time_step_list]
        rainfall_24.extend(rainfall_06)
        for rain_file in rainfall_24:
            output_file= os.path.join(rainfall_path,rain_file.split('/')[-1]+'.grib2')
            #output_file= os.path.join(relpath(rainfall_path,path),rain_file.split('/')[-1]+'.grib2')
            batch_ex="wget -O %s %s" %(output_file,rain_file)
            os.chdir(MAIN_DIRECTORY)
            print(batch_ex)
            p = subprocess.call(batch_ex ,cwd=MAIN_DIRECTORY)
        
    rain_files = [f for f in listdir(rainfall_path) if isfile(join(rainfall_path, f))]
    os.chdir(rainfall_path)
    pattern1='.pgrb2a.0p50.bc_06h'
    pattern2='.pgrb2a.0p50.bc_24h'
    for files in rain_files:
        if pattern2 in files:
            p = subprocess.call('wgrib2 %s -append -netcdf rainfall_24.nc'%files ,cwd=rainfall_path)
            os.remove(files)
        if pattern1 in files:
            p = subprocess.call('wgrib2 %s -append -netcdf rainfall_06.nc'%files ,cwd=rainfall_path)
            os.remove(files)
