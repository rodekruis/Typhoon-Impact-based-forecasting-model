# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:15:52 2020

@author: ATeklesadik
"""
import re
import zipfile
from ftplib import FTP
import shutil
import sys
import os
import xml.etree.ElementTree as ET
import lxml.etree as ET2
from os.path import relpath
from bs4 import BeautifulSoup
import requests
from os import listdir
from os.path import isfile, join
from sys import platform
from settings import fTP_LOGIN, fTP_PASSWORD, uCL_USERNAME, uCL_PASSWORD
import time
import subprocess
import geopandas as gpd
import pandas as pd
from datetime import datetime
from datetime import timedelta

Pacific_basin=['wp','nwp','NWP','west pacific','north west pacific','northwest pacific'] 
php_admin3 = gpd.read_file(path+'data-raw/phl_admin3_simpl2.shp')
dict2={'WH':'windpast','GH':'gustpast','WF':'wind',
    'GF':'gust','WP0':'0_TSprob','WP1':'1_TSprob',
    'WP2':'2_TSprob','WP3':'3_TSprob','WP4':'4_TSprob',
    'WP5':'5_TSprob','WP6':'6_TSprob','WP7':'7_TSprob'}

def create_ucl_metadata(path,uCL_USERNAME,uCL_PASSWORD):
    mytsr_username=uCL_USERNAME
    mytsr_password=uCL_PASSWORD
    tsrlink='https://www.tropicalstormrisk.com/business/checkclientlogin.php?script=true'
    try:
        os.remove(os.path.join(path,"forecast/RodeKruis.xml"))        
        os.remove(os.path.join(path,"forecast/batch_ucl_metadata_download.bat"))
        os.remove(os.path.join(path,"forecast/batch_ucl_download.bat"))
        
    except:
        print("failed to remove old metadata files")
        pass

    lin1='wget --no-check-certificate --keep-session-cookies --save-cookies tsr_cookies.txt --post-data "user=%s&pass=%s" -O loginresult.txt "%s"' %(mytsr_username,mytsr_password,tsrlink)
    lin2='wget --no-check-certificate -c --load-cookies tsr_cookies.txt -O RodeKruis.xml "https://www.tropicalstormrisk.com/business/include/dlxml.php?f=RodeKruis.xml"'
    fname=open(os.path.join(path,"forecast/batch_ucl_metadata_download.bat"),'w')
    #fname=open("forecast/batch_step1.bat",'w')
    fname.write(lin1+'\n')
    fname.write(lin2+'\n')
    fname.close()

    #os.chdir('forecast')
    try:
        #p = subprocess.check_call(["sh","./batch_step1.sh"])
        p = subprocess.Popen("%sforecast/batch_ucl_metadata_download.bat" % path,cwd=os.path.join(path,"forecast/"))
        stdout, stderr = p.communicate()
        print("download metadata")
    except:    #except  subprocess.CalledProcessError as e:
        print("failed to download metadata")
        pass
        #raise ValueError(str(stderr))
    # p = subprocess.Popen(["sh","./batch_step1.sh"])
    # stdout, stderr = p.communicate()  
    
def download_ucl_data(path,Input_folder,uCL_USERNAME,uCL_PASSWORD):
    """
    download ucl data
    """
    if not os.path.exists(os.path.join(Input_folder,'UCL/')):
        os.makedirs(os.path.join(Input_folder,'UCL/'))
    
    
    #path_input=os.path.join(Input_folder,'UCL/') #Input_folder
    # parser = etree.XMLParser(recover=True) 
    parser2 = ET2.XMLParser(recover=True)#lxml is better in handling error in xml files 
    kml_files=[]
    fname=open(os.path.join(path,"forecast/batch_ucl_download.bat"),'w')  
    fname.write(':: runfile'+'\n')
    TSRPRODUCT_FILENAMEs={}    
    ucl_path=relpath(Input_folder, os.path.join(path,'forecast/')).replace('\\','/')
    
    try:
        tree = ET2.parse(os.path.join(path,'forecast/RodeKruis.xml'),parser=parser2)
        root = tree.getroot()
        update=root.find('ActiveStorms/LatestUpdate').text
        print(update)    
        for members in root.findall('ActiveStorms/ActiveStorm'):
            basin=members.find('TSRBasinDesc').text    
            basin_check=basin.lower()
            if basin_check in Pacific_basin:
                print( basin_check)
                StormName=members.find('StormName').text
                StormID=members.find('StormID').text
                TSRPRODUCT_FILENAMEs['%s' % StormID]=StormName
                AdvisoryDate=members.find('AdvisoryDate').text
                TSRProductAvailability=members.find('TSRProductAvailability').text
                TSRProductAvailability=TSRProductAvailability.split(',')
                YYYY=StormID[0:4]
                TSRPRODUCT_FILENAME_O=StormName+'_'+AdvisoryDate+'.zip' 
                TSRPRODUCT_FILENAME=StormID+'_'+'gust'+'_'+AdvisoryDate+'.zip' 
                line='wget --no-check-certificate -c --load-cookies tsr_cookies.txt -O %s/%s/%s "https://www.tropicalstormrisk.com/business/include/dl.php?y=%s&b=NWP&p=%s&f=%s"' %(ucl_path,'UCL',TSRPRODUCT_FILENAME_O,YYYY,'GF',TSRPRODUCT_FILENAME)
                fname.write(line+'\n')
                for items in TSRProductAvailability:  
                    TSRPRODUCT_FILENAME=StormID+'_'+dict2[items]+'_'+AdvisoryDate+'.zip'
                    kml_files.append(TSRPRODUCT_FILENAME)
                    line1='wget --no-check-certificate -c --load-cookies tsr_cookies.txt -O %s "https://www.tropicalstormrisk.com/business/include/dl.php?y=%s&b=NWP&p=%s&f=%s"' %(TSRPRODUCT_FILENAME,YYYY,items,TSRPRODUCT_FILENAME)
                    print(line1)
    except:
        pass   
    fname.close()   
    
    
    ##############################################################################
    ### download data from UCL
    ##############################################################################
    
    try:
        #p = subprocess.check_call("batch_step2.bat" ,cwd=r"forecast")
        #p = subprocess.call(line ,cwd=os.path.join(path,"forecast"))
        p = subprocess.Popen("%sforecast/batch_ucl_download.bat" % path,cwd=os.path.join(path,"forecast/"))
       
        #p = subprocess.Popen("batch_step2.bat",cwd=os.path.join(Input_folder,"UCL"))
     
    except: # stderr as e: #except subprocess.CalledProcessError as e:
        #p = subprocess.call("%s/forecast/batch_step3.bat" % path,cwd=os.path.join(path,"forecast"))
        p = subprocess.Popen("%sforecast/batch_ucl_download.bat" % path,cwd=os.path.join(path,"forecast/"))
        pass
        #raise ValueError(str(e))
    return update
#%%

def extract_ucl_data(Input_folder):
    filname1=[]
    filname1_={}    
    zip_files = [f for f in listdir(os.path.join(Input_folder,'UCL/')) if '.zip' in f]
    zip_file = [f for f in zip_files if f.startswith(Input_folder.split('/')[-4])]
    dir_name = os.path.join(Input_folder,'UCL/')
    for item in zip_file: # loop through items in dir
        if item.endswith(".zip"): # check for ".zip" extension
            file_name = os.path.join(Input_folder,'UCL/%s'%item)
            dir_name1 = file_name.strip('.zip') 
            if not os.path.exists(dir_name1):
                os.makedirs(dir_name1)                
            zip_ref = zipfile.ZipFile(file_name) # create zipfile object
            zip_ref.extractall(dir_name1) # extract file to dir
            zip_ref.close() # close file
            filname1_['%s' %Input_folder.split('/')[-4]]=dir_name1
            filname1.append(dir_name1)
            os.remove(file_name) 
    return filname1_
                
#%%                
def process_ucl_data(path,Input_folder,uCL_USERNAME,uCL_PASSWORD):
    update=download_ucl_data(path,Input_folder,uCL_USERNAME,uCL_PASSWORD)
    time.sleep(10)
    filname1_=extract_ucl_data(Input_folder)
    for key,value in filname1_.items():
        ile_names = [fn for fn in os.listdir(value) if any(fn.endswith(ext) for ext in ['.shp'])]
        gust=os.path.join(value, [f for f in ile_names if f.split('_')[1]=='gust'][0])
        track=os.path.join(value, [f for f in ile_names if f.split('_')[1]=='forecasttrack'][0])
        gust_shp = gpd.read_file(gust)
        php_gust = gpd.sjoin(php_admin3,gust_shp, how="inner", op='intersects')
        php_gust['vmax'] = php_gust['gust'].apply(lambda x : float(x.split(' ')[0])*0.868976242)
        php_gust_mun = php_gust.groupby('adm3_pcode').agg({'vmax': 'max'})
        php_gust_mun.to_csv( os.path.join(Input_folder,'UCL_%s_%s_windspeed.csv' % (Input_folder.split('/')[-3],Input_folder.split('/')[-4])),index=True)
        track_shp = gpd.read_file(track)
        track_gust = gpd.sjoin(track_shp,gust_shp, how="inner", op='intersects')
        dissolve_key=track_gust.columns[0] #key+'_fo' 
        track_gust = track_gust.dissolve(by='%s' % dissolve_key, aggfunc='max')
        ucl_interval=[0,12,24,36,48,72,96,120]
        date_object =datetime.strptime(update, '%H:%M UT, %d %b %Y')    
        date_list=[(date_object + timedelta(hours=i)).strftime("%Y%m%d%H00") for i in ucl_interval]
        track_gust['YYYYMMDDHH']=date_list[:len(track_gust)]
        track_gust.index=track_gust['YYYYMMDDHH']
        track_gust['Lon']=track_gust['geometry'].apply(lambda p: p.x)
        track_gust['Lat']=track_gust['geometry'].apply(lambda p: p.y)
        track_gust['vmax']=track_gust['gust'].apply(lambda p: int(p.split(' ')[0]))*0.868976 #convert mph to knots
        typhoon_fs=pd.DataFrame()
        typhoon_fs[['LAT','LON','VMAX']]=track_gust[['Lat','Lon','vmax']]
        typhoon_fs['STORMNAME']=key
        #typhoon_fs.to_csv( os.path.join(value,'%s_typhoon.csv' % value.split('/')[-1]))
        typhoon_fs.to_csv( os.path.join(Input_folder,'UCL_%s_%s.csv' % (Input_folder.split('/')[-3],Input_folder.split('/')[-4])))
