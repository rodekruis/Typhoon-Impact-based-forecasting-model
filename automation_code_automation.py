#!/bin/sh

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:07:45 2019

@author: ATeklesadik
"""
""
#%% import libraries  

import sys
import os
import pandas as pd
import xml.etree.ElementTree as ET
import lxml.etree as ET2
import subprocess
import feedparser
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import smtplib
from datetime import datetime
from datetime import timedelta
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email import encoders
import smtplib
from smtplib import SMTP_SSL as SMTP
import re
import zipfile
import geopandas as gpd
import fiona
from ftplib import FTP
import shutil
from lxml import etree
from os.path import relpath
from bs4 import BeautifulSoup
import requests
from os import listdir
from os.path import isfile, join
from pybufrkit.decoder import Decoder
from pybufrkit.renderer import FlatTextRenderer
from sys import platform
decoder = Decoder()

 
#%% define functions 
path='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/'
#path='home/fbf'
Pacific_basin=['wp','nwp','NWP','west pacific','north west pacific','northwest pacific']  
dict2={'WH':'windpast','GH':'gustpast','WF':'wind',
    'GF':'gust','WP0':'0_TSprob','WP1':'1_TSprob',
    'WP2':'2_TSprob','WP3':'3_TSprob','WP4':'4_TSprob',
    'WP5':'5_TSprob','WP6':'6_TSprob','WP7':'7_TSprob'}

##Path='home/fbf/'
os.chdir(path)
from settings import *
from secrets import *
from variables import *
 
#%% define functions 
############################
### Functions 
############################

def retrieve_all_gdacs_events():
    """
    Reads in the RSS feed from GDACS and returns all current events in a pandas data frame
    """

    feed = feedparser.parse('feed://gdacs.org/xml/rss.xml')
    events_out = pd.DataFrame(feed.entries)
    return events_out

# https://www.gdacs.org/datareport/resources/TC/1000604/
# https://www.gdacs.org/gts.aspx?eventid=1000605&eventtype=TC
 
def get_specific_events(gdacs_events_in, event_code_in):
    """
    Filters the GDACS events by type of event. Available event codes:
    EQ: earthquake
    TC: tropical cyclone
    DR: drought
    FL: flood
    VO: volcano
    Requires a pandas data frame as input. Returns a pandas data frame
    """
    return gdacs_events_in.query("gdacs_eventtype == '{}'".format(event_code_in))

def get_current_TC_events():
    current_events = retrieve_all_gdacs_events()
    flood_events = get_specific_events(current_events, 'TC')
    return flood_events

def downloadRainfallFiles(destination, ftp, file_pattern='apcp_sfc_'):
    filelist = ftp.nlst()
    for file in filelist:
      if ((file_pattern in file) and file.endswith('.grib2')):
              ftp.retrbinary("RETR " + file, open(os.path.join(destination,'rainfall_forecast.grib2'),"wb").write)
              print(file + " downloaded")
      return

def message(subject,html,textfile,image):
    msg = MIMEMultipart()
    msg['Date'] =  datetime.now().strftime('%d/%m/%Y %H:%M')
    msg['Subject'] = subject
    part = MIMEText(html, "html")
    msg.attach(part)
    # attaching text file to email body
    if textfile:
        fp = open(textfile[1:-2], 'rb')
        msg1 = MIMEMultipart('plain')
        msg1.set_payload(fp.read())
        fp.close()
        encoders.encode_base64(msg1)
        msg1.add_header('Content-Disposition','attachment', filename="impact.csv")
        msg.attach(msg1)
    # attaching image to email body
    if image:
        fp = open(image[1:-2], 'rb')
        image = MIMEImage(fp.read())
        fp.close()
        image.add_header('Content-Disposition','attachment', filename="impact.png")
        msg.attach(image)
    return msg.as_string()

def sendemail_gmail(from_addr, to_addr_list, cc_addr_list,login,password, message,smtpserver='smtp.gmail.com:587'):
    header  = 'From: %s\n' % from_addr
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'Cc: %s\n' % ','.join(cc_addr_list) 
    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login,password)
    problems = server.sendmail(from_addr, to_addr_list, message)# message)
    server.quit()
    return problems

def delete_old_files():
    if not os.path.exists("forecast"):
        os.makedirs("home/fbf/forecast")
    old_files=os.listdir("/home/fbf/forecast")        
    for item in old_files:
        item2=os.path.join("home/fbf/forecast", item)
        if os.path.isdir(item2):
            shutil.rmtree(item2)
        else:
            os.remove(os.path.join("home/fbf/forecast", item))


#%% define functions     
def sendemail(from_addr, to_addr_list, cc_addr_list, message, login, password, smtpserver=sMTP_SERVER):
    header  = 'From: %s\n' % from_addr
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'Cc: %s\n' % ','.join(cc_addr_list)
    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login,password)
    problems = server.sendmail(from_addr, to_addr_list, message)
    server.quit()
    return problems


    
#%% define functions 
def check_for_active_typhoon_in_PAR():
    
    ##############################################################################
    ### Get events from GDACS
    ##############################################################################
    #Tropical Cyclone Advisory Domain(TCAD) 4°N 114°E, 28°N 114°E, 28°N 145°E and 4°N 145°N.
    
   # PAR=np.array([[145, 35], [145, 5], [115, 5], [115, 35],[145, 35]])  # PAR area
    #TCDA=np.array([[155, 38], [155, 4], [100, 5], [100, 48],[155, 48]])  # TCDA area
    TCDA=np.array([[145, 28], [145, 4], [114, 5], [114, 28],[145, 28]])  # TCDA area
    Pacific_basin=['wp','nwp','NWP','west pacific','north west pacific','northwest pacific']   

    polygon = Polygon(TCDA) # create polygon
    event_tc=get_current_TC_events()
    Activetyphoon=[]

    for ind,row in event_tc.iterrows():
        p_cor=np.array(row['where']['coordinates'])
        point =Point(p_cor[0],p_cor[1])
        #print(point.within(polygon)) # check if a point is in the polygon 
        if point.within(polygon):
            eventid=row['gdacs_eventid']
            Activetyphoon.append(row['gdacs_eventname'][:row['gdacs_eventname'].rfind('-')])
            #print(row['gdacs_eventname'][:row['gdacs_eventname'].rfind('-')])      
    return Activetyphoon

    
#%%     
def download_rainfall_nomads(Input_folder,path,Alternative_data_point):
    """
    download rainfall 
    
    """
    if not os.path.exists(os.path.join(Input_folder,'rainfall/')):
        os.makedirs(os.path.join(Input_folder,'rainfall/'))
    
    rainfall_path=os.path.join(Input_folder,'rainfall/')
 
    url='https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.%s/'% Input_folder.split('/')[-3][:-2] #datetime.now().strftime("%Y%m%d")
    url2='https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.%s/'% Alternative_data_point #datetime.now().strftime("%Y%m%d")
    
    def listFD(url, ext=''):
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')
        return [url + node.get('href') for node in soup.find_all('a') if node.get('href').split('/')[-2] in ['00','06','12','18']]#.endswith(ext)]
    
    try:        
        base_url=listFD(url, ext='')[-1]
        base_url_hour=base_url+'prcp_bc_gb2/geprcp.t%sz.pgrb2a.0p50.bc_' % base_url.split('/')[-2]
        time_step_list=['06','12','18','24','30','36','42','48','54','60','66','72']
        rainfall_24=[base_url_hour+'24hf0%s'%t for t in time_step_list]
        rainfall_06=[base_url_hour+'06hf0%s'%t for t in time_step_list]
        rainfall_24.extend(rainfall_06)
        for rain_file in rainfall_24:
            print(rain_file)
            output_file= os.path.join(relpath(rainfall_path,path),rain_file.split('/')[-1]+'.grib2')
            #output_file= os.path.join(rainfall_path,rain_file.split('/')[-1]+'.grib2')
            batch_ex="wget -O %s %s" %(output_file,rain_file)
            os.chdir(path)
            print(batch_ex)
            p = subprocess.call(batch_ex ,cwd=path)
    except:
        base_url=listFD(url2, ext='')[-1]
        base_url_hour=base_url+'prcp_bc_gb2/geprcp.t%sz.pgrb2a.0p50.bc_' % base_url.split('/')[-2]
        time_step_list=['06','12','18','24','30','36','42','48','54','60','66','72']
        rainfall_24=[base_url_hour+'24hf0%s'%t for t in time_step_list]
        rainfall_06=[base_url_hour+'06hf0%s'%t for t in time_step_list]
        rainfall_24.extend(rainfall_06)
        for rain_file in rainfall_24:
            #output_file= os.path.join(rainfall_path,rain_file.split('/')[-1]+'.grib2')
            output_file= os.path.join(relpath(rainfall_path,path),rain_file.split('/')[-1]+'.grib2')
            batch_ex="wget -O %s %s" %(output_file,rain_file)
            os.chdir(path)
            print(batch_ex)
            p = subprocess.call(batch_ex ,cwd=path)
        
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

#%%                    


#%% define functions 

def download_rainfall(Input_folder):
    """
    download rainfall 
    
    """
    #s.makedirs(os.path.join(Input_folder,'rainfall/'))
    if not os.path.exists(os.path.join(Input_folder,'rainfall/')):
        os.makedirs(os.path.join(Input_folder,'rainfall/'))
    
    rainfall_path=os.path.join(Input_folder,'rainfall/')      
    download_day = datetime.today()
    year_=str(download_day.year)
    ftp = FTP('ftp.cdc.noaa.gov')
    ftp.login(user='anonymous', passwd = 'anonymous')
    path1='/Projects/Reforecast2/%s/' % year_
    ftp.cwd(path1)
    folderlist = ftp.nlst()
    path1_='%s/' % folderlist[-1]
    ftp.cwd(path1_)
    folderlist = ftp.nlst()
    try:
        path2='%s/c00/latlon/' % folderlist[-1]
        ftp.cwd(path2)
        filelist = ftp.nlst()
        for file in filelist:
            if ((file_pattern in file) and file.endswith('.grib2')):
                ftp.retrbinary("RETR " + file, open(os.path.join(rainfall_path,'rainfall_forecast.grib2'),"wb").write)
                print(file + " downloaded")
        #downloadRainfallFiles(rainfall_path,ftp)
        rainfall_error=False
    except:
        rainfall_error=True
        pass
    ftp.quit()

#%% define functions  
def jtcw_data(Input_folder):
    parser = ET2.XMLParser(recover=True)#
    output=[] 
    index_list=[' WARNING POSITION:',' 12 HRS, VALID AT:',' 24 HRS, VALID AT:',' 36 HRS, VALID AT:',' 48 HRS, VALID AT:',' 72 HRS, VALID AT:']
    index_list_id=[]
    index_list_wd=[]
    
    jtwc_content = BeautifulSoup(requests.get('https://www.metoc.navy.mil/jtwc/rss/jtwc.rss').content,parser=parser,features="lxml")#'html.parser')
    try:    
        for channel in jtwc_content.find_all('channel'):
            for item in channel.find_all('item'):
                for li in item.find_all('li'):
                    for href_li in li.find_all('a',href=True):
                        if href_li.text =='TC Warning Text ':
                            output.append(href_li['href'])        
        jtwc_content = BeautifulSoup(requests.get(output[0]).content,'html.parser')#parser=parser,features="lxml")#'html.parser')
        jtwc_=re.sub(' +', ' ', jtwc_content.text)
        listt=jtwc_.split('\r\n')
        listt=listt[listt.index(' WARNING POSITION:'):]        
        for i in index_list:
            index_list_id.append(listt[listt.index(i)+1].replace("NEAR ", "").replace("---", ","))            
        for i in listt:
            if (' '.join(i.split()[0:3])=='MAX SUSTAINED WINDS'):
                i_l=i.replace(",", "").split()
                index_list_wd.append(','.join([i_l[-5],i_l[-2]]))              
        jtwc_df = pd.DataFrame(index_list_id)
        jtwc_df['wind']=index_list_wd
        #jtwc_.split('\r\n')[2].strip('/')  name of the event 
        jtwc_df.to_csv(os.path.join(Input_folder,'JTCW_%s_%s.csv'%(Input_folder.split('/')[-3],Input_folder.split('/')[-4])),index=True)  
        #jtwc_df.to_csv('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/philipiness/jtwc_df.csv',index=False)
    except:
        pass
#%% 
def HK_data(Input_folder):
    HKfeed =  BeautifulSoup(requests.get('https://www.weather.gov.hk/wxinfo/currwx/tc_list.xml').content,parser=parser,features="lxml")#'html.parser')
    trac_data=[]
    try:
        HK_track =  BeautifulSoup(requests.get(HKfeed.find('tropicalcycloneurl').text).content,parser=parser,features="lxml")#'html.parser')    
        for WeatherReport in HK_track.find_all('weatherreport'):
            for forecast in WeatherReport.find_all('pastinformation'):
                l2=[forecast.find('index').text,forecast.find('latitude').text,forecast.find('longitude').text,forecast.find('time').text]
                print(l2)
                trac_data.append(l2)
                last_item=forecast.find('time').text
            for forecast in WeatherReport.find_all('forecastinformation'):
                l2=[forecast.find('index').text,forecast.find('latitude').text,forecast.find('longitude').text,last_item]
                print(l2)
                trac_data.append(l2)
        trac_data.to_csv(os.path.join(Input_folder,'HK_%s_%s.csv'%(Input_folder.split('/')[-3],Input_folder.split('/')[-4])),index=True)  
    except:
        pass
#%%
def download_ecmwf(Input_folder,filepatern):
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
            

#%%
def pre_process_ecmwf(Input_folder):
    """
    preprocess ecmwf forecast data downloaded above
    """
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
#%% define functions 

def create_ucl_metadata(path):
    mytsr_username=uCL_USERNAME
    mytsr_password=uCL_PASSWORD
    tsrlink='https://www.tropicalstormrisk.com/business/checkclientlogin.php?script=true'
    try:
        os.remove(os.path.join(path,"forecast/RodeKruis.xml"))        
        os.remove(os.path.join(path,"forecast/batch_ucl_metadata_download.bat"))
        os.remove(os.path.join(path,"forecast/batch_ucl_download.bat"))
        
    except:
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
    except:    #except  subprocess.CalledProcessError as e:
        pass
        #raise ValueError(str(stderr))
    # p = subprocess.Popen(["sh","./batch_step1.sh"])
    # stdout, stderr = p.communicate()   


#%%
def download_ucl_data(path,Input_folder):
    """
    download ucl data
    """
 
    create_ucl_metadata(path)
    if not os.path.exists(os.path.join(Input_folder,'UCL/')):
        os.makedirs(os.path.join(Input_folder,'UCL/'))
    
    
    #path_input=os.path.join(Input_folder,'UCL/') #Input_folder
    # parser = etree.XMLParser(recover=True) 
    parser2 = ET2.XMLParser(recover=True)#lxml is better in handling error in xml files 
    
    try:
        tree = ET2.parse(os.path.join(path,'forecast/RodeKruis.xml'),parser=parser2)
        root = tree.getroot()
        update=root.find('ActiveStorms/LatestUpdate').text
        print(update)
    except:
        pass        
    kml_files=[]
    #fname=open("forecast/batch_step2.bat",'w')
    fname=open(os.path.join(path,"forecast/batch_ucl_download.bat"),'w')  
    fname.write(':: runfile'+'\n')
    TSRPRODUCT_FILENAMEs={}
    ucl_path=relpath(Input_folder, os.path.join(path,'forecast/')).replace('\\','/')
    
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
    
    filname1=[]
    filname1_={}
    if not listdir(os.path.join(Input_folder,'UCL/'))==[]:
        zip_files = [f for f in listdir(os.path.join(Input_folder,'UCL/')) if '.zip' in f][0]
        for key, value in TSRPRODUCT_FILENAMEs.items():   # check for the storm name make this for all 
            with zipfile.ZipFile(os.path.join(Input_folder,'UCL/',zip_files), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(Input_folder,'UCL/%s'%zip_files.strip('.zip')))
                
            filname1.append(os.path.join(Input_folder,'UCL/%s/'%TSRPRODUCT_FILENAME_O.strip('.zip')))
            filname1_['%s' %value]=os.path.join(Input_folder,'UCL/%s/'%TSRPRODUCT_FILENAME_O.strip('.zip') )
        
        #fname=open(os.path.join(path,'forecast/Input/',"typhoon_info_for_model2.csv"),'w')
        #fname.write('filename,event'+'\n')
        for key,value in filname1_.items():
            ile_names = [fn for fn in os.listdir(value) if any(fn.endswith(ext) for ext in ['.shp'])]
            gust=os.path.join(value, [f for f in ile_names if f.split('_')[1]=='gust'][0])
            track=os.path.join(value, [f for f in ile_names if f.split('_')[1]=='forecasttrack'][0])
            gust_shp = gpd.read_file(gust)
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
            typhoon_fs['STORMNAME']=StormName
            #typhoon_fs.to_csv( os.path.join(value,'%s_typhoon.csv' % value.split('/')[-1]))
            typhoon_fs.to_csv( os.path.join(Input_folder,'UCL_%s_%s.csv' % (Input_folder.split('/')[-3],Input_folder.split('/')[-4])))
            #typhoon_fs.to_csv( os.path.join(Input_folder,'UCL_%s_%s.csv' % (AdvisoryDate,StormName)))
            line_=value+'/%s_typhoon.csv' % value.split('/')[-1]+','+ value.split('/')[-1] #StormName #
            #fname.write(line_+'\n')
        
        #fname.close()







#%% define functions 
############################
# Start Main Script 
############################

def run_main_script(path):
    print('---------------------AUTOMATION SCRIPT STARTED---------------------------------')
    print(str(datetime.now()))
    Activetyphoon=check_for_active_typhoon_in_PAR()
    forecast_available_fornew_typhoon= False#'False'
    ##############################################################################
    ### download metadata from UCL
    ##############################################################################
    if not Activetyphoon==[]:
        Active_typhoon=True#'True'
        #delete_old_files()
        

        
        for typhoons in Activetyphoon:                
            # Activetyphoon=['KAMMURI']
            #############################################################
            #### make input output directory for model 
            #############################################################
            fname=open(os.path.join(path,'forecast/Input/',"typhoon_info_for_model.csv"),'w')
            fname.write('source,filename,event,time'+'\n')

            Alternative_data_point=(datetime.strptime(datetime.now().strftime("%Y%m%d%H"), "%Y%m%d%H")-timedelta(hours=24)).strftime("%Y%m%d")
                 
            Input_folder=os.path.join(path,'forecast/Input/%s/%s/Input/'%(typhoons,datetime.now().strftime("%Y%m%d%H")))
            Output_folder=os.path.join(path,'forecast/Output/%s/%s/Output/'%(typhoons,datetime.now().strftime("%Y%m%d%H")))
            
            if not os.path.exists(Input_folder):
                os.makedirs(Input_folder)
            if not os.path.exists(Output_folder):
                os.makedirs(Output_folder)             
           
            #############################################################
            #### download ecmwf
            #############################################################
            filepatern='_track_%s_'% typhoons
            download_ecmwf(Input_folder,filepatern)
            pre_process_ecmwf(Input_folder)
            #############################################################
            #### download ucl data
            #############################################################
  
            download_ucl_data(path,Input_folder)
                
            #############################################################
            #### download rainfall 
            #############################################################
            
            #rainfall_path=Input_folder
            download_rainfall_nomads(Input_folder,path,Alternative_data_point)
            #download_rainfall(Input_folder)
            line_='UCL,'+'%sUCL_%s_%s.csv' % (Input_folder,Input_folder.split('/')[-3],typhoons )+','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H") #StormName #
            if os.path.exists('%sUCL_%s_%s.csv' % (Input_folder,Input_folder.split('/')[-3],typhoons )):
                fname.write(line_+'\n')
                forecast_available_fornew_typhoon=True#'True'
            
            line_='ECMWF,'+'%sECMWF_%s_%s.csv' % (Input_folder,Input_folder.split('/')[-3],typhoons )+','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H")  #StormName #
            if os.path.exists('%sECMWF_%s_%s.csv' % (Input_folder,Input_folder.split('/')[-3],typhoons )):
                fname.write(line_+'\n')
                forecast_available_fornew_typhoon=True#'True'
            line_='Rainfall,'+'%sRainfall' % Input_folder +','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H")  #StormName #
            #line_='Rainfall,'+'%sRainfall/' % Input_folder +','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H") #StormName #
            fname.write(line_+'\n')
            
            #############################################################
            #### download HK data
            #############################################################
            
            #HK_data(Input_folder)
            #line_='HK,'+'%s/HK_%s_%s.csv' % (Input_folder,Input_folder.split('/')[-3],typhoons )+','+ typhoons #StormName #
            #fname.write(line_+'\n')
            
            #############################################################
            #### download JTCW data
            #############################################################
            
            #JTCW_data(Input_folder)
            #line_='JTCW,'+'%s/JTCW_%s_%s.csv' % (Input_folder,Input_folder.split('/')[-3],typhoons )+','+ typhoons #StormName #
            #fname.write(line_+'\n')
    
            fname.close()

    
            #############################################################
            #### Run IBF model 
            #############################################################
            if forecast_available_fornew_typhoon and Active_typhoon:
                os.chdir(path)
                
                if platform == "linux" or platform == "linux2": #check if running on linux or windows os
                    # linux
                    try:
                        p = subprocess.check_call(["Rscript", "run_model.R", str(rainfall_error)])
                    except subprocess.CalledProcessError as e:
                        raise ValueError(str(e))
                elif platform == "win32": #if OS is windows edit the path for Rscript
                    try:
                        p = subprocess.check_call(["C:/Program Files/R/R-3.6.3/bin/Rscript", "run_model.R", str(rainfall_error)])
                    except subprocess.CalledProcessError as e:
                        raise ValueError(str(e))
                    
    
        
        
                #############################################################
                ### send email in case of landfall-typhoon
                #############################################################
        
                landfall_typhones=[]
                try:
                    fname2=open("forecast/%s_file_names.csv" % typhoons,'r')
                    for lines in fname2.readlines():
                        print(lines)
                        if (lines.split(' ')[1].split('_')[0]) !='"Nolandfall':
                            if lines.split(' ')[1] not in landfall_typhones:
                                landfall_typhones.append(lines.split(' ')[1])
                    fname2.close()
                except:
                    pass
                
                if not landfall_typhones==[]:
                    image_filename=landfall_typhones[0]
                    data_filename=landfall_typhones[1]
                    html = """\
                    <html>
                    <body>
                    <h1>IBF model run result </h1>
                    <p>Please find below a map and data with updated model run</p>
                    <img src="cid:Impact_Data">
                    </body>
                    </html>
                    """
                    sendemail(from_addr  = EMAIL_FROM,
                            to_addr_list = EMAIL_LIST,
                            cc_addr_list = CC_LIST,
                            message = message(
                                subject='Updated impact map for a new Typhoon in PAR',
                                html=html,
                                textfile=data_filename,
                                image=image_filename),
                            login  = EMAIL_LOGIN,
                            password= EMAIL_PASSWORD)


    print('---------------------AUTOMATION SCRIPT FINISHED---------------------------------')
    print(str(datetime.now()))

#%% run main script

###################
# Run Main Script #
###################
path='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/'
try:
    run_main_script(path)
except Exception as e:
    # Send email in case of error
    print(e)
    html = """<html><body><p>Look in Docker logs on server for more info.</p><p>""" + \
        str(e) + \
        """</p></body></html>""" 
    sendemail_gmail(from_addr  = EMAIL_FROM,
            to_addr_list = EMAIL_LIST_ERROR,
            cc_addr_list = [],
            message = message(
                subject='Error in PHL Typhoon script',
                html=html,
                textfile=False,
                image=False),
            login  = EMAIL_LOGIN,
            password= EMAIL_PASSWORD)



 
#%% Test email


###############################
# Test code for sending email #
###############################

# sendemail(from_addr  = EMAIL_FROM, 
#             to_addr_list = ['jannisvisser@redcross.nl'], 
#             cc_addr_list = [],  
#             message = message(
#                 subject='Test',
#                 html="",
#                 textfile=False,
#                 image=False),
#             login  = EMAIL_LOGIN, 
#             password= EMAIL_PASSWORD)
 
