# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:07:45 2019

@author: ATeklesadik
"""

""
import re
import os, csv, sys 
import scipy
import pandas as pd
import gzip
import xml.etree.ElementTree as ET
from os import listdir
from datetime import datetime
import csv
from subprocess import Popen
from pykml import parser
import feedparser
import rasterio as rio
import numpy as np
from rasterio import mask
from zipfile import ZipFile
from geopy.distance import distance
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import smtplib
 



########################## check current active typhoons in PAR and send email alert 

def sendemail(from_addr, to_addr_list, cc_addr_list,
              subject, message,
              login, password,
              smtpserver='smtp.gmail.com:587'):
    header  = 'From: %s\n' % from_addr
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'Cc: %s\n' % ','.join(cc_addr_list)
    header += 'Subject: %s\n\n' % subject
    message = header + message
 
    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login,password)
    problems = server.sendmail(from_addr, to_addr_list, message)
    server.quit()
    return problems

def retrieve_all_gdacs_events():
    """
    Reads in the RSS feed from GDACS and returns all current events in a pandas data frame
    """

    feed = feedparser.parse('feed://gdacs.org/xml/rss.xml')
    events_out = pd.DataFrame(feed.entries)
    return events_out

# https://www.gdacs.org/datareport/resources/TC/1000604/
    
#https://www.gdacs.org/gts.aspx?eventid=1000605&eventtype=TC
    
from bs4 import BeautifulSoup
import requests


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

PAR=np.array([[145, 35], [145, 5], [115, 5], [115, 35],[145, 35]])  # PAR area
polygon = Polygon(PAR) # create polygon


event_tc=get_current_TC_events()
Active_typhoon='False'
Activetyphoon=[]
for ind,row in event_tc.iterrows():
     p_cor=np.array(row['where']['coordinates'])
     point =Point(p_cor[0],p_cor[1])
     print(point.within(polygon)) # check if a point is in the polygon 
     if point.within(polygon):
         Active_typhoon='True'
         eventid=row['gdacs_eventid']
         Activetyphoon.append(row['gdacs_eventname'].split('-')[0])
         print(row['gdacs_eventname'].split('-')[0])      
r  = requests.get("https://www.gdacs.org/gts.aspx?eventid=%s&eventtype=TC" %eventid)
data = r.text
soup = BeautifulSoup(r.content,'html.parser')
forecast_info=''

for items in soup.find_all('div', class_='alert_section_auto'):
    item=items.find('p', class_="p_summary")
    ll=item.prettify()
    #print(ll.split('\n')[5].split(' ')[3] )
    try:
        if ll.split('\n')[5].split(' ')[3] in Activetyphoon:
            #print(ll.split('\n')[5].split(' ')[1],ll.split('\n')[5].split(' ')[3])
            forecast_info=forecast_info+'\n'+ll
    except:
        pass
        #print(item.get_text())
        #print(item)#.replace("<br/>", "\n").prettify())
print(forecast_info)

if not Activetyphoon==[]:
    sendemail(from_addr  = 'partyphoon@gmail.com', 
               to_addr_list = ['akliludin@gmail.com'], 
               cc_addr_list = [''],  subject  = 'Typhoon in PAR please check for avilable forecast',
               message= forecast_info, 
               login  = 'partyphoon', 
               password= '510typhoonModel')


##################download UCL DATA

mytsr_username="RodeKruis"
mytsr_password="TestRK1"
tsrlink='https://www.tropicalstormrisk.com/business/checkclientlogin.php?script=true'

lin1='wget --no-check-certificate --keep-session-cookies --save-cookies tsr_cookies.txt --post-data "user=%s&pass=%s" -O loginresult.txt "%s"' %(mytsr_username,mytsr_password,tsrlink)
lin2='wget --no-check-certificate -c --load-cookies tsr_cookies.txt -O RodeKruis.xml "https://www.tropicalstormrisk.com/business/include/dlxml.php?f=RodeKruis.xml"'
fname=open("C:\\documents\\philipiness\\temp\\input\\batch_step1.bat",'w')
fname.write(lin1+'\n')
fname.write(lin2+'\n')
fname.close()


os.chdir('C:\\documents\\philipiness\\temp\\input')
p = Popen("batch_step1.bat", cwd=r"C:\\documents\\philipiness\\temp\\input")
stdout, stderr = p.communicate()

kml_files=[]



Pacific_basin=['wp','nwp','west pacific','north west pacific','northwest pacific']   
try:
    tree = ET.parse('C:\\documents\\philipiness\\temp\\input\\RodeKruis.xml')
    root = tree.getroot()
    #model_name=root.find('header/generatingApplication/model/name').text 
except:
    pass        
update=root.find('ActiveStorms/LatestUpdate').text
print(update)
dict2={'WH':'windpast','GH':'gustpast','WF':'wind',
       'GF':'gust','WP0':'0_TSprob','WP1':'1_TSprob',
       'WP2':'2_TSprob','WP3':'3_TSprob','WP4':'4_TSprob',
       'WP5':'5_TSprob','WP6':'6_TSprob','WP7':'7_TSprob'}

fname=open("C:\\documents\\philipiness\\temp\\input\\batch_step2.bat",'w')

for members in root.findall('ActiveStorms/ActiveStorm'):
    basin=members.find('TSRBasinDesc').text    
    basin_check=basin.lower()
    if basin_check in Pacific_basin:
        print( basin_check)
        StormName=members.find('StormName').text
        StormID=members.find('StormID').text
        AdvisoryDate=members.find('AdvisoryDate').text
        TSRProductAvailability=members.find('TSRProductAvailability').text
        TSRProductAvailability=TSRProductAvailability.split(',')
        YYYY=StormID[0:4]
        TSRPRODUCT_FILENAME=StormID+'_'+'gust'+'_'+AdvisoryDate+'.zip'
        line='wget --no-check-certificate -c --load-cookies tsr_cookies.txt -O %s "https://www.tropicalstormrisk.com/business/include/dl.php?y=%s&b=NWP&p=%s&f=%s"' %(TSRPRODUCT_FILENAME,YYYY,'GF',TSRPRODUCT_FILENAME)
        fname.write(line+'\n')
        for items in TSRProductAvailability:  
            TSRPRODUCT_FILENAME=StormID+'_'+dict2[items]+'_'+AdvisoryDate+'.zip'
            kml_files.append(TSRPRODUCT_FILENAME)
            line1='wget --no-check-certificate -c --load-cookies tsr_cookies.txt -O %s "https://www.tropicalstormrisk.com/business/include/dl.php?y=%s&b=NWP&p=%s&f=%s"' %(TSRPRODUCT_FILENAME,YYYY,items,TSRPRODUCT_FILENAME)
            print(line1)
            #fname.write(line+'\n')
fname.close()   

    
####################
p = Popen("batch_step2.bat", cwd=r"C:\documents\philipiness\temp\input")
stdout, stderr = p.communicate()
#######################
filename='C:\\documents\\philipiness\\temp\\input\\%s' %TSRPRODUCT_FILENAME
filename1='C:\\documents\\philipiness\\temp\\input\\%s' %TSRPRODUCT_FILENAME[:-4]

import zipfile
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(filename1)


import geopandas as gpd
import fiona
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

fiona.drvsupport.supported_drivers['KML'] = 'rw' # enable KML support which is disabled by default


gust=filename1+'\\'+TSRPRODUCT_FILENAME[:-4]+'.shp'
track=filename1+'\\'+TSRPRODUCT_FILENAME.split('_')[0]+'_forecasttrack_'+TSRPRODUCT_FILENAME.split('_')[2][:-4]+'.shp'

gust_shp = gpd.read_file(gust)
track_shp = gpd.read_file(track)

track_gust = gpd.sjoin(track_shp,gust_shp, how="inner", op='intersects')
track_gust = track_gust.dissolve(by='201919W_fo', aggfunc='max')
ucl_interval=[0,12,24,36,48,72,96,120]

##############3333downlaod other tyhoon data

date_object = datetime.strptime(TSRPRODUCT_FILENAME.split('_')[2][:-4], "%Y%m%d%H")
date_list=[(date_object + timedelta(hours=i)).strftime("%Y%m%d%H00") for i in ucl_interval]    #s1 = date_object.strftime("%m/%d/%Y, %H:00:00")
track_gust['YYYYMMDDHH']=date_list
track_gust.index=track_gust['YYYYMMDDHH']
track_gust['Lon']=track_gust['geometry'].apply(lambda p: p.x)
track_gust['Lat']=track_gust['geometry'].apply(lambda p: p.y)
track_gust['vmax']=track_gust['gust'].apply(lambda p: int(p.split(' ')[0])*0.868976)

typhoon_fs=pd.DataFrame()
typhoon_fs[['LAT','LON','VMAX']]=track_gust[['Lat','Lon','vmax']]
typhoon_fs['STORMNAME']=StormName
typhoon_fs.to_csv(r'C:\\documents\\philipiness\\temp\\new_typhoon.csv')



from ftplib import FTP
import os
from datetime import date, timedelta

def downloadFiles(destination, file_pattern='apcp_sfc_'):
    filelist = ftp.nlst()
    for file in filelist:
      if file_pattern in file:
              ftp.retrbinary("RETR " + file, open(os.path.join(destination,file),"wb").write)
              print(file + " downloaded")
      return


yesterday = date.today() 
year_=str(yesterday.year)
ftp = FTP('ftp.cdc.noaa.gov')
ftp.login(user='anonymous', passwd = 'anonymous')
#path1='/Projects/Reforecast2/%s/%s/' %(year_,md)
path1='/Projects/Reforecast2/%s/'% year_
ftp.cwd(path1)
folderlist = ftp.nlst()
path1_='%s/' % folderlist[-1]  
ftp.cwd(path1_)
folderlist = ftp.nlst()
path2='%s/c00/latlon/' % folderlist[-1]  
ftp.cwd(path2)
downloadFiles('C:/documents/philipiness/temp')
ftp.quit()
