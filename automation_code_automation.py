#!/bin/sh
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:40:17 2019

@author: ATeklesadik
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:24:03 2019

@author: ATeklesadik
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:07:45 2019

@author: ATeklesadik
"""

""
import sys
import os
import pandas as pd
import xml.etree.ElementTree as ET
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

from settings import *
from secrets import *
 


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

def sendemail(from_addr, to_addr_list, cc_addr_list, message, login, password, smtpserver=SMTP_SERVER):
    header  = 'From: %s\n' % from_addr
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'Cc: %s\n' % ','.join(cc_addr_list)
    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login,password)
    problems = server.sendmail(from_addr, to_addr_list, message)
    server.quit()
    return problems

def delete_old_files():
    if not os.path.exists("/home/fbf/forecast"):
        os.makedirs("/home/fbf/forecast")
    old_files=os.listdir("/home/fbf/forecast")        
    for item in old_files:
        item2=os.path.join("/home/fbf/forecast", item)
        if os.path.isdir(item2):
            shutil.rmtree(item2)
        else:
            os.remove(os.path.join("/home/fbf/forecast", item))

def create_ucl_metadata():
    mytsr_username=UCL_USERNAME
    mytsr_password=UCL_PASSWORD
    tsrlink='https://www.tropicalstormrisk.com/business/checkclientlogin.php?script=true'

    lin1='wget --no-check-certificate --keep-session-cookies --save-cookies tsr_cookies.txt --post-data "user=%s&pass=%s" -O loginresult.txt "%s"' %(mytsr_username,mytsr_password,tsrlink)
    lin2='wget --no-check-certificate -c --load-cookies tsr_cookies.txt -O RodeKruis.xml "https://www.tropicalstormrisk.com/business/include/dlxml.php?f=RodeKruis.xml"'
    fname=open("/home/fbf/forecast/batch_step1.sh",'w')
    fname.write(lin1+'\n')
    fname.write(lin2+'\n')
    fname.close()

    os.chdir('/home/fbf/forecast')
    try:
        p = subprocess.check_call(["sh","./batch_step1.sh"])
    except subprocess.CalledProcessError as e:
        raise ValueError(str(e))
    # p = subprocess.Popen(["sh","./batch_step1.sh"])
    # stdout, stderr = p.communicate()

############################
# Start Main Script 
############################

def run_main_script():

    print('---------------------AUTOMATION SCRIPT STARTED---------------------------------')
    print(str(datetime.now()))

    ##############################################################################
    ### Get events from GDACS
    ##############################################################################

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
            Activetyphoon.append(row['gdacs_eventname'][:row['gdacs_eventname'].rfind('-')])
            print(row['gdacs_eventname'][:row['gdacs_eventname'].rfind('-')])      


    ##############################################################################
    ### download metadata from UCL
    ##############################################################################

    delete_old_files()
    create_ucl_metadata()
    # Activetyphoon=['KAMMURI']
    
    parser = etree.XMLParser(recover=True)
    tree=etree.fromstring('/home/fbf/forecast/RodeKruis.xml', parser=parser)

    Pacific_basin=['wp','nwp','NWP','west pacific','north west pacific','northwest pacific']   
    try:
        tree = ET.parse('/home/fbf/forecast/RodeKruis.xml')
        root = tree.getroot()
        update=root.find('ActiveStorms/LatestUpdate').text
        print(update)
    except:
        pass        

    dict2={'WH':'windpast','GH':'gustpast','WF':'wind',
        'GF':'gust','WP0':'0_TSprob','WP1':'1_TSprob',
        'WP2':'2_TSprob','WP3':'3_TSprob','WP4':'4_TSprob',
        'WP5':'5_TSprob','WP6':'6_TSprob','WP7':'7_TSprob'}

    kml_files=[]
    fname=open("/home/fbf/forecast/batch_step2.sh",'w')
    TSRPRODUCT_FILENAMEs={}
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
            TSRPRODUCT_FILENAME=StormID+'_'+'gust'+'_'+AdvisoryDate+'.zip' 
            line='wget --no-check-certificate -c --load-cookies tsr_cookies.txt -O %s "https://www.tropicalstormrisk.com/business/include/dl.php?y=%s&b=NWP&p=%s&f=%s"' %(TSRPRODUCT_FILENAME,YYYY,'GF',TSRPRODUCT_FILENAME)
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
        p = subprocess.check_call(["sh","./batch_step2.sh"], cwd=r"/home/fbf/forecast")
    except subprocess.CalledProcessError as e:
        raise ValueError(str(e))
    # p = subprocess.Popen(["sh","./batch_step2.sh"])
    # stdout, stderr = p.communicate()

    filname1=[]
    filname1_={}

    for key, value in TSRPRODUCT_FILENAMEs.items():   # check for the storm name make this for all 
        if value in Activetyphoon:
            print(value)
            files = [f for f in os.listdir('/home/fbf/forecast') if re.match(r'%s+.*\.zip'% key, f)]
            print(files)
            with zipfile.ZipFile(os.path.join('/home/fbf/forecast', files[0]), 'r') as zip_ref:
                zip_ref.extractall(os.path.join('/home/fbf/forecast', value))
            filname1.append(os.path.join('/home/fbf/forecast', value))
            filname1_['%s' %key]=os.path.join('/home/fbf/forecast',value )


    fname=open("/home/fbf/forecast/typhoon_info_for_model.csv",'w')
    fname.write('filename,event'+'\n')
    for key,value in filname1_.items():
        ile_names = [fn for fn in os.listdir(value) if any(fn.endswith(ext) for ext in ['.shp'])]
        ile_names.sort()
        gust=os.path.join(value,ile_names[1])
        track=os.path.join(value,ile_names[0])
        gust_shp = gpd.read_file(gust)
        track_shp = gpd.read_file(track)
        track_gust = gpd.sjoin(track_shp,gust_shp, how="inner", op='intersects')
        dissolve_key=key+'_fo'
        track_gust = track_gust.dissolve(by='%s' % dissolve_key, aggfunc='max')
        ucl_interval=[0,12,24,36,48,72,96,120]
        date_object =datetime.strptime(update, '%H:%M UT, %d %b %Y')

        date_list=[(date_object + timedelta(hours=i)).strftime("%Y%m%d%H00") for i in ucl_interval]
        track_gust['YYYYMMDDHH']=date_list[:len(track_gust)]
        track_gust.index=track_gust['YYYYMMDDHH']
        track_gust['Lon']=track_gust['geometry'].apply(lambda p: p.x)
        track_gust['Lat']=track_gust['geometry'].apply(lambda p: p.y)
        track_gust['vmax']=track_gust['gust'].apply(lambda p: int(p.split(' ')[0])*0.868976)
        typhoon_fs=pd.DataFrame()
        typhoon_fs[['LAT','LON','VMAX']]=track_gust[['Lat','Lon','vmax']]
        typhoon_fs['STORMNAME']=StormName
        # typhoon_fs.to_csv( os.path.join(value,'%s_typhoon.csv' % value.split('\\')[-1]))
        # line=value.replace('\\','/')+'/%s_typhoon.csv' % value.split('\\')[-1]+','+ value.split('\\')[-1]
        typhoon_fs.to_csv( os.path.join(value,'%s_typhoon.csv' % value.split('/')[-1]))
        line=value+'/%s_typhoon.csv' % value.split('/')[-1]+','+ value.split('/')[-1]
        fname.write(line+'\n')

    fname.close()

    #############################################################
    #### download rainfall 
    #############################################################

    download_day = datetime.today()
    year_=str(download_day.year)
    ftp = FTP('ftp.cdc.noaa.gov')
    ftp.login(user=FTP_LOGIN, passwd = FTP_PASSWORD)
    path1='/Projects/Reforecast2/%s/' % year_
    ftp.cwd(path1)
    folderlist = ftp.nlst()
    path1_='%s/' % folderlist[-1]  
    ftp.cwd(path1_)
    folderlist = ftp.nlst()
    path2='%s/c00/latlon/' % folderlist[-1]  
    # path2='%s/c00/latlon/' % folderlist[-2]  #ADJUST!!!
    ftp.cwd(path2)
    downloadRainfallFiles('/home/fbf/forecast/',ftp)
    ftp.quit()

    #############################################################
    #### Run IBF model 
    #############################################################

    os.chdir('/home/fbf')
    try:
        p = subprocess.check_call(["Rscript", "/home/fbf/run_model.R"])
    except subprocess.CalledProcessError as e:
        raise ValueError(str(e))
    #p = subprocess.Popen(["sh","/home/fbf/run_typhoon_model.sh"], cwd=r"/home/fbf")
    #stdout, stderr = p.communicate()


    #############################################################
    ### send email in case of landfall-typhoon
    #############################################################

    landfall_typhones=[]
    try:
        fname2=open("/home/fbf/forecast/file_names.csv",'r')
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


###################
# Run Main Script #
###################

try:
    run_main_script()
except Exception as e:
    # Send email in case of error
    print(e)
    html = """<html><body><p>Look in Docker logs on server for more info.</p><p>""" + \
        str(e) + \
        """</p></body></html>"""
    sendemail(from_addr  = EMAIL_FROM,
            to_addr_list = EMAIL_LIST_ERROR,
            cc_addr_list = [],
            message = message(
                subject='Error in PHL Typhoon script',
                html=html,
                textfile=False,
                image=False),
            login  = EMAIL_LOGIN,
            password= EMAIL_PASSWORD)






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
 
