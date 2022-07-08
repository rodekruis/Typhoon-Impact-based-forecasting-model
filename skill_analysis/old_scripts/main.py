# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:01:00 2020

@author: ATeklesadik
"""
import sys
import os
import pandas as pd
import feedparser
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from datetime import datetime
from datetime import timedelta
import smtplib
from smtplib import SMTP_SSL as SMTP
import geopandas as gpd
import fiona
from ftplib import FTP
import shutil
from os.path import relpath
import re
import zipfile
from os.path import relpath
from os import listdir
from os.path import isfile, join
from pybufrkit.decoder import Decoder
from pybufrkit.renderer import FlatTextRenderer
from sys import platform
from io import StringIO
import numpy as np
from bs4 import BeautifulSoup
import subprocess
 
decoder = Decoder()
path='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/'
##Path='home/fbf/'

sys.path.insert(0, path+'lib')
os.chdir(path)
from settings import fTP_LOGIN, fTP_PASSWORD, uCL_USERNAME, uCL_PASSWORD

#from settings import *
from secrets import *
from variables import *
from ecmwfDataDownload import Ecmwf_data
from JtcwDataDownload import Jtcw_data
from uclDataDownload import ucl_data
from notification import Sendemail
from CheckActiveTyphoon import Check_for_active_typhoon
from RainfallDataDownload import Rainfall_data
 
#from typhoons_tracks_forecast import TCForecast 

#%% 


def automation_sript(path):
    print('---------------------AUTOMATION SCRIPT STARTED---------------------------------')
    print(str(datetime.now()))
    Activetyphoon=Check_for_active_typhoon.check_active_typhoon()
    forecast_available_fornew_typhoon= False#'False'
    ##############################################################################
    ### download metadata from UCL
    ##############################################################################
    if not Activetyphoon==[]:
        Active_typhoon=True#'True'
        #delete_old_files()      
        for typhoons in Activetyphoon:            
            #typhoons='SURIGAE'  # to run it manually for any typhoon 
            
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
            Ecmwf_data.ecmwf_data_download(Input_folder,filepatern)
        
            Ecmwf_data.ecmwf_data_process(Input_folder,filepatern)
            #############################################################
            #### download ucl data
            #############################################################
 
      
 
            ucl_data.create_ucl_metadata(path,uCL_USERNAME,uCL_PASSWORD)
            ucl_data.process_ucl_data(path,Input_folder,uCL_USERNAME,uCL_PASSWORD)
  
            #download_ucl_data(path,Input_folder)
                
            #############################################################
            #### download rainfall 
            #############################################################
            
            #rainfall_path=Input_folder
            Rainfall_data.download_rainfall_nomads(Input_folder,path,Alternative_data_point)
            #download_rainfall(Input_folder)
            line_='UCL,'+'%sUCL_%s_%s.csv' % (Input_folder,Input_folder.split('/')[-3],typhoons )+','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H") #StormName #
            if os.path.exists('%sUCL_%s_%s.csv' % (Input_folder,Input_folder.split('/')[-3],typhoons )):
                fname.write(line_+'\n')
                forecast_available_fornew_typhoon=True#'True'
            line_='ECMWF,'+'%sECMWF_%s_%s_ECMF.csv' % (Input_folder,Input_folder.split('/')[-3],typhoons )+','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H")  #StormName #
            if os.path.exists('%sECMWF_%s_%s_ECMF.csv' % (Input_folder,Input_folder.split('/')[-3],typhoons )):
                fname.write(line_+'\n')
                forecast_available_fornew_typhoon=True#'True'
            line_='Rainfall,'+'%sRainfall' % Input_folder +','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H")  #StormName #
            #line_='Rainfall,'+'%sRainfall/' % Input_folder +','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H") #StormName #
            fname.write(line_+'\n')
            
            #############################################################
            #### download HK data
            #############################################################
            #Hk_data.download_hk(Input_folder)
            #line_='HK,'+'%s/HK_%s_%s.csv' % (Input_folder,Input_folder.split('/')[-3],typhoons )+','+ typhoons #StormName #
            #fname.write(line_+'\n')
            #############################################################
            #### download JTCW data
            #############################################################
            Jtcw_data.jtcw_data(Input_folder)
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


 
#%% 
