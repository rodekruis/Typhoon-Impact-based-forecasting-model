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

os.chdir(path+'/lib')
import Check_for_active_typhoon_in_PAR #import check_for_active_typhoon_in_PAR as check_for_active_typhoon_in_PAR
import Create_ucl_metadata #import create_ucl_metadata as create_ucl_metadata
import Delete_old_files #import delete_old_files as delete_old_files
import Download_ecmwf #import download_ecmwf as download_ecmwf
import Download_rainfall #import download_rainfall as download_rainfall
import Download_rainfall_nomads #import download_rainfall_nomads as download_rainfall_nomads
import Download_ucl_data #import download_ucl_data as download_ucl_data
import Hk_data #import hk_data as hk_data
import Jtcw_data #import jtcw_data as jtcw_data
import Pre_process_ecmwf #import pre_process_ecmwf as pre_process_ecmwf
import Sendemail #import sendemail as sendemail
import Sendemail_gmail #import sendemail_gmail as sendemail_gmail
import Run_main_script #import run_main_script as run_main_script


os.chdir(path)
from settings import *
from secrets import *
from variables import *

#%% define functions 

############################
### Functions 
############################
forecast_available_fornew_typhoon= False#'False'
Activetyphoon=Check_for_active_typhoon_in_PAR.check_for_active_typhoon_in_PAR()

if not Activetyphoon==[]:
    Active_typhoon=True
    for typhoons in Activetyphoon:                
        #############################################################
        #### make input output directory for model 
        #############################################################
        fname=open(os.path.join(path,'forecast/Input/',"typhoon_info_for_model.csv"),'w')
        fname.write('source,filename,event,time'+'\n')
        Alternative_data_point=(datetime.strptime(time_now.strftime("%Y%m%d%H"), "%Y%m%d%H")-timedelta(hours=24)).strftime("%Y%m%d")
        Input_folder=os.path.join(path,'forecast/Input/%s/%s/Input/'%(typhoons,time_now.strftime("%Y%m%d%H")))
        Output_folder=os.path.join(path,'forecast/Output/%s/%s/Output/'%(typhoons,time_now.strftime("%Y%m%d%H")))
        if not os.path.exists(Input_folder):
            os.makedirs(Input_folder)
        if not os.path.exists(Output_folder):
            os.makedirs(Output_folder)            
        filepatern='_track_%s_'% typhoons


#%% r
try:
    run_main_script(path)
except Exception as e:
    # Send email in case of error
    print(e)
    html = """<html><body><p>Look in Docker logs on server for more info.</p><p>""" + str(e) + """</p></body></html>""" 
    #sendemail_gmail(from_addr  = EMAIL_FROM,to_addr_list = EMAIL_LIST_ERROR,cc_addr_list = [],message = message(subject='Error in PHL Typhoon script',     html=html, textfile=False,           image=False),          login  = EMAIL_LOGIN,    password= EMAIL_PASSWORD)



 
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
 
