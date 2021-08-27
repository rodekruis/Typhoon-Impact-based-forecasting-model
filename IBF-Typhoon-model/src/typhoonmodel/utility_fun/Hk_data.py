import re
import shutil
import sys
import os
from lxml import etree
from os.path import relpath
from bs4 import BeautifulSoup
import requests
from os import listdir
from os.path import isfile, join
from sys import platform

def download_hk(Input_folder):
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