# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:22:22 2019

@author: ATeklesadik
"""

import os, csv, sys 
import scipy
import pandas as pd
import gzip
import xml.etree.ElementTree as ET
from os import listdir
from datetime import datetime
import re
Pacific_basin=['wp','nwp','west pacific','north west pacific','northwest pacific']
file_list=[]
list_failed=[]
list_st=[]
directory='C:\\globus_data\\lfpw'
file_name=open('C:\\globus_data\\PAR\\lfpw.txt','w')
file_name.write('prod_center,model_name,Mtype,Emember,perturb,ID,cycloneName,cycloneNumber,validt,hour,longitude,latitude,development')
for path, subdirs, files in os.walk(directory):
    for name in files:
        #print (os.path.join(path, name))
        file_list.append(os.path.join(path, name))


for data in file_list:
    if data.endswith('.xml'):
        try:
            tree = ET.parse(data)
            root = tree.getroot()
            #model_name=root.find('header/generatingApplication/model/name').text 
        except:
            list_failed.append(data)
            pass
                
    elif data.endswith('.gz'):
        try:
            tree = ET.parse(gzip.open(data))  
            root = tree.getroot()
            #model_name=root.find('header/generatingApplication/model/name').text 
        except:
            list_failed.append(data)
            pass
    #root = tree.getroot()

    prod_center=root.find("header/productionCenter").text
    model_name=root.find('header/generatingApplication/model/name').text
    prod_center=re.sub('\s+',' ',prod_center)
    for members in root.findall('data'):
        Emember=members.attrib.get('member')
        perturb=members.attrib.get('control')
        Mtype=members.attrib.get('type')
        if Mtype in ['forecast','ensembleForecast']:
            for members2 in members.findall('disturbance'):
                ID=members2.attrib.get('ID')
                cycloneName=members2.find('cycloneName').text if members2 is not None else None
                cycloneNumber=members2.find('cycloneNumber').text if members2 is not None else None 
                basin=members2.find('basin').text if members2 is not None else None
                basin_check=basin.lower()
                for members3 in members2.findall('fix'):
                    hour=members3.attrib.get('hour')
                    validt= members3.find('validTime').text if members3 is not None else None
                    latitude= members3.find('latitude').text if members3 is not None else None
                    longitude= members3.find('longitude').text if members3 is not None else None
                    for members4 in members3.findall('cycloneData'):
                        development= members4.find('development').text if members4 is not None else None
                    if basin_check in Pacific_basin:
                        line=[prod_center,model_name ,Mtype,Emember,perturb,ID,cycloneName,cycloneNumber,validt,hour,longitude,latitude,development]
                        for el in line:
                            file_name.write(str(el))
                            file_name.write(',')
                        file_name.write('\n')
file_name.close()
#date_object = datetime.strptime(validt, "%Y-%m-%dT%H:%M:%SZ")
#s1 = date_object.strftime("%m/%d/%Y, %H:%M:%S")
import csv
csv_columns = list_st[0].keys()
csv_file = "C:\\globus_data\\PAR\\lfpw_all_1.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in list_st:
            writer.writerow(data)
except IOError:
    print("I/O error") 