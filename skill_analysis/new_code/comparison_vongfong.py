
# coding: utf-8

# In[ ]:


import re
import os, csv, sys 
import scipy
import pandas as pd
import gzip
import xml.etree.ElementTree as ET
from os import listdir
from datetime import datetime
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import random
from netCDF4 import Dataset
import itertools
import geopy.distance

from functions import *


# ## Create pandaframe from xml files and save it as csv file

# In[ ]:


## Parameters

cyclone_names = ['vongfong']

data_folder = '../data/vongfong_all_times/'
results_folder = '../CSVs/vongfong/'
figures_folder = '../figures/vongfong/'

delete_previous_results_files = 'y'
save_csv_files = 'y'  # one per model


# In[ ]:


# Create folders

for folder_name in [data_folder, results_folder, figures_folder]:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

        
# Initialise lists

file_list = []
file_list_short = []
list_failed1 = []
list_total = []
institutes = []
    

# Create list with filenames and delete previously created results files
    
for file in os.listdir(data_folder):
    if '.xml' in file:
        if '.xml' in file[-4:]:
            xml_file_short = file
        else:
            xml_file_short = file[:-3]  
        if xml_file_short not in file_list_short:
            file_list.append(data_folder+file)
            file_list_short.append(xml_file_short)
            institute = file.split('_')[3].lower()
            if institute not in institutes:
                institutes.append(institute)
                 
if delete_previous_results_files == 'y':
    for cyclone_name in cyclone_names:
        try:
            os.remove(results_folder+cyclone_name+'_all.csv')
        except:
            pass


# In[ ]:


# Loop over list of filenames         
        
for data in file_list:

    # Institute name from filename
    institute_name = os.path.basename(data).split('_')[3].lower()

    # Read file
    if data.endswith('.xml'):
        try:
            tree = ET.parse(data)
            root = tree.getroot()
        except:
            list_failed1.append(data)
            pass

    elif data.endswith('.gz'):
        try:
            tree = ET.parse(gzip.open(data))  
            root = tree.getroot()
        except:
            list_failed1.append(data)
            pass

    # Check how many files could not be parsed
    try:
        model_name=root.find('header/generatingApplication/model/name').text 
    except:
        model_name='NAN'
        pass
    print(len(list_failed1))
    prod_center=root.find('header/productionCenter').text
    baseTime=root.find('header/baseTime').text

    ## Create one dictonary for each time point, and append it to a list
    
    for members in root.findall('data'):
        
        Mtype=members.get('type')
         
        for members2 in members.findall('disturbance'):
            try: 
                cyclone_name = [name.text.lower().strip() for name in members2.findall('cycloneName')]
            except:
                cyclone_name = [' ']
                
            if cyclone_name and cyclone_name[0] in cyclone_names:
                
                list_data = []
        
                if Mtype in ['forecast','ensembleForecast']:
                    for members3 in members2.findall('fix'):
                        tem_dic={}
                        tem_dic['Mtype']=[Mtype]
                        tem_dic['institute_name']=[institute_name.lower()]
                        tem_dic['product']=[re.sub('\s+',' ',prod_center).strip().lower()]
                        if model_name != 'NAN':
                            tem_dic['model_name']=[model_name.lower()]
                        else:
                            tem_dic['model_name'] = tem_dic['product']
                        tem_dic['basin'] = [name.text for name in members2.findall('basin')]
                        tem_dic['cycloneName'] = cyclone_name
                        tem_dic['cycloneNumber'] = [name.text for name in members2.findall('cycloneNumber')]
                        tem_dic['ensemble']=[members.get('member')]#[member]
                        tem_dic['cyc_speed'] = [name.text for name in members3.findall('cycloneData/maximumWind/speed')]
                        #tem_dic['cyc_speed'] = [name.text for name in members3.findall('cycloneData/minimumPressure/pressure')]
                        tem_dic['cyc_cat'] = [name.text for name in members3.findall('cycloneData/development')]
                        time = [name.text for name in members3.findall('validTime')]
                        tem_dic['time'] = ['/'.join(time[0].split('T')[0].split('-'))+', '+time[0].split('T')[1][:-1]]
                        tem_dic['lat'] = [name.text for name in members3.findall('latitude')]
                        tem_dic['lon']= [name.text for name in members3.findall('longitude')]                
                        tem_dic['vhr']=[members3.get('hour')]
    #                     validt=tem_dic['validt'][0].split('-')[0]+tem_dic['validt'][0].split('-')[1]+tem_dic['validt'][0].split('-')[2][:2]+tem_dic['validt'][0].split('-')[2][3:5]
    #                     date_object = datetime.strptime(validt, "%Y%m%d%H")
    #                     s1 = date_object.strftime("%m/%d/%Y, %H:00:00")
    #                     tem_dic['time']=[s1]
    #                     validt2=members2.get('ID').split('_')[0]
    #                     date_object = datetime.strptime(validt2, "%Y%m%d%H")
    #                     s2 = date_object.strftime("%m/%d/%Y, %H:00:00")
    #                     tem_dic['validt2']=[s2] 
                        tem_dic['forecast_time'] = ['/'.join(baseTime.split('T')[0].split('-'))+', '+baseTime.split('T')[1][:-1]]
                        tem_dic1 = dict( [(k,''.join(str(e).lower().strip() for e in v)) for k,v in tem_dic.items()])
                        list_data.append(tem_dic1)
                        list_total.append(tem_dic1)
                    
                elif Mtype=='analysis':

                    tem_dic={}
                    tem_dic['Mtype']=['analysis']
                    tem_dic['institute_name']=[institute_name.lower()]
                    tem_dic['product']=[re.sub('\s+',' ',prod_center).strip().lower()]
                    if model_name != 'NAN':
                        tem_dic['model_name']=[model_name.lower()]
                    else:
                        tem_dic['model_name'] = tem_dic['product']
                    tem_dic['basin']= [name.text for name in members2.findall('basin')]
                    tem_dic['cycloneName'] = cyclone_name
                    tem_dic['cycloneNumber'] = [name.text for name in members2.findall('cycloneNumber')]
                    tem_dic['ensemble']=['NAN']
                    tem_dic['cyc_speed'] = [name.text for name in members2.findall('cycloneData/maximumWind/speed')]
                    #tem_dic['cyc_speed'] = [name.text for name in members2.findall('cycloneData/minimumPressure/pressure')]
                    tem_dic['cyc_cat'] = [name.text for name in members2.findall('cycloneData/development')]
                    time = [name.text for name in members2.findall('fix/validTime')]
                    tem_dic['time'] = ['/'.join(time[0].split('T')[0].split('-'))+', '+time[0].split('T')[1][:-1]]
                    tem_dic['lat'] = [name.text for name in members2.findall('fix/latitude')]
                    tem_dic['lon']= [name.text for name in members2.findall('fix/longitude')]
                    tem_dic['vhr']=[members2.get('hour')]
    #                 validt=tem_dic['validt'][0].split('-')[0]+tem_dic['validt'][0].split('-')[1]+tem_dic['validt'][0].split('-')[2][:2]+tem_dic['validt'][0].split('-')[2][3:5]
    #                 date_object = datetime.strptime(validt, "%Y%m%d%H")
    #                 s1 = date_object.strftime("%m/%d/%Y, %H:00:00")
    #                 tem_dic['validt']=[s1]
    #                 validt2=members2.get('ID').split('_')[0]
    #                 date_object = datetime.strptime(validt2, "%Y%m%d%H")
    #                 s2 = date_object.strftime("%m/%d/%Y, %H:00:00")
    #                 tem_dic['validt2']=[s2]
                    tem_dic['forecast_time'] = ['/'.join(baseTime.split('T')[0].split('-'))+', '+baseTime.split('T')[1][:-1]]
                    tem_dic1 = dict( [(k,''.join(str(e).lower().strip() for e in v)) for k,v in tem_dic.items()])
                    list_data.append(tem_dic1)
                    list_total.append(tem_dic1)

                # Save the databases to the csv files (one for each institute)
                if save_csv_files == 'y':

                    # Define csv file
                    csv_file = results_folder+cyclone_name[0]+'_all.csv'

                    # Headers
                    csv_columns = tem_dic1.keys()

                    if os.path.exists(csv_file):
                        append_write = 'a' # append if already exists
                    else:
                        append_write = 'w' # make a new file if not

                    # Write data
                    try:
                        with open(csv_file, append_write) as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                            if append_write == 'w':
                                writer.writeheader()
                            for row in list_data:
                                writer.writerow(row)
                    except IOError:
                        print("I/O error")
            
            
## Create pandas dataframe
df_store=pd.DataFrame(list_total)
df_store = df_store[['Mtype', 'institute_name', 'product', 'model_name', 'basin', 'cycloneName', 'cycloneNumber', 'ensemble', 'cyc_speed', 'cyc_cat', 'time', 'lat', 'lon', 'vhr', 'forecast_time']]

