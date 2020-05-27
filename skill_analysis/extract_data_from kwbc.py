# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:22:22 2019

@author: ATeklesadik
"""

import re
import os, csv, sys 
import scipy
import pandas as pd
import gzip
import xml.etree.ElementTree as ET
from os import listdir
from datetime import datetime
import csv

Pacific_basin=['wp','nwp','west pacific','north west pacific','northwest pacific']
file_list=[]
list_failed=[]
list_st=[]
directory2='C:\\globus_data\\CENS_CMC_GEFS_GFS'

for path, subdirs, files in os.walk(directory2):
    for name in files:
        #print (os.path.join(path, name))
        file_list.append(os.path.join(path, name))
df_store=pd.DataFrame()

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
    try:
        model_name=root.find('header/generatingApplication/model/name').text 
    except:
        model_name='NAN'
        pass
    #root = tree.getroot()
    print(len(list_failed))
    prod_center=root.find('header/productionCenter').text
    model_name=root.find('header/generatingApplication/model/name').text 
    for members in root.findall('data'):
        Mtype=members.get('type')
        if Mtype in ['forecast','ensembleForecast']:
            for members2 in members.findall('disturbance'):
                basin=members2.find('basin').text
                basin_check=basin.lower()
                if basin_check in Pacific_basin:

                    for members3 in members2.findall('fix'):
                        tem_dic={}
                        tem_dic['Mtype']=[Mtype]
                        tem_dic['model_name']=[model_name]
                        tem_dic['pro']=[prod_center.split('\n')[0]]
                        tem_dic['basin']=[members2.find('basin').text]
                        tem_dic['cycloneName'] = [(name.text).lower() for name in members2.findall('cycloneName')]
                        tem_dic['cycloneNumber'] = [name.text for name in members2.findall('cycloneNumber')]
                        tem_dic['ensamble']=[members.get('member')]#[member]
                        tem_dic['cyc_speed'] = [name.text for name in members3.findall('cycloneData/maximumWind/speed')]
                        tem_dic['cyc_pressure'] = [name.text for name in members3.findall('cycloneData/minimumPressure/pressure')]
                        tem_dic['cyc_cat'] = [name.text for name in members3.findall('cycloneData/development')]
                        tem_dic['validt'] = [name.text for name in members3.findall('validTime')]
                        tem_dic['lat'] = [name.text for name in members3.findall('latitude')]
                        tem_dic['lon']= [name.text for name in members3.findall('longitude')]                
                        tem_dic['vhr']=[members3.get('hour')]
                        tem_dic1 = dict( [(k,''.join(str(e) for e in v)) for k,v in tem_dic.items()])
                        validt=members2.get('ID').split('_')[0]
                        date_object = datetime.strptime(validt, "%Y%m%d%H")
                        s1 = date_object.strftime("%m/%d/%Y, %H:00:00")
                        tem_dic['validt2']=[s1]  
                        #tem_dic1 = dict( [(k,v) for k,v in tem_dic.items() if len(v)>0])
                        #df_tem=pd.DataFrame.from_dict(tem_dic1)
                        list_st.append(tem_dic1)
                        #df_store=df_store.append(tem_dic1)

        elif Mtype=='analysis':
            for members2 in members.findall('disturbance'):
                basin=members2.find('basin').text
                basin_check=basin.lower()
                if basin_check in Pacific_basin:
                    tem_dic={}
                    tem_dic['Mtype']=['analysis']
                    tem_dic['model_name']=[model_name]
                    tem_dic['pro']=[prod_center.split('\n')[0]]
                    tem_dic['basin']=[members2.find('basin').text]
                    tem_dic['cycloneName'] = [name.text for name in members2.findall('cycloneName')]
                    tem_dic['cycloneNumber'] = [name.text for name in members2.findall('cycloneNumber')]
                    tem_dic['ensamble']=['NAN']
                    tem_dic['cyc_speed'] = [name.text for name in members2.findall('cycloneData/maximumWind/speed')]
                    tem_dic['cyc_pressure'] = [name.text for name in members2.findall('cycloneData/minimumPressure/pressure')]
                    tem_dic['cyc_cat'] = [name.text for name in members2.findall('cycloneData/development')]
                    tem_dic['validt'] = [name.text for name in members2.findall('fix/validTime')]
                    tem_dic['lat'] = [name.text for name in members2.findall('fix/latitude')]
                    tem_dic['lon']= [name.text for name in members2.findall('fix/longitude')]
                    tem_dic['vhr']=[members2.get('hour')]
                    tem_dic1 = dict( [(k,''.join(str(e) for e in v)) for k,v in tem_dic.items()])
                    validt=members2.get('ID').split('_')[0]
                    date_object = datetime.strptime(validt, "%Y%m%d%H")
                    s1 = date_object.strftime("%m/%d/%Y, %H:00:00")
                    tem_dic['validt2']=[s1]  
                    #tem_dic1 = dict( [(k,v) for k,v in tem_dic.items() if len(v)>0])
                    #df_tem=pd.DataFrame.from_dict(tem_dic1)
                    #df_store=df_store.append(tem_dic1)
                    list_st.append(tem_dic1)


#date_object = datetime.strptime(validt, "%Y-%m-%dT%H:%M:%SZ")
#s1 = date_object.strftime("%m/%d/%Y, %H:%M:%S")
import csv
csv_columns = list_st[0].keys()
csv_file = "C:\\globus_data\\PAR\\kwbc_all2.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in list_st:
            writer.writerow(data)
except IOError:
    print("I/O error") 