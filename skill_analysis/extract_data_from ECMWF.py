# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 01:12:47 2019

@author: ATeklesadik
"""
import os, csv, sys 
import scipy

import xml.etree.ElementTree as ET
from os import listdir

data = ('C:/globus_data/z_tigge_c_ecmf_20180906000000_ifs_glob_prod_all_glo.xml')  
data2 = ('C:/globus_data/babj/2008/20080726/z_tigge_c_babj_20080726000000_GEFS_glob_prod_etctr_nwp.xml')

directory='C:\\globus_data\\kwbc'
file_list=[]
for path, subdirs, files in os.walk(directory):
    for name in files:
        print (os.path.join(path, name))
        file_list.append(os.path.join(path, name))

data2=file_list[1]
tree = ET.parse(data)
root = tree.getroot()
prod_center=root.find('header/productionCenter').text.split('\n')[0]
for members in root.findall('data'):
    Mtype=members.get('type')
    ensamble=members.get('member')
    cyc_speed = [name.text for name in members.findall('disturbance/cloneData/maximumWind/speed')]
    basin = [name.text for name in members.findall('disturbance/basin')]
    cyc_na = [name.text for name in members.findall('disturbancecy/cycloneName')]
    cyc_num = [name.text for name in members.findall('disturbancecy/cycloneNumber')]
    validt = [name.text for name in members.findall('disturbancecy/fix/validTime')]
    lat = [name.text for name in members.findall('disturbancecy/fix/latitude')]
    lon= [name.text for name in members.findall('disturbancecy/fix/longitude')]
# check the data type, there are three types of data in the forecast and each has a differnt 
#structure


for members in root.findall('data'):
    Mtype=members.get('type')
    if Mtype=='ensembleForecast':
        for members2 in members.findall('disturbance'):
            basin=members2.find('basin').text
            cycloneName=members.find('cycloneName')
            ensamble=members.get('member')#[member]
            #print vhr
            for members3 in members2.findall('fix'):
                validt=members3.find('validTime').text
                vhr=members3.get('hour')
                ws=members3.find('cycloneData/maximumWind/speed').text
                lat=ws=members3.find('cycloneData/maximumWind/latitude').text
                lon=members3.find('cycloneData/maximumWind/longitude').text
                print (basin,cycloneName,Mtype,ensamble,validt,vhr,lat,lon,ws)
    elif Mtype=='forecast':
        for members2 in members.findall('disturbance'):
            cyc_speed = [name.text for name in members2.findall('cloneData/maximumWind/speed')]
            cyc_cat = [name.text for name in members2.findall('cycloneData/development')]
            basin = [name.text for name in members2.findall('basin')]
            cyc_na = [name.text for name in members2.findall('cycloneName')]
            cyc_num = [name.text for name in members2.findall('cycloneNumber')]
            validt = [name.text for name in members2.findall('fix/validTime')]
            lat = [name.text for name in members2.findall('fix/latitude')]
            lon= [name.text for name in members2.findall('fix/longitude')]
            #ensamble=members.get('member')#[member]
            vhr=members2.findall('fix').get('hour')
            #print vhr
            for members3 in members2.findall('fix'):
                validt=members3.find('validTime').text
                vhr=members3.get('hour')
                ws=members3.find('cycloneData/maximumWind/speed').text
                lat=ws=members3.find('cycloneData/maximumWind/latitude').text
                lon=members3.find('cycloneData/maximumWind/longitude').text
                #print (basin,cycloneName,Mtype,validt,vhr,lat,lon,ws)
    elif Mtype=='analysis':
        tem_dic={}
        tem_dic['cyc_speed'] = [name.text for name in members.findall('disturbance/cloneData/maximumWind/speed')]
        tem_dic['cyc_cat = [name.text for name in members.findall('disturbance/cycloneData/development')]
        tem_dic['basin = [name.text for name in members.findall('disturbance/basin')]
        tem_dic['cyc_na = [name.text for name in members.findall('disturbancecy/cycloneName')]
        tem_dic['cyc_num = [name.text for name in members.findall('disturbancecy/cycloneNumber')]
        tem_dic[' validt = [name.text for name in members.findall('disturbancecy/fix/validTime')]
        tem_dic['lat = [name.text for name in members.findall('disturbancecy/fix/latitude')]
        tem_dic['lon= [name.text for name in members.findall('disturbancecy/fix/longitude')]
        
        for members2 in members.findall('disturbance'):
            basin=members2.find('basin')
            cycloneName=members2.find('cycloneName')
            #ensamble=members.get('member')#[member]
            #print vhr
            for members3 in members2.findall('fix'):
                validt=members3.find('validTime').text
                lat=ws=members3.find('latitude').text
                lon=members3.find('longitude').text
                #print (basin,cycloneName,Mtype,validt,lat,lon,ws)

    #else:        ws='NA'