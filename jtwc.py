from bs4 import BeautifulSoup
import re
import requests
import pandas as pd
import lxml.etree as ET2
import numpy as np
parser = ET2.XMLParser(recover=True)#
output=[] 
index_list=[' WARNING POSITION:',' 12 HRS, VALID AT:',' 24 HRS, VALID AT:',' 36 HRS, VALID AT:',' 48 HRS, VALID AT:',' 72 HRS, VALID AT:']
index_list_id=[]
index_list_wd=[]

path='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/forecast/'

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
    jtwc_df.to_csv('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/philipiness/jtwc_df.csv',index=False)
except:
    pass

#%%
HKfeed =  BeautifulSoup(requests.get('https://www.weather.gov.hk/wxinfo/currwx/tc_list.xml').content,parser=parser,features="lxml")#'html.parser')


HK_track =  BeautifulSoup(requests.get(HKfeed.find('tropicalcycloneurl').text).content,parser=parser,features="lxml")#'html.parser')
trac_data=[]
try:    
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
except:
    pass

#%%
from pybufrkit.decoder import Decoder
from pybufrkit.renderer import FlatTextRenderer
decoder = Decoder()
#%%
from ftplib import FTP
import os
path='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/forecast/ecmwf/'
filepatern='tropical_cyclone_track_NURI'

ftp=FTP("dissemination.ecmwf.int")
ftp.login("wmo","essential")
files=ftp.nlst()
ftp.cwd(os.path.join(ftp.pwd(),files[-1]))
file_list=ftp.nlst()

#%%
for files in file_list:
    if filepatern in files:
        ftp.retrbinary("RETR " +files,open(os.path.join(path,files),'wb').write)
        
#%%
from os import listdir
from os.path import isfile, join
ecmwf_files = [f for f in listdir(path) if isfile(join(path, f))]
#%%
for ecmwf_file in ecmwf_files:
    with open(os.path.join(path,ecmwf_file), 'rb') as bin_file:
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
    except:
        pass

            
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
    WIND[['time','wind_kmh']].to_csv(os.path.join(path,f_name+'_wind.csv'),index=False)
    LATITUDE[['time','loction','lat']].to_csv(os.path.join(path,f_name+'_latitude.csv'),index=False)
    LONGITUDE[['time','loction','lon']].to_csv(os.path.join(path,f_name+'_longitude.csv'),index=False)

    date_object ='%04d%02d%02d%02d'%(int(YEAR[0]),int(MONTH[0]),int(DAY[0]),int(HOUR[0]))
    date_object=datetime.strptime(date_object, "%Y%m%d%H")
    
    wind=WIND[['time','wind_kmh']] 
    wind['time']=wind.time.astype(int)
    wind['YYYYMMDDHH']=wind['time'].apply(lambda x: (date_object + timedelta(hours=x)).strftime("%Y%m%d%H") )
    
    lon=LONGITUDE[['time','loction','lon']]
    lon['time']=lon.time.astype(int)
    lon['YYYYMMDDHH']=lon['time'].apply(lambda x: (date_object + timedelta(hours=x)).strftime("%Y%m%d%H") )
    lon=lon[lon['loction'].isin(['1','4'])]
    lon=lon[~((lon.time == 0) & (lon.loction=='4'))]
    
    lat=LATITUDE[['time','loction','lat']]
    lat['time']=lat.time.astype(int)
    lat['YYYYMMDDHH']=lat['time'].apply(lambda x: (date_object + timedelta(hours=x)).strftime("%Y%m%d%H") )
    lat=lat[lat['loction'].isin(['1','4'])]
    lat=lat[~((lat.time == 0) & (lat.loction=='4'))]
    
    df_forecast=lon[['lon','YYYYMMDDHH']].set_index('YYYYMMDDHH').join(lat[['lat','YYYYMMDDHH']].set_index('YYYYMMDDHH'), on='YYYYMMDDHH').join(wind[['wind_kmh','YYYYMMDDHH']].set_index('YYYYMMDDHH'),on='YYYYMMDDHH')
    df_forecast=df_forecast.rename(columns={"lat": "LAT", "lon": "LON", "wind_kmh": "VMAX"})
    df_forecast['STORMNAME']=STORMNAME
    df_forecast.to_csv(os.path.join(path,f_name+'_forecast.csv'),index=True)
#%%

#%%
##### Meteorological attribute significance
##### Location of maximum wind	3
##### Location of the storm in the analysis	5
##### Location of the storm in the perturbed analysis	4
##### Outer limit or edge of storm	2
##### Storm centre	1





