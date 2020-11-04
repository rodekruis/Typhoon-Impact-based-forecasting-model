# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 20:28:23 2019

@author: ATeklesadik
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from datetime import datetime
def con_date(vt):
    try:
        cdd=(datetime.strptime(vt, "%Y-%m-%dT%H:00:00Z")).strftime("%Y-%m-%d %H:00:00")
    except:
        cdd='None'
        pass
    return cdd
def calculate_dis(row):
    return (np.sqrt((row['lat_delta']*row['lat_delta']) + (row['lon_delta']*row['lon_delta'])))*111
dtype={'Mtype':str,'pro':str,'basin':str,'cycloneName':str,'cycloneNumber':object,'ensamble':object,'cyc_speed':object,'cyc_pressure':object,'cyc_cat':str,'validt':object,'lat':np.float64,'lon':np.float64,'vhr':object,'validt2':np.datetime64}

sim = pd.read_table('C:\\globus_data\\PAR\\egrr_all.csv',sep=',',parse_dates=True)
sim=sim[sim.cycloneName.notnull()]
sim=sim[sim['cycloneName'] !='NAMELESS']
sim["cycloneName"] = (sim["cycloneName"].astype('str')).map(lambda x: x.lower())
sim=sim[['Mtype', 'model_name','ensamble', 'validt','cycloneName', 'cycloneNumber', 'cyc_speed', 'cyc_cat', 'lat', 'lon', 'vhr', 'validt2']]
#sim=sim[sim['vhr'] != 'None']

#sim2=sim.set_index(['validt2','cycloneName','vhr','ensamble'])
# kbc   ['CENS', 'CMC', 'GEFS', 'GFS']
#rjtd (['JMA-GEPS', 'JMA-GSM', 'JMA-TEPS', 'JMA-WEPS', 'NAN']
Obs = pd.read_table('C:\\globus_data\\PAR\\observation.txt',sep=',')
typhon_list = pd.read_table('C:\\globus_data\\PAR\\typhoon_dates.txt',sep='\t',parse_dates=True)
Obs["cycloneName"] = (Obs["Name"].astype('str')).map(lambda x: x.lower())
observed=np.unique(Obs["cycloneName"].values)
sim=sim[sim["cycloneName"].isin(observed)]

sim["validt"] = sim["validt"].map(lambda x: con_date(x))
steps_vhr=['6', '12', '18', '24', '30', '36', '42', '48', '54', '60', '66', '72', '78', '84', '90', '96','102', '108', '114', '120', '126', '132', '138', '144']
sim=sim[sim['vhr'].isin(steps_vhr)]
sim['vhr']=sim['vhr'].astype(np.int32)


sim=sim[sim['ensamble']!='None']
sim['ensamble']=pd.to_numeric(sim['ensamble'],errors='coerce')

#, '126', '132', '138', '144'np.unique(sim_forcast['vhr'].values)['pro', 'model_name', 'cycloneName', 'ensamble', 'cyc_speed', 'cyc_pressure', 'cyc_cat', 'validt', 'lat', 'lon', 'vhr', 'validt2']
sim['latitude']=pd.to_numeric(sim['lat'],errors='coerce')
sim['longitude']=pd.to_numeric(sim['lon'],errors='coerce')
sim['cyc_speed']=pd.to_numeric(sim['cyc_speed'],errors='coerce')

sim=sim[sim['latitude'] > 0 ]
sim=sim[sim['longitude'] > 0 ]
sim=sim[sim['cyc_speed'] >= 0 ]

sim1=sim.set_index(['model_name','cycloneName', 'validt', 'ensamble','vhr',])

sim_df= sim1[~sim1.index.duplicated()][['cyc_speed','longitude','latitude']]
import xarray as xr
sim_xr=xr.Dataset.from_dataframe(sim_df)
sim_xr.to_netcdf("C:\\globus_data\\PAR\\egrr.nc",'w','NETCDF4')

typhon_list['validt']=typhon_list['Start_D'].astype(np.datetime64)
typhon_list["cycloneName"] = (typhon_list["wmo_name"].astype('str')).map(lambda x: x.lower())
typhon_list['year']=typhon_list['validt'].map(lambda x: x.year) 


list_1=typhon_list['cycloneName1'].values

sim=sim[sim['cycloneName1'].isin(list_1)]
sim=sim[sim['vhr'].isin(steps_vhr)]

Obs['validt']=Obs['ISO_time'].astype(np.datetime64)
Obs['year']=Obs['validt'].map(lambda x: x.year)

sim["validt"] = sim["validt"].map(lambda x: con_date(x))
Obs["cycloneName1"] = (Obs["Name"].astype('str')).map(lambda x: x.lower())
Obs["speed_ms"] = (Obs["Wind(WMO)kt"].astype('float')).map(lambda x: x*0.5144444)
typhon_list["id"] = typhon_list["cycloneName1"]+typhon_list["year"].map(str) 
sim["id"] = sim["cycloneName1"]+sim["year"].map(str) 
Obs["id"] = Obs["cycloneName1"]+Obs["year"].map(str) 
typhones= typhon_list['id'].values

Obs["speed_ms"] = (Obs["Wind(WMO)kt"].astype('float')).map(lambda x: x*0.5144444)
columns_ob=[ 'Nature', 'Latitude','Longitude', 'Wind(WMO)kt', 'Pres(WMO)mb', 'validt']
columns_sim=['ensamble','cyc_speed', 'cyc_pressure', 'cyc_cat', 'validt', 'lat', 'lon', 'vhr']

sim_WD=sim[sim['id'].isin(np.unique(Obs['id'].values))]

OBS_WD=Obs[Obs['id'].isin(np.unique(sim_WD['id'].values))]


sim_anly=sim_WD[sim_WD['Mtype']=='analysis']
sim_forcast=sim_WD[sim_WD['Mtype']=='forecast']

sim_enforecast=sim_WD[sim_WD['Mtype']=='ensembleForecast']

listof_dic=[]
typhones= np.unique(sim_WD['id'].values)

ensambles=np.unique(sim_enforecast['ensamble'].values)
ls_ens=[]


(datetime.strptime('2011-04-04T12:00:00Z', "%Y-%m-%dT%H:00:00Z")).strftime("%Y-%m-%d %H:00:00")


for anitem in typhones:
    print(anitem)
    lat_diff={}
    lon_diff={}
    disp={}
    del_speed={}
    a1=sim_enforecast[sim_enforecast['id']==anitem]
    b1=Obs[Obs['id']==anitem]
    a1=a1[['cycloneName','cyc_speed', 'cyc_cat', 'validt2','validt', 'lat', 'lon','vhr']]
    b1=b1[['Name','Nature', 'speed_ms', 'validt', 'Latitude', 'Longitude']]
    for items in steps_vhr:
        #print(items)
        try:            
            a=a1[a1['vhr']==items]
            a=a.set_index('validt2')
            b=b1[['validt', 'Latitude','Longitude','speed_ms']]
            b=b.set_index('validt')
            df2=pd.concat([a, b], axis=1)
            #df3=df2[df2['lat'] > 0]
            df3=df2[df2.lat.notnull()]
            df3['lat_delta']=df3['lat']-df3['Latitude']
            df3['speed_del']=df3['cyc_speed']-df3['speed_ms']
            df3['lon_delta']=df3['lon']-df3['Longitude']
            df3['abs_del']=df3.apply(calculate_dis, axis=1)
            del_speed[items]=df3['speed_del'].values 
            disp[items]=df3['abs_del'].values 
            ls_ens.append(df3)
        except:
            print('%s vhr doesnt exist' %items )
        print('adding to list')
        listof_dic.append(disp)
l_con={}
for em in steps_vhr:
    
    tm_l=[]
    for item in listof_dic:
        if em in item.keys():
            tm_l.append(item[em])
    ss=np.concatenate(np.array(tm_l)).ravel()
    l_con[em]=ss    
        


steps_vhr1=['6', '12', '18', '24', '30', '36', '42', '48', '54', '60', '66', '72']
df1=pd.DataFrame(dict([(k,pd.Series(v))for k,v in l_con.items()]))

fig, ax = plt.subplots(figsize=(14, 8))
fig.canvas.set_window_title('A Boxplot Example')
fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
df1.boxplot(column=steps_vhr1,ax=ax,showfliers=False)

ax.set_axisbelow(True)
ax.set_title('ECMF Error in Track position for different lead time')
ax.set_xlabel('Lead time')
ax.set_ylabel('Track Error(km)')



listof_dic2=[]
typhones= np.unique(sim_WD['id'].values)

ensambles=np.unique(sim_enforecast['ensamble'].values)

ls_ens2=[]

for anitem in typhones:
    print(anitem)
    lat_diff={}
    lon_diff={}
    disp={}
    del_speed={}
    a1=sim_enforecast[sim_enforecast['id']==anitem]
    b1=Obs[Obs['id']==anitem]
    b=b1.set_index('validt')
    #a1=a1[['cycloneName','cyc_speed', 'cyc_cat', 'validt2','validt', 'lat', 'lon','vhr']]
    #b1=b1[['Name','Nature', 'speed_ms', 'validt', 'Latitude', 'Longitude']]
    for items in steps_vhr:
        #print(items)
        if items in a1['vhr'].values:
            a=a1[a1['vhr']==items]
            for item2 in ensambles:
                if item2 in a1['ensamble'].values:
                    a2=a[a['ensamble']==item2]
                    a2=a2.set_index('validt2')             
                    df2=pd.concat([a2, b], axis=1)
                    df3=df2[df2.lat.notnull()]
                    df3=df2[df2['lat'] > 0]
                    df3['lat_delta']=df3['lat']-df3['Latitude']
                    df3['speed_del']=df3['cyc_speed']-df3['speed_ms']
                    df3['lon_delta']=df3['lon']-df3['Longitude']
                    df3['abs_del']=df3.apply(calculate_dis, axis=1)
                    del_speed[item2]=df3['speed_del'].values 
                    tag=items+'_'+str(item2)
                    disp[tag]=df3['abs_del'].values
                    ls_ens2.append(df3)
            listof_dic2.append(disp)
            print('%s vhr doesnt exist' %items )
            #print('adding to list')
            #listof_dic2.append(disp)
        
        
l_con2={}
for em in steps_vhr:
    
    tm_l=[]
    for item in listof_dic2:
        if em in item.keys():
            tm_l.append(item[em])
    ss=np.concatenate(np.array(tm_l)).ravel()
    l_con2[em]=ss    
        


steps_vhr1=['6', '12', '18', '24', '30', '36', '42', '48', '54', '60', '66', '72']
df1=pd.DataFrame(dict([(k,pd.Series(v))for k,v in l_con2.items()]))

fig, ax = plt.subplots(figsize=(14, 8))
fig.canvas.set_window_title('A Boxplot Example')
fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
df1.boxplot(column=steps_vhr1,ax=ax,showfliers=False)

ax.set_axisbelow(True)
ax.set_title('ECMF Error in Track position for different lead time')
ax.set_xlabel('Lead time')
ax.set_ylabel('Track Error(km)')

