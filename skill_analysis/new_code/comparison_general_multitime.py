
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
import geopandas as gpd
import pickle

from functions import *


# ## Create pandaframe from xml files

# In[ ]:


## Parameters

data_obs_folder = '../data/obs/'
UCL_folder = '../data/UCL_all/'
results_folder = '../CSVs/multicyclone/'
figures_folder = '../figures/multicyclone/'

include_UCL_data = 'y'


# ## Observations

# In[ ]:


df_obs = pd.read_csv(data_obs_folder+'best_track_27ty.csv')

cyclone_names_upper = [x for x in sorted(list(set(df_obs['STORMNAME'])))]+['vongfong']

cyclone_names_upper.remove('KETSANA')
cyclone_names_upper.remove('FENGSHEN')
cyclone_names_upper.remove('DURIAN')
cyclone_names_upper.remove('NOCK-TEN')

cyclone_names = [x.lower().strip() for x in cyclone_names_upper]


# In[ ]:


obs = {}

for c, cyclone_name in enumerate(cyclone_names):
    cyclone_name_upper = cyclone_names_upper[c]
    obs[cyclone_name] = {}
    if cyclone_name == 'vongfong':
        obs[cyclone_name]['track'] = np.array([[9.6,128.8],[10.1,128.8],[10.8,129],[11.4,128.8],[11.9,129.2],[11.8,128.9],[12,128.5],[12.2,127.9],[12.1,127],[12.1,126.2],[12.2,125.3],[12.3,124.9],[12.5,123.6],[13.3,122.7],[14.1,121.9],[15.1,121.4],[16.2,121.1],[18,120],[19.3,120.5]])
        time_string = ['1118', '1200', '1206', '1212', '1218', '1300', '1306', '1312', '1318', '1400', '1406', '1412', '1418', '1500', '1506', '1512', '1518', '1600', '1606'] 
        obs[cyclone_name]['date_time'] = ['2020/05/'+t[:2]+', '+t[2:]+':00:00' for t in time_string]
    else:
        obs[cyclone_name]['track'] = np.transpose(np.vstack((np.array(df_obs[df_obs['STORMNAME']==cyclone_name_upper]['LAT']),np.array(df_obs[df_obs['STORMNAME']==cyclone_name_upper]['LON']))))
        date_time = [str(x) for x in list(df_obs[df_obs['STORMNAME']==cyclone_name_upper]['YYYYMMDDHH'])]
        obs[cyclone_name]['date_time'] = [dt[:4]+'/'+dt[4:6]+'/'+dt[6:8]+', '+dt[8:10]+':'+dt[10:12]+':00' for dt in date_time]


# ## Parameters

# In[ ]:


time_separation = 6
time_limit_list = list(np.linspace(6,72,72/time_separation, dtype=int))
# time_limit_list = [72]

institutes_selections = [['kwbc'], ['rjtd'], ['egrr'], ['ecmf'], ['kwbc', 'rjtd', 'egrr', 'ecmf'], ['kwbc', 'egrr', 'ecmf'], ['kwbc', 'rjtd', 'ecmf'], ['kwbc', 'ecmf']]
selections_names = ['kwbc', 'rjtd', 'egrr', 'ecmf', 'full', 'w_rjtd', 'w_egrr', 'w_rjtd_egrr']

results = {}


# In[ ]:


time_limit_list


# # If the file exists You can jump to load dictionary results

# ## Structure the results as a dictionary (for each time_limit)

# Create a dictionary out of the results, with the following sctructure:
# 
# - number identifying institute and model:
#     - institute-model name
#     - cyclone name
#         - number identifying the forecast time
#             - forecast time (date and time at which the forecast was started)
#             - ensemble
#                 - start
#                 - lat
#                 - lon
#             - ensemble_mean
#                 - start
#                 - lat
#                 - lon
#             - number_members_ensemble
#         - number forecast times

# In[ ]:


# Loop over time_limit_list
for time_limit in time_limit_list:

    print('Time limit: '+str(time_limit))

    # Initialise the dictonary
    results_t = {}

    # Create a list with time points
    nhours_list = list(np.linspace(0,time_limit,time_limit/time_separation+1, dtype=int))

    # Loop over cyclone names
    for cyclone_name in cyclone_names:

        # Initialise subdictionary
        results_t[cyclone_name] = {}

        # Restrict the dataframe to the specific cyclone
        df_cyclone = pd.read_csv(results_folder+cyclone_name+'_all.csv')
        
        # Retrieve model names from the dataframe (the function set removes duplicates from a list)
        institute_names = list(set(df_cyclone['institute_name']))

        # Initialise model_num (so that the first number will actually be 0)
        model_num = -1

        # Loop over institutes
        for institute_name in institute_names:

            # Restrict the dataframe to the specific institute
            df_institute = df_cyclone[df_cyclone['institute_name'] == institute_name]

            # Retrieve model names from the dataframe (the function set removes duplicates from a list)
            model_names = list(set(df_institute['model_name']))

            # Loop over models
            for model_name in model_names:

                model_num += 1

                # Initialise subdictionary and assign name
                results_t[cyclone_name][str(model_num)] = {}
                results_t[cyclone_name][str(model_num)]['model_name'] = institute_name.upper()+' - '+model_name

    #             print(model_num, results_t[cyclone_name][str(model_num)]['model_name'])

                # Restrict the dataframe to the specific model
                df_model = df_institute[df_institute['model_name'] == model_name]

                # Restrict the dataframe to the ensemble forecasts
                df_ensemble = df_model[df_model['Mtype'] == 'ensembleforecast']

                # For each model, each cyclone, retrieve the list of forecast times
                forecast_time_list = sorted(list(set(df_ensemble['forecast_time'])))
                results_t[cyclone_name][str(model_num)]['num_forecast_times'] = len(forecast_time_list)

                # Loop over forecast times
                for forecast_time_num, forecast_time in enumerate(forecast_time_list):

                    # Initialise lists for lat and lon of paths
                    lat_ensemble_list = []
                    lon_ensemble_list = []
                    date_ensemble_list = []

                    # Initialise subdictionary
                    results_t[cyclone_name][str(model_num)][str(forecast_time_num)] = {}
                    results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['forecast_time'] = forecast_time

                    # Restrict the dataframe to the specific forecast time
                    df_time = df_ensemble[df_ensemble['forecast_time'] == forecast_time]

                    # Initialise subdictionary ensembles
                    results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble'] = {}

                    # For each model, each cyclone, each forecast time, retrieve the list of ensembles
                    ensemble_list = sorted([int(x) for x in list(set(df_time['ensemble']))])
                    results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['num_members_ensemble'] = len(ensemble_list)

                    # Loop over the member of the ensemble
                    for member in ensemble_list:

                        # Restrict the dataframe to the ensemble
                        df_member = df_time[df_time['ensemble'] == str(member)]

                        # Initialise lists for lat and lon of paths
                        lat_member = []
                        lon_member = []
                        date_member = []
                        date_int_member = []

                        # Loop over the time points
                        for nhours in nhours_list:
                            # Assign lat and lon
                            try:
                                lat_member.append(float(df_member[df_member['vhr']==str(nhours)]['lat']))
                                lon_member.append(float(df_member[df_member['vhr']==str(nhours)]['lon']))
                                date_member.append(list(df_member[df_member['vhr']==str(nhours)]['time'])[0])
                            except:
                                lat_member.append(np.nan)
                                lon_member.append(np.nan)
                                date_member.append('')

                        lat_ensemble_list.append(lat_member)
                        lon_ensemble_list.append(lon_member)
                        date_ensemble_list.append(date_member)

                    # Store lats and lons in the dictonary (as arrays)
                    try:
                        results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble']['start'] = list(df_member[df_member['vhr']==str(0)]['time'])[0]
                    except:
                        results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble']['start'] = ''

                    results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble']['lat'] = np.array(lat_ensemble_list)
                    results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble']['lon'] = np.array(lon_ensemble_list)
                    results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble']['date'] = date_ensemble_list

        results_t[cyclone_name]['number_models'] = model_num+1

    ## Crete average path over each ensemble

    for cyclone_name in cyclone_names:
        for model_num in range(results_t[cyclone_name]['number_models']):
                for forecast_time_num in range(results_t[cyclone_name][str(model_num)]['num_forecast_times']):
                    lat_ens = np.array(results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble']['lat'], dtype=np.float)
                    lon_ens = np.array(results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble']['lon'], dtype=np.float)
                    results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble_mean'] = {}
                    results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble_mean']['start'] = results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble']['start']
                    results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble_mean']['lat'] = np.nanmean(lat_ens,0)
                    results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble_mean']['lon'] = np.nanmean(lon_ens,0)
                    try:
                        results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble_mean']['date'] = [sum_date_time(results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble']['start'], hours=nhours) for nhours in nhours_list]
                    except:
                        results_t[cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble_mean']['date'] = ''
                        
                        
    results[str(time_limit)] = results_t


# ## UCL data

# In[ ]:


for time_limit in time_limit_list:
    
    print('Time limit: '+str(time_limit))
    
    ##### UCL data    
    
    if include_UCL_data == 'y':
        
        for cyclone_name in cyclone_names:
        
            results[str(time_limit)][cyclone_name]['UCL'] = {}

            UCL_cyclone_folder = UCL_folder+cyclone_name+'/'

            n_file = -1

            for UCL_filename in os.listdir(UCL_cyclone_folder):

                if 'forecasttrack' in UCL_filename and '.shp' in UCL_filename:

                    n_file += 1

                    shapefile = gpd.read_file(UCL_cyclone_folder+UCL_filename)

                    date_time_UCL = UCL_filename.split('_')[-1][:-4]

                    year_UCL = date_time_UCL[0:4]
                    month_UCL = date_time_UCL[4:6]
                    day_UCL = date_time_UCL[6:8]
                    hour_UCL = date_time_UCL[8:10]

                    month_UCL = str(int(month_UCL)+1) if int(month_UCL)+1>9 else '0'+str(int(month_UCL)+1)

                    forecast_time = year_UCL+'/'+month_UCL+'/'+day_UCL+', '+hour_UCL+':00:00'

                    lead_times = np.array([0,  12,  24,  36,  48,  72,  96, 120])

                    try:
                        if lead_times[-1] > time_limit:
                            ind_lead_times_max = np.where(lead_times==time_limit)[0][0]
                            lead_times = lead_times[:ind_lead_times_max+1]
                        else:
                            ind_lead_times_max = len(lead_times)

                        results[str(time_limit)][cyclone_name]['UCL'][str(n_file)] = {}  
                        results[str(time_limit)][cyclone_name]['UCL'][str(n_file)]['forecast_time'] = forecast_time
                        results[str(time_limit)][cyclone_name]['UCL'][str(n_file)]['lat'] = np.array(shapefile.geometry.y)[:ind_lead_times_max+1]
                        results[str(time_limit)][cyclone_name]['UCL'][str(n_file)]['lon'] = np.array(shapefile.geometry.x)[:ind_lead_times_max+1]
                        results[str(time_limit)][cyclone_name]['UCL'][str(n_file)]['date'] = [sum_date_time(forecast_time, hours=nhours) for nhours in lead_times]

                    except:

                        results[str(time_limit)][cyclone_name]['UCL'][str(n_file)] = {}
                        results[str(time_limit)][cyclone_name]['UCL'][str(n_file)]['forecast_time'] = forecast_time
                        results[str(time_limit)][cyclone_name]['UCL'][str(n_file)]['lat'] = np.nan
                        results[str(time_limit)][cyclone_name]['UCL'][str(n_file)]['lon'] = np.nan
                        results[str(time_limit)][cyclone_name]['UCL'][str(n_file)]['date'] = ''

            results[str(time_limit)][cyclone_name]['UCL']['num_forecast_times'] = n_file+1


# ## Calculate multimodel means

# In[ ]:


for time_limit in time_limit_list:
    
    print('Time limit: '+str(time_limit))

    ## List of all forecast times (where each one appears only once)

    for cyclone_name in cyclone_names:
        forecast_time_list = [] 
        for model_num in range(results[str(time_limit)][cyclone_name]['number_models']):
                for forecast_time_num in range(results[str(time_limit)][cyclone_name][str(model_num)]['num_forecast_times']):
                    forecast_time_list.append(results[str(time_limit)][cyclone_name][str(model_num)][str(forecast_time_num)]['forecast_time'])

        results[str(time_limit)][cyclone_name]['forecast_time_list'] = sorted(list(set(forecast_time_list)))


    ## Create means for each selection of ensemble of models

    for cyclone_name in cyclone_names:
        results[str(time_limit)][cyclone_name]['multimodel'] = {}
        for s,selection in enumerate(institutes_selections):
            results[str(time_limit)][cyclone_name]['multimodel'][selections_names[s]] = {}
            for forecast_time_num_general,forecast_time in enumerate(results[str(time_limit)][cyclone_name]['forecast_time_list']):
                n = 0
                lats = np.nan
                lons = np.nan
                for model_num in range(results[str(time_limit)][cyclone_name]['number_models']):
                    if results[str(time_limit)][cyclone_name][str(model_num)]['model_name'].split('-')[0].strip().lower() in selection:
                        for forecast_time_num in range(results[str(time_limit)][cyclone_name][str(model_num)]['num_forecast_times']):
                            if results[str(time_limit)][cyclone_name][str(model_num)][str(forecast_time_num)]['forecast_time'] == forecast_time:
                                lat_ens = np.array(results[str(time_limit)][cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble']['lat'], dtype=np.float)
                                lon_ens = np.array(results[str(time_limit)][cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble']['lon'], dtype=np.float)
                                if n == 0:
                                    lats = lat_ens
                                    lons = lon_ens
                                else:
                                    lats = np.concatenate((lats,lat_ens),axis=0)
                                    lons = np.concatenate((lons,lon_ens),axis=0)
                                n += 1
                results[str(time_limit)][cyclone_name]['multimodel'][selections_names[s]][str(forecast_time_num_general)] = {}
                results[str(time_limit)][cyclone_name]['multimodel'][selections_names[s]][str(forecast_time_num_general)]['forecast_time'] = forecast_time
                results[str(time_limit)][cyclone_name]['multimodel'][selections_names[s]][str(forecast_time_num_general)]['date'] = [sum_date_time(forecast_time, hours=nhours) for nhours in nhours_list]
                results[str(time_limit)][cyclone_name]['multimodel'][selections_names[s]][str(forecast_time_num_general)]['lat'] = np.nanmean(lats,0)
                results[str(time_limit)][cyclone_name]['multimodel'][selections_names[s]][str(forecast_time_num_general)]['lon'] = np.nanmean(lons,0)


# ## Calculate errors respect to observations

# In[ ]:


for time_limit in time_limit_list:
    
    print('Time limit: '+str(time_limit))

    # For each model

    for cyclone_name in cyclone_names:
#         print('Cyclone: '+cyclone_name+'\n')
        for model_num in range(results[str(time_limit)][cyclone_name]['number_models']):

            distance_initial_list = []
            distance_final_list = []

            for forecast_time_num in range(results[str(time_limit)][cyclone_name][str(model_num)]['num_forecast_times']):

                date_forecast = results[str(time_limit)][cyclone_name][str(model_num)][str(forecast_time_num)]['forecast_time']
                dates_list = results[str(time_limit)][cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble_mean']['date']

                for date_obs in obs[cyclone_name]['date_time']:

                    if date_obs in dates_list:

                        ind_final = dates_list.index(date_obs)

                        try:
                            lat_final = results[str(time_limit)][cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble_mean']['lat'][ind_final]
                            lon_final = results[str(time_limit)][cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble_mean']['lon'][ind_final]
                        except:
                            lat_final = np.nan
                            lon_final = np.nan

                        ind_final_obs = obs[cyclone_name]['date_time'].index(date_obs)
                        lat_final_obs = obs[cyclone_name]['track'][ind_final_obs,0]
                        lon_final_obs = obs[cyclone_name]['track'][ind_final_obs,1]

                        date_initial = sum_date_time(date_obs, hours=-time_limit)

                        if date_forecast == date_initial:

                            ind_initial = dates_list.index(date_forecast)

                            try:
                                lat_initial = results[str(time_limit)][cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble_mean']['lat'][ind_initial]
                                lon_initial = results[str(time_limit)][cyclone_name][str(model_num)][str(forecast_time_num)]['ensemble_mean']['lon'][ind_initial]
                            except:
                                lat_initial = np.nan
                                lon_initial = np.nan

                            try:
                                ind_initial_obs = obs[cyclone_name]['date_time'].index(date_forecast)
                                lat_initial_obs = obs[cyclone_name]['track'][ind_initial_obs,0]
                                lon_initial_obs = obs[cyclone_name]['track'][ind_initial_obs,1]
                            except:
                                ind_initial_obs = np.nan
                                lat_initial_obs = np.nan
                                lon_initial_obs = np.nan

                            # Coordinates and distance

                            coords_final_theor = (lat_final_obs, lon_final_obs)
                            coords_final_model = (lat_final, lon_final)

                            coords_initial_theor = (lat_initial_obs, lon_initial_obs)
                            coords_initial_model = (lat_initial, lon_initial)

                            try:
                                distance_initial = geopy.distance.distance(coords_initial_theor, coords_initial_model).km
                            except:
                                distance_initial = np.nan

                            try:
                                distance_final = geopy.distance.distance(coords_final_theor, coords_final_model).km
                            except:
                                distance_final = np.nan

                            distance_initial_list.append(distance_initial)
                            distance_final_list.append(distance_final)

#                             print(results[str(time_limit)][cyclone_name][str(model_num)]['model_name'], date_forecast, '\t', distance_initial, distance_final)

            results[str(time_limit)][cyclone_name][str(model_num)]['distance_initial'] = np.nanmean(distance_initial_list)
            results[str(time_limit)][cyclone_name][str(model_num)]['distance_final'] = np.nanmean(distance_final_list)


    # For each multimodel ensemble
#         print('\n')

        for s in range(len(selections_names)):

            distance_initial_list = []
            distance_final_list = []

            for forecast_time_num_general in range(len(results[str(time_limit)][cyclone_name]['forecast_time_list'])):

                date_forecast = results[str(time_limit)][cyclone_name]['multimodel'][selections_names[s]][str(forecast_time_num_general)]['forecast_time']
                dates_list = results[str(time_limit)][cyclone_name]['multimodel'][selections_names[s]][str(forecast_time_num_general)]['date']

                for date_obs in obs[cyclone_name]['date_time']:

                    if date_obs in dates_list:

                        ind_final = dates_list.index(date_obs)

                        try:
                            lat_final = results[str(time_limit)][cyclone_name]['multimodel'][selections_names[s]][str(forecast_time_num_general)]['lat'][ind_final]
                            lon_final = results[str(time_limit)][cyclone_name]['multimodel'][selections_names[s]][str(forecast_time_num_general)]['lon'][ind_final]
                        except:
                            lat_final = np.nan
                            lon_final = np.nan

                        ind_final_obs = obs[cyclone_name]['date_time'].index(date_obs)
                        lat_final_obs = obs[cyclone_name]['track'][ind_final_obs,0]
                        lon_final_obs = obs[cyclone_name]['track'][ind_final_obs,1]

                        date_initial = sum_date_time(date_obs, hours=-time_limit)

                        if date_forecast == date_initial:

                            ind_initial = dates_list.index(date_forecast)

                            try:
                                lat_initial = results[str(time_limit)][cyclone_name]['multimodel'][selections_names[s]][str(forecast_time_num_general)]['lat'][ind_initial]
                                lon_initial = results[str(time_limit)][cyclone_name]['multimodel'][selections_names[s]][str(forecast_time_num_general)]['lon'][ind_initial]
                            except:
                                lat_initial = np.nan
                                lon_initial = np.nan

                            try:
                                ind_initial_obs = obs[cyclone_name]['date_time'].index(date_forecast)
                                lat_initial_obs = obs[cyclone_name]['track'][ind_initial_obs,0]
                                lon_initial_obs = obs[cyclone_name]['track'][ind_initial_obs,1]
                            except:
                                ind_initial_obs = np.nan
                                lat_initial_obs = np.nan
                                lon_initial_obs = np.nan

                            # Coordinates and distance

                            coords_final_theor = (lat_final_obs, lon_final_obs)
                            coords_final_model = (lat_final, lon_final)

                            coords_initial_theor = (lat_initial_obs, lon_initial_obs)
                            coords_initial_model = (lat_initial, lon_initial)

                            try:
                                distance_initial = geopy.distance.distance(coords_initial_theor, coords_initial_model).km
                            except:
                                distance_initial = np.nan

                            try:
                                distance_final = geopy.distance.distance(coords_final_theor, coords_final_model).km
                            except:
                                distance_final = np.nan

                            distance_initial_list.append(distance_initial)
                            distance_final_list.append(distance_final)

#                             print(selections_names[s], date_forecast, '\t', distance_initial, distance_final)

            results[str(time_limit)][cyclone_name]['multimodel'][selections_names[s]]['distance_initial'] = np.nanmean(distance_initial_list)
            results[str(time_limit)][cyclone_name]['multimodel'][selections_names[s]]['distance_final'] = np.nanmean(distance_final_list)


    # For UCL
        if include_UCL_data == 'y':
#         print('\n')

            distance_initial_list = []
            distance_final_list = []

            for forecast_time_num_general in range(results[str(time_limit)][cyclone_name]['UCL']['num_forecast_times']):

                date_forecast = results[str(time_limit)][cyclone_name]['UCL'][str(forecast_time_num_general)]['forecast_time']
                dates_list = results[str(time_limit)][cyclone_name]['UCL'][str(forecast_time_num_general)]['date']

                for date_obs in obs[cyclone_name]['date_time']:

                    if date_obs in dates_list:

                        ind_final = dates_list.index(date_obs)

                        try:
                            lat_final = results[str(time_limit)][cyclone_name]['UCL'][str(forecast_time_num_general)]['lat'][ind_final]
                            lon_final = results[str(time_limit)][cyclone_name]['UCL'][str(forecast_time_num_general)]['lon'][ind_final]
                        except:
                            lat_final = np.nan
                            lon_final = np.nan

                        ind_final_obs = obs[cyclone_name]['date_time'].index(date_obs)
                        lat_final_obs = obs[cyclone_name]['track'][ind_final_obs,0]
                        lon_final_obs = obs[cyclone_name]['track'][ind_final_obs,1]

                        date_initial = sum_date_time(date_obs, hours=-time_limit)

                        if date_forecast == date_initial:

                            ind_initial = dates_list.index(date_forecast)

                            try:
                                lat_initial = results[str(time_limit)][cyclone_name]['UCL'][str(forecast_time_num_general)]['lat'][ind_initial]
                                lon_initial = results[str(time_limit)][cyclone_name]['UCL'][str(forecast_time_num_general)]['lon'][ind_initial]
                            except:
                                lat_initial = np.nan
                                lon_initial = np.nan

                            try:
                                ind_initial_obs = obs[cyclone_name]['date_time'].index(date_forecast)
                                lat_initial_obs = obs[cyclone_name]['track'][ind_initial_obs,0]
                                lon_initial_obs = obs[cyclone_name]['track'][ind_initial_obs,1]
                            except:
                                ind_initial_obs = np.nan
                                lat_initial_obs = np.nan
                                lon_initial_obs = np.nan

                            # Coordinates and distance

                            coords_final_theor = (lat_final_obs, lon_final_obs)
                            coords_final_model = (lat_final, lon_final)

                            coords_initial_theor = (lat_initial_obs, lon_initial_obs)
                            coords_initial_model = (lat_initial, lon_initial)

                            try:
                                distance_initial = geopy.distance.distance(coords_initial_theor, coords_initial_model).km
                            except:
                                distance_initial = np.nan

                            try:
                                distance_final = geopy.distance.distance(coords_final_theor, coords_final_model).km
                            except:
                                distance_final = np.nan

                            distance_initial_list.append(distance_initial)
                            distance_final_list.append(distance_final)

#                             print('UCL', date_forecast, '\t', distance_initial, distance_final)

            results[str(time_limit)][cyclone_name]['UCL']['distance_initial'] = np.nanmean(distance_initial_list)
            results[str(time_limit)][cyclone_name]['UCL']['distance_final'] = np.nanmean(distance_final_list)


# ## Calculate errors averaging over all the times considered

# In[ ]:


results['average_time'] = {}

for cyclone_name in cyclone_names:
    
    # Initialise dictionary for the specific cyclone
    results['average_time'][cyclone_name] = {}
    
    # Single models
    for model_num in range(results[str(time_limit_list[0])][cyclone_name]['number_models']):
        results['average_time'][cyclone_name][str(model_num)] = {}
        results['average_time'][cyclone_name][str(model_num)]['distance_final'] = np.nanmean([results[str(time_limit_list[i])][cyclone_name][str(model_num)]['distance_final'] for i in range(len(time_limit_list))])
        results['average_time'][cyclone_name][str(model_num)]['distance_initial'] = np.nanmean([results[str(time_limit_list[i])][cyclone_name][str(model_num)]['distance_initial'] for i in range(len(time_limit_list))])
    
    # Multimodels
    results['average_time'][cyclone_name]['multimodel'] = {}
    
    for s in range(len(selections_names)):
        results['average_time'][cyclone_name]['multimodel'][selections_names[s]] = {}
        results['average_time'][cyclone_name]['multimodel'][selections_names[s]]['distance_final'] = np.nanmean([results[str(time_limit_list[i])][cyclone_name]['multimodel'][selections_names[s]]['distance_final'] for i in range(len(time_limit_list))])
        results['average_time'][cyclone_name]['multimodel'][selections_names[s]]['distance_initial'] = np.nanmean([results[str(time_limit_list[i])][cyclone_name]['multimodel'][selections_names[s]]['distance_initial'] for i in range(len(time_limit_list))])
    
    # UCL
    results['average_time'][cyclone_name]['UCL'] = {}
    results['average_time'][cyclone_name]['UCL']['distance_final'] = np.nanmean([results[str(time_limit_list[i])][cyclone_name]['UCL']['distance_final'] for i in range(len(time_limit_list))])
    results['average_time'][cyclone_name]['UCL']['distance_initial'] = np.nanmean([results[str(time_limit_list[i])][cyclone_name]['UCL']['distance_initial'] for i in range(len(time_limit_list))])
    


# ## Save dictionary results

# In[ ]:


with open('results.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ## Load dictionary results

# In[ ]:


with open('results.pickle', 'rb') as handle:
    results = pickle.load(handle)


# # Plots single cyclone

# In[ ]:


cyclone_name = 'nesat'


# ## Plots single models

# In[ ]:


# Plot errors depending on leading times

include_average_time = 'y'

plt.rcParams['font.size'] = 20
width = 0.05  # the width of the bars

labels = []
err = {}
rects = {}

if include_average_time == 'y':
    time_limit_list_tot = [str(x) for x in time_limit_list]+['average_time']
else:
    time_limit_list_tot = [str(x) for x in time_limit_list]

for t in range(len(time_limit_list_tot)):
    err[str(t)] = []

title_string = 'Cyclone: '+cyclone_name

fig, ax = plt.subplots(1, 1, figsize=(30,12))
results_t[cyclone_name]['number_models']

for s in range(results[time_limit_list_tot[0]][cyclone_name]['number_models']):
    labels.append(results[time_limit_list_tot[0]][cyclone_name][str(s)]['model_name'])
    for t in range(len(time_limit_list_tot)):
        err[str(t)].append(results[time_limit_list_tot[t]][cyclone_name][str(s)]['distance_final'])
        
if include_UCL_data == 'y':
    labels.append('UCL')
    for t in range(len(time_limit_list_tot)):
        err[str(t)].append(results[time_limit_list_tot[t]][cyclone_name]['UCL']['distance_final'])
        

x = np.arange(len(labels))  # the label locations

if len(time_limit_list_tot) % 2:
    positions = list(range(-int(len(time_limit_list_tot)/2),0))+[0]+list(range(1,int(len(time_limit_list_tot)/2)+1))
else:
    positions = list(np.arange(-int(len(time_limit_list_tot)/2)+0.5,0,1))+list(np.arange(0.5,int(len(time_limit_list_tot)/2),1))

for t in range(len(time_limit_list_tot)):
    rects[str(t)] = ax.bar(x + width*positions[t], err[str(t)], width, label=time_limit_list_tot[t])

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Extreme sync index ('+str(std_dev)+' std dev)');
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Error (km)', labelpad = 20)
ax.set_xlabel('Models', labelpad = 20)
ax.yaxis.grid()
ax.legend();

ttl = ax.set_title(title_string, fontweight='bold')
ttl.set_position([0.5, 1.05])

plt.tight_layout()
# fig.savefig(figures_folder+'error_singlemodels_by_model_'+cyclone_name+'.pdf', format='pdf', bbox_inches = 'tight', pad_inches = 0)


# In[ ]:


# Plot errors depending on leading times

include_average_time = 'y'

plt.rcParams['font.size'] = 25
width = 0.07  # the width of the bars

labels = []
err = {}
rects = {}

if include_average_time == 'y':
    time_limit_list_tot = [str(x) for x in time_limit_list]+['average_time']
else:
    time_limit_list_tot = [str(x) for x in time_limit_list]

model_names = [results[str(time_limit_list[0])][cyclone_name][str(s)]['model_name'] for s in range(results[str(time_limit_list[0])][cyclone_name]['number_models'])]

if include_UCL_data == 'y':
    tot_selection_len = results[str(time_limit_list[0])][cyclone_name]['number_models']+1
    tot_selection_names = model_names+['UCL']
else:
    tot_selection_len = results[str(time_limit_list[0])][cyclone_name]['number_models']
    tot_selection_names = model_names

for s in range(tot_selection_len):
    err[str(s)] = []

title_string = 'Cyclone: '+cyclone_name

fig, ax = plt.subplots(1, 1, figsize=(30,12))


for t in range(len(time_limit_list_tot)):
    labels.append(time_limit_list_tot[t])
    for s in range(tot_selection_len):
        if s < tot_selection_len-1:
            err[str(s)].append(results[time_limit_list_tot[t]][cyclone_name][str(s)]['distance_final'])
        else:
            err[str(s)].append(results[time_limit_list_tot[t]][cyclone_name]['UCL']['distance_final'])

            
x = np.arange(len(labels))  # the label locations

if tot_selection_len % 2:
    positions = list(range(-int(tot_selection_len/2),0))+[0]+list(range(1,int(tot_selection_len/2)+1))
else:
    positions = list(np.arange(-int(tot_selection_len/2)+0.5,0,1))+list(np.arange(0.5,int(tot_selection_len/2),1))

for s in range(tot_selection_len):
    rects[str(s)] = ax.bar(x + width*positions[s], err[str(s)], width, label=str(tot_selection_names[s]))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Error (km)', labelpad = 20)
ax.set_xlabel('Lead time (hours)', labelpad = 20)
ax.yaxis.grid()
ax.legend();

ttl = ax.set_title(title_string, fontweight='bold')
ttl.set_position([0.5, 1.05])

plt.tight_layout()
fig.savefig(figures_folder+'error_singlemodels_by_leadtime_'+cyclone_name+'.pdf', format='pdf', bbox_inches = 'tight', pad_inches = 0)


# In[ ]:


# Plot errors depending on leading times, initial distance

include_average_time = 'y'

plt.rcParams['font.size'] = 20
width = 0.05  # the width of the bars

labels = []
err = {}
rects = {}

if include_average_time == 'y':
    time_limit_list_tot = [str(x) for x in time_limit_list]+['average_time']
else:
    time_limit_list_tot = [str(x) for x in time_limit_list]

for t in range(len(time_limit_list_tot)):
    err[str(t)] = []

title_string = 'Cyclone: '+cyclone_name

fig, ax = plt.subplots(1, 1, figsize=(30,12))
results_t[cyclone_name]['number_models']

for s in range(results[time_limit_list_tot[0]][cyclone_name]['number_models']):
    labels.append(results[time_limit_list_tot[0]][cyclone_name][str(s)]['model_name'])
    for t in range(len(time_limit_list_tot)):
        err[str(t)].append(results[time_limit_list_tot[t]][cyclone_name][str(s)]['distance_initial'])
        
if include_UCL_data == 'y':
    labels.append('UCL')
    for t in range(len(time_limit_list_tot)):
        err[str(t)].append(results[time_limit_list_tot[t]][cyclone_name]['UCL']['distance_initial'])
        

x = np.arange(len(labels))  # the label locations

if len(time_limit_list_tot) % 2:
    positions = list(range(-int(len(time_limit_list_tot)/2),0))+[0]+list(range(1,int(len(time_limit_list_tot)/2)+1))
else:
    positions = list(np.arange(-int(len(time_limit_list_tot)/2)+0.5,0,1))+list(np.arange(0.5,int(len(time_limit_list_tot)/2),1))

for t in range(len(time_limit_list_tot)):
    rects[str(t)] = ax.bar(x + width*positions[t], err[str(t)], width, label=time_limit_list_tot[t])

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Extreme sync index ('+str(std_dev)+' std dev)');
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Error (km)', labelpad = 20)
ax.set_xlabel('Models', labelpad = 20)
ax.yaxis.grid()
ax.legend();

ttl = ax.set_title(title_string, fontweight='bold')
ttl.set_position([0.5, 1.05])

plt.tight_layout()


# ## Plots multimodel

# In[ ]:


# Plot errors depending on leading times

include_average_time = 'y'

plt.rcParams['font.size'] = 30
width = 0.05  # the width of the bars

labels = []
err = {}
rects = {}

if include_average_time == 'y':
    time_limit_list_tot = [str(x) for x in time_limit_list]+['average_time']
else:
    time_limit_list_tot = [str(x) for x in time_limit_list]

for t in range(len(time_limit_list_tot)):
    err[str(t)] = []

title_string = 'Cyclone: '+cyclone_name

fig, ax = plt.subplots(1, 1, figsize=(30,12))


for s in range(len(selections_names)):
    labels.append(selections_names[s])
    for t in range(len(time_limit_list_tot)):
        err[str(t)].append(results[time_limit_list_tot[t]][cyclone_name]['multimodel'][selections_names[s]]['distance_final'])
        
if include_UCL_data == 'y':
    labels.append('UCL')
    for t in range(len(time_limit_list_tot)):
        err[str(t)].append(results[time_limit_list_tot[t]][cyclone_name]['UCL']['distance_final'])
        

x = np.arange(len(labels))  # the label locations

if len(time_limit_list_tot) % 2:
    positions = list(range(-int(len(time_limit_list_tot)/2),0))+[0]+list(range(1,int(len(time_limit_list_tot)/2)+1))
else:
    positions = list(np.arange(-int(len(time_limit_list_tot)/2)+0.5,0,1))+list(np.arange(0.5,int(len(time_limit_list_tot)/2),1))

for t in range(len(time_limit_list_tot)):
    rects[str(t)] = ax.bar(x + width*positions[t], err[str(t)], width, label=time_limit_list_tot[t])

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Extreme sync index ('+str(std_dev)+' std dev)');
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Error (km)', labelpad = 20)
ax.set_xlabel('Models', labelpad = 20)
ax.yaxis.grid()
ax.legend();

ttl = ax.set_title(title_string, fontweight='bold')
ttl.set_position([0.5, 1.05])

plt.tight_layout()
# fig.savefig(figures_folder+'error_multimodels_by_model_'+cyclone_name+'.pdf', format='pdf', bbox_inches = 'tight', pad_inches = 0)


# In[ ]:


# Plot errors depending on leading times

include_average_time = 'y'

plt.rcParams['font.size'] = 25
width = 0.08  # the width of the bars

labels = []
err = {}
rects = {}

if include_average_time == 'y':
    time_limit_list_tot = [str(x) for x in time_limit_list]+['average_time']
else:
    time_limit_list_tot = [str(x) for x in time_limit_list]

if include_UCL_data == 'y':
    tot_selection_len = len(selections_names)+1
    tot_selection_names = selections_names+['UCL']
else:
    tot_selection_len = len(selections_names)
    tot_selection_names = selections_names

for s in range(tot_selection_len):
    err[str(s)] = []

title_string = 'Cyclone: '+cyclone_name

fig, ax = plt.subplots(1, 1, figsize=(30,12))


for t in range(len(time_limit_list_tot)):
    labels.append(time_limit_list_tot[t])
    for s in range(tot_selection_len):
        if s < tot_selection_len-1:
            err[str(s)].append(results[time_limit_list_tot[t]][cyclone_name]['multimodel'][selections_names[s]]['distance_final'])
        else:
            err[str(s)].append(results[time_limit_list_tot[t]][cyclone_name]['UCL']['distance_final'])

            
x = np.arange(len(labels))  # the label locations

if tot_selection_len % 2:
    positions = list(range(-int(tot_selection_len/2),0))+[0]+list(range(1,int(tot_selection_len/2)+1))
else:
    positions = list(np.arange(-int(tot_selection_len/2)+0.5,0,1))+list(np.arange(0.5,int(tot_selection_len/2),1))

for s in range(tot_selection_len):
    rects[str(s)] = ax.bar(x + width*positions[s], err[str(s)], width, label=str(tot_selection_names[s]))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Error (km)', labelpad = 20)
ax.set_xlabel('Lead time (hours)', labelpad = 20)
ax.yaxis.grid()
ax.legend();

ttl = ax.set_title(title_string, fontweight='bold')
ttl.set_position([0.5, 1.05])

plt.tight_layout()
# fig.savefig(figures_folder+'error_multimodels_by_leadtime_'+cyclone_name+'.pdf', format='pdf', bbox_inches = 'tight', pad_inches = 0)


# ## Multicyclone plots

# In[ ]:


# Several panels

plt.rcParams['font.size'] = 15

Tot = len(cyclone_names)
Cols = 3

# Compute Rows required

Rows = Tot // Cols 
Rows += Tot % Cols

# Create a Position index

Position = range(1,Tot + 1)

fig = plt.figure(figsize=(30,50))

for k in range(Tot):
    
    cyclone_name = cyclone_names[k]

    # add every single subplot to the figure with a for loop
    ax = fig.add_subplot(Rows,Cols,Position[k])
  
    labels = []
    err = {}
    rects = {}

    if include_average_time == 'y':
        time_limit_list_tot = [str(x) for x in time_limit_list]+['average_time']
    else:
        time_limit_list_tot = [str(x) for x in time_limit_list]

    if include_UCL_data == 'y':
        tot_selection_len = len(selections_names)+1
        tot_selection_names = selections_names+['UCL']
    else:
        tot_selection_len = len(selections_names)
        tot_selection_names = selections_names

    for s in range(tot_selection_len):
        err[str(s)] = []

    title_string = 'Cyclone: '+cyclone_name

#     fig, ax = plt.subplots(1, 1, figsize=(30,12))


    for t in range(len(time_limit_list_tot)):
        labels.append(time_limit_list_tot[t])
        for s in range(tot_selection_len):
            if s < tot_selection_len-1:
                err[str(s)].append(results[time_limit_list_tot[t]][cyclone_name]['multimodel'][selections_names[s]]['distance_final'])
            else:
                err[str(s)].append(results[time_limit_list_tot[t]][cyclone_name]['UCL']['distance_final'])


    x = np.arange(len(labels))  # the label locations

    if tot_selection_len % 2:
        positions = list(range(-int(tot_selection_len/2),0))+[0]+list(range(1,int(tot_selection_len/2)+1))
    else:
        positions = list(np.arange(-int(tot_selection_len/2)+0.5,0,1))+list(np.arange(0.5,int(tot_selection_len/2),1))

    for s in range(tot_selection_len):
        rects[str(s)] = ax.bar(x + width*positions[s], err[str(s)], width, label=str(tot_selection_names[s]))        
        
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    if k % Cols == 0:
        ax.set_ylabel('Error (km)', labelpad = 20)
    
    if any(k==kk for kk in [Tot-1,Tot-2,Tot-3]):
        ax.set_xlabel('Lead time (hours)', labelpad = 20)
    
    ax.yaxis.grid()
    
    if k == Tot-1:
        ax.legend();

    ttl = ax.set_title(title_string, fontweight='bold')

fig.tight_layout()
# plt.show()
# plt.tight_layout(pad = 1.5, w_pad = 15)


# #### Create multicyclone data

# In[ ]:


time_limit_list_tot = [str(x) for x in time_limit_list]+['average_time']

for t in time_limit_list_tot:
    results[t]['multimodel'] = {}
    for s in range(len(selections_names)):
        r = []
        for cyclone_name in cyclone_names:
            r.append(results[t][cyclone_name]['multimodel'][selections_names[s]]['distance_final'])
        results[t]['multimodel'][selections_names[s]] = {}
        r = np.array(r)
        results[t]['multimodel'][selections_names[s]]['distance_final'] = r[~np.isnan(r)]
        
    r = []
    for cyclone_name in cyclone_names:
        r.append(results[t][cyclone_name]['UCL']['distance_final'])
    results[t]['UCL'] = {}
    r = np.array(r)
    results[t]['UCL']['distance_final'] = r[~np.isnan(r)]


# #### Final plot

# In[ ]:


selections_names


# In[ ]:


selections_better_names = ['KWBC', 'RJTD', 'EGRR', 'ECMWF', 'KWBC + RJTD + EGRR + ECMWF', 'KWBC + EGRR + ECMWF', 'KWBC + RJTD + ECMWF', 'KWBC + ECMWF']


# In[ ]:


plt.rcParams['font.size'] = 15
width = 0.08
include_average_time = 'y'

colors = ['#ED254E','#F9DC5C','#97EAD2','#731963','#40F99B','#A020F0','#D2691E','#00008B','#FF8C00','#ADFF2F','#FFFF00']
colors = ['#8c510a','#bf812d','#dfc27d','#f6e8c3','#f5f5f5','#c7eae5','#80cdc1','#35978f','#01665e']
# '#FF3864','#FF00FF','#556B2F','#B8860B',
fig, ax = plt.subplots(1, 1, figsize=(30,12))

labels = []
err = {}
rects = {}

if include_average_time == 'y':
    time_limit_list_tot = [str(x) for x in time_limit_list]+['average_time']
else:
    time_limit_list_tot = [str(x) for x in time_limit_list]

if include_UCL_data == 'y':
    tot_selection_len = len(selections_better_names)+1
    tot_selection_names = selections_better_names+['UCL']
else:
    tot_selection_len = len(selections_better_names)
    tot_selection_names = selections_better_names

for s in range(tot_selection_len):
    err[str(s)] = []

title_string = 'Multicyclone ('+str(len(cyclone_names))+' cyclones)'

#     fig, ax = plt.subplots(1, 1, figsize=(30,12))


for t in range(len(time_limit_list_tot)):
    labels.append(time_limit_list_tot[t])
    for s in range(tot_selection_len):
        if s < tot_selection_len-1:
            err[str(s)].append(results[time_limit_list_tot[t]]['multimodel'][selections_names[s]]['distance_final'])
        else:
            err[str(s)].append(results[time_limit_list_tot[t]]['UCL']['distance_final'])


x = np.arange(len(labels))  # the label locations

if tot_selection_len % 2:
    positions = list(range(-int(tot_selection_len/2),0))+[0]+list(range(1,int(tot_selection_len/2)+1))
else:
    positions = list(np.arange(-int(tot_selection_len/2)+0.5,0,1))+list(np.arange(0.5,int(tot_selection_len/2),1))
    
# Order selection by median of average times
selection_ordered = sorted(zip([np.median(err[str(s)][-1]) for s in range(tot_selection_len)], list(range(tot_selection_len))))
selection_ordered = [x[1] for x in selection_ordered]

for i,s in enumerate(selection_ordered):
    rects[str(s)] = ax.boxplot(err[str(s)], positions = x + width*positions[i], widths = width*0.7, patch_artist=True)
    
    r = random.random()
    b = random.random()
    g = random.random()
    c = (r, g, b)
    c = colors[i]
    
#     for item in ['whiskers', 'fliers', 'medians', 'caps']:
#             plt.setp(rects[str(s)][item], color=c)
            
    plt.setp(rects[str(s)]['medians'], color='black')
    plt.setp(rects[str(s)]["boxes"], facecolor=c)
    plt.setp(rects[str(s)]["fliers"], markeredgecolor=c)
    
#     rects[str(s)]['boxes'].set_facecolor('blue')

#         bplot1 = ax1.boxplot(all_data,
#                      vert=True,  # vertical box alignment
#                      patch_artist=True,  # fill with color
#                      labels=labels)  # will be used to label x-ticks

#         boxplot(A, positions = [1, 2], widths = 0.6)


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylabel('Error (km)', labelpad = 20)

ax.set_ylim([-10,700])

ax.set_xlabel('Lead time (hours)', labelpad = 20)

ax.yaxis.grid()

ax.legend([rects[str(s)]["boxes"][0] for s in selection_ordered], [tot_selection_names[s] for s in selection_ordered], loc='upper right')

ttl = ax.set_title(title_string, fontweight='bold')

# plt.show()
# plt.tight_layout(pad = 1.5, w_pad = 15)
# fig.savefig(figures_folder+'error_multicyclone_by_leadtime.pdf', format='pdf', bbox_inches = 'tight', pad_inches = 0.3)
# fig.savefig(figures_folder+'error_multicyclone_by_leadtime.png', format='png', dpi=300, bbox_inches = 'tight', pad_inches = 0.3)


# ## Print length data vector

# In[ ]:


for cyclone_name in cyclone_names:
    print('\n')
    for s in range(results[time_limit_list_tot[0]][cyclone_name]['number_models']):
        try:
            print(results[time_limit_list_tot[0]][cyclone_name][str(s)]['model_name'], results[time_limit_list_tot[0]][cyclone_name][str(s)][str(0)]['num_members_ensemble'])
        except:
            pass

