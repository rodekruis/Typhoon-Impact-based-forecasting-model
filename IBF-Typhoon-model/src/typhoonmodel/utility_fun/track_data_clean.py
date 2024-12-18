import os
import math
import numpy as np
import xarray as xr
import pandas as pd
from climada.hazard.tc_tracks import estimate_roci,estimate_rmw
def track_data_clean(forcast_df):
   
    track = xr.Dataset(
        data_vars={
            'max_sustained_wind': ('time', pd.Series(forcast_df.max_sustained_wind.values).interpolate().tolist()),
            'environmental_pressure': ('time', forcast_df.environmental_pressure.values),
            'central_pressure': ('time', pd.Series(forcast_df.central_pressure.values).interpolate().tolist()),
            'lat': ('time', pd.Series(forcast_df.lat.values).interpolate().tolist()),
            'lon': ('time', pd.Series(forcast_df.lon.values).interpolate().tolist()),
            #'radius_max_wind': ('time', pd.Series(forcast_df.radius_max_wind.values).interpolate().tolist()),
            'max_radius':('time', pd.Series(forcast_df.max_radius.values).interpolate().tolist()),  
            'radius_max_wind': ('time', pd.Series(estimate_rmw(forcast_df.radius_max_wind.values, forcast_df.central_pressure.values)).interpolate().tolist()),
            #'radius_oci':('time', estimate_roci(forcast_df.radius_max_wind.values, forcast_df.central_pressure.values)), 
            #'time_step':('time', forcast_df.time_step.values),
            'time_step':('time', np.full(len(forcast_df.time.values),0.5).tolist())
        },
        coords={
            'time': forcast_df.time.values,
        },
        attrs={
            'max_sustained_wind_unit':forcast_df.max_sustained_wind_unit,
            'central_pressure_unit':forcast_df.central_pressure_unit,
            'name': forcast_df.name,
            'sid': forcast_df.sid,#+str(forcast_df.ensemble_number),
            'orig_event_flag': forcast_df.orig_event_flag,
            'data_provider': forcast_df.data_provider,
            'id_no': forcast_df.id_no,
            'ensemble_number': forcast_df.ensemble_number,
            'is_ensemble':forcast_df.is_ensemble,
            'forecast_time': forcast_df.forecast_time,
            'basin': forcast_df.basin,
            'category': forcast_df.category,
        }
    )

    
    track = track.resample(time="0.5H").interpolate("linear") #in
    track = track.set_coords(['lat', 'lon'])
    return track
    
def track_data_force_HRS(forcast_df,HRS_SPEED):
    track = xr.Dataset(
        data_vars={
            'max_sustained_wind': ('time', pd.Series(HRS_SPEED).interpolate().tolist()),
            'environmental_pressure': ('time', forcast_df.environmental_pressure.values),
            'central_pressure': ('time', pd.Series(forcast_df.central_pressure.values).interpolate().tolist()),
            'lat': ('time', pd.Series(forcast_df.lat.values).interpolate().tolist()),
            'lon': ('time', pd.Series(forcast_df.lon.values).interpolate().tolist()),
            'radius_max_wind': ('time', pd.Series(forcast_df.radius_max_wind.values).interpolate().tolist()),
            'radius_max_wind':('time', estimate_rmw(forcast_df.radius_max_wind.values, forcast_df.central_pressure.values)),  
            'radius_oci':('time', estimate_roci(forcast_df.radius_max_wind.values, forcast_df.central_pressure.values)), 
            'time_step':('time', forcast_df.time_step.values),
        },
        coords={
            'time': forcast_df.time.values,
        },
        attrs={
            'max_sustained_wind_unit': 'm/s',
            'central_pressure_unit': 'mb',
            'name': forcast_df.name,
            'sid': forcast_df.sid,#+str(forcast_df.ensemble_number),
            'orig_event_flag': forcast_df.orig_event_flag,
            'data_provider': forcast_df.data_provider,
            'id_no': forcast_df.id_no,
            'ensemble_number': forcast_df.ensemble_number,
            'is_ensemble':forcast_df.is_ensemble,
            'forecast_time': forcast_df.forecast_time,
            'basin': forcast_df.basin,
            'category': forcast_df.category,
        }
    )
   
 
    track = track.set_coords(['lat', 'lon'])
    return track
