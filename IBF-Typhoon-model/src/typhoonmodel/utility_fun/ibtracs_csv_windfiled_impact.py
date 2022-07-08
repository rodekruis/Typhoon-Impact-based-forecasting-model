# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 20:29:04 2022

@author: ATeklesadik
"""
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from os.path import relpath
from geopandas.tools import sjoin
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import nearest_points
import datetime as datetime
 
#wor_dir="/home/fbf/src"
wor_dir='C:/Typhoon_IBF/Typhoon-Impact-based-forecasting-model/IBF-Typhoon-model/src'
#wor_dir="C:\\Typhoon_IBF\\Typhoon-Impact-based-forecasting-model\\IBF-Typhoon-model\\src"
#sys.path.insert(0, wor_dir+'\\IBF-Typhoon-model\\src')
os.chdir(wor_dir)
# Importing local libraries       
from climada.hazard import Centroids, TropCyclone, TCTracks
from climada.hazard.tc_tracks import estimate_roci, estimate_rmw
from climada.hazard.tc_tracks_forecast import TCForecast

wor_dir="C:\\Typhoon_IBF\\Typhoon-Impact-based-forecasting-model\\IBF-Typhoon-model"
#wor_dir="/home/fbf"
os.chdir(wor_dir)
cdir = os.getcwd()
 
 
 

def csv_nc_tracks(data,typhoon):
        dta_dict=data.query('SID==@typhoon').to_dict('list') 
        track = xr.Dataset(
            data_vars={
                "time_step": ("time", np.full_like(dta_dict['SID'], 3, dtype=float)),
                "max_sustained_wind": (
                    "time", dta_dict['USA_WIND'],  # conversion from kn to meter/s
                ),
                "environmental_pressure": (
                    "time",
                    [1010]*len(dta_dict['SID']),
                ),
                "central_pressure": ("time", [1005]*len(dta_dict['SID'])),
                "lat": ("time", dta_dict['LAT']),
                "lon": ("time", dta_dict['LON']),
                "radius_max_wind": ("time",dta_dict['USA_RMW']),
                "radius_oci": ("time", dta_dict['USA_ROCI']),
                "basin": ("time", ['WP']*len(dta_dict['USA_RMW'])),
                
                
            },
            coords={"time": pd.date_range("1900-01-01", periods=len(dta_dict['SID']), freq="3H"),},
            attrs={
                "max_sustained_wind_unit": "m/s",
                "central_pressure_unit": "mb",
                "name": typhoon,
                "sid": typhoon,  # +str(forcast_df.ensemble_number),
                "orig_event_flag": True,
                "data_provider": 'ibtracs_usa',
                "id_no": typhoon,
                "basin": 'wp',
                "category": np.nan,         
                
            },
        )
        track = track.set_coords(["lat", "lon"])
        return track


 
 
 

file_name = "data/wind_data/input/phl_admin3_simpl2.geojson"
path = os.path.join(cdir, file_name)
admin = gpd.read_file(path)

minx, miny, maxx, maxy = admin.total_bounds
print(minx, miny, maxx, maxy)

cent = Centroids()
cent.set_raster_from_pnt_bounds((minx, miny, maxx, maxy), res=0.05)
cent.check()
cent.plot()
plt.show()
plt.close()

 
def match_values_lat(x):

    return df_admin["lat"][df_admin["centroid_id"] == x].values[0]

def match_values_lon(x):

    return df_admin["lon"][df_admin["centroid_id"] == x].values[0]

def match_values_geo(x):

    return df_admin["geometry"][df_admin["centroid_id"] == x].values[0]

df = pd.DataFrame(data=cent.coord)
df["centroid_id"] = "id" + (df.index).astype(str)
centroid_idx = df["centroid_id"].values
ncents = cent.size
df = df.rename(columns={0: "lat", 1: "lon"})
df_ = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
df_.crs = {"init": "epsg:4326"}
df_ = df_.to_crs("EPSG:4326")
df_admin = sjoin(df_, admin, how="left")

# To remove points that are in water
df_admin = df_admin.dropna()



df_admin = df_admin[
    [
        "adm3_en",
        "adm3_pcode",
        "adm2_pcode",
        "adm1_pcode",
        "glat",
        "glon",
        "lat",
        "lon",
        "centroid_id",
    ]
]
 
threshold = 10

#%%
######################################################
####### START PROCESSING
#########################################################
 
typhoon_filename = os.path.join(wor_dir, "data/wind_data/input/ibtracs.WP.list.v04r00.csv")

ibtracs = pd.read_csv(typhoon_filename, delimiter=",")

data=ibtracs.filter(['SID', 'SEASON', 'NUMBER', 'BASIN', 'NAME', 'ISO_TIME','NATURE', 'LAT', 'LON','USA_WIND','USA_PRES','USA_ROCI','USA_RMW'])
data = data.iloc[1: , :]
data['Date']= pd.to_datetime(data['ISO_TIME'])

for cols in ['SEASON', 'NUMBER', 'LAT', 'LON','USA_WIND','USA_PRES','USA_ROCI','USA_RMW']:
    data[cols] = pd.to_numeric(data[cols], errors='coerce')
    
data['USA_WIND']=0.514444*data['USA_WIND'].values    



#%%%

def adjust_tracks(forcast_df):
        track = xr.Dataset(
            data_vars={
                "max_sustained_wind": (
                    "time",
                    0.514444
                    * forcast_df.max_sustained_wind.values,  # conversion from kn to meter/s
                ),
                "environmental_pressure": (
                    "time",
                    forcast_df.environmental_pressure.values,
                ),
                "central_pressure": ("time", forcast_df.central_pressure.values),
                "lat": ("time", forcast_df.lat.values),
                "lon": ("time", forcast_df.lon.values),
                "radius_max_wind": ("time", forcast_df.radius_max_wind.values),
                "radius_oci": ("time", forcast_df.radius_oci.values),
                "time_step": (
                    "time",
                    np.full_like(forcast_df.time_step.values, 3, dtype=float),
                ),
            },
            coords={"time": forcast_df.time.values,},
            attrs={
                "max_sustained_wind_unit": "m/s",
                "central_pressure_unit": "mb",
                "name": forcast_df.name,
                "sid": forcast_df.sid,  # +str(forcast_df.ensemble_number),
                "orig_event_flag": forcast_df.orig_event_flag,
                "data_provider": forcast_df.data_provider,
                "id_no": forcast_df.id_no,
                "basin": forcast_df.basin,
                "category": forcast_df.category,
            },
        )
        track = track.set_coords(["lat", "lon"])
        return track
    
 
sel_ibtracs = TCTracks()
# Set year range for which data should be collected
sel_ibtracs.read_ibtracs_netcdf(provider="usa", year_range=(1980, 2022), basin="WP", correct_pres=True)
tracks = TCTracks()
tracks.data = [adjust_tracks(tr) for tr in sel_ibtracs.data]
tracks.equal_timestep(0.5)



#%%

data=data.query('SEASON>1980')
events_SID=np.unique(data.SID.values)
#%%
for tr in tracks.data:
    print(tr.name)
    trackss = TCTracks()
    trackss.data = [tr]
    
#for typhoon_name in events_SID:
    

    #tracks2=csv_nc_tracks(data,typhoon_name)
    
    
    #trackss = TCTracks()
    #trackss.data = [ tracks2  ]        
    #trackss.equal_timestep(0.5)   


    if trackss.data !=[]:
        # define a new typhoon class
        typhoon = TropCyclone()
        typhoon.set_from_tracks(trackss, cent, store_windfields=True)

        list_intensity = []
        distan_track = []
        windfield = typhoon.windfields
        nsteps = windfield[0].shape[0]
        centroid_id = np.tile(centroid_idx, nsteps)
        intensity_3d = windfield[0].toarray().reshape(nsteps, ncents, 2)
        intensity = np.linalg.norm(intensity_3d, axis=-1).ravel()

        timesteps = np.repeat(trackss.data[0].time.values, ncents)
        timesteps = timesteps.reshape((nsteps, ncents)).ravel()
        inten_tr = pd.DataFrame(
            {"centroid_id": centroid_id, "value": intensity, "timestamp": timesteps,}
        )

        inten_tr = inten_tr[inten_tr.value > threshold]

        inten_tr["storm_id"] = tr.sid
        list_intensity.append(inten_tr)

    df_intensity = pd.concat(list_intensity)
    df_intensity = pd.merge(df_intensity, df_admin, how="outer", on="centroid_id")

    df_intensity = df_intensity.dropna()
    
    if len(df_intensity.index > 1):
        # Obtains the maximum intensity for each municipality and storm_id combination & also return the count (how often the combination occurs in the set)
        df_intensity = ( df_intensity[df_intensity["value"].gt(0)].groupby(["adm3_pcode", "storm_id"], as_index=False).agg({"value": ["count", "max"]}))
        # rename columns
        df_intensity.columns = [ x for x in ["adm3_pcode", "storm_id", "value_count", "v_max"]  ]

    
        # Obtains the minimum track distance for each municipality and storm_id combination
    
 
        typhhon_df =df_intensity# pd.merge( df_intensity, df_track_, how="left", on=["adm3_pcode", "storm_id"]        )

        duplicate = typhhon_df[
            typhhon_df.duplicated(subset=["adm3_pcode", "storm_id"], keep=False)
        ]
        if len(duplicate) != 0:
            print("There are duplicates, please check")

        file_name = (
            "data/wind_data/output_ibtracks2/"
            + tr.sid
            + "_windgrid_output_ibttracs.csv"
        )
        path = os.path.join(cdir, file_name)
        typhhon_df.to_csv(path)
        
        
#%%%

#Adding the wind data only for events with impact data

wind_path = os.path.join(wor_dir,"data/wind_data/output_ibtracks2")
 
dfs=[]

    

#df_wind_final= pd.DataFrame(columns=["typhoon", "Mun_Code", "v_max","dis_track_min"])

for file in os.listdir(wind_path):
    
    typhoon=file.split('_')[0]

    # Path to the rainfall excel sheet per typhoon
    f_path = os.path.join(wind_path,  file )
    
    df_temp = pd.read_csv(f_path)
    
    if len(df_temp.index)>1:
        df_temp.rename(columns={'adm3_pcode':'Mun_Code'},inplace=True)
        df_wind=df_temp[['storm_id','v_max','Mun_Code']].groupby("Mun_Code").agg({'v_max':'max'})
        df_wind.reset_index(inplace=True)
        df_wind['typhoon']=typhoon
        #df_wind['typhoon_name']=df_wind['typhoon'].apply(lambda x:(x[:-4].upper()+'-'+x[-4:]))
        dfs.append(df_wind)   
   

df_wind_final = pd.concat(dfs,axis=0, join='outer')      
 
#%%

f_path = os.path.join(wor_dir,"data/all_predisaster_indicators.csv") 
df_predisasters = pd.read_csv(f_path)


def division(x, y):
    try:
        value =100* (x / y)
        
    except:
        value = np.nan
    
    return 100 if value>100 else value


# Setting the new damage threshold

df_predisasters["vulnerable_groups"] = df_predisasters.apply(lambda x: division(x["vulnerable_groups"], x["Total Pop"]), axis=1).values
#df_total["pantawid_pamilya_beneficiary"] = df_total.apply(lambda x: division(x["Total # of Active HHs"], x["Housing Units"]), axis=1).values
df_predisasters["pantawid_pamilya_beneficiary"] = df_predisasters.apply(lambda x: division(x["pantawid_total_pop"], x["Total Pop"]), axis=1).values

df_total=pd.merge(df_wind_final, df_predisasters,  how='left', left_on='Mun_Code', right_on = 'Mun_Code')
df_total=df_total[df_total['v_max'].notnull()]

#%%
df_total=df_total.filter(['Mun_Code', 'typhoon', 'v_max', 'landslide_per', 'stormsurge_per',
                 'Bu_p_inSSA','Bu_p_LS', 'Red_per_LSbldg', 'Or_per_LSblg',
                 'Yel_per_LSSAb','RED_per_SSAbldg', 'OR_per_SSAbldg', 
                 'Yellow_per_LSbl', 'mean_slope','mean_elevation_m', 
                 'ruggedness_stdev', 'mean_ruggedness','slope_stdev',
                 'poverty_perc', 'with_coast', 'coast_length','Housing Units', 'Strong Roof/Strong Wall',
                 'Strong Roof/Light Wall', 'Strong Roof/Salvage Wall',
                 'Light Roof/Strong Wall', 'Light Roof/Light Wall',
                 'Light Roof/Salvage Wall', 'Salvaged Roof/Strong Wall',
                 'Salvaged Roof/Light Wall', 'Salvaged Roof/Salvage Wall',
                 'vulnerable_groups', 'pantawid_pamilya_beneficiary','perc_dmg'])


df_total.rename(columns ={           'v_max':'HAZ_v_max',
                                    'landslide_per':'GEN_landslide_per',
                                    'stormsurge_per':'GEN_stormsurge_per',
                                    'Bu_p_inSSA':'GEN_Bu_p_inSSA',
                                    'Bu_p_LS':'GEN_Bu_p_LS',
                                     'Red_per_LSbldg':'GEN_Red_per_LSbldg',
                                    'Or_per_LSblg':'GEN_Or_per_LSblg',
                                     'Yel_per_LSSAb':'GEN_Yel_per_LSSAb',
                                    'RED_per_SSAbldg':'GEN_RED_per_SSAbldg',
                                     'OR_per_SSAbldg':'GEN_OR_per_SSAbldg',
                                    'Yellow_per_LSbl':'GEN_Yellow_per_LSbl',
                                     'mean_slope':'TOP_mean_slope',
                                    'mean_elevation_m':'TOP_mean_elevation_m',
                                     'ruggedness_stdev':'TOP_ruggedness_stdev',
                                    'mean_ruggedness':'TOP_mean_ruggedness',
                                     'slope_stdev':'TOP_slope_stdev',
                                     'poverty_perc':'VUL_poverty_perc',
                                    'with_coast':'GEN_with_coast',
                                     'coast_length':'GEN_coast_length',
                                     'Housing Units':'VUL_Housing_Units',
                                    'Strong Roof/Strong Wall':"VUL_StrongRoof_StrongWall",
                                    'Strong Roof/Light Wall':'VUL_StrongRoof_LightWall',
                                    'Strong Roof/Salvage Wall':'VUL_StrongRoof_SalvageWall',
                                    'Light Roof/Strong Wall':'VUL_LightRoof_StrongWall',
                                    'Light Roof/Light Wall':'VUL_LightRoof_LightWall',
                                    'Light Roof/Salvage Wall':'VUL_LightRoof_SalvageWall',
                                    'Salvaged Roof/Strong Wall':'VUL_SalvagedRoof_StrongWall',
                                    'Salvaged Roof/Light Wall':'VUL_SalvagedRoof_LightWall',
                                    'Salvaged Roof/Salvage Wall':'VUL_SalvagedRoof_SalvageWall',
                                    'vulnerable_groups':'VUL_vulnerable_groups',
                                    'pantawid_pamilya_beneficiary':'VUL_pantawid_pamilya_beneficiary'},inplace=True)



selected_columns=['Mun_Code', 'typhoon', 'HAZ_v_max','GEN_landslide_per', 'GEN_stormsurge_per',
'GEN_Bu_p_inSSA','GEN_Bu_p_LS', 'GEN_Red_per_LSbldg', 'GEN_Or_per_LSblg','GEN_Yel_per_LSSAb',
'GEN_RED_per_SSAbldg', 'GEN_OR_per_SSAbldg','GEN_Yellow_per_LSbl', 'TOP_mean_slope', 'TOP_mean_elevation_m',
'TOP_ruggedness_stdev', 'TOP_mean_ruggedness', 'TOP_slope_stdev','VUL_poverty_perc', 'GEN_with_coast',
'GEN_coast_length','VUL_Housing_Units', 'VUL_StrongRoof_StrongWall','VUL_StrongRoof_LightWall',
'VUL_StrongRoof_SalvageWall','VUL_LightRoof_StrongWall', 'VUL_LightRoof_LightWall','VUL_LightRoof_SalvageWall',
'VUL_SalvagedRoof_StrongWall','VUL_SalvagedRoof_LightWall', 'VUL_SalvagedRoof_SalvageWall','VUL_vulnerable_groups',
'VUL_pantawid_pamilya_beneficiary','DAM_perc_dmg']


df_total = df_total.filter(selected_columns)

#%%%


import numpy as np
import random
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
import os
from sklearn.feature_selection import RFECV
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
    KFold)
import numpy as np
from numpy.lib.function_base import average
import pandas as pd
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error,
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix,
    make_scorer)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
 
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import importlib
import os
from sklearn.feature_selection import (
    SelectKBest,
    SequentialFeatureSelector,
    RFE,
    mutual_info_regression,
    f_regression,
    mutual_info_classif)
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.inspection import permutation_importance
import xgboost as xgb
import pickle
import openpyxl
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import geopandas as gpd

 
combined_input_data=pd.read_csv("data/model_input/df_modelinput_july.csv") 
tphoon_events=combined_input_data[['typhoon','DAM_perc_dmg']].groupby('typhoon').size().to_dict()
### for hyper parameter optimization we looped over typhoon events with at least 100 data entries  

 
def set_zeros(x):
    x_max = 25
    y_max = 50    
    v_max = x[0]
    rainfall_max = x[1]
    damage = x[2]
    if pd.notnull(damage):
        value = damage
    elif v_max > x_max or rainfall_max > y_max:
        value =damage
    elif (v_max < np.sqrt((1- (rainfall_max**2/y_max ** 2))*x_max ** 2)):
        value = 0
    #elif ((v_max < x_max)  and  (rainfall_max_6h < y_max) ):
    #elif (v_max < x_max ):
    #value = 0
    else:
        value = np.nan
    return value

combined_input_data["DAM_perc_dmg"] = combined_input_data[["HAZ_v_max", "HAZ_rainfall_Total", "DAM_perc_dmg"]].apply(set_zeros, axis="columns")


selected_features_xgb_regr= [#'HAZ_rainfall_max_24h',
                                       'HAZ_v_max',
                                       #'HAZ_dis_track_min',
                                       'TOP_mean_slope',
                                       'TOP_mean_elevation_m',
                                       'TOP_ruggedness_stdev',
                                       'TOP_mean_ruggedness',
                                       'TOP_slope_stdev',
                                       'VUL_poverty_perc',
                                       'GEN_with_coast',
                                       'VUL_Housing_Units',
                                       'VUL_StrongRoof_StrongWall',
                                       'VUL_StrongRoof_LightWall',
                                       'VUL_StrongRoof_SalvageWall',
                                       'VUL_LightRoof_StrongWall',
                                       'VUL_LightRoof_LightWall',
                                       'VUL_SalvagedRoof_StrongWall',
                                       'VUL_SalvagedRoof_LightWall',
                                       'VUL_SalvagedRoof_SalvageWall',
                                       'VUL_vulnerable_groups',
                                       'VUL_pantawid_pamilya_beneficiary']



# split data into train and test sets

SEED2 = 314159265
SEED = 31 
 
test_size = 0.1

# Full dataset for feature selection

combined_input_data_ = combined_input_data[combined_input_data['DAM_perc_dmg'].notnull()]


X = combined_input_data_[selected_features_xgb_regr]
y = combined_input_data_["DAM_perc_dmg"]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)





from sklearn.metrics import mean_squared_error,mean_absolute_error

 
reg = xgb.XGBRegressor(base_score=0.5,
             booster='gbtree', 
             subsample=0.8,
             eta=0.05,
             max_depth=8, 
             colsample_bylevel=1,
             colsample_bynode=1, 
             colsample_bytree=1,
             early_stopping_rounds=10, 
             eval_metric=mean_absolute_error,
             gamma=1, 
             objective='reg:squarederror',
             gpu_id=-1, 
             grow_policy='depthwise', 
             learning_rate=0.025,
             min_child_weight=1,
             n_estimators=200,        
             random_state=42,
             tree_method="hist",
             )

eval_set=[(X_train, y_train), ( X_test, y_test)]

reg.fit(X, y, eval_set=eval_set)

#%%%



y_pred = reg.predict(X)
combined_input_data_['DMG_predicted']=y_pred


#%%%


admin_df =gpd.read_file('C:/Typhoon_IBF/Typhoon-Impact-based-forecasting-model/IBF-Typhoon-model/data/wind_data/input/phl_admin3_simpl2.geojson')
 

IMPACT_DF0 = pd.merge(combined_input_data_, df_predisasters[['Housing Units','Mun_Code']],  how='left', left_on='Mun_Code', right_on = 'Mun_Code') 
IMPACT_DF0['Hu']=0.01*IMPACT_DF0['Housing Units']

impact_scenarios=['DAM_perc_dmg', 'DMG_predicted']

IMPACT_DF0.loc[:,impact_scenarios] = IMPACT_DF0.loc[:,impact_scenarios].multiply(IMPACT_DF0.loc[:, 'Hu'], axis="index")
IMPACT_DF0[impact_scenarios] = IMPACT_DF0[impact_scenarios].astype('int')

IMPACT_DF0.rename(columns={'DAM_perc_dmg':'DAM_number_Hous_DROMIC', 'DMG_predicted':'DAM_number_Hous_Model'},inplace=True)




#%%


X_all = df_total[selected_features_xgb_regr]
 

y_pred = reg.predict(X_all)
df_total['No_Totally_DMG_BLD']=y_pred


IMPACT_DF1 = pd.merge(df_total, df_predisasters[['Housing Units','Mun_Code']],  how='left', left_on='Mun_Code', right_on = 'Mun_Code') 
IMPACT_DF1['Hu']=0.01*IMPACT_DF1['Housing Units']

impact_scenarios=['No_Totally_DMG_BLD']

IMPACT_DF1.loc[:,impact_scenarios] = IMPACT_DF1.loc[:,impact_scenarios].multiply(IMPACT_DF1.loc[:, 'Hu'], axis="index")
IMPACT_DF1[impact_scenarios] = IMPACT_DF1[impact_scenarios].astype('int')
 
IMPACT_DF1[['Mun_Code','typhoon','No_Totally_DMG_BLD']].to_csv(os.path.join(wor_dir,'results','loos_table_ibtrcs.csv'))



