""""
Obtaining the relevant track info by filtering all typhoon tracks shape file. 
Input:
(1) Shape file (and corresponding files) of full typhoon tracks. 
    Obtained from: https://www.ncdc.noaa.gov/ibtracs/index.php?name=ib-v4-access
(2) Excel sheet with typhoons to be filtered and their SID as 'storm_id'
Output:
(1) Shape file of the filtered typhoon tracks
"""

#%% Loading Libraries
import os
import geopandas as gpd
import pandas as pd


#%% Loading data
os.chdir("c:\\Users\\Marieke\\GitHub\\Typhoon_IBF_Rice_Damage_Model")
cdir = os.getcwd()

name = "IBF_typhoon_model\\data\\gis_data\\typhoon_tracks\\IBTrACS.WP.list.v04r00.lines.shp"
path = os.path.join(cdir, name)
df_tracks = gpd.read_file(path)

name = "IBF_typhoon_model\\data\\data_overview.xlsx"
path = os.path.join(cdir, name)
df_typhoons = pd.read_excel(path, sheet_name="typhoon_overview")

#%% Creating filtered file
sid_list = df_typhoons["storm_id"]
df_tracks_filtered = df_tracks[df_tracks["SID"].isin(sid_list)]

#%% Saving df to shape file
path = "IBF_typhoon_model\\data\\gis_data\\typhoon_tracks\\tracks_filtered.shp"
df_tracks_filtered.to_file(path)
