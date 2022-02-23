#%% Loading libraries
import pandas as pd
import numpy as np
import random
import os
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from pandas.core.dtypes.missing import isnull
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import openpyxl

os.chdir("C:\\Users\\Marieke\\GitHub\\Typhoon_IBF_Rice_Damage_Model")
cdir = os.getcwd()


"""
Loading Data
"""
#%% Damage sheet
file_name = (
    "IBF_typhoon_model\\data\\restricted_data\\combined_input_data\\input_data_05.xlsx"
)
path = os.path.join(cdir, file_name)
df_total = pd.read_excel(path, engine="openpyxl")

# typhoon overview sheet
file_name = "IBF_typhoon_model\\data\\data_overview.xlsx"
path = os.path.join(cdir, file_name)
df_typhoons = pd.read_excel(path, sheet_name="typhoon_overview", engine="openpyxl")

# typhoon tracks shapefile
file_name = "IBF_typhoon_model\\data\\gis_data\\typhoon_tracks\\tracks_filtered.shp"
path = os.path.join(cdir, file_name)
df_tracks = gpd.read_file(path)

# map for philippines
name = "IBF_typhoon_model\\data\\phl_administrative_boundaries\\phl_admbnda_adm3.shp"
path = os.path.join(cdir, name)
df_phil = gpd.read_file(path)

# world map
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# Initial Info
sid_dict = dict(zip(df_typhoons["name_year"], df_typhoons["storm_id"]))
typhoon_dict = dict(zip(df_typhoons["storm_id"], df_typhoons["name_year"]))

sid_list = df_total["storm_id"].unique().tolist()
typhoons = df_total["typhoon"].unique().tolist()


"""
Creating damages figures: Creating single figures in loop and saving
PERCENTAGE LOSS
"""
# region
#%% Loop through all typhoons
column_color = "perc_loss"
vmin = 0
vmax = 1
cmap = "Reds"
output_folder = "IBF_typhoon_model\\data\\figures\\track_images_damage"
output_file_name = "_track_damage"

for typhoon in typhoons:

    print(typhoon)
    df_plot = df_total[df_total["typhoon"] == typhoon]
    df_plot = pd.merge(
        df_plot, df_phil, how="left", left_on=["mun_code"], right_on=["ADM3_PCODE"]
    )
    gpd_plot = gpd.GeoDataFrame(df_plot)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)
    ax.set_aspect("equal")

    minx, miny, maxx, maxy = df_phil.total_bounds

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis("off")
    ax.set_title(f"Percentage Loss for {typhoon}")

    df_phil.plot(ax=ax, color="whitesmoke", zorder=1)

    gpd_plot.plot(
        ax=ax,
        column=column_color,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=2,
        legend=True,
        legend_kwds={"shrink": 0.5},
    )

    df_tracks[df_tracks["SID"] == sid_dict[typhoon]].plot(
        ax=ax, zorder=4, linestyle=":", linewidth=2, color="black"
    )

    path = os.path.join(cdir, output_folder, typhoon + output_file_name)

    fig.savefig(path)
    plt.close()


print("Done")
# endregion

"""
Creating damages figures: Creating single figures in loop and saving
MAXIMUM 6 HOUR RAINFALL
"""
# region
#%% Loop through all typhoons
column_color = "rainfall_max_6h"
vmin = 0
vmax = 46
cmap = "Blues"
output_folder = "IBF_typhoon_model\\data\\figures\\track_images_rainfall"
output_file_name = "_track_max_6h_rainfall"

path = os.path.join(cdir, "IBF_typhoon_model/data/rainfall_data/rainfall_max_6h.csv")
df_rainfall_max_6h = pd.read_csv(path)

for typhoon in typhoons:

    print(typhoon)
    df_plot = df_rainfall_max_6h[df_rainfall_max_6h["typhoon"] == typhoon]
    df_plot = pd.merge(
        df_plot, df_phil, how="left", left_on=["mun_code"], right_on=["ADM3_PCODE"]
    )
    gpd_plot = gpd.GeoDataFrame(df_plot)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)
    ax.set_aspect("equal")

    minx, miny, maxx, maxy = df_phil.total_bounds

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis("off")
    ax.set_title(f"Maximum 6 hour rainfall in mm/h for {typhoon}")

    df_phil.plot(ax=ax, color="whitesmoke", zorder=1)

    gpd_plot.plot(
        ax=ax,
        column=column_color,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=2,
        legend=True,
        legend_kwds={"shrink": 0.5},
    )

    df_tracks[df_tracks["SID"] == sid_dict[typhoon]].plot(
        ax=ax, zorder=4, linestyle=":", linewidth=2, color="black"
    )

    path = os.path.join(cdir, output_folder, typhoon + output_file_name)

    fig.savefig(path)
    plt.close()


print("Done")
# endregion

"""
Creating damages figures: Creating single figures in loop and saving
MAXIMUM SUSTAINED WIND
"""
# region
#%% Loop through all typhoons
column_color = "v_max"
vmin = 0
vmax = 87
cmap = "Blues"
output_folder = "IBF_typhoon_model\\data\\figures\\track_images_wind"
output_file_name = "_track_vmax"

# path = os.path.join(cdir, "IBF_typhoon_model/data/rainfall_data/rainfall_max_6h.csv")
# df_rainfall_max_6h = pd.read_csv(path)

for typhoon in typhoons:

    print(typhoon)

    path = (
        "IBF_typhoon_model\\data\\wind_data\\output\\"
        + typhoon
        + "_windgrid_output.csv"
    )
    df_plot = pd.read_csv(os.path.join(cdir, path))

    df_plot = pd.merge(
        df_plot, df_phil, how="left", left_on=["adm3_pcode"], right_on=["ADM3_PCODE"]
    )
    gpd_plot = gpd.GeoDataFrame(df_plot)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)
    ax.set_aspect("equal")

    minx, miny, maxx, maxy = df_phil.total_bounds

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis("off")
    ax.set_title(f"Maximum wind speed in m/s for {typhoon}")

    df_phil.plot(ax=ax, color="whitesmoke", zorder=1)

    gpd_plot.plot(
        ax=ax,
        column=column_color,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=2,
        legend=True,
        legend_kwds={"shrink": 0.5},
    )

    df_tracks[df_tracks["SID"] == sid_dict[typhoon]].plot(
        ax=ax, zorder=4, linestyle=":", linewidth=2, color="black"
    )

    path = os.path.join(cdir, output_folder, typhoon + output_file_name)

    fig.savefig(path)
    plt.close()


print("Done")
# endregion

"""
Creating damages figures: Creating single figures in loop and saving
PERCENTAGE DAMAGE AND BUFFER
"""
# region

#%% Adding a track buffer
# Plotting the damages and tracks with a buffer of 500km, to show that allmost all damage fall within this area
# Used to confirm that predictions only need to be made for municipalities that are within 500km distance from the track
buffer = 500000
df_tracks_buff = df_tracks.copy()
df_tracks_buff = df_tracks_buff.to_crs("EPSG:25395")

df_tracks_buff["geometry"] = df_tracks_buff.buffer(buffer)
df_tracks_buff = df_tracks_buff.to_crs("EPSG:4326")

column_color = "perc_loss"
vmin = 0
vmax = 1
cmap = "Reds"
output_folder = "IBF_typhoon_model\\data\\figures\\track_images_damage_buffer"
output_file_name = "_track_damage_buffer"

#%% Loop through all typhoons
for typhoon in typhoons:

    print(typhoon)
    df_plot = df_total[df_total["typhoon"] == typhoon]
    df_plot = pd.merge(
        df_plot, df_phil, how="left", left_on=["mun_code"], right_on=["ADM3_PCODE"]
    )
    gpd_plot = gpd.GeoDataFrame(df_plot)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)
    ax.set_aspect("equal")

    minx, miny, maxx, maxy = df_phil.total_bounds

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis("off")
    ax.set_title(f"Percentage Loss and track distance of {buffer/1000}km for {typhoon}")

    df_phil.plot(ax=ax, color="whitesmoke", zorder=1)

    gpd_plot.plot(
        ax=ax, column=column_color, cmap="Reds", vmin=vmin, vmax=vmax, zorder=2
    )

    df_tracks[df_tracks["SID"] == sid_dict[typhoon]].plot(
        ax=ax, zorder=4, linestyle=":", linewidth=2, color="black"
    )

    df_tracks_buff[df_tracks_buff["SID"] == sid_dict[typhoon]].plot(
        ax=ax, alpha=0.01, zorder=3, color="blue"
    )

    path = os.path.join(cdir, output_folder, typhoon + output_file_name)
    fig.savefig(path)

    plt.close()


print("Done")
# endregion

"""
Creating damages figures: Creating single figures in loop and saving
PERCENTAGE LOSS: ABOVE OR BELOW 30%
"""
# region
#%% Loop through all typhoons
column_color = "damage_above_30"
vmin = -1
vmax = 1
cmap = "Reds"
output_folder = "IBF_typhoon_model\\data\\figures\\track_images_binary"
output_file_name = "_binary_damage"

for typhoon in typhoons:

    print(typhoon)
    df_plot = df_total[df_total["typhoon"] == typhoon]
    df_plot = pd.merge(
        df_plot, df_phil, how="left", left_on=["mun_code"], right_on=["ADM3_PCODE"]
    )
    gpd_plot = gpd.GeoDataFrame(df_plot)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)
    ax.set_aspect("equal")

    minx, miny, maxx, maxy = df_phil.total_bounds

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis("off")
    ax.set_title(f"Damage higher than 30% for {typhoon}")

    df_phil.plot(ax=ax, color="whitesmoke", zorder=1)

    gpd_plot.plot(
        ax=ax,
        column=column_color,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=2,
        legend=True,
        categorical=True,
    )

    df_tracks[df_tracks["SID"] == sid_dict[typhoon]].plot(
        ax=ax, zorder=4, linestyle=":", linewidth=2, color="black"
    )

    path = os.path.join(cdir, output_folder, typhoon + output_file_name)

    fig.savefig(path)
    plt.close()


print("Done")
# endregion


# %%
