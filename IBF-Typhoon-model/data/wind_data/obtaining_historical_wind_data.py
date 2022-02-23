#%% import libraries from local directory
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from os.path import relpath
from pybufrkit.decoder import Decoder
from geopandas.tools import sjoin
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import nearest_points

os.chdir("C:\\Users\\Marieke\\GitHub\\Typhoon_IBF_Rice_Damage_Model")
cdir = os.getcwd()
decoder = Decoder()

# Importing local libraries
sys.path.insert(0, os.path.join(cdir, "\\IBF_typhoon_model\\data\\wind_data\\climada"))

from climada.hazard import Centroids, TropCyclone, TCTracks
from climada.hazard.tc_tracks import estimate_roci, estimate_rmw
from climada.hazard.tc_tracks_forecast import TCForecast


#%%######################################################
####### VARIABLES TO SET
#########################################################
year_range = (2006, 2011)

# Laoding PH admin file and setting centroids
# Instead of one grid point for each municipalities: uses a range of gridpoints
# calculates value for each point
# Eventually select a specific point in municipality based on condition
# maximum windspeed and minimum track distance
file_name = "IBF_typhoon_model\\data\\wind_data\\input\\phl_admin3_simpl2.geojson"
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

# Loads an excel sheet with Local_name, International_Name and year
file_name = "IBF_typhoon_model\\data\\wind_data\\input\\typhoon_events_test.csv"
path = os.path.join(cdir, file_name)
typhoon_events = pd.read_csv(path)

df_typhoons = pd.read_csv(path)

typhoon_events = []
for index, row in df_typhoons.iterrows():
    typhoon_events.append(str(row["International_Name"]).upper() + str(row["year"]))

#%%######################################################
####### START PROCESSING
#########################################################

for typhoon_name in typhoon_events:

    typoon_event = [typhoon_name]
    print(typoon_event)

    sel_ibtracs = TCTracks()
    # Set year range for which data should be collected
    sel_ibtracs.read_ibtracs_netcdf(
        provider="usa", year_range=year_range, basin="WP", correct_pres=False
    )

    Typhoons = TCTracks()
    # Select typhoons that are in the typhoon event sheet
    Typhoons.data = [
        tr for tr in sel_ibtracs.data if (tr.name + tr.sid[:4]) in typoon_event
    ]

    # Plot the typhoon track
    ax = Typhoons.plot()
    ax.get_legend()._loc = 1
    ax.set_title(typoon_event, fontsize=14)
    plt.show()
    plt.close()

    # Select names and storm id's of storms
    names = [[tr.name, tr.sid] for tr in Typhoons.data]

    df = pd.DataFrame(data=cent.coord)
    df["centroid_id"] = "id" + (df.index).astype(str)
    centroid_idx = df["centroid_id"].values
    ncents = cent.size
    df = df.rename(columns={0: "lat", 1: "lon"})

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

    tracks = TCTracks()
    tracks.data = [adjust_tracks(tr) for tr in Typhoons.data]
    tracks.equal_timestep(0.5)

    # define a new typhoon class
    TYphoon = TropCyclone()
    TYphoon.set_from_tracks(tracks, cent, store_windfields=True)

    # plot intensity
    TYphoon.plot_intensity(event=Typhoons.data[0].sid)
    plt.show()
    plt.close()

    df = pd.DataFrame(data=cent.coord)
    df["centroid_id"] = "id" + (df.index).astype(str)
    centroid_idx = df["centroid_id"].values
    ncents = cent.size
    df = df.rename(columns={0: "lat", 1: "lon"})
    threshold = 0.1

    list_intensity = []
    distan_track = []

    for tr in tracks.data:
        print(tr.name)

        track = TCTracks()
        typhoon = TropCyclone()
        track.data = [tr]
        typhoon.set_from_tracks(track, cent, store_windfields=True)
        windfield = typhoon.windfields
        nsteps = windfield[0].shape[0]
        centroid_id = np.tile(centroid_idx, nsteps)
        intensity_3d = windfield[0].toarray().reshape(nsteps, ncents, 2)
        intensity = np.linalg.norm(intensity_3d, axis=-1).ravel()

        timesteps = np.repeat(tr.time.values, ncents)
        timesteps = timesteps.reshape((nsteps, ncents)).ravel()
        inten_tr = pd.DataFrame(
            {"centroid_id": centroid_id, "value": intensity, "timestamp": timesteps,}
        )

        inten_tr = inten_tr[inten_tr.value > threshold]

        inten_tr["storm_id"] = tr.sid
        list_intensity.append(inten_tr)
        distan_track1 = []
        for index, row in df.iterrows():
            dist = np.min(
                np.sqrt(
                    np.square(tr.lat.values - row["lat"])
                    + np.square(tr.lon.values - row["lon"])
                )
            )
            distan_track1.append(dist * 111)
        dist_tr = pd.DataFrame({"centroid_id": centroid_idx, "value": distan_track1})
        dist_tr["storm_id"] = tr.sid
        distan_track.append(dist_tr)

    df_ = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    df_.crs = {"init": "epsg:4326"}
    df_ = df_.to_crs("EPSG:4326")
    df_admin = sjoin(df_, admin, how="left")
    # To remove points that are in water
    df_admin = df_admin.dropna()

    # For some municipalities, there is missing data
    # For these, fill with the value of the nearest centroid with data observed
    mun_missing = admin[admin["adm3_pcode"].isin(df_admin["adm3_pcode"]) == False]
    print(f"There are {len(mun_missing)} missing municipalities")

    mun_missing_points = gpd.GeoDataFrame(
        mun_missing, geometry=gpd.points_from_xy(mun_missing.glon, mun_missing.glat)
    )
    pts3 = df_admin.geometry.unary_union

    def near(point, pts=pts3):
        # find the nearest point and return the corresponding Place value
        nearest = df_admin.geometry == nearest_points(point, pts)[1]
        # display(df_admin[nearest].centroid_id.values[0])
        return df_admin[nearest].centroid_id.values[0]

    mun_missing_points["centroid_id"] = mun_missing_points.apply(
        lambda row: near(row.geometry), axis=1
    )

    def match_values_lat(x):

        return df_admin["lat"][df_admin["centroid_id"] == x].values[0]

    def match_values_lon(x):

        return df_admin["lon"][df_admin["centroid_id"] == x].values[0]

    def match_values_geo(x):

        return df_admin["geometry"][df_admin["centroid_id"] == x].values[0]

    mun_missing_points["lat"] = mun_missing_points["centroid_id"].apply(
        match_values_lat
    )
    mun_missing_points["lon"] = mun_missing_points["centroid_id"].apply(
        match_values_lon
    )
    mun_missing_points["geometry"] = mun_missing_points["centroid_id"].apply(
        match_values_geo
    )

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

    mun_missing_points = mun_missing_points[
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

    df_admin = pd.concat([df_admin, mun_missing_points])
    mun_missing = admin[admin["adm3_pcode"].isin(df_admin["adm3_pcode"]) == False]
    print(f"There are {len(mun_missing)} missing municipalities")

    df_intensity = pd.concat(list_intensity)
    df_intensity = pd.merge(df_intensity, df_admin, how="outer", on="centroid_id")

    df_intensity = pd.concat(list_intensity)
    df_intensity = pd.merge(df_intensity, df_admin, how="outer", on="centroid_id")

    df_intensity = df_intensity.dropna()

    # Obtains the maximum intensity for each municipality and storm_id combination & also return the count (how often the combination occurs in the set)
    df_intensity = (
        df_intensity[df_intensity["value"].gt(0)]
        .groupby(["adm3_pcode", "storm_id"], as_index=False)
        .agg({"value": ["count", "max"]})
    )
    # rename columns
    df_intensity.columns = [
        x for x in ["adm3_pcode", "storm_id", "value_count", "v_max"]
    ]

    df_track = pd.concat(distan_track)
    df_track = pd.merge(df_track, df_admin, how="outer", on="centroid_id")
    df_track = df_track.dropna()

    # Obtains the minimum track distance for each municipality and storm_id combination
    df_track_ = df_track.groupby(["adm3_pcode", "storm_id"], as_index=False).agg(
        {"value": "min"}
    )
    df_track_.columns = [x for x in ["adm3_pcode", "storm_id", "dis_track_min"]]
    typhhon_df = pd.merge(
        df_intensity, df_track_, how="left", on=["adm3_pcode", "storm_id"]
    )

    # Check if there are duplicates for municipality and storm_id
    duplicate = typhhon_df[
        typhhon_df.duplicated(subset=["adm3_pcode", "storm_id"], keep=False)
    ]
    if len(duplicate) != 0:
        print("There are duplicates, please check")

    file_name = (
        "IBF_typhoon_model\\data\\wind_data\\output\\"
        + typhoon_name.lower()
        + "_windgrid_output.csv"
    )
    path = os.path.join(cdir, file_name)
    typhhon_df.to_csv(path)


# %%
