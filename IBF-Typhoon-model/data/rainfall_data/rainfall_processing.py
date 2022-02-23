#%% Loading libraries
from numpy.lib.shape_base import _expand_dims_dispatcher
import os
import pandas as pd
import datetime
import math

os.chdir("C:\\Users\\Marieke\\GitHub\\Typhoon_IBF_Rice_Damage_Model")
cdir = os.getcwd()

#%% Loading information sheet
typhoon_metadata_filename = os.path.join(
    cdir, "IBF_typhoon_model/data/rainfall_data/input/metadata_typhoons.csv"
)
typhoon_metadata = pd.read_csv(typhoon_metadata_filename, delimiter=",")

# To make sure the dates can be converted to date type
for i in range(len(typhoon_metadata)):
    typhoon_metadata["startdate"][i] = typhoon_metadata["startdate"][i].replace(
        "/", "-"
    )
    typhoon_metadata["enddate"][i] = typhoon_metadata["enddate"][i].replace("/", "-")
    typhoon_metadata["landfalldate"][i] = typhoon_metadata["landfalldate"][i].replace(
        "/", "-"
    )

typhoon_metadata["landfall_date_time"] = (
    typhoon_metadata["landfalldate"] + "-" + typhoon_metadata["landfall_time"]
)

typhoons = typhoon_metadata["typhoon"]

"""
Obtain 6h maximum rainfall in mm/h
"""
#%% Processing the data into an excel sheet
time_frame = 12  # in half hours
mov_window = 12  # in half hours
before_landfall_h = 72  # how many hours before landfall to include
num_intervals = math.floor((2 * 72 - time_frame) / mov_window) + 1

df_rainfall_final = pd.DataFrame(columns=["typhoon", "mun_code", "rainfall_max_6h"])

for typhoon in typhoons:

    # Getting typhoon info
    df_info = typhoon_metadata[typhoon_metadata["typhoon"] == typhoon]
    landfall = df_info["landfall_date_time"].values[0]
    landfall = datetime.datetime.strptime(landfall, "%d-%m-%Y-%H:%M:%S")

    # End date is landfall date
    # Start date is 72 hours before landfall date
    end_date = landfall
    start_date = end_date - datetime.timedelta(hours=before_landfall_h)

    # Loading the data
    file_name = (
        "IBF_typhoon_model\\data\\rainfall_data\\output_hhr\\" + typhoon + "_matrix.csv"
    )
    path = os.path.join(cdir, file_name)
    df_rainfall = pd.read_csv(path)

    # Convert column names to date format
    for col in df_rainfall.columns[1:]:
        date_format = datetime.datetime.strptime(col, "%Y%m%d-S%H%M%S")
        df_rainfall = df_rainfall.rename(columns={col: date_format})

    df_mean_rainfall = pd.DataFrame({"mun_code": df_rainfall["ADM3_PCODE"]})

    for i in range(num_intervals):

        start = start_date + datetime.timedelta(minutes=i * mov_window * 30)
        end = (
            start_date
            + datetime.timedelta(minutes=i * mov_window * 30)
            + datetime.timedelta(minutes=time_frame * 30)
        )

        available_dates = [
            date for date in df_rainfall.columns[1:] if (date >= start) & (date < end)
        ]

        # To check if there is data available for all needed dates
        if len(available_dates) != time_frame:
            print("There are less files available than would be needed: please inspect")
            print(typhoon, start, end, len(available_dates))

        df_mean_rainfall[i] = df_rainfall[available_dates].mean(axis="columns")

    df_mean_rainfall["rainfall_max_6h"] = df_mean_rainfall.max(axis="columns")

    df_rainfall_single = df_mean_rainfall[["mun_code", "rainfall_max_6h"]]
    df_rainfall_single["typhoon"] = typhoon
    df_rainfall_final = df_rainfall_final.append(df_rainfall_single)


file_path = "IBF_typhoon_model\\data\\rainfall_data\\rainfall_max_6h.csv"  # in mm/h
df_rainfall_final.to_csv(file_path, index=False)


"""
Obtain 24h maximum rainfall in mm/h
"""
#%% Processing the data into an excel sheet
time_frame = 48  # in half hours
mov_window = 12  # in half hours
before_landfall_h = 72  # how many hours before landfall to include
num_intervals = math.floor((2 * 72 - time_frame) / mov_window) + 1

df_rainfall_final = pd.DataFrame(columns=["typhoon", "mun_code", "rainfall_max_24h"])

for typhoon in typhoons:

    # Getting typhoon info
    df_info = typhoon_metadata[typhoon_metadata["typhoon"] == typhoon]
    landfall = df_info["landfall_date_time"].values[0]
    landfall = datetime.datetime.strptime(landfall, "%d-%m-%Y-%H:%M:%S")

    # End date is landfall date
    # Start date is 72 hours before landfall date
    end_date = landfall
    start_date = end_date - datetime.timedelta(hours=before_landfall_h)

    # Loading the data
    file_name = (
        "IBF_typhoon_model\\data\\rainfall_data\\output_hhr\\" + typhoon + "_matrix.csv"
    )
    path = os.path.join(cdir, file_name)
    df_rainfall = pd.read_csv(path)

    # Convert column names to date format
    for col in df_rainfall.columns[1:]:
        date_format = datetime.datetime.strptime(col, "%Y%m%d-S%H%M%S")
        df_rainfall = df_rainfall.rename(columns={col: date_format})

    df_mean_rainfall = pd.DataFrame({"mun_code": df_rainfall["ADM3_PCODE"]})

    for i in range(num_intervals):

        start = start_date + datetime.timedelta(minutes=i * mov_window * 30)
        end = (
            start_date
            + datetime.timedelta(minutes=i * mov_window * 30)
            + datetime.timedelta(minutes=time_frame * 30)
        )

        available_dates = [
            date for date in df_rainfall.columns[1:] if (date >= start) & (date < end)
        ]

        # To check if there is data available for all needed dates
        if len(available_dates) != time_frame:
            print("There are less files available than would be needed: please inspect")
            print(typhoon, start, end, len(available_dates))

        df_mean_rainfall[i] = df_rainfall[available_dates].mean(axis="columns")

    df_mean_rainfall["rainfall_max_24h"] = df_mean_rainfall.max(axis="columns")

    df_rainfall_single = df_mean_rainfall[["mun_code", "rainfall_max_24h"]]
    df_rainfall_single["typhoon"] = typhoon
    df_rainfall_final = df_rainfall_final.append(df_rainfall_single)


file_path = "IBF_typhoon_model\\data\\rainfall_data\\rainfall_max_24h.csv"  # in mm/h

df_rainfall_final.to_csv(file_path, index=False)

# %%
