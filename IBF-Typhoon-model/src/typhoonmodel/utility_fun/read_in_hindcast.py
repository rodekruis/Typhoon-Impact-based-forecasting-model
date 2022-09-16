from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

ONE_MIN_TO_TEN_MIN = 0.84


def read_in_hindcast(typhoon_name: str, remote_dir: str, local_directory: str):
    # Read in the hindcast csv
    filename = Path(local_directory) / f"{typhoon_name.lower()}_all.csv"
    forecast_time = datetime.strptime(remote_dir, "%Y%m%d%H%M%S")

    df = pd.read_csv(filename)
    for cname in ["time", "forecast_time"]:
        df[cname] = pd.to_datetime(df[cname])
    df = df[df["mtype"].isin(["ensembleforecast", "forecast"])]
    df = df[df["forecast_time"] == forecast_time]

    # Format into a list of tracks
    tracks = []
    for ensemble, group in df.groupby("ensemble"):

        is_ensemble = 'False' if ensemble == 'none' else 'True'

        time_step = (group["time"].values[1] - group["time"].values[0]).astype('timedelta64[h]')
        time_step = pd.to_timedelta(time_step).total_seconds() / 3600

        coords = dict(
             time=(["time"], group["time"]),
        )
        data_vars=dict(
            max_sustained_wind=(["time"], group['speed'] / ONE_MIN_TO_TEN_MIN),
            central_pressure=(["time"], group['pressure']),
            lat=(["time"], group["lat"]),
            lon=(["time"], group["lon"]),
            time_step=(["time"], [time_step] * len(group)),
            radius_max_wind = (["time"], [np.nan] * len(group)),
            environmental_pressure = (["time"], [1010.] * len(group)),
        )
        attrs=dict(
            max_sustained_wind_unit="m/s",
            central_pressure_unit="mb",
            name=typhoon_name.upper(),
            data_provider="ECMWF",
            ensemble_number=ensemble,
            is_ensemble=is_ensemble,
            forecast_time=forecast_time,
            basin="W - North West Pacific",
            sid=typhoon_name,
            orig_event_flag=False,
            id_no=None,
            category=None
        )
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs).set_coords(["lat", "lon"])
        tracks.append(ds)
    return tracks
