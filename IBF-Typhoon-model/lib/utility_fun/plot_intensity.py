import os

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from climada.hazard import TropCyclone


def plot_inensity(typhoon: TropCyclone, event: str, output_dir: str, date_dir: str, typhoon_name: str):
    # figure needs to be instantiated with projection so that axis has "set_extent" method
    fig, ax = plt.subplots(figsize=(9, 13), subplot_kw=dict(projection=ccrs.PlateCarree()))
    typhoon.plot_intensity(event=event, axis=ax)
    output_filename = os.path.join(output_dir, f"intensity_{date_dir}_{typhoon_name}")
    fig.savefig(output_filename)
