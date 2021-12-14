import os

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from climada.hazard import TropCyclone


def plot_inensity(typhoon: TropCyclone, event: str, output_dir: str, date_dir: str, typhoon_name: str):
    # figure needs to be instantiated with projection so that axis has "set_extent" method
    ax = typhoon.plot_intensity(event=event)
    output_filename = os.path.join(output_dir, f"intensity_{date_dir}_{typhoon_name}")
    ax.figure.savefig(output_filename)
