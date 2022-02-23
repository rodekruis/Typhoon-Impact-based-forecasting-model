"""
Module to process the data for a typhoon that is required for feature building and
running the machine-learning pipeline.
There are a number of things that this module does:
(i) The methods download_gpm and download_srtm download the data for rainfall and elevation, respectively.
(ii) These data are then used, along with the windspeed and track data to obtain average values per municipality.
These data are then outputted into a CSV file for each typhoon that it is run for. It needs to be run
ONLY ONCE for each typhoon.
REQUIRED INPUTS:
(i) Typhoon name.
(ii) Start/Landfall/End date for the typhoon.
(iii) Landfall time
(iiii) IMERG data type: early, late or final: the data respository also needs to be checked, lest files are moved away
                        and the data doesn't exist anymore.
OUTPUTS:
    (i) Downloaded GPM data
    (ii) CSV file with daily rainfall (mm/24h)
"""
#%% Import Libraries
import datetime as dt
import ftplib
import gzip
import os
import zipfile
from ftplib import FTP_TLS
import ssl
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from fiona.crs import from_epsg
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterstats import zonal_stats
import shutil

#%% Functions used
def date_range(start_date, end_date):
    return [
        str(start_date + dt.timedelta(days=x))
        for x in range((end_date - start_date).days + 1)
    ]


def unzip(zip_file, destination):
    os.makedirs(destination, exist_ok=True)

    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(destination)

    return


def reproject_file(gdf, file_name, force_epsg):

    print(
        "Reprojecting %s to EPSG %i...\n" % (file_name, force_epsg), end="", flush=True
    )
    gdf = gdf.to_crs(epsg=force_epsg)

    return gdf


def reproject_raster(
    src_array, src_transform, src_epsg, dst_epsg, src_nodata=-32768, dst_nodata=-32768
):
    """ Method to re-project a data-frame in the digital elevation model (DEM) to EPSG format.
    :param src_array: the data in DEM format
    :type src_array: pandas data-frame
    :param src_transform:
    :type src_transform:
    :param src_epsg:
    :type src_epsg:
    :param dst_epsg:
    :type dst_epsg:
    :param src_nodata:
    :type src_nodata:
    :param dst_nodata:
    :type dst_nodata:
    :raises:
    :returns:
    """
    src_height, src_width = src_array.shape
    dst_affine, dst_width, dst_height = calculate_default_transform(
        from_epsg(src_epsg),
        from_epsg(dst_epsg),
        src_width,
        src_height,
        *array_bounds(src_height, src_width, src_transform)
    )

    dst_array = np.zeros((dst_width, dst_height))
    dst_array.fill(dst_nodata)

    reproject(
        src_array,
        dst_array,
        src_transform=src_transform,
        src_crs=from_epsg(src_epsg),
        dst_transform=dst_affine,
        dst_crs=from_epsg(dst_epsg),
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        resampling=Resampling.nearest,
    )

    return dst_array, dst_affine


def download_gpm(start_date, end_date, download_path, type_imerg):

    """ Method that downloads gpm files.
    This method looks in the data repositories of NASA for rainfall data.
    :param start_date: A date object denoting the START date to search for rainfall data.
    :param end_date: A date object denoting the END date to search for rainfall data.
    :param download_path: A string denoting where the data should be downloaded to.
    :param type_imerg: Hart-coded st
    :returns: file_list: List of the downloaded files
    :raises: ftplib.allerrors
    """
    # Username and password for logging in
    # Can create own account on NASA site
    ppm_username = "mvanbrussel@rodekruis.nl"
    base_url = ""

    # Connection to the site, if pasting in chrome: https://arthurhouhttps.pps.eosdis.nasa.gov/
    # Directory to where the data is saved
    if type_imerg == "final":
        base_url = "arthurhou.pps.eosdis.nasa.gov"

    date_list = date_range(start_date, end_date)
    file_list = []

    # A string denoting where the data should be downloaded to
    os.makedirs(download_path, exist_ok=True)

    print("Connecting to: %s...\n" % base_url, end="", flush=True)

    FTP_TLS.ssl_version = ssl.PROTOCOL_TLSv1_2

    with FTP_TLS(base_url) as ftp:

        try:
            ftp.login(ppm_username, ppm_username)
            print("Login OK...\n", end="", flush=True)
        except ftplib.all_errors as e:
            print("Error logging in:\n", e, end="", flush=True)

        for date in date_list:
            print(date)
            d, m, y = reversed(date.split("-"))
            day_path = os.path.join(download_path, y + m + d)
            # Make a folder for each day, to save GPM data
            os.makedirs(day_path, exist_ok=True)

            if type_imerg == "final":
                data_dir_final = (
                    "/pub/gpmdata/" + str(y) + "/" + str(m) + "/" + str(d) + "/gis"
                )  # data_dir/yyyy/
                ftp.cwd(data_dir_final)

            for entry in ftp.mlsd():

                file_name = entry[0]

                if (
                    type_imerg == "final"
                    and file_name.endswith(("tif", "tfw"))
                    and entry[0][3:6] == "DAY"
                ):

                    file_path = os.path.join(day_path, file_name)

                    if file_name.endswith("tif"):
                        print("Retrieving %s...\n" % file_name, end="", flush=True)
                        file_list.append(file_path)
                        if os.path.isfile(file_path):
                            print("found locally...\n", end="", flush=True)
                    if not os.path.isfile(file_path):
                        with open(file_path, "wb") as write_file:
                            ftp.retrbinary("RETR " + file_name, write_file.write)
                    if file_name.endswith("tif.gz"):
                        new_file_name = os.path.splitext(file_path)[0]
                        with gzip.open(file_path, "rb") as file_in:
                            file_data = file_in.read()
                            with open(new_file_name, "wb") as file_out:
                                file_out.write(file_data)
                        file_list.append(new_file_name)
    return file_list


def cumulative_rainfall(
    admin_geometry, start_date, end_date, download_path, type_imerg
):

    """ Method to calcualte the cumulative amount of rainfall from the typhoon.
    :param admin_geometry:
    :type admin_geometry:
    :param start_date: A date object signifying the start date of the typhoon.
    :param end_date: A date object signifying the end date of the typhoon.
    :param download_path: A string denoting where the data should be downloaded to.
    :param type_imerg: A string denoting the data file type: "early" or "final"
    :returns: df_rainfall = dataframe with the daily rainfall in mm for each municipality and date (mm/24h)
    :raises:
    """
    # File_list is a list of tif files for each date
    file_list = download_gpm(start_date, end_date, download_path, type_imerg)

    if not file_list:
        force_download = True
        file_list = download_gpm(
            start_date, end_date, download_path, type_imerg, force_download
        )

    sum_rainfall = []
    file_list = sorted(file_list)

    if file_list:

        print("Reading GPM data...\n", end="", flush=True)

        transform = ""
        # creating dataframe to save daily rainfall per municipality
        df_rainfall = pd.DataFrame(admin_gdf["ADM3_PCODE"])

        for input_raster in file_list:

            print(input_raster)

            with rasterio.open(input_raster) as src:

                array = src.read(1)
                transform = src.transform
                sum_rainfall = zonal_stats(
                    admin_geometry,
                    array,
                    stats="mean",
                    nodata=29999,
                    all_touched=True,
                    affine=transform,
                )

                sum_rainfall = [i["mean"] for i in sum_rainfall]

                # data is 0.1 mm/h:
                # / 10 give mm/h because the factor in the tif file is 10
                def convert(x):
                    return x / 10 * 24

                sum_rainfall_converted = [convert(item) for item in sum_rainfall]

                # Column name: obtain date from input_raster name
                # String after which date can be found
                identify_str = "3IMERG."
                str_index = input_raster.find(identify_str)
                len_date = 8
                column_name = input_raster[
                    str_index
                    + len(identify_str) : str_index
                    + len(identify_str)
                    + len_date
                ]

                df_rainfall[column_name] = sum_rainfall_converted

    else:
        print(
            "No files were found/downloaded from the appropriate folder. Please investigate further.\n"
        )
        pass

    return df_rainfall


def process_tyhoon_data(typhoon_to_process, typhoon_name):
    """ Method to process the data for typhoons.
    :param typhoon_to_process: A dictionary instance containing all required information about the data for the typhoon.
    :param typhoon_name: The name of the typhoon (can be just passed through as the dictionary key.
    """

    typhoon = typhoon_to_process.get("typhoon")

    # Start/End date for precipitation data, get from the dictionary
    start_date = min(typhoon_to_process.get("dates"))
    end_date = max(typhoon_to_process.get("dates"))
    print("start_date is:", start_date, "end date of typhoon is:", end_date)

    # IMERG data type, either "early" (6hr), "late" (18hr) or "final" (4 months),
    # see https://pps.gsfc.nasa.gov/Documents/README.GIS.pdf
    imerg_type = typhoon_to_process.get("imerg_type")  # "early"
    print("imerg_type:", imerg_type)

    # Specify P Coded column in administrative boundaries file
    p_code = "ADM3_PCODE"

    # Specify output file names
    output_matrix_csv_name = typhoon_name + "_matrix.csv"

    # Output will be in this CRS, datasets are reprojected if necessary
    force_epsg = 32651  # UTM Zone 51N

    t0 = dt.datetime.now()

    # Specify the names to save the GPM data (folder) and the output file
    subfolder = typhoon + "/"
    gpm_path = os.path.join(gpm_folder_path, subfolder, "GPM")
    output_matrix_csv_file = os.path.join(output_path, output_matrix_csv_name)

    # Calculating cumulative rainfall
    if not imerg_type == "trmm":
        df_rainfall = cumulative_rainfall(
            admin_geometry_wgs84, start_date, end_date, gpm_path, imerg_type
        )

    # Move the rainfall data into the other CSV file.
    if output_matrix_csv_name:
        print(
            "Exporting output to %s...\n" % output_matrix_csv_name, end="", flush=True
        )
        df_rainfall.to_csv(output_matrix_csv_file, index=False)

    t_total = dt.datetime.now()
    print("Completed in %fs\n" % (t_total - t0).total_seconds(), end="", flush=True)


###################################################################
### FILL IN INPUT HERE
###################################################################

#%% Setting path and workspace_admin directory
os.chdir("C:\\Users\\Marieke\\GitHub\\Typhoon_IBF_Rice_Damage_Model")
cdir = os.getcwd()

# Typhoons for which to run
typhoons = ["kammuri2019"]
typhoons = [
    "ketsana2009",
    "kompasu2010",
    "nock-ten2011",
    "nanmadol2011",
    "nalgae2011",
    "utor2013",
    "mangkhut2018",
    "yutu2018",
    "nakri2019",
    "danas2019",
    "kammuri2019",
    "vamco2020",
    "haima2016",
    "koppu2015",
    "goni2015",
    "molave2020",
    "usagi2013",
    "nari2013",
    "kai-tak2012",
    "krosa2013",
    "matmo2014",
    "fung-wong2014",
    "melor2015",
    "sarika2016",
    "pakhar2017",
    "khanun2017",
    "rammasun2014",
    "kalmaegi2014",
    "mujigae2015",
    "maliksi2018",
    "vongfong2020",
    "saudel2020",
    "goni2020",
    "nock-ten2016",
    "durian2006",
    "nesat2011",
    "bopha2012",
    "trami2013",
    "haiyan2013",
    "tokage2016",
    "lingling2014",
    "atsani2020",
    "tembin2017",
    "washi2011",
    "linfa2015",
    "hagupit2014",
    "jangmi2014",
    "phanfone2019",
    "kai-tak2017",
    "aere2011",
    "haikui2017",
    "meari2011",
    "mekkhala2015",
    "mirinae2009",
    "parma2009",
    "krovanh2020",
]

# Setting the number of days prior to the landfall data for which to collect data
days_to_landfall = 3

# Setting path to save the GPM data
gpm_file_name = "IBF_typhoon_model/data/rainfall_data/output_daily/gpm"
gpm_folder_path = os.path.join(cdir, gpm_file_name)

# Setting path to save the obtained DataFrames
output_file_name = "IBF_typhoon_model/data/rainfall_data/output_daily"
output_path = os.path.join(cdir, output_file_name)

# Default = FALSE
# IMPORTANT: setting to TRUE means that all downloaded GPM files will be deleted and re-downloaded
delete_folders = False

# Setting directory for administrative boundaries shape file
admin_file_name = (
    "IBF_typhoon_model/data/phl_administrative_boundaries/phl_admbnda_adm3.shp"
)
admin_file = os.path.join(cdir, admin_file_name)

# Setting directory for typhoon metadata csv file
typhoon_metadata_filename = os.path.join(
    cdir, "IBF_typhoon_model/data/rainfall_data/input/metadata_typhoons.csv"
)
typhoon_metadata = pd.read_csv(typhoon_metadata_filename, delimiter=",")

#%%##################################################################
### START OF PROCESSING
###################################################################

# To make sure the dates can be converted to date type
for i in range(len(typhoon_metadata)):
    typhoon_metadata["startdate"][i] = typhoon_metadata["startdate"][i].replace(
        "/", "-"
    )
    typhoon_metadata["enddate"][i] = typhoon_metadata["enddate"][i].replace("/", "-")
    typhoon_metadata["landfalldate"][i] = typhoon_metadata["landfalldate"][i].replace(
        "/", "-"
    )

# Creating a dictionary for the typhoons, with corresponding information
typhoon_metadata = typhoon_metadata.set_index("typhoon").to_dict()
typhoons_dict = dict()
i = 0
for typhoon in typhoons:
    case = typhoon
    typhoons_dict[case] = {
        "typhoon": typhoon,
        "dates": [
            dt.datetime.strptime(
                typhoon_metadata["startdate"][typhoon], "%d-%m-%Y"
            ).date(),
            dt.datetime.strptime(
                typhoon_metadata["enddate"][typhoon], "%d-%m-%Y"
            ).date(),
            dt.datetime.strptime(
                typhoon_metadata["landfalldate"][typhoon], "%d-%m-%Y"
            ).date()
            - dt.timedelta(days=days_to_landfall),
        ],
        "imerg_type": typhoon_metadata["imerg_type"][typhoon],
    }
    i = i + 1

# Loading admin-files only once instead of every time
print("Importing adminfile...\n", end="", flush=True)
admin_gdf = gpd.GeoDataFrame()
try:
    admin_gdf = gpd.GeoDataFrame.from_file(admin_file)
except IOError as ioe:
    print("Could not load file properly", ioe, end="", flush=True)
# Check if CRS is defined and default to WGS 84 if not
if not admin_gdf.crs:
    admin_gdf.crs = from_epsg(4326)
# Keeping an unprojected copy of admin area geometry in WGS84 to speed up raster calculations
# .geometry --> gives the geometry column of the dataframe
if admin_gdf.crs != "EPSG:4326":
    admin_geometry_wgs84 = reproject_file(admin_gdf.geometry, admin_file_name, 4326)
else:
    admin_geometry_wgs84 = admin_gdf.geometry
    # Output will be in this CRS, datasets are reprojected if necessary
    force_epsg = 32651  # UTM Zone 51N
# Check CRS of each layer and reproject if necessary
if admin_gdf.crs != force_epsg:
    admin_gdf = reproject_file(admin_gdf, admin_file_name, force_epsg)

#%%##################################################################
### START OF PROCESSING
###################################################################
for key in typhoons_dict:

    print("Processing typhoon data for:", key)

    # Removing the typhoon folder if it is present, so code can be re-run without errors
    if delete_folders == True:
        dir_temp = os.path.join(
            cdir, "IBF_typhoon_model/data/rainfall_data/output_daily/gpm", key, "GPM"
        )
        try:
            shutil.rmtree(dir_temp)
            print(key, " GPM folder removed")
        except:
            print("No ", key, " GPM folder present")

    process_tyhoon_data(typhoons_dict[key], key)


# %%
