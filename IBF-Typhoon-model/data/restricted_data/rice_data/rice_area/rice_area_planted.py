"""
This code can be used to process several raster (.tif) files from PRiSM, obtaining information on the rice area planted over time. For each PRiSM file loaded, it performs zonal statistics to obtain the area planted in each municipality.
Input:
(1) Folder with PRiSM files to process: 
    Currently, all files without the .sml and .hdr  extension are read and processed. 
    Make sure to check that all files in the folder are in the correct extension
(2) Vector layer of Philippines:
    Shapefile of the philippines at administrative level 3 (municipality level)
(3) Pixel conversion
    To convert from output obtained from zonal histogram to area in hectare. Check the PRiSM .tif file to determine the size of one pixel. Find conversion factor to convert to hectares. Note, make sure to check that the pixel size for each input file is the same
Output:
(1) Area planted excel sheet
    Excel sheet with for each municipality and available date, the planted area in HA
"""

#%% Libraries
from qgis.core import QgsProject
from qgis.core import QgsProcessing
import processing
from qgis.core import *
from qgis.analysis import QgsNativeAlgorithms
import processing
from processing.core.Processing import Processing
import os
from os import listdir
from os.path import isfile, join
import pandas as pd


#%% Loading data
os.chdir("C:\\Users\\Marieke\\GitHub\\Typhoon_IBF_Rice_Damage_Model")
cdir = os.getcwd()

# Setting factor to convert to area in Hectares
# Convert to HA (pixel size 20 x 20 in meters)
conversion_factor = 0.04

# Directory of folder with all PRISM files to process
folder_name = (
    "IBF_typhoon_model\\data\\restricted_data\\rice_data\\rice_area\\PRISM_source_data"
)
folder_path = os.path.join(cdir, folder_name)
files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

# Keeping only the Tif files (diregarding sml and hdr) --> make sure only .sml .hdr and tif
file_list = [file for file in files if file.endswith((".sml", ".hdr")) == False]

# Vector layer
file_name = (
    "IBF_typhoon_model\\data\\phl_administrative_boundaries\\phl_admbnda_adm3.shp"
)
shp_path = os.path.join(cdir, file_name)
shp = QgsVectorLayer(shp_path, "zonepolygons", "ogr")


#%% Create df to save info
columns = [f.name() for f in shp.fields()]
columns_types = [f.typeName() for f in shp.fields()]

row_list = []
for f in shp.getFeatures():
    row_list.append(dict(zip(columns, f.attributes())))

df_total = pd.DataFrame(row_list, columns=columns)
df_total = df_total[["ADM3_PCODE"]]

#%% Obtaining zonal histogram for each file
for file in file_list:

    print(file)

    # Loading raster layer
    raster_path = os.path.join(
        cdir,
        "IBF_typhoon_model\\data\\restricted_data\\rice_data\\rice_area\\PRISM_source_data",
        file,
    )
    raster = QgsRasterLayer(raster_path)

    # Obtaining zonal histogram
    Processing.initialize()
    QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

    input_raster = raster
    input_vector = shp

    params = {
        "COLUMN_PREFIX": "HISTO_",
        "INPUT_RASTER": input_raster,
        "INPUT_VECTOR": input_vector,
        "OUTPUT": "TEMPORARY_OUTPUT",
        "RASTER_BAND": 1,
    }

    result = processing.run("native:zonalhistogram", params)
    layer = result["OUTPUT"]

    # convert legend to dictionary
    legend = raster.legendSymbologyItems()
    key = []
    value = []

    for i in range(2, len(legend)):

        list_str = legend[i][0].split()
        key.append("HISTO_" + list_str[0])
        value.append(list_str[2])

    value_date = pd.to_datetime(pd.Series(value), format="%d-%b-%Y")
    legend_dict = dict(zip(key, value_date))

    # Turning attribute table into dataframe
    columns = [f.name() for f in layer.fields()]
    columns_types = [f.typeName() for f in layer.fields()]

    row_list = []
    for f in layer.getFeatures():
        row_list.append(dict(zip(columns, f.attributes())))

    df_histo = pd.DataFrame(row_list, columns=columns)

    # Only keeping histogram columns and municipality code
    columns = [col for col in df_histo.columns if col in key]
    columns.append("ADM3_PCODE")
    df_histo_filtered = df_histo[columns]

    # Renaming columns to corresponding plant date
    df_histo_filtered = df_histo_filtered.rename(columns=legend_dict)

    # Merging with the total dataframe
    df_total = pd.merge(df_total, df_histo_filtered, on="ADM3_PCODE")

# Convert to area in hectare
df_total.loc[:, df_total.columns != "ADM3_PCODE"] = (
    df_total.loc[:, df_total.columns != "ADM3_PCODE"] * conversion_factor
)


#%% Save output into excel file
file_name = "IBF_typhoon_model\\data\\restricted_data\\rice_data\\rice_area\\rice_area_planted.xlsx"
path = os.path.join(cdir, file_name)
df_total.to_excel(path, index=False)

