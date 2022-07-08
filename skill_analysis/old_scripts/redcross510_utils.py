#############################
# collected some functions I frequently use for 510 related stuff 
#
#############################
import os 
import sys 
import pandas as pd 
import numpy as np 
from datetime import datetime
import zipfile 
import geopandas as gpd  # dealing with GIS data:
import rasterio  # dealing with GIS data:
import rasterio.mask    # dealing with GIS data:
import requests  # accessing URLs and handling 
from requests.auth import HTTPBasicAuth   # accessing URLs and handling 
from bs4 import BeautifulSoup # accessing URLs and handling 

###################################
# Downloading files from the web  #
###################################
def download_files_url(url, username, password, path_download_file):
    # --- use responce package to access the URL. Responce package has simple 'login' method --- 
    response = requests.get(url ,auth=HTTPBasicAuth(username, password))
    
    # --- write contents (the file) to disc ---
    with open(path_download_file, "wb") as downloadFile:
        downloadFile.write(response.content)
    return  

def extract_zip_archive(zip_archive, destination_dir):
    with zipfile.ZipFile(zip_archive, 'r') as ZipArchive:
        ZipArchive.extractall(path = destination_dir)
    return 

###############################################################
# Zonal statistics: Aggregate GIS data by polygon boundaries  #
###############################################################
def zonal_statistics(rasterfile, shapefile, 
                    minval=-np.inf,
                    maxval=+np.inf,
                    aggregate=np.mean, 
                    nameKey = None,
                    pcodeKey = None,
                    polygonKey = 'geometry'): 
    
    '''
    Perform zonal statistics on raster data ('.tif') , based on polygons defined in shape file ('.shp')
    
    INPUT:
    - rasterfile: path to TIFF file 
    - shapefile : path to .shp file 
    - aggregate: A Python function that returns a single numnber. Will use this to aggregate values per polygon. 
    - minval / maxval : Physical boundaries of quantity encoded in TIFF file. Values outside this range are usually reserved to denote special terrain/areas in image
    - nameKey / pcodeKey : column names in shape file that contain unique identifiers for every polygon 
    - polygonKey : by default geopandas uses the 'geometry' column to store the polygons 
    
    
    OUTPUT:
    table (DataFrame) with the one-number metric (aggregate) for every zone defined in the provided shape file
    '''
    
    aggregates_of_zones = []
    
    # ---- open the shape file and access info needed --- 
    shapeData = gpd.read_file(shapefile)
    shapes = list(shapeData[polygonKey])
    if nameKey:
        names = list(shapeData[nameKey])
    if pcodeKey:
        pcodes = list(shapeData[pcodeKey])
        
    # --- open the raster image data --- 
    with rasterio.open(rasterfile, 'r') as src:
        img = src.read(1)
        
        # --- for every polygon: mask raster image and calculate value --- 
        for shape in shapes: 
            out_image, out_transform = rasterio.mask.mask(src, [shape], crop=True)
            
            # --- show the masked image (shows non-zero value within boundaries, zero outside of boundaries shape) ----
            img = out_image[0, :, :]

            # --- Only use physical values  ----
            data = img[(img >= minval) & (img <= maxval)]
            
            #--- determine metric: Must be a one-number metric for every polygon ---
            aggregates_of_zones.append( aggregate(data) )
    
    # --- store output --- 
    zonalStats = pd.DataFrame()
    if nameKey:
        zonalStats['name'] = names
    if pcodeKey:
        zonalStats['pcode'] = pcodes
    zonalStats['value'] = aggregates_of_zones
    
    return zonalStats


