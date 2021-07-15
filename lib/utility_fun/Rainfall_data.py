import os
import urllib.request
import urllib.error
import requests
import subprocess
import logging

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def url_is_alive(url):
    """
    Checks that a given URL is reachable.
    :param url: A URL
    :rtype: bool
    """
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.error.HTTPError:
        return False


def download_rainfall_nomads(Input_folder, path, Alternative_data_point, use_wgrib2 = False):
    """

    download rainfall 
    """
    rainfall_path = os.path.join(Input_folder, 'rainfall/')
    if not os.path.exists(rainfall_path):
        os.makedirs(rainfall_path)

    url_base = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.'
    url1 = f"{url_base}{Input_folder.split('/')[-3][:-2]}/"  # Use the timestamp of the input folder for the query
    url2 = f"{url_base}{Alternative_data_point}/"  # Yesterday's date


    try:
        logger.info("Trying to get rainfall from today's date")
        get_grib_files(url1, path, rainfall_path)
    except Exception as e:
        logger.warning(f"Encountered error: {e}, using yesterday's date instead")
        get_grib_files(url2, path, rainfall_path)

    # TODO: turning this off until we figure out if it's really needed and then install wgrib2
    if use_wgrib2:
        rain_files = [f for f in os.listdir(rainfall_path) if os.path.isfile(os.path.join(rainfall_path, f))]
        os.chdir(rainfall_path)
        for files in rain_files:
            for hour in ['06', '24']:
                pattern = f'.pgrb2a.0p50.bc_{hour}h'
                if pattern in files:
                    subprocess.call(['wgrib2', files, '-append', '-netcdf', 'rainfall_24.nc'], cwd=rainfall_path)
                    os.remove(files)


def get_grib_files(url, path, rainfall_path):
    base_urls = []
    logger.info(f'base_urls')
    for items in listFD(url):
        if url_is_alive(items+'prcp_bc_gb2/'):
            base_urls.append(items)
    base_url = base_urls[-1]
    base_url_hour = base_url+'prcp_bc_gb2/geprcp.t%sz.pgrb2a.0p50.bc_' % base_url.split('/')[-2]
    time_step_list = ['06', '12', '18', '24', '30', '36', '42', '48', '54', '60', '66', '72']
    rainfall_24 = [base_url_hour+'24hf0%s' % t for t in time_step_list]
    rainfall_06 = [base_url_hour+'06hf0%s' % t for t in time_step_list]
    rainfall_24.extend(rainfall_06)
    for rain_file in rainfall_24:
        logger.info(f'Rain file: {rain_file}')
        output_file = os.path.join(os.path.relpath(rainfall_path, path), rain_file.split('/')[-1]+'.grib2')
        batch_ex = ["wget", "-O", output_file, rain_file]
        logger.info(f'Running command {batch_ex}')
        os.chdir(path)
        subprocess.call(batch_ex, cwd=path)


def listFD(url):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + node.get('href')
            for node in soup.find_all('a')
            if node.get('href').split('/')[-2]
            in ['00', '06', '12', '18']]
