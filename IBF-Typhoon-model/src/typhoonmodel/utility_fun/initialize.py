import logging

import cartopy
from cartopy.io.shapereader import NEShpDownloader


CARTOPY_SOURCE_TEMPLATE = 'https://naturalearth.s3.amazonaws.com/{resolution}_{category}/ne_{resolution}_{name}.zip'


def setup_logger():
    # Set up logger
    logging.root.handlers = []
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename='ex.log')
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def setup_cartopy():
    # Configures cartopy to download NaturalEarth shapefiles from S3 instead of naciscdn
    target_path_template = NEShpDownloader.default_downloader().target_path_template
    downloader = NEShpDownloader(url_template=CARTOPY_SOURCE_TEMPLATE,
                                 target_path_template=target_path_template)
    cartopy.config['downloaders'][('shapefiles', 'natural_earth')] = downloader
