# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:01:00 2020

@author: ATeklesadik
"""
from datetime import datetime
import logging

from pybufrkit.decoder import Decoder
import click

from typhoonmodel.utility_fun import initialize
from typhoonmodel.utility_fun.forecast_process import Forecast

decoder = Decoder()
initialize.setup_logger()
logger = logging.getLogger(__name__)


@click.command()
@click.option('--path', default='./', help='main directory')
@click.option('--remote_directory', default=None,
              help='remote directory for ECMWF forecast data, format: YYYYMMDDhhmmss. In the case of hindcasts,'
                   'the target timestamp of the forecast run.')
@click.option('--use-hindcast', is_flag=True, help='Use hindcast instead of latest ECMWF. Need to specify'
                                                   'local directory and use remote directory for timestamp')
@click.option('--local-directory', default=None,
              help='local directory containing with ECMWF hindcasts')
@click.option('--typhoonname', default=None, help='name for active typhoon')
@click.option('--no-azure', is_flag=True, help="Don't push to Azure lake")
@click.option('--debug', is_flag=True, help='setting for DEBUG option')
def main(path, remote_directory, use_hindcast, local_directory, typhoonname, no_azure, debug):
    initialize.setup_cartopy()
    start_time = datetime.now()
    print('---------------------AUTOMATION SCRIPT STARTED---------------------------------')
    print(str(start_time))
    print('---------------------check for active typhoons---------------------------------')
    print(str(start_time))
    remote_dir = remote_directory
    if debug:
        typhoonname = 'MEGI'
        remote_dir = '20220412000000'
        logger.info(f"DEBUGGING piepline for typhoon {typhoonname}")
    if use_hindcast:
        if not remote_directory or not local_directory or not typhoonname:
            logger.error("If you want to use a hindcast, you need to specify remote directory"
                         "(for the forecast timestamp), a local directory, and the typhoon name")
    Forecast(path, remote_dir, typhoonname, countryCodeISO3='PHP', admin_level=3, no_azure=no_azure,
             use_hindcast=use_hindcast, local_directory=local_directory)
    print('---------------------AUTOMATION SCRIPT FINISHED---------------------------------')
    print(str(datetime.now()))


if __name__ == "__main__":
    main()
