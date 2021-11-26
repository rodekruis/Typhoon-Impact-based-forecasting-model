import os
import glob
import pandas as pd
import numpy as np
import requests
import datetime
from azure.storage.file import FileService, ContentSettings
import logging



def get_api_token():

    # load credentials to IBF API
    IBF_API_URL = os.environ["IBF_API_URL"]
    ADMIN_LOGIN = os.environ["ADMIN_LOGIN"]
    ADMIN_PASSWORD = os.environ["ADMIN_PASSWORD"]

    # log in to IBF API
    login_response = requests.post(f'{IBF_API_URL}/api/user/login',
                                   data=[('email', ADMIN_LOGIN), ('password', ADMIN_PASSWORD)])
    token = login_response.json()['user']['token']

    return IBF_API_URL, token



def to_api(IBF_API_URL, token, admin_level, layer, exposure_place_codes, Activetyphoon, leadtime_str):
    '''
    Function to post layers into IBF System.
    
    '''

    # prepare layer
    exposure_data = {"countryCodeISO3": "PHL"}
    exposure_data["exposurePlaceCodes"] = exposure_place_codes
    exposure_data["adminLevel"] = admin_level
    exposure_data["leadTime"] = leadtime_str
    exposure_data["dynamicIndicator"] = layer
    exposure_data["disasterType"] = "typhoon"
    exposure_data["eventName"] = Activetyphoon
    
    # upload layer
    r = requests.post(f'{IBF_API_URL}/api/admin-area-dynamic-data/exposure',
                        json=exposure_data,
                        headers={'Authorization': 'Bearer ' + token,
                                'Content-Type': 'application/json',
                                'Accept': 'application/json'})
    if r.status_code >= 400:
        # logging.error(f"PIPELINE ERROR AT EMAIL {email_response.status_code}: {email_response.text}")
        # print(r.text)
        raise ValueError()



def post_noevent():
    '''
    Function to use when there is no active typhoon
    '''

    logging.info('post_output: sending output to dashboard')

    # log in to IBF API
    IBF_API_URL, token = get_api_token()

    layer = 'alert_threshold'
    admin_level = 0
    trigger_place_codes = {'placeCode': 'PH', # TODO: alert at national level, not municipalities-> check placeCode
                           'amount': 0}
    to_api(IBF_API_URL, token, admin_level, layer, trigger_place_codes, leadtime_str)



def post_output(Output_folder, Activetyphoon, debug_post=False):
    '''
    Function to post all layers into IBF System.
    For every layer, the function calls IBF API and post the layer in the format of json.
    The layers are alert_threshold (drought or not drought per provinces), population_affected and ruminants_affected.
    
    '''

    logging.info('post_output: sending output to dashboard')
    
    start_time = datetime.datetime.now()

    # log in to IBF API
    IBF_API_URL, token = get_api_token()


    # data reading 
    if debug_post: # get mock data in debug mode
        # access storage blob
        data_folder = './data_dummy/'
        file_service = FileService(account_name=os.environ["AZURE_STORAGE_ACCOUNT"], protocol='https', connection_string=os.environ["AZURE_CONNECTING_STRING"])
        file_service.get_file_to_path('forecast', 'dummy', "rain_data.csv", data_folder + "rain_data.csv")
        file_service.get_file_to_path('forecast', 'dummy', "ecmwf_hrs_track.csv", data_folder + "ecmwf_hrs_track.csv")
        file_service.get_file_to_path('forecast', 'dummy', "landfall.csv", data_folder + "landfall.csv")
        file_service.get_file_to_path('forecast', 'dummy', "Average_Impact_2021090810_CONSON.csv", data_folder + "Average_Impact_2021090810_CONSON.csv")
        file_service.get_file_to_path('forecast', 'dummy', "DREF_TRIGGER_LEVEL_2021090810_CONSON_trigger.csv", data_folder + "DREF_TRIGGER_LEVEL_2021090810_CONSON_trigger.csv")

        # read data
        df_rain = pd.read_csv(os.path.join(data_folder, "rain_data.csv"))
        df_wind = pd.read_csv(os.path.join(data_folder,'ecmwf_hrs_track.csv'))
        df_landfall = pd.read_csv(os.path.join(data_folder,'landfall.csv'))
        df_impact = pd.read_csv(os.path.join(data_folder, "Average_Impact_2021090810_CONSON.csv"))
        df_trigger = pd.read_csv(os.path.join(data_folder, "DREF_TRIGGER_LEVEL_2021090810_CONSON_trigger.csv"))
    
    else: # get real data
        df_rain = pd.read_csv(os.path.join(Output_folder, "rain_data.csv"))
        df_wind = pd.read_csv(os.path.join(Output_folder,'ecmwf_hrs_track.csv'))
        df_landfall = pd.read_csv(glob.glob(os.path.join(Output_folder,'landfall*.csv'))[0])
        df_impact = pd.read_csv(glob.glob(os.path.join(Output_folder, "Average_Impact_*.csv"))[0])
        df_trigger = pd.read_csv(glob.glob(os.path.join(Output_folder, "DREF_TRIGGER_LEVEL_*.csv"))[0])


    Activetyphoon = Activetyphoon[0]

    # leadtime
    leadtime = df_landfall['time_for_landfall'].values[0]
    leadtime_str = str(leadtime) + '-hour'


    # check alert threshold
    trigger = check_trigger(df_trigger, Activetyphoon)
    layer = "alert_threshold"
    if trigger == 1:
        df_impact[layer] = np.where(df_impact['impact']>0, 1, 0) # mark 1 when impact>0
    else:
        df_impact[layer] = 0


    # rainfall
    layer = "rainfall"
    admin_level = 3
    df_rain_exposure_place_codes = []
    for ix, row in df_rain.iterrows():
        exposure_entry = {"placeCode": row['Mun_Code'],
                          "amount": row['max_24h_rain']}
        df_rain_exposure_place_codes.append(exposure_entry)
    to_api(IBF_API_URL, token, admin_level, layer, df_rain_exposure_place_codes, Activetyphoon, leadtime_str)


    # # windspeed
    # layer = "windspeed"
    # admin_level = 3
    # df_wind_exposure_place_codes = []
    # for ix, row in df_wind.iterrows():
    #     exposure_entry = {"placeCode": row['adm3_pcode'],
    #                       "amount": row['VMAX']}
    #     df_wind_exposure_place_codes.append(exposure_entry)
    # to_api(IBF_API_URL, token, admin_level, layer, df_wind_exposure_place_codes, Activetyphoon, leadtime_str)


    # # landfall location
    # layer = "landfall"
    # admin_level = 0
    # df_wind_exposure_place_codes = []
    # for ix, row in df_landfall.iterrows():
    #     exposure_entry = {"lat": row['landfall_point_lat'],
    #                       "lon": row['landfall_point_lon']}
    #     df_wind_exposure_place_codes.append(exposure_entry)
    # to_api(IBF_API_URL, token, admin_level, layer, df_wind_exposure_place_codes, Activetyphoon, leadtime_str)


    # houses_affected
    layer = "houses_affected"
    admin_level = 3
    df_impact['impact'] = df_impact['impact'].fillna(0)
    df_impact_exposure_place_codes = []
    for ix, row in df_impact.iterrows():
        exposure_entry = {"placeCode": row['adm3_pcode'],
                          "amount": row['impact']}
        df_impact_exposure_place_codes.append(exposure_entry)
    to_api(IBF_API_URL, token, admin_level, layer, df_impact_exposure_place_codes, Activetyphoon, leadtime_str)
    

    # probability within 50 km
    layer = "prob_within_50km"
    admin_level = 3
    df_impact['probability_dist50'] = df_impact['probability_dist50'].fillna(0)
    df_impact['probability_dist50'] = df_impact['probability_dist50']/100
    df_impact_exposure_place_codes = []
    for ix, row in df_impact.iterrows():
        exposure_entry = {"placeCode": row['adm3_pcode'],
                          "amount": row['probability_dist50']}
        df_impact_exposure_place_codes.append(exposure_entry)
    to_api(IBF_API_URL, token, admin_level, layer, df_impact_exposure_place_codes, Activetyphoon, leadtime_str)


    # alert layer
    layer = "alert_threshold"
    admin_level = 3
    df_impact_exposure_place_codes = []
    for ix, row in df_impact.iterrows():
        exposure_entry = {"placeCode": row['adm3_pcode'],
                          "amount": row[layer]}
        df_impact_exposure_place_codes.append(exposure_entry)
    to_api(IBF_API_URL, token, admin_level, layer, df_impact_exposure_place_codes, Activetyphoon, leadtime_str)



    # typhoon track
    # layer = 'track'
    admin_level = 3
    df_wind['YYYYMMDDHH'] = pd.to_datetime(df_wind['YYYYMMDDHH'], format='%Y%m%d%H%M').dt.strftime('%m-%d-%Y %H:%M:%S')
    wind_track = []
    for ix, row in df_wind.iterrows():
        exposure_entry = {"lon": row['LON'],
                          "lat": row['LAT'],
                          "timestampOfTrackpoint": row['YYYYMMDDHH']}
        wind_track.append(exposure_entry)
        # prepare layer
    exposure_data = {"countryCodeISO3": "PHL"}
    exposure_data["leadTime"] = leadtime_str
    exposure_data["trackpointDetails"] = wind_track
    
    # upload layer
    r = requests.post(f'{IBF_API_URL}/api/typhoon-track',
                        json=exposure_data,
                        headers={'Authorization': 'Bearer ' + token,
                                'Content-Type': 'application/json',
                                'Accept': 'application/json'})
    if r.status_code >= 400:
        # logging.error(f"PIPELINE ERROR AT EMAIL {email_response.status_code}: {email_response.text}")
        # print(r.text)
        raise ValueError()


    logging.info('post_output: done')



def check_trigger(df_trigger, Activetyphoon):
    '''
    Function to check if the event should be triggered based on the impact

    '''
    
    trigger = []
    if df_trigger[df_trigger['Typhoon_name']==Activetyphoon]['>=100k'].values >= 50:
        trigger.append(1)
    else:
        trigger.append(0)
    if df_trigger[df_trigger['Typhoon_name']==Activetyphoon]['>=80k'].values >= 60:
        trigger.append(1)
    else:
        trigger.append(0)
    if df_trigger[df_trigger['Typhoon_name']==Activetyphoon]['>=70k'].values >= 70:
        trigger.append(1)
    else:
        trigger.append(0)
    if df_trigger[df_trigger['Typhoon_name']==Activetyphoon]['>=50k'].values >= 80:
        trigger.append(1)
    else:
        trigger.append(0)
    if df_trigger[df_trigger['Typhoon_name']==Activetyphoon]['>=30k'].values >= 95:
        trigger.append(1)
    else:
        trigger.append(0)
    
    if 1 in trigger:
        trigger_alert = 1
    else:
        trigger_alert = 0

    return trigger_alert

