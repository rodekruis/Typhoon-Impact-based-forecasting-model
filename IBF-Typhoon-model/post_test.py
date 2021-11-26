import os
import glob
import pandas as pd
import json
import numpy as np
import requests
import datetime
# from azure.storage.blob import FileService, ContentSettings
import logging



def to_api(admin_level, layer, exposure_place_codes, leadtime_str):
    '''
    Function to post layers into IBF System.
    
    '''

    # prepare layer
    exposure_data = {'countryCodeISO3': 'PHL'}
    exposure_data['exposurePlaceCodes'] = exposure_place_codes
    exposure_data["adminLevel"] = admin_level
    exposure_data["leadTime"] = leadtime_str
    exposure_data["dynamicIndicator"] = layer
    exposure_data["disasterType"] = 'typhoon'
    
    print(exposure_data)
    # with open(layer + '.json', 'w') as f:
    #     json.dump(exposure_data, f)
    
    # # upload layer
    # r = requests.post(f'{IBF_API_URL}/api/admin-area-dynamic-data/exposure',
    #                     json=exposure_data,
    #                     headers={'Authorization': 'Bearer '+ token,
    #                             'Content-Type': 'application/json',
    #                             'Accept': 'application/json'})
    # if r.status_code >= 400:
    #     # logging.error(f"PIPELINE ERROR AT EMAIL {email_response.status_code}: {email_response.text}")
    #     # print(r.text)
    #     raise ValueError()



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



def post_output(Output_folder, Activetyphoon):
    '''
    Function to post all layers into IBF System.
    For every layer, the function calls IBF API and post the layer in the format of json.
    The layers are alert_threshold (drought or not drought per provinces), population_affected and ruminants_affected.
    
    '''

    logging.info('post_output: sending output to dashboard')
    
    start_time = datetime.datetime.now()

    # log in to IBF API
    # IBF_API_URL, token = get_api_token()


    # data reading 
    if debug: # get mock data in debug mode
        # access storage blob
        data_folder = './data_dummy'
        # file_service = FileService(account_name=os.environ["AZURE_STORAGE_ACCOUNT"], protocol='https', connection_string=os.environ["AZURE_CONNECTING_STRING"])
        # file_service.get_file_to_path('forecast', 'dummy', "rain_data.csv", data_folder + "rain_data.csv", \
        #     content_settings=ContentSettings(content_type='text/csv'))
        # file_service.get_file_to_path('forecast', 'dummy', "windfield.csv", data_folder + "windfield.csv", \
        #     content_settings=ContentSettings(content_type='text/csv'))
        # file_service.get_file_to_path('forecast', 'dummy', "rain_data.csv", data_folder + "rain_data.csv", \
        #     content_settings=ContentSettings(content_type='text/csv'))
        # file_service.get_file_to_path('forecast', 'dummy', "rain_data.csv", data_folder + "rain_data.csv", \
        #     content_settings=ContentSettings(content_type='text/csv'))

        # read data
        df_rain = pd.read_csv(os.path.join(data_folder, "rain_data.csv"))
        df_wind = pd.read_csv(os.path.join(data_folder,'windfield.csv'))
        df_landfall = pd.read_csv(os.path.join(data_folder,'landfall.csv'))
        df_impact = pd.read_csv(os.path.join(data_folder, "Average_Impact_2021091014_CHANTHU.csv"))
        # df_impact = df_impact.replace('NA', np.nan)
        df_trigger = pd.read_csv(os.path.join(data_folder, "DREF_TRIGGER_LEVEL_2021091014_CHANTHU.csv"))
    else: # get real data
        date_dir = start_time.strftime("%Y%m%d%H")
        # Output_folder = os.path.join(path, f'forecast/Output/{date_dir}/Output/')
        df_rain = pd.read_csv(os.path.join(Output_folder, "rainfall/rain_data.csv"), index=False)
        df_wind = pd.read_csv(os.path.join(Output_folder,'windfield.csv'), index=False)
        df_impact = pd.read_csv(glob.glob(os.path.join(Output_folder, "Average_Impact_*.csv")), index=False)
        df_trigger = pd.read_csv(glob.glob(os.path.join(Output_folder, "DREF_TRIGGER_LEVEL_*.csv")), index=False)


    # leadtime
    leadtime = df_landfall['time_for_landfall'].item()
    print(leadtime)


    # check alert threshold
    trigger = check_trigger(df_trigger, Activetyphoon)
    layer = 'alert_threshold'
    df_impact[layer] = np.where(df_impact['impact']>0, 1, 0) # mark 1 when impact>0
    admin_level = 0


    # rainfall
    layer = 'rainfall'
    admin_level = 3
    df_rain_exposure_place_codes = []
    for ix, row in df_rain.iterrows():
        exposure_entry = {'placeCode': row['Mun_Code'],
                          'amount': round(row['max_24h_rain'], 2)}
        df_rain_exposure_place_codes.append(exposure_entry)
    to_api(admin_level, layer, df_rain_exposure_place_codes, leadtime)


    # windspeed
    layer = 'windspeed'
    admin_level = 3
    df_wind_exposure_place_codes = []
    for ix, row in df_wind.iterrows():
        exposure_entry = {'placeCode': row['adm3_pcode'],
                          'amount': round(row['v_max'], 2)}
        df_wind_exposure_place_codes.append(exposure_entry)
    to_api(admin_level, layer, df_wind_exposure_place_codes, leadtime)

    # landfall location
    layer = 'landfall'
    admin_level = 0
    df_wind_exposure_place_codes = []
    for ix, row in df_landfall.iterrows():
        exposure_entry = {'lat': row['landfall_point_lat'],
                          'lon': row['landfall_point_lon']}
        df_wind_exposure_place_codes.append(exposure_entry)
    to_api(admin_level, layer, df_wind_exposure_place_codes, leadtime)
    
    
    # impact
    layer = 'house_affected'
    admin_level = 3
    df_impact_exposure_place_codes = []
    for ix, row in df_impact.iterrows():
        exposure_entry = {'placeCode': row['adm3_pcode'],
                          'amount': round(row['impact'], 2)}
        df_impact_exposure_place_codes.append(exposure_entry)
    to_api(admin_level, layer, df_impact_exposure_place_codes, leadtime)
    

    # probability within 50 km
    layer = 'prob_within_50km'
    admin_level = 3
    df_impact_exposure_place_codes = []
    for ix, row in df_impact.iterrows():
        exposure_entry = {'placeCode': row['adm3_pcode'],
                          'amount': round(row['probability_dist50'], 2)}
        df_impact_exposure_place_codes.append(exposure_entry)
    to_api(admin_level, layer, df_impact_exposure_place_codes, leadtime)


    # alert layer
    layer = 'alert_threshold'
    admin_level = 3
    df_impact_exposure_place_codes = []
    for ix, row in df_impact.iterrows():
        exposure_entry = {'placeCode': row['adm3_pcode'],
                          'amount': row[layer]}
        df_impact_exposure_place_codes.append(exposure_entry)
    to_api(admin_level, layer, df_impact_exposure_place_codes, leadtime)

    logging.info('post_output: done')


    # landfall time


    # landfall location




def check_trigger(df_trigger, Activetyphoon):
    '''
    Function to check if the event should be triggered based on the impact

    '''
    
    trigger = []
    if df_trigger[df_trigger['Typhoon_name']==Activetyphoon]['>=100k'].values >= 50:
        trigger.append(1)
    else:
        trigger.append(0)
    if df_trigger[df_trigger['Typhoon_name']==Activetyphoon]['>=80k'].item() >= 60:
        trigger.append(1)
    else:
        trigger.append(0)
    if df_trigger[df_trigger['Typhoon_name']==Activetyphoon]['>=70k'].item() >= 70:
        trigger.append(1)
    else:
        trigger.append(0)
    if df_trigger[df_trigger['Typhoon_name']==Activetyphoon]['>=50k'].item() >= 80:
        trigger.append(1)
    else:
        trigger.append(0)
    if df_trigger[df_trigger['Typhoon_name']==Activetyphoon]['>=30k'].item() >= 95:
        trigger.append(1)
    else:
        trigger.append(0)
    
    if 1 in trigger:
        trigger_alert = 1
    else:
        trigger_alert = 0

    return trigger_alert




debug = True
