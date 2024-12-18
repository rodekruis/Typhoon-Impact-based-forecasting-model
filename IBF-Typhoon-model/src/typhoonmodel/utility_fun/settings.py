import os
from datetime import datetime, timedelta
import shutil
from pathlib import Path
##################
## LOAD SECRETS ##
##################
'''
# 1. Try to load secrets from Azure key vault (i.e. when running through Logic App) if user has access
try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient   
    az_credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
    secret_client = SecretClient(vault_url='https://ibf-flood-keys.vault.azure.net', credential=az_credential)
    
    AZURE_STORAGE_ACCOUNT=secret_client.get_secret("AZURE-STORAGE-ACCOUNT").value
    AZURE_CONNECTING_STRING=secret_client.get_secret("AZURE-CONNECTING-STRING").value   
     
    ADMIN_LOGIN = secret_client.get_secret("ADMIN-LOGIN").value
    ADMIN_PASSWORD=secret_client.get_secret("IBF-PRO-PASSWORD").value
    IBF_API_URL=secret_client.get_secret("IBF-URL").value
 
 
   
   # UCL_USERNAME=secret_client.get_secret("UCL-USERNAME").value
   # UCL_PASSWORD=secret_client.get_secret("UCL-PASSWORD").value  
    
    DATALAKE_STORAGE_ACCOUNT_NAME = secret_client.get_secret("DATALAKE-STORAGE-ACCOUNT-NAME").value
    DATALAKE_STORAGE_ACCOUNT_KEY = secret_client.get_secret("DATALAKE-STORAGE-ACCOUNT-KEY").value
    DATALAKE_STORAGE_ACCOUNT_KEY_IBFSYSTEM=secret_client.get_secret("DATALAKE-STORAGE-ACCOUNT-KEY-IBFSYSTEM").value
    DATALAKE_API_VERSION = '2018-11-09'
 
 
 


except Exception as e:
    print('No access to Azure Key vault, skipping.')

#2. Try to load secrets from env-variables (i.e. when using Github Actions)
try:
    import os  
    
    ADMIN_LOGIN = os.environ['ADMIN_LOGIN']
    IBF_API_URL=os.environ['IBF_API_URL']
    ADMIN_PASSWORD = os.environ['ADMIN_PASSWORD']    
    #PHP_PASSWORD=os.environ['IBF_PASSWORD']
    DATALAKE_STORAGE_ACCOUNT_NAME = os.environ['DATALAKE_STORAGE_ACCOUNT_NAME']        
    DATALAKE_STORAGE_ACCOUNT_KEY = os.environ["DATALAKE_STORAGE_ACCOUNT_KEY"]
  
    print('Environment variables found.')
   
    DATALAKE_API_VERSION = '2021-06-08'
    #DATALAKE_STORAGE_ACCOUNT_KEY_IBFSYSTEM = os.environ["DATALAKE_STORAGE_ACCOUNT_KEY2"]
    
except Exception as e:
    print('No environment variables found.')
    
''' 
# 3. If 1. and 2. both fail, then assume secrets are loaded via secrets.py file (when running locally). If neither of the 3 options apply, this script will fail.
try:
    from typhoonmodel.utility_fun.secrets import *
except ImportError:
    print('No secrets file found.')



countryCodes=['PHL']

# COUNTRY SETTINGS
SETTINGS_SECRET = {
    "PHL": {
        "IBF_API_URL":IBF_API_URL,# IBF_API_URL, 
        "ADMIN_LOGIN": ADMIN_LOGIN,
        "ADMIN_PASSWORD": ADMIN_PASSWORD,
        #"UCL_USERNAME": UCL_USERNAME,
        #"UCL_PASSWORD": UCL_PASSWORD,
        #"AZURE_STORAGE_ACCOUNT": AZURE_STORAGE_ACCOUNT,
        #"AZURE_CONNECTING_STRING": AZURE_CONNECTING_STRING,
        "admin_level": 3,
        "mock": False,
        "mock_nontrigger_typhoon_event": 'nontrigger_scenario',
        "if_mock_trigger": True,
        "mock_trigger_typhoon_event": 'trigger_scenario',
        "notify_email": True
    },
}



#Trigger Values
EAPTrigger='Average' #Average

dref_probabilities_10 = {
            "50": [50,'Moderate'],
            "70": [70,'High'],
            "90":[90,'Very High'],
        }

dref_probabilities = {
    "80k": [80000, 0.5],
    "50k": [50000, 0.6],
    "10k": [10000, 0.8],
    "5k": [5000, 0.95],
}

dref_probabilities_old = {
    "100k": [100000, 0.5],
    "80k": [80000, 0.6],
    "70k": [70000, 0.7],
    "50k": [50000, 0.8],
    "30k": [30000, 0.95],
}
        
cerf_probabilities = {
    "80k": [80000, 0.5],
    "50k": [50000, 0.6],
    "30k": [30000, 0.7],
    "10k": [10000, 0.8],
    "5k": [5000, 0.95],
}               

START_probabilities_old = {
    'PH166700000':{
        "8k": [8000, 0.8],
        "10k": [10000, 0.8],
        "15k": [15000, 0.7],
        "20k": [20000, 0.5],
        "25k": [25000, 0.5], 
        },
    'PH021500000':{
        "16k": [16000, 0.8],
        "18k": [18000, 0.8],
        "20k": [20000, 0.7],
        "25k": [25000, 0.5],
        "28.5k": [28500, 0.5], 
        },
    'PH082600000':{
        "18k": [18000, 0.8],
        "20k": [20000, 0.8],
        "25k": [25000, 0.7],
        "30k": [30000, 0.6],
        "40k": [40000, 0.5],
        }, }


HI_probabilities = {
    'PH050500000':{
        "15k": [15000, 0.8],
        "24.5k": [24500, 0.7],
        "36k": [36000, 0.5]
        }
}

provinces_names={'PH166700000':'SurigaoDeLnorte','PH021500000':'Cagayan','PH082600000':'EasternSamar'}   

START_probabilities = {
    'PH166700000':{ #SurigaoDeLnorte
        "8k": [8000, 0.8],
        "17k": [17000, 0.8],
        "25k": [25000, 0.7],
        "34k": [34000, 0.5],
        "37k": [37000, 0.5], 
        },
    'PH021500000':{ #Cagayan
        "35k": [35000, 0.8],
        "49k": [49000, 0.8],
        "55k": [55000, 0.7],
        "59k": [59000, 0.5],
        "62k": [62000, 0.5], 
        },
    'PH082600000':{ #EasternSamar
        "23k": [23000, 0.8],
        "42k": [42000, 0.8],
        "53k": [53000, 0.7],
        "64k": [64000, 0.5],
        "70k": [70000, 0.5],
        }, }
#### PAR SETTINGS
'''
This is the smallest and innermost monitoring domain, whose boundary is closest to the Philippine Islands.
The exact dimensions of this domain are the area of the Western North Pacific bounded by imaginary lines 
connecting the coordinates: 5°N 115°E, 15°N 115°E, 21°N 120°E, 25°N 120°E, 25°N 135°E and 5°N 135°E. 
The western boundary of the PAR is closer to the coastline of the country than the eastern boundary.
The eastern PAR boundary is several hundred kilometers away from the nearest coastline in the eastern part 
of the country and completely encloses the East Philippine Sea. Tropical Cyclones inside the PAR warrants the
issuance of Severe Weather Bulletin, the highest level of warning information issued for tropical cyclones.
'''
parBox=[5,115,25,135]


start_time = datetime.now()
### to run data pipeline for a specific event
#ecmwf_remote_directory='20230526000000'#'20221014000000'#''#(start_time - timedelta(hours=24)).strftime("%Y%m%d120000")
#Active_Typhoon_event_list=['NALGAE']

 
### to run data pipeline for a specific event
#ecmwf_remote_directory='20241113180000'

ecmwf_remote_directory=None
Active_Typhoon_event_list=[]

High_resoluation_only_Switch=False

if ecmwf_remote_directory==None:
    forecastTime = datetime.utcnow()
    uploadTime = datetime.now()
    uploadTime = uploadTime.strftime("%Y-%m-%dT%H:%M:%SZ")
else:
    forecastTime = datetime.strptime(ecmwf_remote_directory, "%Y%m%d%H%M%S")
    uploadTime = forecastTime.strftime("%Y-%m-%dT%H:%M:%SZ")


typhoon_event_name=None

ECMWF_CORRECTION_FACTOR=1

ECMWF_LATENCY_LEADTIME_CORRECTION=8 
longtiude_limit_leadtime=120 # if track pass this point consider it has made landfall 

WIND_SPEED_THRESHOLD=0
Wind_damage_radius=300 #will be updated based on maximum_radius varaible from model 
Show_Areas_on_IBF_radius=400

Alternative_data_point = (start_time - timedelta(hours=24)).strftime("%Y%m%d")  
data_point = start_time.strftime("%Y%m%d")      
 
 
 
###################
## PATH SETTINGS ##
###################


#MAIN_DIRECTORY='/home/fbf/'


MAIN_DIRECTORY ='./'# str(Path(__file__).parent.absolute())

 

#MAIN_DIRECTORY='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/IBF_TYPHOON_DATA_PIPELINE/IBF-Typhoon-model/'

ADMIN_PATH =MAIN_DIRECTORY+'data/gis_data/phl_admin3.geojson'
ADMIN4_PATH =MAIN_DIRECTORY+'data/gis_data/adm4_centers.geojson'
ADMIN3_PATH = MAIN_DIRECTORY+'data/gis_data/adm3_centers.csv'

maxDistanceFromCoast=2000 # max (km) distance to consider lead time calculation 

PRE_DISASTER_INDICATORS = MAIN_DIRECTORY+'data/pre_disaster_indicators/all_predisaster_indicators.csv'
CENTROIDS_PATH = MAIN_DIRECTORY+'data/gis_data/centroids_windfield.geojson' 
 
Input_folder = MAIN_DIRECTORY+ 'forecast/Input/'
Output_folder = MAIN_DIRECTORY+ 'forecast/Output/'
ECMWF_folder = Input_folder+'ECMWF/'

rainfall_path =MAIN_DIRECTORY+ 'forecast/rainfall/' 

mock_data_path = MAIN_DIRECTORY+'data/mock/'
ML_model_input = MAIN_DIRECTORY+'data/model_input/df_modelinput_july.csv'
logoPath = MAIN_DIRECTORY+'/data/logos/combined_logo.png'


for dir_path in [Input_folder,Output_folder,rainfall_path,ECMWF_folder]:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)



readmefilePath=os.path.join(Output_folder, "readme.txt")


lines = ['Readme\n',
         '1.In the csv file *_admin3_leadtime.csv column name Potential_leadtime refers to\n  the time until the typhoon reaches the point closest to this municipality',
         '',
         '2.In the csv file *_dref_trigger_status_10_percent.csv column name\n\t Threshold =\t\t The name of EAP threshold (based on probability or Average)\n\t Scenario=\t\t probability values for the threshold (NA for average)\n\t Trigger status=\t If EAP will be activated based on the specific scenario',
         '']



with open(readmefilePath, 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
        
        
    
    
Population_Growth_factor=1.15 #(1+0.02)^7 adust 2015 census data by 2%growth for the pst 7 years 

Housing_unit_correction={'year':['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022'],
                        'facor':[0.88,0.89,0.91,0.92,0.93,0.95,0.96,0.97,0.99,1.00,1.01,1.03,1.04,1.06,1.07,1.09,1.10]}



