
# coding: utf-8

# In[ ]:


import requests
import re
import os, csv, sys 
import scipy
import pandas as pd
import gzip
import xml.etree.ElementTree as ET
from os import listdir
from datetime import datetime
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import random
from netCDF4 import Dataset
import itertools

from functions import *


# In[ ]:


def check_status(file_num, file_num_tot, start):
    percent_complete = (file_num/file_num_tot)*100
    end = time.time()
    sys.stdout.write('\r%.3f %s, %s: %.1f' % (percent_complete, '% Completed', 'time elapsed', end-start))
    sys.stdout.flush()


# ## Parameters

# In[ ]:


email = # 'your email'
pswd = # 'your password'

data_obs_folder = '../data/obs/'
data_models_folder = '../data/temporarily_downloaded/'
results_folder = '../CSVs/multicyclone/'

delete_previous_results_files = 'y'
save_csv_files = 'y'  # one per model


# ## Cyclone names from observation files

# In[ ]:


df_obs = pd.read_csv(data_obs_folder+'best_track_27ty.csv')

cyclone_names = [x.lower().strip() for x in sorted(list(set(df_obs['STORMNAME'])))]#+['vongfong']

p = {}
p['durian'] = ['durian', 'reming']
p['fengshen'] = ['fengshen', 'frank']
p['ketsana'] = ['ketsana', 'ondoy']
p['conson'] = ['conson', 'basyang']
p['nesat'] = ['nesat', 'pedring']
p['bopha'] = ['bopha', 'pablo']
p['utor'] = ['utor', 'labuyo']
p['trami'] = ['trami', 'maring']
p['usagi'] = ['usagi', 'odette']
p['nari'] = ['nari', 'santi']
p['krosa'] = ['krosa', 'vinta']
p['haiyan'] = ['haiyan', 'yolanda']
p['lingling'] = ['lingling', 'agaton']
p['rammasun'] = ['rammasun', 'glenda']
p['kalmaegi'] = ['kalmaegi', 'luis']
p['fung-wong'] = ['fung-wong', 'fungwong', 'fung wong', 'fung_wong', 'mario']
p['hagupit'] = ['hagupit', 'ruby']
p['mekkhala'] = ['mekkhala', 'amang']
p['noul'] = ['noul', 'dodong']
p['goni'] = ['goni', 'ineng']
p['mujigae'] = ['mujigae', 'kabayan']
p['koppu'] = ['koppu', 'lando']
p['melor'] = ['melor', 'nona']
p['sarika'] = ['sarika', 'karen']
p['vongfong'] = ['vongfong', 'ambo']
p['haima'] = ['haima', 'lawin']
p['nock-ten'] = ['nock-ten', 'nockten', 'nock ten', 'nock_ten', 'nina']
p['mangkhut'] = ['mangkhut', 'ompong']

cyclone_possible_names = [p[x] for x in cyclone_names]


# ## Authentication

# In[ ]:


try:
    import getpass
    input = getpass.getpass
except:
    try:
        input = raw_input
    except:
        pass
    
values = {'email' : email, 'passwd' : pswd, 'action' : 'login'}
login_url = 'https://rda.ucar.edu/cgi-bin/login'

ret = requests.post(login_url, data=values)
if ret.status_code != 200:
    print('Bad Authentication')
    print('ret.text')
    exit(1)
    
dspath = 'https://rda.ucar.edu/data/ds330.3/'


# ## Create filelist to download

# In[ ]:


institute_list = ['RJTD', 'ecmf', 'egrr', 'kwbc']
model_list = [['GSM','GEM','TEM','WFM'],['ifs'],['mogreps'],['CENS','CMC','GEFS','GFS']]
model_spec_list = [[['tcan_nwp','tctr_nwp'],['etctr_nwp'],['etctr_nwp'],['etctr_nwp']],[['all_glo']],[['etctr_glo']],[['etctr_glo','esttr_glo'],['tctr_glo','sttr_glo'],['etctr_glo','esttr_glo'],['tctr_glo','sttr_glo']]]

year_list = [str(x) if x>9 else '0'+str(x) for x in range(2006, 2021)]
year_list.remove('2007')
year_list.remove('2017')
year_list.remove('2018')

month_list = [['11','12'],['06'],['09'],['07'],['09'],['11','12'],['08','09','10','10','11'],['01','02','07','09','11','12'],['01','05','08','09','10','12'],['10'],['12'],['05']]
day_list_preliminary = [[[27,30],[1,2]],[[17,27]],[[24,30]],[[11,18]],[[23,30]],[[25,30],[1,9]],[[8,24],[16,24],[8,16],[27,31],[1,11]],[[10,31],[1,2],[9,20],[10,25],[29,30],[1,12]],[[13,21],[2,16],[13,30],[29,30],[1,21],[10,17]],[[13,19]],[[19,29]],[[10,18]]]
hour_list = ['00','06','12','18']

day_list = []

for y in day_list_preliminary:
    months = []
    for m in y:
        days = [str(x) if x>9 else '0'+str(x) for x in range(m[0], m[1]+1)]
        months.append(days)
    day_list.append(months)
    
filelist = []

for i,institute in enumerate(institute_list):
    for mod,model in enumerate(model_list[i]):
        for model_spec in model_spec_list[i][mod]:
            for y,year in enumerate(year_list):
                for m,month in enumerate(month_list[y]):
                    for day in day_list[y][m]:
                        for hour in hour_list:
                            filename = institute.lower()+'/'+year+'/'+year+month+day+'/z_tigge_c_'+institute+'_'+year+month+day+hour+'0000_'+model+'_glob_prod_'+model_spec+'.xml'
                            filelist.append(filename)


# ## Create folders

# In[ ]:


# Create folders

for folder_name in [data_models_folder, results_folder]:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

        
# Initialise lists

file_list = []
file_list_short = []
list_failed1 = []
list_total = []
institutes = []
    

# Create list institutes
    
institutes = [x.lower().strip() for x in institute_list]
                 
if delete_previous_results_files == 'y':
    for cyclone_name in cyclone_names:
        try:
            os.remove(results_folder+cyclone_name+'_all.csv')
        except:
            pass


# In[ ]:


# filelist = filelist[6227:]


# In[ ]:


# file_num


# ## Actual download and processing

# In[ ]:


import time
start = time.time()
filelist_actual = []

for file_num, file in enumerate(filelist):
    filename = dspath + file
    check_status(file_num, len(filelist), start)
    outfile = data_models_folder + os.path.basename(filename)      
    r = requests.head(filename, cookies = ret.cookies, allow_redirects=False)
    if r.status_code == 200:
        filelist_actual.append(file)
        req = requests.get(filename, cookies = ret.cookies, allow_redirects=False)
        open(outfile, 'wb').write(req.content)
        CXML_to_csv(outfile, cyclone_names, cyclone_possible_names, results_folder)
        os.remove(outfile)
    else:
        r_gz = requests.head(filename+'.gz', cookies = ret.cookies, allow_redirects=False)
        if r_gz.status_code == 200:
            filelist_actual.append(file+'.gz')
            req_gz = requests.get(filename+'.gz', cookies = ret.cookies, allow_redirects=False)
            open(outfile+'.gz', 'wb').write(req_gz.content)
            CXML_to_csv(outfile+'.gz', cyclone_names, cyclone_possible_names, results_folder)
            os.remove(outfile+'.gz')

