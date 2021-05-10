# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 20:28:23 2019

@author: ATeklesadik
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from datetime import datetime
import xarray as xr

import seaborn as sns
def con_date(vt):
    try:
        cdd=(datetime.strptime(vt, "%Y-%m-%dT%H:00:00Z")).strftime("%Y-%m-%d %H:00:00")
    except:
        cdd='None'
        pass
    return cdd
def calculate_dis(row):
    return (np.sqrt((row['lat_delta']*row['lat_delta']) + (row['lon_delta']*row['lon_delta'])))*111




steps_vhr1=['6', '12', '18', '24', '30', '36', '42', '48', '54', '60', '66', '72']








df1 = pd.read_table('C:\\globus_data\\PAR\\calculated_rations\\data_delta.csv',sep=',')
df1['delta_x']=df1['delta_x']*111
df1=df1[df1['vhr'].isin(steps_vhr1)]

df2=data3[data3['vhr'].isin(steps_vhr1)]

df2=df1[df1['ensamble']!=99]


plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(18, 12))
#ax.subplots_adjust(left=0.2, wspace=0.025,hspace=0.1, top=0.90)
#ax = sns.boxplot(x="vhr", y="delta_x",hue="model_name", data=df1,showfliers=False,palette=sns.color_palette("cubehelix", n_colors=11),linewidth=1)

ax = sns.boxplot(x="vhr", y="delta_x",hue="model_name", data=df2,showfliers=False,palette=sns.diverging_palette(170, 20, s=90, l=55, n=9,center='light'),linewidth=1)

major_ticks = np.arange(0, 1000, 200)
minor_ticks = np.arange(0, 100, 50)
# Or if you want different settings for the grids:
ax.grid(which='minor', alpha=0.3)
ax.grid(which='major', alpha=0.5)
#ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.set_ylim([-10,800])
#sns.despine(left=True)
#sns.despine(offset=5, trim=True);
ax.set_title('Error in Track position for different lead time',fontsize=14)
ax.set_xlabel('Lead time',fontsize=14)
ax.set_ylabel('Track Error(km)',fontsize=14)

#

fig.savefig('C:\\globus_data\\PAR\\calculated_rations\\track_error2.jpg', format='jpg', dpi=400)





fig, ax = plt.subplots(figsize=(16, 10))
#ax.subplots_adjust(left=0.2, wspace=0.025,hspace=0.1, top=0.90)
ax = sns.boxplot(x="vhr", y="delta_speed",hue="model_name", data=df1,showfliers=False,palette=sns.color_palette("RdBu", n_colors=7),linewidth=1)

#major_ticks = np.arange(0, 1000, 200)
#minor_ticks = np.arange(0, 1000, 50)
# Or if you want different settings for the grids:
#ax.grid(which='minor', alpha=0.3)
#ax.grid(which='major', alpha=0.5)
#ax.set_yticks(major_ticks)
#ax.set_yticks(minor_ticks, minor=True)
#ax.set_ylim([-10,800])
#sns.despine(left=True)
#sns.despine(offset=5, trim=True);
ax.set_title('ERROR IN FORECASTED MAXIMUM WIND SPEED (km/hr) AT DIFFERENT LEAD TIME',fontsize=14)
ax.set_xlabel('LEAD TIME(Hrs)',fontsize=14)
ax.set_ylabel('ERROR IN MAXIMUM WIND SPEED (km/hr)',fontsize=14)
plt.legend(loc='upper left')
#

fig.savefig('C:\\globus_data\\PAR\\calculated_rations\\speed_error.jpg', format='jpg', dpi=400)





















data_list2.columns
fig, ax = plt.subplots(figsize=(14, 8))
ax = sns.boxplot(x="vhr", y="cyc_speed",hue="model_name", data=data_list2,showfliers=False)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
#ax.set_ylim([-10,1000])

#major_ticks = np.arange(0, 2000, 200)
#minor_ticks = np.arange(0, 2000, 100)
# Or if you want different settings for the grids:
#ax.grid(which='minor', alpha=0.3)
#ax.grid(which='major', alpha=0.5)

#sns.despine(left=True)
#sns.despine(offset=5, trim=True);
ax.set_title('Error in forecasted Max speed for different lead time')
ax.set_xlabel('Lead time')
ax.set_ylabel('Max speed Error(m/s)')



