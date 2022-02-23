# Rainfall data

This folder contains the needed data and scripts for obtaining and processing the rainfall data. The downloaded data is the 'Global Precipitation Measurement' obtained from NASA. Information on the data, what it entails and how it can be obtained can be found at the bottom. There are two types of rainfall that can be downloaded and processed: daily and half hourly rainfall. Due to the size of the downloaded GPM files, they are not included on the GitHub, but can be downloaded locally.

## Daily Rainfall

Containing the daily average precipitation rates. With this the cummulative or maximum daily rainfall can be obtained efficiently. Only once .Tif file per day has to be downloaded. 

**Scripts**
- rainfall_daily <br>
    Collects the daily rainfall in mm (mm/24h) for the range of the typhoon. This is saved into a csv file with the municipalities and daily rainfall per column


**Input**
- metadata_typhoon <br>
    csv file with all necessarey typhoon information

**Output**
- for each typhoon: a csv file with the daily rainfall in mm per municipality (mm/24h)

## HHR (half hourly) Rainfall

Containing the precipitation rates on an half hourly interval. This can be used to obtain more diverse rainfall variables, such as the maximum 6 hour precipitation rate. For every day, 48 files need to be downloaded. It therefore takes longer, but also provides more detailed information

**Scripts**
- rainfall_hhr <br>
    Collects the half hourly rainfall precipitation rate in mm/h for the range of the typhoon

**Input**
- metadata_typhoon <br>
    csv file with all necessarey typhoon information

**Output**
- for each typhoon: a csv file with half hourly precipitation rate in mm/h

## Rainfall Processing

This scripts uses the collected Half Hourly Rainfall data for each typhoon to obtain the maximum precipitation rate in a certain time interfall and a rolling window. This is currently done to obtain the following two variables. The timeframe for which the data is obtained covers the 72 hours before the typhoon made landfall. This was done as the collected data when making predictions for a new typhoon covers the 72 hours before a typhoon makes landfall. If a typhoon didn't make landfall (in the Philippines) the date at which the typhoon was closest to the shore was used as the landfall date.

1. Maximum 6 hour rainfall in mm/h
2. Maximum 24 hour rainfall in mm/h

**Output**
- rainfall_max_6h : csv sheet with the maximum 6 hour precipitation rate (mm/h) for each municipality and typhoon
- rainfall_max_24h : csv sheet with the maximum 24 hour precipitation rate (mm/h) for each municipality and typhoon

## Information Links

- NASA Global Precipitation Measurement: https://gpm.nasa.gov/data/directory
- The precipitation processing system at NASA Goddard: https://gpm.nasa.gov/sites/default/files/2020-06/IMERG-GIS-Readme_4_22_20.pdf
- Accessing the data: https://gpm.nasa.gov/sites/default/files/2021-01/jsimpsonhttps_retrieval.pdf
