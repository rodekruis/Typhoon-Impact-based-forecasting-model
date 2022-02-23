# Wind Data

This folder contains the scripts used for obtaining and processing wind input data. It makes use of the Climada package in the 'climada' folder. 

- **obtaining_historical_wind_data** : the script used to obtain the wind grids and minimum track distance for all typhoons and municipalities. It uses a method that creates centroids on the map and calculates the wind speed and track distance for each centroid. Next, for each municipality, the minimum track distance of all centroids within the municipality and the maximum windspeed is taken. 
- **windspeed**: m/s - 1 minute average
- **track distance**: km
- **input : geojson file** of philippines administrative boundaries
- **input : typhoon_event** = excel sheet with info on all typhoons for which the data should be obtained (local_name, international_name, year)
- **output** : folder with for each typhoon a csv file with the minimum track distance and maximum windspeed per municipality

Note: the climada package sets threshold on which data should be collect (track distance, intensity etc.). This is set and can be changed in: *climada/hazard/trop_cyclone.py*