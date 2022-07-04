# Saved Models

This folder contains the saved optimal model for each task, which can be downloaded and used to make predictions. To do so, collect and process the required input variables (the selected features). A description of the input variables can be found [here](https://github.com/rodekruis/Typhoon_IBF_Rice_Damage_Model) and more information on how to collect the data can be found in the corresponding [data folder](https://github.com/rodekruis/Typhoon_IBF_Rice_Damage_Model/tree/main/(IBF_typhoon_model/data).) 

## Binary Classification

**Selected features RF Binary:**
- rainfall_max_6h
- dis_track_min
- vmax


**Selected parameters RF Binary:**
- max_depth = 20
- min_samples_leaf = 5
- min_samples_split = 15
- n_estimators = 250

## Multiclass Classification

**The selected features ares:**
- mean_slope
- mean_elevation_m
- ruggedness_stdev
- mean_ruggedness
- slope_stdev
- area_km2
- poverty_perc
- perimeter
- glat
- glon
- rainfall_max_6h
- rainfall_max_24h
- dis_track_min
- vmax

**Selected Parameters in RF multiclass:** 
- max_depth = None
- min_samples_leaf = 3
- min_samples_split = 15
- n_estimators = 50

## Regression

**Selected features RF Regression:**
- mean_slope
- mean_elevation_m
- ruggedness_stdev
- mean_ruggedness
- area_km2
- coast_length
- poverty_perc
- perimeter
- glat
- glon
- coast_peri_ratio
- rainfall_max_6h
- rainfall_max_24h
- dis_track_min
- vmax


**Selected Parameters RF Regression:**
- max_depth = 20
- min_samples_leaf = 1
- min_samples_split = 8
- n_estimators = 100