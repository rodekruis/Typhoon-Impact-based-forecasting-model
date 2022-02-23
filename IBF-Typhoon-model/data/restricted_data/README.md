# Restricted Data

The content of this folder can be requested by contacting 510. It contains the input data for the models and a folder containing rice data.


## Combined Input Data

This folder contains the processed data that is used as model input. It is the dataframe that is generated in 'data_preparation_notebook'. To use: place the 'combined_input_data' folder in the 'restricted_data' folder.

## Rice Data

To use: place the 'rice_data' folder in the 'restricted_data' folder.

### Rice Area

To calculate the percentage loss, the standing rice area at typhoon occurrence is needed. To obtain this, data from PRiSM was used. Raster files with the area planted at specific dates for the 2016 - 2019 period was obtained. Using the 'rice_area_plant.py' script, the area planted (in ha) in each municipality and provided data is obtained. This is save into the excel file 'rice_area_planted.xlsx'. 

### Rice Losses

This folder contains the source data on rice losses and a file in which this input data is cleaned per region.