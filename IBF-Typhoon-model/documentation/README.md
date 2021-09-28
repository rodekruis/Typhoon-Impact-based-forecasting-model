# Steps to develop the housing Damag prediction model 

1. Download and Process rainfall data for Historical typhoon events
2. Download and process typhoon data (wind speed/ Track distance) for historical event
3. Process pre disaster indicators and Damage/Loss data (Prepare model input)
4. Build Machin learning model a) model training and hyper paramerer optimization b) optimal model result

## Installation

1. Clone the repo
2. Change `/IBF-Typhoon-model/src/settings.py.template` to `settings.py` and fill in the necessary passwords.
3. Find `data.zip` in https://rodekruis.sharepoint.com/sites/510-CRAVK-510/_layouts/15/guestaccess.aspx?guestaccesstoken=HadTB1h%2FWiVluDiortTyd3%2F9rSc0MdjS2yub9GEntCs%3D&docid=2_0013b102f095246fdab4ff4ce03b12933&rev=1&e=eDGoN5 and unzip in /IBF-Typhoon-model/data.
4. Install Docker-compose [follow this link](https://docs.docker.com/desktop/windows/install/)

## Model data preparation 


Data sets used for training a damage prediction model were collected from different sources through desk research and in-country visits of key stakeholders. It is essential to have these datasets with national spatial coverage and at the same aggregation level, As not all data sets -in particular historical damage counts- are available at barangay level, which is the lowest administrative unit in the Philippines, the typhoon model can only be developed at the next administrative level, which is a municipality level. The main datasets used for the model are:- 

- Typhoon hazard data
	* rainfall
	* wind speed  
- Pre disaster indicators 
- Data related to topography 
- Damage and loss data for historical events 



### Download and process rainfall data

Raifall is one of the hazards associated with Typhoons. For the model we used observed rainfall data like the one below  ![example rainfall](https://eoimages.gsfc.nasa.gov/images/imagerecords/52000/52366/philippines_mpa_2011275.png)

* [This](https://github.com/rodekruis/typhoon_house_damage_model/blob/master/IBF-Typhoon-model/documentation/notebooks/rainfalldownload.ipynb) jupyter notebook contains scripts to download and process historical rainfall data 

### Download and process typhoon data

* [This](https://github.com/rodekruis/typhoon_house_damage_model/blob/master/IBF-Typhoon-model/documentation/notebooks/windfield.ipynb) jupyter notebook contains scripts to download and process typhoon wind data

### Process data for pre disaser indicators, damage loss data and prepare model input

* [This](https://github.com/rodekruis/typhoon_house_damage_model/blob/master/IBF-Typhoon-model/documentation/notebooks/pre_disaster_indicators.ipynb) jupyter notebook contains scripts to process pre disaster indicators, damage and topography datasets. The scripts will is also used to prepare final model input.

## Machin learning model

### Binary Classification

* [This](https://github.com/rodekruis/typhoon_house_damage_model/blob/master/IBF-Typhoon-model/documentation/notebooks/classification_model.ipynb) jupyter notebook contains scripts for classification Model training and Hyper paramerer optimization
* [This](https://github.com/rodekruis/typhoon_house_damage_model/blob/master/IBF-Typhoon-model/documentation/notebooks/classification_model_result.ipynb) jupyter notebook contains scripts to view Classification Model result

### Regression

* [This](https://github.com/rodekruis/typhoon_house_damage_model/blob/master/IBF-Typhoon-model/documentation/notebooks/regression_model.ipynb) jupyter notebook contains scripts for Regression Model training and Hyper paramerer optimization
* [This](https://github.com/rodekruis/typhoon_house_damage_model/blob/master/IBF-Typhoon-model/documentation/notebooks/Regression_model_result.ipynb) jupyter notebook contains scripts to view Regression model result
