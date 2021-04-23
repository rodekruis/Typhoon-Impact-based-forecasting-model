# Typhoon Impact forecasting model

This tool was developed as trigger mechanism for the typhoon Early action protocol of the philipiness resdcross FbF project. The model will predict the potential damage of a typhoon before landfall, the prediction will be percentage of completely damaged houses per manuciplaity.
The tool is available under the [GPL license](https://github.com/rodekruis/Typhoon-Impact-based-forecasting-model/blob/master/LICENSE)

# Methodology


## 1. Typhoon model:

NLRC 510 has developed the impact prediction model for typhoons in the Philippines. This model provides information on predicted impact of approaching typhoons.Starting from the first weather forecast for an active typhoons were made available by ECMWF 510 typhoon model predicts potential impacts of the typhoon using a proxy variable ‘% of completely damaged houses’ per municipality.

* 510 Typhoon model was trained based on data from 27 past typhoons in the last 15 years in the Philippines, for which detailed damage reports were available through [NDRRMC](https://www.ndrrmc.gov.ph/) For these events, we also collected explanatory indicators, such as wind speed, rainfall (event-specific) and building materials of houses (PH national census) etc..

* From Typhoon Track wind fields were calculated based on [Willoughby](https://journals.ametsoc.org/view/journals/mwre/132/12/mwr2831.1.xml) methodology. The codes for appling these methedology can be found [here](https://github.com/rodekruis/TYPHOONTRACK2GRIDPOINT) We built a statistical model, which could explain differences in damage on the basis of differences in wind speed and building materials (etc.)

*In the case of new typhoon, we are dealing with an upcoming typhoon, which is still in the open sea and probably making landfall in the next 120hrs..
*	Our wind speed source for forecast wind speed data is ECMWF
*	This forecasted wind speed (and typhoon track) are plugged as input into the above-mentioned prediction model, which - still in combination with building materials - lead to the predicted damage class per municipality that can be seen in the map.

*	Note that the results are strongly dependent on the input of windspeed, which is itself still an unknown. (see accuracy below).

## 2. How to use this product:

* The map contains damage classes (1-5) per municipality. As such, we advise to put priority on municipalities in damage class 5, and depending on available resources continue with class 4, etc.
*	This damage class is based on ‘% of houses that are completely damaged’. As priority might also be based on exposure and vulnerability, we have added to the Excel a couple of relevant indicators, from the Community Risk Assessment dashboard.
*	PRC can decide if and how to combine these various features. If needed, 510 can be asked for assistance of course.

## 3. Important notes:

*	ACCURACY: it should be realized that during the course of the coming 3 days, the typhoon might change course, or increase/decrease in terms of strength. This will affect the quality of these predictions. 
*	This damage prediction is only about completely damaged houses, not about partially damaged houses.
*	We only included municipalities that are within 100km of the forecasted typhoon track, as we have seen from previous typhoons (with comparable wind speeds) that damage figures outside of this area are low.

## 4. Sources

*	The wind speed is provided by ECMWF. It is the ‘maximum 10-minute sustained wind speed’. An average of this is calculated per municipality.
*	From Typhoon Track wind fields were calculated based on  Willoughby  methodology 
*	Typhoon track (from which ‘distance to typhoon track’ per municipality is calculated), is provided by ECMWF as well.
*	Additionally, various wall and roof type categories from the Philippines national census. The model uses 2010 census data, as it was developed using this data (2015 census data on municipality level only became available in 2018). The 2015 census data could not be easily plugged in, because of some differences in roof/wall categories. We believe that this would not change the result much though, as even if there are large differences from 2010 to 2015, these would still be dominated by wind speed effects in the model.
*	All additional indicators, that are added to the Excel table (population, poverty) are derived from the Community Risk Assessment dashboard (Go to [this](https://dashboard.510.global/#!/community_risk?country=PHL) link and click ‘Export to CSV’ on top-right.) The sources for these indicators can be found in the dashboard itself.



# Trigger Model Automation 

Automation of Triggr model for Forecast-based Financing in the Philippines

## Description

Organize process of running priority models.
The following steps have been implemented for the pilot Phase automation

### Step 1

Check for active typhoon events in the Philippines area. This is done first by checking  for disaster events  listed on GDACS website https://www.gdacs.org/ , from those active events select typhoon events.

### Step 2

From those typhoon events check if they are located in the Philippines responsibility (PAR) area defined by the following binding box PAR=[[145, 35], [145, 5], [115, 5], [115, 35],[145, 35]]  which is approximately the area shown in the figure below.

### Step 3

Extract name and other relevant information for the typhoon event located in PAR

### Step 4

Download Typhoon forecast data from UCL and Rainfall forecast Data from NOAA

### Step 5

Run the trigger model based on input data for the new typhoon 

### Step 6

Send an automated email with information on forecasted impact of the new typhoon for users identified by the local FbF team. For the pilot phase this information will be sent via a temporary email account created for the trigger model. The tentative list for automated email recipients has five email addresses.

### Step 7

Repeat the above steps 1 to 6 every 6 hours – until landfall.

# Instructions to Run /udapte the path for the directory

Prerequisites 
* Docker installed
* Docker-settings: set memory of containers to at least 2GB

Retrieve code and move in repository:
```
git clone https://github.com/rodekruis/Typhoon-Impact-based-forecasting-model.git
cd Typhoon-Impact-based-forecasting-model
```
create a lib/setting.py -file: # s
```
sudo vim lib/setting.py cp 
```
.. and retrieve the correct credentials from someone who knows. 

Start up application:
```
docker build -t fbf-phv2 .
docker run --rm --name=fbf-phv2 -v ${PWD}:/home/fbf -it fbf-phv2 bash
```
If entering the container a 2nd time or later:
```
docker exec -it fbf-phv2 /bin/bash
```
or (if unstarted)
```
docker start -i fbf-phv2
```
To edit files within the container first install vim 
```
apt-get update
apt-get install vim
```


To start code manually from inside container
```
python3 main.py
```

To inspect the logs (e.g. when getting an email about errors), run from inside the container:
```
nano /var/log/cron.log
```
(Scroll down with Ctrl+V) 


## Imitate typhoon-scenario

Most times, there will be no ongoing typhoon. If you want to simulate a typhoon-scenario for testing/development purposes, you can change the following lines in main.py:
```
 
  # Activetyphoon=['KAMMURI']
```
to
```
 
  Activetyphoon=['KAMMURI']
```
and run
```
  cp Rodekruis-example.xml forecast/Rodekruis.xml
```

