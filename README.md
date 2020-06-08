# Typhoon Impact forecasting model 
This tool was developed as trigger mechanism for the typhoon Early action protocol of the philipiness resdcross FbF project. The model will predict the potential damage of a typhoon before landfall, the prediction will be percentage of completely damaged houses per manuciplaity.
The tool is available under the [GPL license](https://github.com/rodekruis/Typhoon-Impact-based-forecasting-model/blob/master/LICENSE)

# Trigger Model Automation 
Automation of Triggr model for Forecast-based Financing in the Philippines
## Description :
Organize process of running priority models.
The following steps have been implemented for the pilot Phase automation

### Step 1: 
Check for active typhoon events in the Philippines area. This is done first by checking  for disaster events  listed on GDACS website https://www.gdacs.org/ , from those active events select typhoon events.

### Step 2: 
From those typhoon events check if they are located in the Philippines responsibility (PAR) area defined by the following binding box PAR=[[145, 35], [145, 5], [115, 5], [115, 35],[145, 35]]  which is approximately the area shown in the figure below.
   
### Step 3: 
Extract name and other relevant information for the typhoon event located in PAR
### Step 4: 
Download Typhoon forecast data from UCL and Rainfall forecast Data from NOAA
### Step 5: 
Run the trigger model based on input data for the new typhoon 
### Step 6: 
Send an automated email with information on forecasted impact of the new typhoon for users identified by the local FbF team. For the pilot phase this information will be sent via a temporary email account created for the trigger model. The tentative list for automated email recipients has five email addresses. 
### Step 7: 
Repeat the above steps 1 to 6 every 6 hours – until landfall.


# Instructions to Run # udapte the path for the directory 

Prerequisites 
* Docker installed
* Docker-settings: set memory of containers to at least 2GB

Retrieve code and move in repository:
```
git clone https://github.com/rodekruis/Typhoon-Impact-based-forecasting-model.git
cd Typhoon-Impact-based-forecasting-model
```
Copy secrets-file: # secrets file has been updated to variables.py 
```
cp secrets.py.template secrets.py
```
.. and retrieve the correct credentials from someone who knows. 

Start up application:
```
docker build -t fbf-ph .
docker run --name=fbf-ph -v ${PWD}:/home/fbf -it fbf-ph
```
If entering the container a 2nd time or later:
```
docker exec -it fbf-ph /bin/bash
```
or (if unstarted)
```
docker start -i fbf-ph
```

To start code manually from inside container
```
python3 automation_code_automation.py
```

To inspect the logs (e.g. when getting an email about errors), run from inside the container:
```
nano /var/log/cron.log
```
(Scroll down with Ctrl+V) 


## Imitate typhoon-scenario

Most times, there will be no ongoing typhoon. If you want to simulate a typhoon-scenario for testing/development purposes, you can change the following lines in automation_code_automation.py:
```
  delete_old_files()
  create_ucl_metadata()
  # Activetyphoon=['KAMMURI']
```
to
```
  # delete_old_files()
  # create_ucl_metadata()
  Activetyphoon=['KAMMURI']
```
and run
```
  cp Rodekruis-example.xml forecast/Rodekruis.xml
```

