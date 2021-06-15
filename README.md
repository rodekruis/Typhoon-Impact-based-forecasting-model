# Typhoon Impact forecasting model

This tool was developed as trigger mechanism for the typhoon Early action protocol of the philipiness resdcross FbF project. The model will predict the potential damage of a typhoon before landfall, the prediction will be percentage of completely damaged houses per manuciplaity.
The tool is available under the [GPL license](https://github.com/rodekruis/Typhoon-Impact-based-forecasting-model/blob/master/LICENSE)

## Pipeline
    The main script for the pipeline is mainpipeline.py 
	After setting the python R inviroment listed in the requirement file 
	mainpipeline.py can be run from a command line via python3 mainpipeline.py... 
	
	
## Instructions to Build,Update and Run Docker image

  Prerequisites 
  Docker installed
  Docker-settings: set memory of containers to at least 2GB

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
docker build -t fbf-phv3 .
docker run --rm --name=fbf-phv3 -v ${PWD}:/home/fbf -it fbf-phv3 bash
```
If entering the container a 2nd time or later:
```
docker exec -it fbf-phv3 /bin/bash
```
or (if unstarted)
```

docker start -i fbf-phv3
```
To edit files within the container first install vim 
```
apt-get update
apt-get install vim
```


To start code manually from inside container
```
python3 mainpipeline.py
```

To inspect the logs (e.g. when getting an email about errors), run from inside the container:
```
nano /var/log/cron.log
```
(Scroll down with Ctrl+V) 


