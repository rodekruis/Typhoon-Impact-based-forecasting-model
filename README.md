# Typhoon Impact forecasting model

This tool was developed as a trigger mechanism for the typhoon Early action protocol of the Philippines Red Cross 
FbF project. The model will predict the potential damage of a typhoon before landfall, and the prediction will be 
percentage of completely damaged houses per municipality.
The tool is available under the 
[GPL license](https://github.com/rodekruis/Typhoon-Impact-based-forecasting-model/blob/master/LICENSE)

## Installation

To run the pipeline, you need access to an FTP server. 
If you or your organization is interested in using the pipeline, 
please contact [510 Global](https://www.510.global/contact-us/)
to obtain the credentials.  You will receive a file called `settings.py`, which you need to place in 
the `IBF-Typhoon-model/lib` directory.

### Without Docker

The main script for the pipeline is `IBF-Typhoon-model/scr/pipeline.py`.
It can in principle be run locally by setting up your local Python environment using
the `requirements.txt` file, and installing all the R packages listed in the `Dockerfile`.
However, we suggest using the docker image instead.

### With Docker

You will need to have docker installed, and you should modify `Docker-settings`
to set the container memory to at least 2GB

####  Build the image

To build the docker image, run:
```
docker build -t rodekruis510/typhoonibf .
```
## Running
To spin up and enter the docker container, create a directory where you would like
the output data to go. Then run:
```
docker run --rm -it --name=fbf-phv3 -v $path-to-output-directory:/home/fbf/forecast rodekruis510/typhoonibf bash
```

To run the pipeline, enter the container and execute:
```
run-typhoon-model [OPTIONS]

Options:
  --path TEXT             main directory defult 
  --remote_directory TEXT                  remote directory 
  --typhoonname TEXT               name for active typhoon
```

When there is an active typhoon in the PAR polygon the model will run for the active typhoons,
unless you specify a remote directory and typhoon name. 
