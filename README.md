# Typhoon Impact forecasting model

This tool was developed as trigger mechanism for the typhoon Early action protocol of the philipiness resdcross 
FbF project. The model will predict the potential damage of a typhoon before landfall, the prediction will be 
percentage of completely damaged houses per manuciplaity.
The tool is available under the 
[GPL license](https://github.com/rodekruis/Typhoon-Impact-based-forecasting-model/blob/master/LICENSE)

## Installation

To run the pipeline, you need access to an FTP server. 
If you or your organization is interested in using the pipeline, 
please contact [510 Global](https://www.510.global/contact-us/)
to obtain the credentials.  You will receive a file called `settings.py`, which you need to place in 
the `lib` directory.

### Without Docker

The main script for the pipeline is IBF-Typhoon-model/mainpipeline.py.
It can in principle be run locally by setting up your local Python environment using
the `IBF-Typhoon-model/requirements.txt` file, and installing all the R packages listed in the `IBF-Typhoon-model/Dockerfile`.
However, we suggest using the docker image instead.

### With Docker

You will need to have docker installed, and you should modify `Docker-settings`
to set the container memory to at least 2GB

####  Get the image

The docker image is available at 
[typhoonibf:latest](https://hub.docker.com/repository/docker/rodekruis510/typhoonibf/).
To download it, run:
```
docker pull rodekruis510/typhoonibf
```

You can also build it yourself. To do so, first download 
[this version of ncdf4](https://cran.r-project.org/src/contrib/Archive/ncdf4/ncdf4_1.13.tar.gz)
and place the tarball in the top-level directory.
Then run:
```
docker build -t rodekruis510/typhoonibf .
```
Create a docker container
```
docker run --name typhoonibf rodekruis510/typhoonibf

```
Run and access the container
```
docker run -it --entrypoint /bin/bash rodekruis510/typhoonibf
```
## Running
To spin up and enter the docker container, execute:
```
docker run --rm -it --name=fbf-phv3 -v ${PWD}:/home/fbf  rodekruis510/typhoonibf bash
```
To run the pipeline, enter the container and execute:
```
python3 mainpipeline.py
```
Run with options
```
run-typhoon-model [OPTIONS]

Options:
  --path TEXT             main directory
  --remote_dir_ TEXT                  remote directory
  --active_typhoon TEXT               name for active typhoon

```

If you need to inspect the log files, you can find them in `/var/log/cron.log`.
