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
to obtain the credentials.  You will receive a file called `secrets`, which you need to place in 
the top-level directory.

### Without Docker

The main code for the pipeline is `IBF-Typhoon-model/src/`, which can in principle be run locally,
however we do recommend using Docker if you can.
To run locally:

1. Enter the `IBF-Typhoon-model` directory and install the required Python packages, as
    well as the `typhoonmodel` package:
    ```
    pip install -r requirements.pip
    pip install .
    ```
2. Install all packages listed at the top of the R script `run_model_v2.R`.
3. Ensure that all the parameters from the `secrets` file have been exported as environment variables
4. Execute:
    ```
    run-typhoon-model [OPTIONS]

    Options:
      --path TEXT             main directory defult 
      --remote_directory TEXT                  remote directory 
      --typhoonname TEXT               name for active typhoon
    ```
    When there is an active typhoon in the PAR polygon the model will run for the active typhoons,
    unless you specify a remote directory and typhoon name. 

### With Docker

You will need to have `docker` and `docker-compose` installed.
You need to create an environment variable called `TYPHOONMODEL_OUTPUT` that contains
the path to where you would like the model run output data to go.

#### Running 

To build and run the image, ensure you are in the top-level directory and execute:
```
docker-compose up --build


```
When you are finished, run
```
docker-compose down
```
to remove any docker container(s).

### To run model Documentation 
Excute:

```
docker run -p 8888:8888 fbf-phv3
```
