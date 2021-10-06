
# Table of contents

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Typhoon Impact forecasting model">Typhoon Impact forecasting model</a>
    </li>
	    <li>
      <a href="#Installation">Installation</a>
    </li>
	<li>
      <a href="#Running Without Docker">Running pipeline Without Docker</a>
    </li>
    <li>
      <a href="#Running With Docker">Running pipeline With Docker</a>
      <ul>
        <li><a href="#Build Container">Build and Run Container</a></li>
     </ul>
    </li>
    <li>
	<a href="#Tutorial">Tutorial on Buidling Model</a>
	 <ul>
        <li><a href="#Run Tutorial">Run Jupyter notebook Tutorial from Docker cotainer</a></li>
     </ul>
	</li>
    <li><a href="#Acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- Typhoon Impact forecasting model -->
## Typhoon Impact forecasting model

This tool was developed as a trigger mechanism for the typhoon Early action protocol of the Philippines Red Cross FbF project. The model will predict the potential damage of a typhoon before landfall, and the prediction will be percentage of completely damaged houses per municipality. The tool is available under the [GPL license](https://github.com/rodekruis/Typhoon-Impact-based-forecasting-model/blob/master/LICENSE)

To run the pipeline, you need access to an Data.zip, and credentiials for 510 Datalake and FTP server. If you or your organization is interested in using the pipeline, 
please contact [510 Global](https://www.510.global/contact-us/) to obtain the credentials. You will receive a file called `secrets`, which you need to place in the top-level directory.


<!-- Installation -->
## Installation

1. Clone the repo
2. Change `/IBF-Typhoon-model/src/settings.py.template` to `settings.py` and fill in the necessary passwords.
3. Download [data.zip](https://rodekruis.sharepoint.com/sites/510-CRAVK-510/_layouts/15/guestaccess.aspx?guestaccesstoken=HadTB1h%2FWiVluDiortTyd3%2F9rSc0MdjS2yub9GEntCs%3D&docid=2_0013b102f095246fdab4ff4ce03b12933&rev=1&e=eDGoN5) from sharepoint and unzip in /IBF-Typhoon-model/data
4. Install Docker-compose [follow this link](https://docs.docker.com/desktop/windows/install/)
5. Enamble [volume sharing](https://forums.docker.com/t/filesharing-not-enabled-volume-sharing-is-not-enabled-on-the-settings-screen-in-docker-desktop-error-125/95332) for docker compose  



<!-- Running pipeline Without Docker -->
## Running Without Docker


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

<!-- Running Pipeline With Docker -->
## Running With Docker

You will need to have `docker` and `docker-compose` installed.
You need to create an environment variable called `TYPHOONMODEL_OUTPUT` that contains
the path to where you would like the model run output data to go.

<!-- Build and Run Container -->
### Build Container

To build and run the image, ensure you are in the top-level directory and execute:
```
docker-compose up --build


```
When you are finished, run
```
docker-compose down
```
to remove any docker container(s).


<!-- Tutorials -->
## Tutorial


[Here](https://github.com/rodekruis/Typhoon-Impact-based-forecasting-model/tree/model_dev/IBF-Typhoon-model/documentation) you can find a guideline, a Jupyter Notebook Tutorial. Folow the step in this page to understand the workflow used to build the model 

<!-- Run Jupyter notebook Tutorial from Docker cotainer-->
### Run Tutorial

This will be running inside the docker container

Excute:

```
docker run -p 8888:8888 fbf-phv3
```
<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

- [Germen Red Cross](https://www.drk.de/en/)
- [Philippines Red Cross](https://redcross.org.ph/)
- [Rode Kruis](https://www.rodekruis.nl/)
