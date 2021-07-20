FROM ubuntu:18.04
ARG DEBIAN_FRONTEND=noninteractive

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"


# Set up basics
RUN apt-get update
RUN apt-get install -y software-properties-common nano sudo wget
RUN apt-get install -y python3-pip
RUN apt-get install -y python3.7
RUN apt-get update && apt-get install -y libspatialindex-dev

RUN apt-get install -y libgdal-dev
RUN apt-get install -y libproj-dev
RUN apt-get install -y libgeos-dev
RUN apt-get install -y libv8-dev 
RUN apt-get install -y libudunits2-dev 
RUN apt-get install -y libfontconfig1-dev


#RUN apt-get install r-base
#RUN apt-get install r-cran-ncdf4


RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.9.2-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.9.2-Linux-x86_64.sh 
#RUN conda --version


# update pip
RUN python3.7 -m pip install pip --upgrade
RUN python3.7 -m pip install wheel


# copy files
RUN mkdir --parents /home/fbf/
WORKDIR /home/fbf/

# install R and R-packages
RUN apt update -qq
RUN apt install -y apt-transport-https software-properties-common
RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable -y && apt-get update -y
RUN apt-get install -y gdal-bin=3.0.4+dfsg-1~bionic0
#RUN apt-get install -y libgdal-dev=2.2.2+dfsg-1~xenial1 	

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository -y 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/'
RUN apt-get update
RUN apt-get install -y r-base

# Set ...
RUN mkdir ~/.R
COPY Makevars /home/fbf/
RUN cp /home/fbf/Makevars ~/.R/Makevars

# Install R-packages
RUN Rscript -e "install.packages('stringr', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('ggplot2', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('dplyr', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('tidyr', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('gridExtra', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('tmap', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('viridis', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('maps', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('ggmap', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('httr', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('sf', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('geojsonsf', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('raster', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('rgdal', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('ranger', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('caret', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('RFmarkerDetector', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('kernlab', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('MLmetrics', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('plyr', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('lubridate', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('readr', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('rNOMADS', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('xgboost', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('huxtable', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('ncdf4_1.13.tar.gz', dependencies=FALSE, verbose=TRUE, repos=NULL, type='source')"

RUN conda install -c conda-forge proj==7.0.0
RUN conda install -c anaconda pytables
RUN conda install -c conda-forge matplotlib
RUN conda install -c conda-forge matplotlib-base
RUN conda install -c conda-forge cartopy
RUN conda install dask
# Put this here for now but move later
RUN apt-get install -y python3-eccodes
# install python dependencies
COPY requirements.txt /home/fbf/
RUN python3.7 -m pip install --no-cache-dir -r requirements.txt

COPY data /home/fbf/data
COPY data-raw /home/fbf/data-raw
COPY lib /home/fbf/lib
COPY lib_r /home/fbf/lib_r
COPY logos /home/fbf/logos
COPY lib_r /home/fbf/lib_r
COPY models /home/fbf/models
COPY mainpipeline.py /home/fbf/
COPY setup.py /home/fbf/
COPY run_model_V2.R /home/fbf/
COPY __init__.py /home/fbf/
# set up cronjob
# COPY crontab /etc/cron.d/crontab
# RUN chmod 0644 /etc/cron.d/crontab
# RUN crontab /etc/cron.d/crontab
# RUN touch /var/log/cron.log
# CMD cron && tail -f /var/log/cron.log
RUN pip install .


