FROM ubuntu:18.04
ARG DEBIAN_FRONTEND=noninteractive

# Set up basics
RUN apt-get update
RUN apt-get install -y software-properties-common nano sudo wget
RUN apt-get install -y python3-pip
RUN apt-get update && apt-get install -y libspatialindex-dev

# update pip
RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install wheel

# copy files
RUN mkdir --parents /home/fbf/
WORKDIR /home/fbf/

# prerequisite script for R-package 'tmap'
COPY tmap_ubuntu_installation_18.sh  /home/fbf/
RUN chmod +x tmap_ubuntu_installation_18.sh && bash -c "./tmap_ubuntu_installation_18.sh"

COPY ncdf4_1.13.tar.gz  /home/fbf/
R CMD INSTALL /home/fbf/ncdf4_1.9.tar.gz

# install R and R-packages
RUN apt install -y apt-transport-https software-properties-common
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository -y 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
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
RUN Rscript -e "install.packages('raster', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('rgdal', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('ranger', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('caret', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('randomForest', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('rlang', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('RFmarkerDetector', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('AUCRF', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('kernlab', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('ROCR', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('MASS', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('glmnet', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('MLmetrics', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('plyr', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('lubridate', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('readr', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('rNOMADS', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('ncdf4', repos='http://cran.us.r-project.org')"


# install python dependencies
COPY requirements.txt /home/fbf/
RUN pip install -r requirements.txt

# set up cronjob
# COPY crontab /etc/cron.d/crontab
# RUN chmod 0644 /etc/cron.d/crontab
# RUN crontab /etc/cron.d/crontab
# RUN touch /var/log/cron.log
# CMD cron && tail -f /var/log/cron.log



