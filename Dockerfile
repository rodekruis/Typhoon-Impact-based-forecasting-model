FROM rocker/r-ubuntu:20.04

# Set up main directory
RUN mkdir --parents /home/fbf/forecast
ENV HOME /home/fbf
WORKDIR $HOME

# Install required system packages
# python3-eccodes is required for python cfgrib
# libproj-dev & libgeos-dev required for python cartopy
# libspatialindex-dev required for python Rtree (which is required for geopandas sjoin)
# libudunits2-dev required for R units (which is required for R tmap)
# libssl-dev required for R s2 (which is required for tmap)
# libgdal-dev required for R sf

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-eccodes \
    libproj-dev \
    libgeos-dev \
    libspatialindex-dev \
    libudunits2-dev \
    libssl-dev \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*


# Install additional R-packages
 RUN Rscript -e \
    "install.packages(c('tmap', 'dplyr', 'tidyr', 'sf', 'geojsonsf', 'raster', \
    'rlang', 'lubridate', 'ncdf4', 'xgboost', 'huxtable', 'readr'))"

# Install Python dependencies
COPY requirements.txt /home/fbf/
RUN pip install --no-cache-dir -r requirements.txt

#Install Jupyter 
RUN pip3 install jupyter

# Copy code and install
ADD IBF-Typhoon-model .
RUN pip install .


# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

#command that starts up the notebook at the end of the Dockerfile.
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]


