FROM rocker/geospatial:4.1.0

# Set up main directory
RUN mkdir --parents /home/fbf/
WORKDIR /home/fbf/

# Install additional R-packages
RUN Rscript -e \
    "install.packages(c('ggmap', 'RFmarkerDetector', 'kernlab', 'MLmetrics',  'rNOMADS', 'xgboost', 'huxtable'))"

# Install packages needed for Python
RUN apt-get update && apt-get install -y \
    python3-pip \
    libspatialindex-dev \
    python3-eccodes \
    && rm -rf /var/lib/apt/lists/*


# Install additional dependencies with pip
COPY requirements.txt /home/fbf/
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and install
ADD IBF-Typhoon-model .
RUN pip install .
