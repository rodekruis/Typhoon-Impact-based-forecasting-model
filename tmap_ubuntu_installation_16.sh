# install fundamental spatial libraries (needed for sf, sp, rgdal, rgeos)
add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable
apt-get update
apt-get install -y libudunits2-dev libgdal-dev libgeos-dev libproj-dev 

# install v8, needed for the R package V8 which reverse imports geojsonio and rmapshaper -> tmaptools -> tmap
apt-get install -y libv8-3.14-dev

# install jq, needed for the R package jqr whith reverse imports: geojson -> geojsonio -> rmapshaper -> tmaptools -> tmap
add-apt-repository -y ppa:opencpu/jq
apt-get update -q
apt-get install -y libjq-dev

# install libraries needed for the R package protolite, which reverse imports: geojson -> geojsonio -> rmapshaper -> tmaptools -> tmap
apt-get install -y libprotobuf-dev protobuf-compiler

# other libraries
apt-get install -y libssl-dev
apt-get install -y libcairo2-dev