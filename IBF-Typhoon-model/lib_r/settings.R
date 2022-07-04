crs1 <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0" 
e <- raster::extent(114,145,4,28) # clip data to polygon around PAR

material_variable2 <- read.csv("data-raw/pre_disaster_indicators/material_variable2.csv")
data_matrix_new_variables <- read.csv("data-raw/landuse_stormsurge/data_matrix_new_variables.csv")
geo_variable <- read.csv("data-raw/topography/geo_variable.csv")
grid_points_adm3<-read.csv("data-raw/gis_data/grid_points_admin3_v2.csv", sep=",")

php_admin3 <- geojsonsf::geojson_sf('data-raw/gis_data/phl_admin3_simpl2.geojson')
php_admin1 <- geojsonsf::geojson_sf('data-raw/gis_data/phl_admin1_gadm_pcode.geojson')
php_admin_buffer <- geojsonsf::geojson_sf('data-raw/gis_data/phl_admin1_buffer.geojson')
