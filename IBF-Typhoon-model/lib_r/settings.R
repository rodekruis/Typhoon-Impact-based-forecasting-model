crs1 <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0" 
e <- extent(114,145,4,28) # clip data to polygon around PAR

material_variable2 <- read.csv("data/material_variable2.csv")
data_matrix_new_variables <- read.csv("data/data_matrix_new_variables.csv")
geo_variable <- read.csv("data/geo_variable.csv")
grid_points_adm3<-read.csv("data-raw/grid_points_admin3_v2.csv", sep=",")

php_admin3 <- geojsonsf::geojson_sf('data-raw/phl_admin3_simpl2.geojson')
php_admin1 <- geojsonsf::geojson_sf('data-raw/phl_admin1_gadm_pcode.geojson')
php_admin_buffer <- geojsonsf::geojson_sf('data-raw/phl_admin1_buffer.geojson')
