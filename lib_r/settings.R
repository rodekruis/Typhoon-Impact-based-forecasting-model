crs1 <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0" 
e <- extent(114,145,4,28) # clip data to polygon around PAR

#main_directory<-'C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/'
#main_directory<-'home/fbf/'

php_admin3 <- st_read(dsn=paste0(main_directory,'data-raw'),layer='phl_admin3_simpl2')
php_admin1 <- st_read(dsn=paste0(main_directory,'data-raw'),layer='phl_admin1_gadm_pcode') 
php_admin_buffer <- st_read(dsn=paste0(main_directory,'data-raw'),layer='phl_admin1_buffer')

material_variable2 <- read.csv(paste0(main_directory,"data/material_variable2.csv"))
data_matrix_new_variables <- read.csv(paste0(main_directory,"data/data_matrix_new_variables.csv"))
geo_variable <- read.csv(paste0(main_directory,"data/geo_variable.csv"))
grid_points_adm3<-read.csv(paste0(main_directory,"data-raw/grid_points_admin3.csv"), sep=",")


