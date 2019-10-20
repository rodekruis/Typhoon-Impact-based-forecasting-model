library(stringr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(tmap)
library(viridis)
library(maps)
library(ggmap)
library(httr)
library(sf)
source('C:/documents/philipiness/Typhoons/model/new_model/typhoon_track2point.R')

#------------------------- new typhoon ---------------------------------
# ------------------------ Read data -----------------------------------
# track wind should have a unit of knots
# read data
TRACK_DATA <- read.csv("C:/documents/philipiness/Typhoons/model/new_model/new_typhoon.csv")
grid_points_adm3<-read.csv("C:/documents/philipiness/Typhoons/model/new_model/grid_points_admin3.csv", sep=",")

TRACK_DATA<- TRACK_DATA %>% 
  dplyr::mutate(typhoon = paste0(TRACK_DATA$STORMNAME,substr(TRACK_DATA$YYYYMMDDHH, 1, 4)))

landmask <- readr::read_csv("C:/documents/philipiness/Typhoons/model/new_model/stormwindmodel/data-raw/landseamask_ph1.csv",col_names = c("longitude", "latitude", "land")) %>%
  dplyr::mutate(land = factor(land, levels = c(1, 0), labels = c("land", "water")))

TRACK_DATA1<-TRACK_DATA

library(raster)
library(rgdal)  ## 

##
mean <- brick("C://documents//philipiness//temp//apcp_sfc_2019101700_mean.grib2")
con <- brick("C://documents//philipiness//temp//apcp_sfc_2019101800_c00.grib2")

names(con) = paste("rain",outer(1:45,'0',paste,sep="-"),sep="-")
 

crs1 <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0" 
wshade <- readOGR('C:/documents/philipiness/Typhoons/model/new_model/data-raw', layer = 'phl_admin3_simpl2')
wshade <- spTransform(wshade, crs1)

e <- extent(110,150,5,40) # clip data to polygon around PAR

ewataSub <- crop(con, e)

rainfall <-extract(ewataSub, y=wshade, method='bilinear',fun=max,df=TRUE,sp=TRUE) 

rainfall_<-rainfall %>%
  dplyr::mutate(Mun_Code=adm3_pcode,rainfall_24h=rain.1.0 + rain.2.0 + rain.3.0 + rain.4.0 + rain.5.0 + rain.6.0+rain.7.0 + rain.8.0 + rain.9.0 + rain.10.0 + rain.11.0 + rain.12.0) %>%
  dplyr::select(Mun_Code,rainfall_24h)


# ------------------------ calculate  variables  -----------------------------------

fulltrack<-create_full_track(hurr_track=TRACK_DATA1, tint = 6)

Mydata<-fulltrack %>%
  dplyr::mutate(index= 1:nrow(fulltrack))%>%
  dplyr::select(tclat,tclon,index,typhoon_name,vmax,date)


my_track <- st_as_sf(Mydata, coords = c("tclon", "tclat"), crs = "+init=epsg:4326")


wind_grids <- get_grid_winds(hurr_track=TRACK_DATA1, grid_df=grid_points_adm3, tint = 3,gust_duration_cut = 15, sust_duration_cut = 15)

wind_grids<- wind_grids %>%
  dplyr::mutate(typhoon_name=as.vector(TRACK_DATA$STORMNAME)[1],fips = as.numeric(substr(gridid, 3, 11)),vmax_gust_mph=vmax_gust*2.23694,vmax_sust_mph=vmax_sust*2.23694) %>%
  dplyr::select(typhoon_name,fips,gridid,vmax_gust,vmax_gust_mph,vmax_sust,vmax_sust_mph,gust_dur,sust_dur,dist_track,glon,glat)


wind_grid<-wind_grids%>%
  dplyr::mutate(Mun_Code=gridid)%>%
  dplyr::select(-fips,-gridid)




# BUILD DATA MATRIC FOR NEW TYPHOON 

setwd('C://documents//philipiness//Typhoons//model//new_model')

material_variable2 <- read.csv("data/material_variable2.csv")
data_matrix_new_variables <- read.csv("data/data_matrix_new_variables.csv")
geo_variable <- read.csv("data/geo_variable.csv")
data_new_typhoon<-geo_variable %>%
  left_join(material_variable2 %>% dplyr::select(-Region,-Province,-Municipality_City), by = "Mun_Code") %>%
  left_join(data_matrix_new_variables , by = "Mun_Code") %>%
  left_join(wind_grid , by = "Mun_Code")  %>%
  left_join(rainfall_ , by = "Mun_Code")

source('C:/documents/philipiness/Typhoons/model/new_model/data_cleaning_forecast.R')
data_new_typhoon<-clean_typhoon_forecast_data(data_new_typhoon)

data <- data_new_typhoon %>%
  dplyr::select(-GEN_typhoon_name,
    #-GEN_mun_code,
    -contains('DAM_'),
    -GEN_mun_name) %>%
  na.omit() # Randomforests don't handle NAs, you can impute in the future 




# load the model
RANDOMFOREST_model <- readRDS("C://documents//philipiness//Typhoons//model//new_model//final_model.rds")



rm.predict.pr <- predict(RANDOMFOREST_model, data = data, predict.all = FALSE, num.trees = optimal_model$num.trees, type = "response",
                         se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)


FORECASTED_IMPACT<-as.data.frame(rm.predict.pr$predictions) %>%
  dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact=rm.predict.pr$predictions) %>%
  left_join(data , by = "index") %>%
  dplyr::select(GEN_mun_code,impact)

#write.csv(data_new_typhoon, file = 'C:/documents/philipiness/Typhoons/model/new_model/output/new_typhoon.csv')

# load the rr model
RANDOMFOREST_model <- readRDS("C://documents//philipiness//Typhoons//model//new_model//final_model_regression.rds")

rm.predict.pr <- predict(RANDOMFOREST_model, data = df, predict.all = FALSE, num.trees = optimal_model$num.trees, type = "response",
                         se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)
FORECASTED_IMPACT_rr<-as.data.frame(rm.predict.pr$predictions) %>%
  dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact=rm.predict.pr$predictions) %>%
  left_join(df , by = "index") %>%
  dplyr::select(GEN_mun_code,impact)


# ------------------------ import boundary layer   -----------------------------------
php_admin3 <- st_read(dsn='C:/documents/philipiness/temp/stormwindmodel/data-raw',layer='phl_admin3_simpl2')


php_admin3<-php_admin3 %>%
  dplyr::mutate(long = glon,lat=glat,group=as.numeric(substr(adm2_pcode, 3, 6)),GEN_mun_code=adm3_pcode,fips=as.numeric(substr(adm3_pcode, 3, 11)))%>%
  dplyr::select(adm3_en,GEN_mun_code,adm2_pcode,fips,long,lat,group,geometry)

php_data2<-php_admin3 %>%
  left_join(FORECASTED_IMPACT, by = "GEN_mun_code")

my_track<-my_track %>%
  dplyr::mutate(date2 =format(my_track$date,"%B %d")) #, %H

# tmap
tmap_mode(mode = "view")
tm_shape(php_data2) + tm_polygons(col = "vmax_sust_mph", border.col = "black",lwd = 0.1,lyt='dotted')+
  tm_shape(my_track) + tm_symbols(size=0.05,border.alpha = .5,col="red") +
  #tm_shape(my_track) + tm_symbols(size=0.01,border.alpha = 1,col="red") +
  tm_text("date2", col="date2",legend.col.show = F,clustering = TRUE) +
  tm_view(set.view = c(lon = 122, lat = 20, zoom = 5)) +
  tm_format("World")
# tmap

#write.table(track_full_grid, file = "C:/SOFTWARES/stormwindmodel/track_full_grid.csv",row.names=FALSE, na="", sep=",")

st_write(my_track, dsn = "C:/documents/philipiness/temp/output/full_track.shp", layer='full_track')

