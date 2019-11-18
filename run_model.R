#!/usr/bin/env Rscript
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

library(raster)
library(rgdal)  ## 
################################

library(ranger)
library(caret)
library(randomForest)
library(rlang)
library(RFmarkerDetector)
library(AUCRF)
library(kernlab)
library(ROCR)
library(MASS)
library(glmnet)
library(MLmetrics)
library(plyr)

library(lubridate)

#------------------------- new typhoon ---------------------------------
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
# ------------------------ calculate  variables  -----------------------------------
typhoon_events <- read.csv('forecast/typhoon_info_for_model.csv')
grid_points_adm3<-read.csv("data-raw/grid_points_admin3.csv", sep=",")
# ------------------------ import boundary layer   -----------------------------------
php_admin3 <- st_read(dsn='data-raw',layer='phl_admin3_simpl2')
php_admin1 <- st_read(dsn='data-raw',layer='phl_admin1_gadm_pcode')

source('prepare_typhoon_input.R')

for(i in 1:nrow(typhoon_events)){
  
  TRACK_DATA<- read.csv(as.character(typhoon_events$filename[i]))
  
  TRACK_DATA1<- TRACK_DATA %>% 
    dplyr::mutate(typhoon = paste0(TRACK_DATA$STORMNAME,substr(TRACK_DATA$YYYYMMDDHH, 1, 4)))
  
  fulltrack<-create_full_track(hurr_track=TRACK_DATA1, tint = 3)
  
  Mydata<-fulltrack %>% 
    dplyr::mutate(index= 1:nrow(fulltrack))%>%
    dplyr::select(tclat,tclon,index,typhoon_name,vmax,date)
  
  my_track <- st_as_sf(Mydata, coords = c("tclon", "tclat"), crs = "+init=epsg:4326")
  
  # check if there is a landfall
  Landfall_check <- st_intersection(php_admin1, my_track)
  
  if (nrow(Landfall_check) > 0){
    
    Typhoon_stormname<-as.character(typhoon_events$event[i])
    Landfall_point<-Landfall_check[1,]
    
    #######################################################################
    # caculate time for landfll
    
    dt<-lubridate::with_tz(lubridate::ymd_hms(format(Landfall_check$date[1], format="%Y-%m-%d %H:%M:%S")), "PST8PDT")
    dt1<-lubridate::with_tz(lubridate::ymd_hm(format(TRACK_DATA$YYYYMMDDHH[1], format="%Y%m%d%H%M")), "PST8PDT")
    time_for_landfall<- as.numeric(difftime(dt,dt1,units="hours"))
    
    
    wind_grids <- get_grid_winds(hurr_track=TRACK_DATA1, grid_df=grid_points_adm3, tint = 3,gust_duration_cut = 15, sust_duration_cut = 15)
    
    wind_grids<- wind_grids %>%
      dplyr::mutate(typhoon_name=as.vector(TRACK_DATA$STORMNAME)[1],fips = as.numeric(substr(gridid, 3, 11)),vmax_gust_mph=vmax_gust*2.23694,vmax_sust_mph=vmax_sust*2.23694) %>%
      dplyr::select(typhoon_name,fips,gridid,vmax_gust,vmax_gust_mph,vmax_sust,vmax_sust_mph,gust_dur,sust_dur,dist_track,glon,glat)
    
    
    wind_grid<-wind_grids%>%
      dplyr::mutate(Mun_Code=gridid)%>%
      dplyr::select(-fips,-gridid)
    
    
    
    # track wind should have a unit of knots
    
    
    ##
    
    NOAA_rain <- brick("forecast/rainfall_forecast.grib2")
    
    names(NOAA_rain) = paste("rain",outer(1:45,'0',paste,sep="-"),sep="-")
    
    
    crs1 <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0" 
    wshade <- readOGR('data-raw', layer = 'phl_admin3_simpl2')
    wshade <- spTransform(wshade, crs1)
    
    e <- extent(110,150,5,40) # clip data to polygon around PAR
    
    NOAA_rain <- crop(NOAA_rain, e)
    
    rainfall <-extract(NOAA_rain, y=wshade, method='bilinear',fun=max,df=TRUE,sp=TRUE) 
    
    rainfall_<-rainfall@data %>%
      dplyr::mutate(Mun_Code=adm3_pcode,rainfall_24h=rain.1.0 + rain.2.0 + rain.3.0 + rain.4.0 + rain.5.0 + rain.6.0+rain.7.0 + rain.8.0 + rain.9.0 + rain.10.0 + rain.11.0 + rain.12.0) %>%
      dplyr::select(Mun_Code,rainfall_24h)
    
    
    
    
    # BUILD DATA MATRIC FOR NEW TYPHOON 
    
    setwd('C:/documents/philipiness/Typhoons/model/new_model/input')
    
    material_variable2 <- read.csv("data/material_variable2.csv")
    data_matrix_new_variables <- read.csv("data/data_matrix_new_variables.csv")
    geo_variable <- read.csv("data/geo_variable.csv")
    data_new_typhoon<-geo_variable %>%
      left_join(material_variable2 %>% dplyr::select(-Region,-Province,-Municipality_City), by = "Mun_Code") %>%
      left_join(data_matrix_new_variables , by = "Mun_Code") %>%
      left_join(wind_grid , by = "Mun_Code")  %>%
      left_join(rainfall_ , by = "Mun_Code")
    
    source('C:/documents/philipiness/Typhoons/model/new_model/input/data_cleaning_forecast.R')
    data_new_typhoon<-clean_typhoon_forecast_data(data_new_typhoon)
    
    data <- data_new_typhoon %>%
      dplyr::select(-GEN_typhoon_name,
                    -GEO_n_households,
                    #-GEN_mun_code,
                    -contains('DAM_'),
                    -GEN_mun_name) %>%
      na.omit() # Randomforests don't handle NAs, you can impute in the future 
    
    
    
    
    
    
    ##############################################################################
    # load the rr model
    
    mode_classification <- readRDS("C:/documents/philipiness/Typhoons/model/new_model/input/models/final_model.rds")
    
    rm.predict.pr <- predict(mode_classification, data = data, predict.all = FALSE, num.trees = mode_classification$num.trees, type = "response",
                             se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)
    
    FORECASTED_IMPACT<-as.data.frame(rm.predict.pr$predictions) %>%
      dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact_threshold_passed=rm.predict.pr$predictions) %>%
      left_join(data , by = "index") %>%
      dplyr::select(GEN_mun_code,impact_threshold_passed,WEA_dist_track)
    
    
    
    mode_continious <- readRDS("C:/documents/philipiness/Typhoons/model/new_model/input/models/final_model_regression.rds")
    
    rm.predict.pr <- predict(mode_continious, data = data, predict.all = FALSE, num.trees = mode_continious$num.trees, type = "response",
                             se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)
    
    FORECASTED_IMPACT_rr<-as.data.frame(rm.predict.pr$predictions) %>%
      dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact=rm.predict.pr$predictions) %>%
      left_join(data , by = "index") %>%
      dplyr::select(GEN_mun_code,impact,WEA_dist_track)
    
    
    
    
    php_admin3<-php_admin3 %>%  dplyr::mutate(GEN_mun_code=adm3_pcode)%>%  dplyr::select(adm3_en,GEN_mun_code,geometry)
    
    php_admin3<-php_admin3 %>%  left_join(FORECASTED_IMPACT, by = "GEN_mun_code")
    php_admin3<-php_admin3 %>%  left_join(FORECASTED_IMPACT_rr, by = "GEN_mun_code")
    
    php_admin3<-php_admin3 %>% dplyr::arrange(WEA_dist_track.x) %>%
      dplyr::mutate(impact=ifelse(WEA_dist_track.x > 100,NA,impact),impact_threshold_passed =ifelse(WEA_dist_track.x > 100,NA,impact_threshold_passed))
    
    Impact<-php_admin3  %>%   dplyr::select(GEN_mun_code,impact,impact_threshold_passed,WEA_dist_track.x)
    Impact <- Impact %>% st_set_geometry(NULL)
    
    ####################################################################################################
    #  calculate damage quantile. ----
    
    
    ntile_na <- function(x,n)
    {
      notna <- !is.na(x)
      out <- rep(NA_real_,length(x))
      out[notna] <- ntile(x[notna],n)
      return(out)
    }
    
    
    php_admin4 <- php_admin3 %>% 
      mutate(dam_perc_comp_prediction_lm_quantile = ntile_na(impact,5))  %>% filter(WEA_dist_track.x < 300)
    
    region2<-extent(php_admin4)
    php_admin3<-php_admin3 %>% arrange(WEA_dist_track.x)
    #region<- st_bbox(php_admin3[1,])
    #typhoon_region = st_bbox(c(xmin = region[["xmin"]]-2, xmax = 2+region[["xmax"]],  ymin = region[["ymin"]]-2, ymax =2+ region[["ymax"]]),     crs = st_crs(php_admin1)) %>% st_as_sfc()
    
    typhoon_region = st_bbox(c(xmin =as.vector(region2@xmin), xmax = as.vector(region2@xmax),
                               ymin = as.vector(region2@ymin), ymax =as.vector(region2@ymax)),
                             crs = st_crs(php_admin1)) %>% st_as_sfc()
    
    
    
    
    # ------------------------ make a map   -----------------------------------
    
    
    
    
    subtitle =paste0("Predicted damage per Manucipality
Source of wind speed forecast Tropical Storm Risk (UCL)
Only municipalities within 100km of forecasted typhoon track are included
Prediction is about completely damaged houses only\n",'Estimated time to Landfall: ',dt,' PST (',time_for_landfall,' hrs)')
    
    tmap_mode(mode = "plot")
    impact_map= 
      tm_shape(php_admin4) + 
      tm_fill(col = "impact",showNA=FALSE, border.col = "black",lwd = 3,lyt='dotted',
              breaks = c(0,0.1,1,2,5,9.5,10),
              title='Predicted % of Damaged ',
              labels=c(' No Damage',' < 1%',' 1 to 2%',' 2 to 5%',' 5 to 10%',' > 10%'),
              palette = c('#ffffff','#fdd0a2','#fdae6b','#fd8d3c','#e6550d','#a63603')) + #,style = "cat")+
      
      tm_borders(col = NA, lwd = .25, lty = "solid", alpha = .25, group = NA) +
      #tm_polygons(col = "dam_perc_comp_prediction_lm_quantile", border.col = "black",lwd = 0.1,lyt='dotted',style = "cat")+
      tm_shape(my_track) + tm_symbols(size=0.1,border.alpha = .25,col="blue") +
      tm_shape(Landfall_point) + tm_symbols(size=0.25,border.alpha = .25,col="red") +   
      tm_scale_bar(breaks = c(0, 100, 200), text.size = .5,
                   color.light = "#f0f0f0",
                   position = c(0,0))+
      tm_layout(legend.outside= TRUE, legend.outside.position=c("left"),
                main.title=subtitle, main.title.size=.8,asp=.8)
    
    ph_map = tm_shape(php_admin1) + tm_polygons(border.col = "black",lwd = 0.1,lyt='dotted',alpha =0.2) +
      tm_shape(typhoon_region) + tm_borders(lwd = 2,col='red') +
      tm_compass(type = "8star", position = c("left", "top")) +
      tm_credits("The maps used do not imply \nthe expression of any opinion on \nthe part of the International Federation\n of the Red Cross and Red Crescent Societies\n concerning the legal status of a\n territory or of its authorities.",
                 position = c("left", "BOTTOM"),size = 0.9) +
      tm_layout(inner.margins=c(.04,.03, .02, .01)) 
    
    map1<- tmap_arrange(impact_map,ph_map,ncol = 2,widths = c(.8, .2))
    
    
    
    
    # ------------------------ save to file   -----------------------------------
    
    ## save an image ("plot" mode)
    tmap_save(map1, filename = paste0('C:/documents/philipiness/Typhoons/model/new_model/input/forecast/Impact_',as.vector(TRACK_DATA[1,]$YYYYMMDDHH),'_',as.vector(TRACK_DATA[1,]$STORMNAME),'.png'))
    write.csv(Impact, file = paste0('C:/documents/philipiness/Typhoons/model/new_model/input/forecast/Impact_',as.vector(TRACK_DATA[1,]$YYYYMMDDHH),'_',as.vector(TRACK_DATA[1,]$STORMNAME),'.csv'))
    
    file_names<- c(paste0('C:/documents/philipiness/Typhoons/model/new_model/input/forecast/Impact_',as.vector(TRACK_DATA[1,]$YYYYMMDDHH),'_',as.vector(TRACK_DATA[1,]$STORMNAME),'.png'),
                   paste0('C:/documents/philipiness/Typhoons/model/new_model/input/forecast/Impact_',as.vector(TRACK_DATA[1,]$YYYYMMDDHH),'_',as.vector(TRACK_DATA[1,]$STORMNAME),'.csv'))
    
    write.table(file_names, file = 'forecast/file_names.csv',append=TRUE, col.names = FALSE)
    
  }
  
  
  else{
    
    Typhoon_stormname<-as.character(typhoon_events$event[i])
    
    file_names<- c(paste0('Nolandfall','_',Typhoon_stormname,'.png'),
                   paste0('Nolandfall','_',Typhoon_stormname,'.csv'))
    
    
    
    write.table(file_names, file = 'C:/documents/philipiness/Typhoons/model/new_model/input/forecast/file_names.csv',append=TRUE, col.names = FALSE)
    
    
  }
  
  
}

