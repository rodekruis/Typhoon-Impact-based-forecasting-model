#!/usr/bin/env Rscript
#options(warn=-1)
suppressMessages(library(stringr))
suppressMessages(library(ggplot2))
suppressMessages(library(dplyr))
suppressMessages(library(tidyr))
suppressMessages(library(gridExtra))
suppressMessages(library(tmap))
suppressMessages(library(viridis))
suppressMessages(library(maps))
suppressMessages(library(ggmap))
suppressMessages(library(httr))
suppressMessages(library(sf))
suppressMessages(library(raster))
suppressMessages(library(rgdal))
suppressMessages(library(ranger))
suppressMessages(library(caret))
suppressMessages(library(randomForest))
suppressMessages(library(rlang))
suppressMessages(library(AUCRF))
suppressMessages(library(kernlab))
suppressMessages(library(ROCR))
suppressMessages(library(MASS))
suppressMessages(library(glmnet))
suppressMessages(library(MLmetrics))
suppressMessages(library(plyr))
suppressMessages(library(lubridate))
suppressMessages(library(rNOMADS))
suppressMessages(library(ncdf4))

#args = commandArgs(trailingOnly=TRUE)
#rainfall_error = args[1]

path='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/'
#path='home/fbf/'
####################################################################################################

setwd(path)

source('settings.R')
source('data_cleaning_forecast.R')
source('prepare_typhoon_input.R')

#source('home/fbf/prepare_typhoon_input.R')
#source('home/fbf/settings.R')
#source('home/fbf/data_cleaning_forecast.R')


###########################################################################
# ------------------------ import DATA  -----------------------------------

php_admin3 <- st_read(dsn=paste0(main_directory,'data-raw'),layer='phl_admin3_simpl2')
php_admin1 <- st_read(dsn=paste0(main_directory,'data-raw'),layer='phl_admin1_gadm_pcode')
wshade <- php_admin3
material_variable2 <- read.csv(paste0(main_directory,"data/material_variable2.csv"))
data_matrix_new_variables <- read.csv(paste0(main_directory,"data/data_matrix_new_variables.csv"))
geo_variable <- read.csv(paste0(main_directory,"data/geo_variable.csv"))
grid_points_adm3<-read.csv(paste0(main_directory,"data-raw/grid_points_admin3.csv"), sep=",")

# load the rr model
mode_classification <- readRDS(paste0(main_directory,"models/final_model.rds"))
mode_continious <- readRDS(paste0(main_directory,"models/final_model_regression.rds"))



# load forecast data
typhoon_info_for_model <- read.csv(paste0(main_directory,"/forecast/Input/typhoon_info_for_model.csv"))
#typhoon_events <- read.csv(paste0(main_directory,'/forecast/Input/typhoon_info_for_model.csv')) 

rain_directory<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='Rainfall',]$filename)
UCL_<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='UCL',]$filename)
ECMWF_<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='ECMWF',]$filename)
#HK_<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='HK',]$filename)
#JTCW_<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='JTCW',]$filename)

Typhoon_stormname<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='UCL',]$event)

forecast_time<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='UCL',]$time)
forecast_time<-str_remove_all(forecast_time, "'")

rain_directory<-ifelse(identical(character(0), rain_directory),NULL,rain_directory)
UCL_directory<-ifelse(identical(character(0), UCL_),NA,UCL_)
ECMWF_directory<-ifelse(identical(character(0), ECMWF_),NA,ECMWF_)
#HK_directory<-ifelse(identical(character(0), HK_),NA,HK_)
#JTCW_directory<-ifelse(identical(character(0), JTCW_),NA,JTCW_)






####################################################################################################





#------------------------- define functions ---------------------------------

ntile_na <- function(x,n){
  notna <- !is.na(x)
  out <- rep(NA_real_,length(x))
  out[notna] <- ntile(x[notna],n)
  return(out)
}

####################################################################################################
########### track interpolation

track_interpolation<- function(TRACK_DATA){
  
  ########### track_interpolation ###################
  TRACK_DATA<- TRACK_DATA %>% 
    dplyr::mutate(typhoon = paste0(TRACK_DATA$STORMNAME,substr(TRACK_DATA$YYYYMMDDHH, 1, 4)))
  
  fulltrack<-create_full_track(hurr_track=TRACK_DATA, tint = 1)
  
  Mydata<-fulltrack %>% dplyr::mutate(index= 1:nrow(fulltrack))%>%
    dplyr::select(tclat,tclon,index,typhoon_name,vmax,date)
  
  my_track <- st_as_sf(Mydata, coords = c("tclon", "tclat"), crs = "+init=epsg:4326")
  
  return(my_track)
  
}

####################################################################################################
########### rainfall download function 

Read_rainfall_v2<-function(wshade){
  
  ########### Read_rainfall ###################
  
  
  if(file.exists(paste0(rain_directory, "/rainfall_24.nc") )) {
    
    e <- extent(114,145,4,28) # clip data to polygon around 
    
    #NOAA_rain<-xx*3600*3 # unit conversion kg/m2/s to mm/3hr
    
    #NOAA_rain <- brick(paste0(rain_directory, "/rainfall_06.nc"))
    #names(NOAA_rain) = paste("rain",outer(1:12,'0',paste,sep="-"),sep="-")
    #NOAA_rain <- crop(NOAA_rain, e)
    
    #rainfall <-raster::extract(NOAA_rain, y=wshade, method='bilinear',fun=max,df=TRUE,sp=TRUE) 
    #rainfall_1<-rainfall@data %>%
    #  dplyr::mutate(Mun_Code=adm3_pcode,rain_6h_max=apply(rainfall@data[,7:ncol(rainfall@data)], 1, FUN=max),
    #                rain_cum=apply(rainfall@data[,7:ncol(rainfall@data)], 1, FUN=sum)) %>%
    #  dplyr::select(Mun_Code,rain_6h_max,rain_cum)
    
    NOAA_rain <- brick(paste0(rain_directory, "/rainfall_24.nc"))
    
    NOAA_rain <- crop(NOAA_rain, e)
    
    rainfall <-raster::extract(NOAA_rain, y=wshade, method='bilinear',fun=max,df=TRUE,sp=TRUE)
    
    rainfall_<-rainfall@data %>%
      dplyr::mutate(Mun_Code=adm3_pcode,rainfall_24h=apply(rainfall@data[,7:ncol(rainfall@data)], 1, FUN=max)) %>%
      dplyr::select(Mun_Code,rainfall_24h) #%>% left_join(rainfall_1,by='Mun_Code')
    
    
  }
  
  else{
    
    urls.out <- CrawlModels(abbrev = "gfs_0p25", depth = 2) # to avoid error if GFS model out put is not yet uploaded we use GFS model results for previous time step
    model.parameters <- ParseModelPage(urls.out[2])  #Get a list of forecasts, variables and levels
    levels <- c("surface") #What region of the atmosphere to get data for
    variables <- c("PRATE")  #What data to return
    dir.create(file.path("forecast", "rainfall"), showWarnings = FALSE)
    
    # Get the data
    for (i in 2:length(head(model.parameters$pred,n=72))){
      grib.info <- GribGrab(urls.out[2], model.parameters$pred[i], levels, variables,local.dir = paste0(path, "/forecast/rainfall"))
    }
    file_list <- list.files(path=paste0(path, "/forecast/rainfall"))
    xx <- raster::stack()   # Read each grib file to a raster and stack it to xx  
    
    for (files in file_list)  {
      fn <- file.path(paste0(path, "/forecast/rainfall"), files)
      r2 <- raster(fn)
      x1 <- crop(r2, e)
      xx <- raster::stack(xx, x1)
    }     
    xx[xx < 0] <- NA  # Remove noise from the data
    
    NOAA_rain<-xx*3600*3 # unit conversion kg/m2/s to mm/3hr
    names(NOAA_rain) = paste("rain",outer(1:length(file_list),'0',paste,sep="-"),sep="-")
    rainfall <-extract(NOAA_rain, y=wshade, method='bilinear',fun=max,df=TRUE,sp=TRUE) 
    rainfall_<-rainfall@data %>%
      dplyr::mutate(Mun_Code=adm3_pcode,rainfall_24h=rain.1.0 + rain.2.0 + rain.3.0 + rain.4.0 + rain.5.0 + rain.6.0+rain.7.0 + rain.8.0 + rain.9.0 + rain.10.0 + rain.11.0 + rain.12.0) %>%
      dplyr::select(Mun_Code,rainfall_24h)
  } 
  
  
  return(rainfall_)
  
}


####################################################################################################


Model_input_processing<-function(TRACK_DATA,my_track,TYF,Typhoon_stormname){
  
  ###################### Model_input_processing ##############
  
  #generate tack with smaller time steps 
  my_track <- my_track
  
  Typhoon_stormname<-Typhoon_stormname
  
  
  
  
  TRACK_DATA1<- TRACK_DATA %>% dplyr::mutate(typhoon = paste0(TRACK_DATA$STORMNAME,substr(TRACK_DATA$YYYYMMDDHH, 1, 4)))
  
  wind_grids <- get_grid_winds(hurr_track=TRACK_DATA1, grid_df=grid_points_adm3, tint = 3,gust_duration_cut = 15, sust_duration_cut = 15)
  
  wind_grids<- wind_grids %>%
    dplyr::mutate(typhoon_name=as.vector(TRACK_DATA$STORMNAME)[1],fips = as.numeric(substr(gridid, 3, 11)),vmax_gust_mph=vmax_gust*2.23694,vmax_sust_mph=vmax_sust*2.23694) %>%
    dplyr::select(typhoon_name,fips,gridid,vmax_gust,vmax_gust_mph,vmax_sust,vmax_sust_mph,gust_dur,sust_dur,dist_track,glon,glat)
  
  wind_grid<-wind_grids%>%
    dplyr::mutate(Mun_Code=gridid)%>%
    dplyr::select(-fips,-gridid)
  
  # track wind should have a unit of knots
  rainfall_<-Read_rainfall_v2(wshade)
  
  
  # BUILD DATA MATRIC FOR NEW TYPHOON 
  data_new_typhoon<-geo_variable %>%
    left_join(material_variable2 %>% dplyr::select(-Region,-Province,-Municipality_City), by = "Mun_Code") %>%
    left_join(data_matrix_new_variables , by = "Mun_Code") %>%
    left_join(wind_grid , by = "Mun_Code")  %>%
    left_join(rainfall_ , by = "Mun_Code")
  
  # source('/home/fbf/data_cleaning_forecast.R')
  data_new_typhoon<-clean_typhoon_forecast_data(data_new_typhoon)
  
  data <- data_new_typhoon %>%
    dplyr::select(-GEN_typhoon_name,
                  -GEO_n_households,
                  #-GEN_mun_code,
                  -contains('DAM_'),
                  -GEN_mun_name) %>%
    na.omit() # Randomforests don't handle NAs, you can impute in the future 
  return(data)
}

####################################################################################################


Check_landfall_time<-function(php_admin1, my_track){
  
  ########## check if there is a landfall #####
  
  Landfall_check <- st_intersection(php_admin1, my_track)
  
  if (nrow(Landfall_check) > 0){
    
    Typhoon_stormname<-Typhoon_stormname
    
    Landfall_check <- st_intersection(php_admin1, my_track)
    Landfall_point<-Landfall_check[1,]
    
    #######################################################################
    # caculate time for landfll
    
    dt<-lubridate::with_tz(lubridate::ymd_hms(format(Landfall_check$date[1], format="%Y-%m-%d %H:%M:%S"),tz="UTC"), tz="Asia/Manila")
    
    dt1<-lubridate::with_tz(lubridate::ymd_hm(format(TRACK_DATA$YYYYMMDDHH[1], format="%Y%m%d%H%M"),tz="UTC"), tz="Asia/Manila")
    
    dt2<-lubridate::with_tz(lubridate::ymd_hm(format(Sys.time(), format="%Y%m%d%H%M"),tz="CEST"), tz="Asia/Manila")
    
    dt2=lubridate::with_tz(lubridate::force_tz(Sys.time()), tz="Asia/Manila")
    
    time_for_landfall<- as.numeric(difftime(dt,dt1,units="hours"))
    etimated_landfall_time<-dt
    
  }
  return(c(Landfall_point,time_for_landfall,etimated_landfall_time))
  
}



####################################################################################################

run_prediction_model<-function(data){
  
  ########## run prediction ##########
  
  rm.predict.pr <- predict(mode_classification, data = data, predict.all = FALSE, num.trees = mode_classification$num.trees, type = "response",
                           se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)
  
  FORECASTED_IMPACT<-as.data.frame(rm.predict.pr$predictions) %>%
    dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact_threshold_passed=rm.predict.pr$predictions) %>%
    left_join(data , by = "index") %>%
    dplyr::select(GEN_mun_code,impact_threshold_passed,WEA_dist_track)
  
  #colnames(FORECASTED_IMPACT) <- c(GEN_mun_code,paste0(TYF,'_impact_threshold_passed'),WEA_dist_track)
  
  rm.predict.pr <- predict(mode_continious, data = data, predict.all = FALSE, num.trees = mode_continious$num.trees, type = "response",
                           se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)
  
  FORECASTED_IMPACT_rr<-as.data.frame(rm.predict.pr$predictions) %>%
    dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact=rm.predict.pr$predictions) %>% 
    dplyr::mutate(priority_index=ntile_na(ifelse(impact<0,NA,impact),5))%>%
    left_join(data , by = "index") %>%
    dplyr::select(GEN_mun_code,impact,priority_index)
  
  df_imact_forecast<-FORECASTED_IMPACT %>% left_join(FORECASTED_IMPACT_rr,by='GEN_mun_code')
  
  #colnames(FORECASTED_IMPACT_rr) <- c(GEN_mun_code,paste0(TYF,'_impact'),paste0(TYF,'_priority_index'))
  
  return(df_imact_forecast)
}

####################################################################################################

Make_maps<-function(php_admin1,php_admin3_,my_track,tc_tracks,TYF,Typhoon_stormname){
  
  ################################### Make maps #########
  
  Landfall_check <- st_intersection(php_admin1, my_track)
  Landfall_point<-Landfall_check[1,]
  
  Landfall_check_1 <- st_intersection(php_admin1,my_track[1,])
  
  if (nrow(Landfall_check_1) == 0){
    dt<-lubridate::with_tz(lubridate::ymd_hms(format(Landfall_check$date[1], format="%Y-%m-%d %H:%M:%S"),tz="UTC"), tz="Asia/Manila")
    
    dt1<-lubridate::with_tz(lubridate::ymd_hm(format(TRACK_DATA$YYYYMMDDHH[1], format="%Y%m%d%H%M"),tz="UTC"), tz="Asia/Manila")
    
    dt2<-lubridate::with_tz(lubridate::ymd_hm(format(Sys.time(), format="%Y%m%d%H%M"),tz="CEST"), tz="Asia/Manila")
    
    dt2=lubridate::with_tz(lubridate::force_tz(Sys.time()), tz="Asia/Manila")
    
    time_for_landfall<- as.numeric(difftime(dt,dt1,units="hours"))
    etimated_landfall_time<-dt
    
    
  }else{
    dt<-NA #lubridate::with_tz(lubridate::ymd_hms(format(Landfall_check$date[1], format="%Y-%m-%d %H:%M:%S"),tz="UTC"), tz="Asia/Manila")
    
    dt1<-lubridate::with_tz(lubridate::ymd_hm(format(TRACK_DATA$YYYYMMDDHH[1], format="%Y%m%d%H%M"),tz="UTC"), tz="Asia/Manila")
    
    dt2<-lubridate::with_tz(lubridate::ymd_hm(format(Sys.time(), format="%Y%m%d%H"),tz="CEST"), tz="Asia/Manila")
    
    #dt2=lubridate::with_tz(lubridate::force_tz(Sys.time()), tz="Asia/Manila")
    
    time_for_landfall<- NA #as.numeric(difftime(dt,dt1,units="hours"))
    etimated_landfall_time<-NA
    
  }
  #######################################################################
  # caculate time for landfll
  
  php_admin4 <- php_admin3_ %>%  dplyr::mutate(dam_perc_comp_prediction_lm_quantile = ntile_na(impact,5)) %>% filter(WEA_dist_track < 300)
  
  region2<-extent(php_admin4)
  
  #php_admin3_<-php_admin3_ %>% arrange(WEA_dist_track)
  
  #region<- st_bbox(php_admin3[1,])
  #typhoon_region = st_bbox(c(xmin = region[["xmin"]]-2, xmax = 2+region[["xmax"]],  ymin = region[["ymin"]]-2, ymax =2+ region[["ymax"]]),     crs = st_crs(php_admin1)) %>% st_as_sfc()
  
  typhoon_region = st_bbox(c(xmin =as.vector(region2@xmin), xmax = as.vector(region2@xmax),
                             ymin = as.vector(region2@ymin), ymax =as.vector(region2@ymax)),
                           crs = st_crs(php_admin1)) %>% st_as_sfc()
  
  
  model_run_time<-lubridate::with_tz(lubridate::force_tz(Sys.time()), tz="Asia/Manila")
  
  subtitle =paste0("Predicted damage per Municipality for ", Typhoon_stormname,'\n',
                   "Impact map generated at:",model_run_time,'\n',
                   "Source of wind speed forecast ",TYF,'\n',
                   "Only Areas within 100km of forecasted track are included",'\n',
                   "Prediction is about completely damaged houses only",'\n',
                   'Expected Landfall at : ',dt,' PST in (',time_for_landfall,' hrs)')
  
  tmap_mode(mode = "plot")
  
  impact_map=tm_shape(php_admin4) + 
    tm_fill(col = "impact",showNA=FALSE, border.col = "black",lwd = 3,lyt='dotted',
            breaks = c(0,0.1,1,2,5,9.5,10),
            title='Predicted % of Damaged ',
            labels=c(' No Damage',' < 1%',' 1 to 2%',' 2 to 5%',' 5 to 10%',' > 10%'),
            palette = c('#ffffff','#fdd0a2','#fdae6b','#fd8d3c','#e6550d','#a63603')) + #,style = "cat")+
    
    tm_borders(col = NA, lwd = .25, lty = "solid", alpha = .25, group = NA) +
    
    tm_shape(tc_tracks) + tm_symbols(col='Data_Provider',size=0.1,border.alpha = .25) +
    
    #tm_layout(legend.show = TRUE, legend.position=c("left", "top"))#, main.title=subtitle, main.title.size=.8,asp=.8)
    
    tm_shape(Landfall_point) + tm_symbols(size=0.25,border.alpha = .25,col="red") +  
    tm_compass(type = "8star", position = c("right", "top")) +
    tm_scale_bar(breaks = c(0, 100, 200), text.size = .5,
                 color.light = "#f0f0f0",
                 position = c(0,.1))+
    tm_credits("The maps used do not imply the expression of any opinion on the part of the International Federation of the \nRed Cross and Red Crescent Societies concerning the legal status of a territory or of its authorities.",
               position = c("left", "BOTTOM"),size = 0.6) + 
    tm_layout(legend.show = FALSE)#legend.outside= TRUE,            legend.outside.position=c("left"),            inner.margins=c(.01,.04, .02, .01),            main.title=subtitle, main.title.size=.8,asp=.8)
  
  
  impact_map2=tm_shape(php_admin4) + 
    
    tm_fill(col = "impact",showNA=FALSE, border.col = "black",lwd = 3,lyt='dotted',
            breaks = c(0,0.1,1,2,5,9.5,10),
            title='Predicted % of Damaged ',
            labels=c(' No Damage',' < 1%',' 1 to 2%',' 2 to 5%',' 5 to 10%',' > 10%'),
            palette = c('#ffffff','#fdd0a2','#fdae6b','#fd8d3c','#e6550d','#a63603')) +
    tm_shape(tc_tracks) + tm_symbols(col='Data_Provider',size=0.1,border.alpha = .25) +
    tm_layout(legend.only = TRUE, legend.position=c("left", "top"))#, main.title=subtitle, main.title.size=.8,asp=.8)
  
  ph_map = tm_shape(php_admin1) + tm_polygons(border.col = "black",lwd = 0.1,lyt='dotted',alpha =0.2) +
    tm_shape(typhoon_region) + tm_borders(lwd = 2,col='red') +
    #tm_credits("The maps used do not imply \nthe expression of any opinion on \nthe part of the International Federation\n of the Red Cross and Red Crescent Societies\n concerning the legal status of a\n territory or of its authorities.",
    # position = c("left", "BOTTOM"),size = 0.9) +
    tm_layout(legend.show = FALSE)#inner.margins=c(.04,.03, .04, .04)) 
  #inner.margins=c(.04,.03, .02, .01), 
  
  
  ph_map2 = tm_shape(php_admin1)+ tm_polygons(border.col = "white",lwd = 0.01,lyt='dotted',alpha =0.2) +
    #tm_shape(typhoon_region) +# tm_borders(lwd = 2,col='red') +
    tm_credits( subtitle,position = c("left", "top"),size = 0.7) +
    tm_logo('combined_logo.png',#https://www.510.global/wp-content/uploads/2017/07/510-LOGO-WEBSITE-01.png', 
            height=3, position = c("right", "top"))+
    tm_layout(legend.show = FALSE)
  
  map1<- tmap_arrange(ph_map,ph_map2,impact_map2,impact_map,nrow=2,ncol = 2,widths = c(.3, .7),heights = c(.3, .7))
  return(map1)
  
}

############################################################################
# FOR EACH FORECASTER interpolae track data
############################################################################

dir.create(file.path(paste0(main_directory,'typhoon_infographic/shapes/', Typhoon_stormname)), showWarnings = FALSE)

typhoon_events<-c(UCL_directory,ECMWF_directory)#,HK_directory,JTCW_directory)

ftrack_geodb=paste0(main_directory,'typhoon_infographic/shapes/',Typhoon_stormname, '/',Typhoon_stormname,'1_',forecast_time,'_track.gpkg')

if (file.exists(ftrack_geodb)){ file.remove(ftrack_geodb)}

 

if (!is.null(typhoon_events)) {
  
  for(forecaster in (typhoon_events))
    {
    if (file.exists(forecaster)){
      
        TRACK_DATA<-read.csv(forecaster)
        TYF<- str_split(str_split(forecaster,"/")[[1]][length(str_split(forecaster,"/")[[1]])],"_")[[1]][1]
        my_track <- track_interpolation(TRACK_DATA) %>% dplyr::mutate(Data_Provider=TYF)
        st_write(obj = my_track,
                 dsn = paste0(main_directory,'typhoon_infographic/shapes/',Typhoon_stormname, '/',Typhoon_stormname,'1_',forecast_time,'_track.gpkg'),
                 layer ='tc_tracks', append = TRUE)
        }
      
      }}

if (file.exists(ftrack_geodb)){
  tc_tracks<-st_read(ftrack_geodb)
}

####################################################################################################
# FOR EACH FORECASTER RUN IMPACT DMODEL , FOR NOW HK AND JTCW ARE EXCLUDED 
####################################################################################################


if (!is.null(typhoon_events)) {
  for(forecaster in (typhoon_events))
    {
    if (file.exists(forecaster)){
      
       

        TRACK_DATA<-read.csv(forecaster)
        TYF<- str_split(str_split(forecaster,"/")[[1]][length(str_split(forecaster,"/")[[1]])],"_")[[1]][1]
        my_track <- track_interpolation(TRACK_DATA) %>% dplyr::mutate(Data_Provider=TYF)
        
        
      
 
      
      
      ####################################################################################################
      # check if there is a landfall
      
      print("chincking landfall")
      Landfall_check <- st_intersection(php_admin1, my_track)
      
      if (nrow(Landfall_check) > 0){
        print("claculating data")
        new_data<-Model_input_processing(TRACK_DATA,my_track,TYF,Typhoon_stormname)
        
        ####################################################################################################
        print("running modesl")
        
        FORECASTED_IMPACT<-run_prediction_model(data=new_data)
        php_admin30<-php_admin3 %>% mutate(GEN_mun_code=adm3_pcode) %>%  left_join(FORECASTED_IMPACT,by="GEN_mun_code")
        php_admin3_<-php_admin30 %>% dplyr::arrange(WEA_dist_track) %>%
          dplyr::mutate(impact=ifelse(WEA_dist_track> 100,NA,impact),
                        impact_threshold_passed =ifelse(WEA_dist_track > 100,NA,impact_threshold_passed))
        
        Impact<-php_admin3_  %>%   dplyr::select(GEN_mun_code,impact,impact_threshold_passed,WEA_dist_track) %>% st_set_geometry(NULL)
        php_admin4 <- php_admin3_ %>%  dplyr::mutate(dam_perc_comp_prediction_lm_quantile = ntile_na(impact,5)) %>% filter(WEA_dist_track < 300)
        
        region2<-extent(php_admin4)
        
        #php_admin3_<-php_admin3_ %>% arrange(WEA_dist_track)
        #region<- st_bbox(php_admin3[1,])
        #typhoon_region = st_bbox(c(xmin = region[["xmin"]]-2, xmax = 2+region[["xmax"]],  ymin = region[["ymin"]]-2, ymax =2+ region[["ymax"]]),     crs = st_crs(php_admin1)) %>% st_as_sfc()
        
        typhoon_region = st_bbox(c(xmin =as.vector(region2@xmin), xmax = as.vector(region2@xmax),
                                   ymin = as.vector(region2@ymin), ymax =as.vector(region2@ymax)),
                                 crs = st_crs(php_admin1)) %>% st_as_sfc()
        
        #Impact2<-php_admin3 %>%  left_join(Impact %>% mutate(imp_thr=impact_threshold_passed,adm3_pcode=GEN_mun_code)%>% dplyr::select(adm3_pcode,impact,imp_thr),by = "adm3_pcode")
        st_write(obj = php_admin4,
                 dsn = paste0(main_directory,'typhoon_infographic/shapes/',Typhoon_stormname, '/',Typhoon_stormname,'_',forecast_time,'_impact.gpkg'),
                 layer =TYF,
                 update = TRUE)
        Landfall_point<-Check_landfall_time(php_admin1, my_track)[1]
        time_for_landfall<- Check_landfall_time(php_admin1, my_track)[2]
        etimated_landfall_time<- Check_landfall_time(php_admin1, my_track)[3]
        
        ####################################################################################################
        print("make mapsl")
        #call map function 
        map1<-Make_maps(php_admin1,php_admin3_,my_track,tc_tracks,TYF,Typhoon_stormname)
        tmap_save(map1, 
                  filename = paste0(main_directory,'forecast/Output/',Typhoon_stormname,'/Impact_',TYF,'_',forecast_time,'_',  Typhoon_stormname,'.png'),
                  width=20, height=24,dpi=600,
                  units="cm")
        
        ####################################################################################################
        # ------------------------ save to file   -----------------------------------
        ## save an image ("plot" mode) paste0(main_directory,'fbf/forecast/Impact_',  as.character(typhoon_events$event[i]),'.png')),
        write.csv(Impact, file = paste0(main_directory,'forecast/Output/',Typhoon_stormname,'/Impact_', TYF,'_',forecast_time,'_' ,Typhoon_stormname,'.csv'))
        #paste0('home/fbf/forecast/Impact_',as.vector(TRACK_DATA[1,]$YYYYMMDDHH),'_',as.vector(TRACK_DATA[1,]$STORMNAME),'.csv'))
        file_names<- c(paste0(main_directory,'forecast/Output/',Typhoon_stormname,'/Impact_',TYF,'_',forecast_time,'_',  Typhoon_stormname,'.png'),
                       paste0(main_directory,'forecast/Output/',Typhoon_stormname,'/Impact_',TYF,'_',forecast_time,'_', Typhoon_stormname,'.csv'))
        # paste0('home/fbf/forecast/Impact_',as.vector(TRACK_DATA[1,]$YYYYMMDDHH),'_',as.vector(TRACK_DATA[1,]$STORMNAME),'.png'),
        # paste0('home/fbf/forecast/Impact_',as.vector(TRACK_DATA[1,]$YYYYMMDDHH),'_',as.vector(TRACK_DATA[1,]$STORMNAME),'.csv'))
        write.table(file_names, file =paste0(main_directory, 'forecast/',Typhoon_stormname,'_',forecast_time,'_file_names.csv'),append=TRUE, col.names = FALSE)
        
      }
      else{
        file_names<- c(paste0('Nolandfall','_',Typhoon_stormname,'.png'), paste0('Nolandfall','_',Typhoon_stormname,'.csv'))
        write.table(file_names, file =paste0(main_directory, 'forecast/Output/',Typhoon_stormname,'_',forecast_time,'_file_names.csv'),append=TRUE, col.names = FALSE)
      }
    } 
  } ############################ if forecaster loop end here 
}  ################## close the typhone loop 

