#!/usr/bin/env Rscript

estimate_landfall<-function(php_admin1,my_track){
  Landfall_check <- st_intersection(php_admin1, my_track)
  Landfall_point <- Landfall_check[1,]
 
  # Number of total linestrings to be created
  ###################################################
  ##########################################################
  
  Landfall_check_1 <- st_intersection(php_admin1,my_track[1,])
  
  if (nrow(Landfall_check_1) > 0){
    dt<-lubridate::with_tz(lubridate::ymd_hms(format(Landfall_check$date[1], format="%Y-%m-%d %H:%M:%S"),tz="UTC"), tz="Asia/Manila")
    
    dt1<-lubridate::with_tz(lubridate::ymd_hm(format(as.character(TRACK_DATA$YYYYMMDDHH[1]), format="%Y%m%d%H%M"),tz="UTC"), tz="Asia/Manila")
    
    dt2<-lubridate::with_tz(lubridate::ymd_hm(format(Sys.time(), format="%Y%m%d%H%M"),tz="Europe/Berlin"), tz="Asia/Manila")
    
    dt2=lubridate::with_tz(lubridate::force_tz(Sys.time()), tz="Asia/Manila")
    
    time_for_landfall<- as.numeric(difftime(dt,dt1,units="hours"))
    etimated_landfall_time<-dt
    
    Landfall_check_1 <- Landfall_check_1 %>%
      st_cast("MULTIPOINT") %>%
      st_cast("POINT")
    point <- st_coordinates(Landfall_check_1)
    point_lon <- point$X
    point_lat <- point$Y
    
  }else{
    dt<-NA #lubridate::with_tz(lubridate::ymd_hms(format(Landfall_check$date[1], format="%Y-%m-%d %H:%M:%S"),tz="UTC"), tz="Asia/Manila")
    
    dt1<-lubridate::with_tz(lubridate::ymd_hm(format(as.character(TRACK_DATA$YYYYMMDDHH[1]), format="%Y%m%d%H%M"),tz="UTC"), tz="Asia/Manila")
    
    dt2<-lubridate::with_tz(lubridate::ymd_hm(format(Sys.time(), format="%Y%m%d%H%M"),tz="Europe/Berlin"), tz="Asia/Manila")
    time_for_landfall<- NA #as.numeric(difftime(dt,dt1,units="hours"))
    etimated_landfall_time<-NA
    
    point_lon <- NA
    point_lat <- NA
  }
  # #######################################################################
  # # caculate time for landfll
  # php_admin4 <- php_admin3_ %>%  dplyr::mutate(dam_perc_comp_prediction_lm_quantile = ntile_na(map_type,5))# %>% filter(WEA_dist_track < 300)
  # region2<-extent(php_admin4)
  
  # model_run_time<-lubridate::with_tz(lubridate::force_tz(Sys.time()), tz="Asia/Manila")
  
  #######################################################################
  # merge land fall location and time
  # landfall = c(time_for_landfall, Landfall_point)
  landfall <- data.frame(time_for_landfall=time_for_landfall, landfall_point_lat=point_lat, landfall_point_lon=point_lon)


  return(landfall)
  
}
