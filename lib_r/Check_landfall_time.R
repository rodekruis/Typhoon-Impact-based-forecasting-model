#!/usr/bin/env Rscript
#options(warn=-1)

Check_landfall_time<-function(php_admin_buffer, my_track){
  
  ########## check if there is a landfall #####
  
  Landfall_check <- st_intersection(php_admin_buffer, my_track)
  
  if (nrow(Landfall_check) > 0){
    
    Typhoon_stormname<-Typhoon_stormname
    
    Landfall_check <- st_intersection(php_admin_buffer, my_track)
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