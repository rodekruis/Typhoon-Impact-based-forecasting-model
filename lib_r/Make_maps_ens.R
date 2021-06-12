#!/usr/bin/env Rscript
#options(warn=-1)
Make_maps_ens<-function(php_admin1,php_admin3,my_track,TYF,Typhoon_stormname){
  
  
  ################################### Make maps #########
  Landfall_check <- st_intersection(php_admin1, my_track)
  Landfall_point<-Landfall_check[1,]
           # Number of total linestrings to be created
           ###################################################
           ##########################################################
  
  Landfall_check_1 <- st_intersection(php_admin1,my_track[1,])
  
  if (nrow(Landfall_check_1) == 0){
    dt<-lubridate::with_tz(lubridate::ymd_hms(format(Landfall_check$date[1], format="%Y-%m-%d %H:%M:%S"),tz="UTC"), tz="Asia/Manila")
    
    dt1<-lubridate::with_tz(lubridate::ymd_hm(format(as.character(TRACK_DATA$YYYYMMDDHH[1]), format="%Y%m%d%H%M"),tz="UTC"), tz="Asia/Manila")
    
    dt2<-lubridate::with_tz(lubridate::ymd_hm(format(Sys.time(), format="%Y%m%d%H%M"),tz="Europe/Berlin"), tz="Asia/Manila")
    
    dt2=lubridate::with_tz(lubridate::force_tz(Sys.time()), tz="Asia/Manila")
    
    time_for_landfall<- as.numeric(difftime(dt,dt1,units="hours"))
    etimated_landfall_time<-dt
    
    
  }else{
    dt<-NA #lubridate::with_tz(lubridate::ymd_hms(format(Landfall_check$date[1], format="%Y-%m-%d %H:%M:%S"),tz="UTC"), tz="Asia/Manila")
    
    dt1<-lubridate::with_tz(lubridate::ymd_hm(format(as.character(TRACK_DATA$YYYYMMDDHH[1]), format="%Y%m%d%H%M"),tz="UTC"), tz="Asia/Manila")
    
    dt2<-lubridate::with_tz(lubridate::ymd_hm(format(Sys.time(), format="%Y%m%d%H%M"),tz="Europe/Berlin"), tz="Asia/Manila")
    time_for_landfall<- NA #as.numeric(difftime(dt,dt1,units="hours"))
    etimated_landfall_time<-NA
    
  }
  #######################################################################
  # caculate time for landfll
  php_admin4 <- php_admin3 %>% filter(WEA_dist_track < 300)
  
  region2<-extent(php_admin4)
 
  typhoon_region = st_bbox(c(xmin =as.vector(region2@xmin), xmax = as.vector(region2@xmax),
                             ymin = as.vector(region2@ymin), ymax =as.vector(region2@ymax)),
                           crs = st_crs(php_admin1)) %>% st_as_sfc()
    
  model_run_time<-lubridate::with_tz(lubridate::force_tz(Sys.time()), tz="Asia/Manila")
  
  subtitle =paste0("Predicted damage per Municipality for ", Typhoon_stormname,'\n',
                   "Impact map generated at:",model_run_time,'\n',
                   "Source of wind speed forecast ",TYF,'\n',
                   "Only Areas within 150km of forecasted track are included",'\n',
                   "Prediction is about completely damaged houses only",'\n',
                   'Expected Landfall at : ',dt,' PST in (',time_for_landfall,' hrs)')
  
  tmap_mode(mode = "plot")
  
  impact_map=tm_shape(php_admin4) + tm_polygons(col = "probability_90k", name='adm3_en',
                                     palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
                                     breaks=c(0,40,50,60,90,100),colorNA=NULL,
                                     labels=c('   < 40%','40 - 50%','50 - 60%','60 - 90%','   > 90%'),
                                     title="Probability for #Dam. Buld >90k",
                                     alpha = 0.75,
                                     border.col = "black",lwd = 0.01,lyt='dotted')+
    tm_borders(col = NA, lwd = .15, lty = "solid", alpha = .5, group = NA) +
    tm_shape(my_track) + tm_symbols(size=0.1,border.alpha = .75,col='#0c2c84')+
    tm_compass(type = "8star", position = c("right", "top")) +
    tm_scale_bar(breaks = c(0, 100, 200), text.size = .5,
                 color.light = "#f0f0f0",
                 position = c(0,.1))+
    tm_credits("The maps used do not imply the expression of any opinion on the part of the International Federation of the \nRed Cross and Red Crescent Societies concerning the legal status of a territory or of its authorities.",
               position = c("left", "BOTTOM"),size = 0.6) + 
    tm_layout(legend.show = FALSE)
    
  impact_map2=tm_shape(php_admin4) + tm_polygons(col = "probability_90k", name='adm3_en',
                                     palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
                                     breaks=c(0,40,50,60,90,100),colorNA=NULL,
                                     labels=c('   < 40%','40 - 50%','50 - 60%','60 - 90%','   > 90%'),
                                     title="Probability for #Dam. Buld >90k",
                                     alpha = 0.75,
                                     border.col = "black",lwd = 0.01,lyt='dotted')+
    tm_shape(my_track) + tm_symbols(size=0.1,border.alpha = .75,col='#0c2c84')+
    tm_layout(legend.only = TRUE, legend.position=c("left", "top"))#, main.title=subtitle, main.title.size=.8,asp=.8)
  
  ph_map = tm_shape(php_admin1) + tm_polygons(border.col = "black",lwd = 0.1,lyt='dotted',alpha =0.2) +
    tm_shape(typhoon_region) + tm_borders(lwd = 2,col='red') +
    tm_layout(legend.show = FALSE)
	
  ph_map2 = tm_shape(php_admin1)+ tm_polygons(border.col = "white",lwd = 0.01,lyt='dotted',alpha =0.2) +
    #tm_shape(typhoon_region) +# tm_borders(lwd = 2,col='red') +
    tm_credits( subtitle,position = c("left", "top"),size = 0.7) +
    tm_logo(paste0(path,'logos/combined_logo.png'),#https://www.510.global/wp-content/uploads/2017/07/510-LOGO-WEBSITE-01.png', 
            height=3, position = c("right", "top"))+
    tm_layout(legend.show = FALSE)
  
  map1<- tmap_arrange(ph_map,ph_map2,impact_map2,impact_map,nrow=2,ncol = 2,widths = c(.3, .7),heights = c(.3, .7))
  return(map1)
  
}