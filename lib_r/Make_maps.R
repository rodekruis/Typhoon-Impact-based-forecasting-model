#!/usr/bin/env Rscript
#options(warn=-1)
Make_maps<-function(php_admin1,php_admin3_,my_track,tc_tracks,TYF,Typhoon_stormname){
  
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
  php_admin4 <- php_admin3_ %>%  dplyr::mutate(dam_perc_comp_prediction_lm_quantile = ntile_na(impact,5)) %>% filter(WEA_dist_track < 300)
  region2<-extent(php_admin4)
 
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
    #tm_shape(Landfall_point) + tm_symbols(size=0.25,border.alpha = .25,col="red") +  
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
    tm_layout(legend.show = FALSE)#inner.margins=c(.04,.03, .04, .04)) 
  
  ph_map2 = tm_shape(php_admin1)+ tm_polygons(border.col = "white",lwd = 0.01,lyt='dotted',alpha =0.2) +
    #tm_shape(typhoon_region) +# tm_borders(lwd = 2,col='red') +
    tm_credits( subtitle,position = c("left", "top"),size = 0.7) +
    tm_logo(paste0(path,'logos/combined_logo.png'),#https://www.510.global/wp-content/uploads/2017/07/510-LOGO-WEBSITE-01.png', 
            height=3, position = c("right", "top"))+
    tm_layout(legend.show = FALSE)
  
  map1<- tmap_arrange(ph_map,ph_map2,impact_map2,impact_map,nrow=2,ncol = 2,widths = c(.3, .7),heights = c(.3, .7))
  return(map1)
  
}


Make_maps_avg<-function(php_admin1,php_admin3_,my_track,TYF,Typhoon_stormname){

  
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
  php_admin4 <- php_admin3_ %>%  dplyr::mutate(dam_perc_comp_prediction_lm_quantile = ntile_na(impact,5))# %>% filter(WEA_dist_track < 300)
  region2<-extent(php_admin4)
  
  typhoon_region = st_bbox(c(xmin =as.vector(region2@xmin),
                             xmax = as.vector(region2@xmax),
                             ymin = as.vector(region2@ymin),
                             ymax =as.vector(region2@ymax)),
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
            palette = c('#ffffe5','#fdd0a2','#fdae6b','#fd8d3c','#e6550d','#a63603')) + #,style = "cat")+
    
    tm_borders(col = NA, lwd = .25, lty = "solid", alpha = .25, group = NA) +
    
    tm_shape(my_track) + tm_symbols(col='Data_Provider',size=0.1,border.alpha = .25) +
    
    #tm_layout(legend.show = TRUE, legend.position=c("left", "top"))#, main.title=subtitle, main.title.size=.8,asp=.8)
    #tm_shape(Landfall_point) + tm_symbols(size=0.25,border.alpha = .25,col="red") +  
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
    tm_shape(my_track) + tm_symbols(col='Data_Provider',size=0.1,border.alpha = .25) +
    tm_layout(legend.only = TRUE, legend.position=c("left", "top"))#, main.title=subtitle, main.title.size=.8,asp=.8)
  
  ph_map = tm_shape(php_admin1) + tm_polygons(border.col = "black",lwd = 0.1,lyt='dotted',alpha =0.2) +
    tm_shape(typhoon_region) + tm_borders(lwd = 2,col='red') +
    #tm_credits("The maps used do not imply \nthe expression of any opinion on \nthe part of the International Federation\n of the Red Cross and Red Crescent Societies\n concerning the legal status of a\n territory or of its authorities.",
    tm_layout(legend.show = FALSE)#inner.margins=c(.04,.03, .04, .04)) 
  
  ph_map2 = tm_shape(php_admin1)+ tm_polygons(border.col = "white",lwd = 0.01,lyt='dotted',alpha =0.2) +
    #tm_shape(typhoon_region) +# tm_borders(lwd = 2,col='red') +
    tm_credits( subtitle,position = c("left", "top"),size = 0.7) +
    tm_logo(paste0(path,'logos/combined_logo.png'),#https://www.510.global/wp-content/uploads/2017/07/510-LOGO-WEBSITE-01.png', 
            height=3, position = c("right", "top"))+
    tm_layout(legend.show = FALSE)
  
  map1<- tmap_arrange(ph_map,ph_map2,impact_map2,impact_map,nrow=2,ncol = 2,widths = c(.3, .7),heights = c(.3, .7))
  return(map1)
  
}