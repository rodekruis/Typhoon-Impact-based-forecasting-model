#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
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


rainfall_error = args[1]

path='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/'
#path='home/fbf/'

main_directory<-path

####################################################################################################




###########################################################################
# ------------------------ import DATA  -----------------------------------
setwd(path)
source('lib_r/settings.R')
source('lib_r/data_cleaning_forecast.R')
source('lib_r/prepare_typhoon_input.R')

source('lib_r/prepare_typhoon_input.R')
source('lib_r/track_interpolation.R')
source('lib_r/Read_rainfall_v2.R')
source('lib_r/Model_input_processing.R')
source('lib_r/run_prediction_model.R')
source('lib_r/Make_maps.R')

source('lib_r/Check_landfall_time.R')

#source('home/fbf/prepare_typhoon_input.R')
#source('home/fbf/settings.R')
#source('home/fbf/data_cleaning_forecast.R')


#------------------------- define functions ---------------------------------

ntile_na <- function(x,n){
  notna <- !is.na(x)
  out <- rep(NA_real_,length(x))
  out[notna] <- ntile(x[notna],n)
  return(out)
}

####################################################################################################


############################################################################
# FOR EACH FORECASTER interpolae track data
############################################################################

dir.create(file.path(paste0(main_directory,'typhoon_infographic/shapes/', Typhoon_stormname)), showWarnings = FALSE)

typhoon_events<-c(UCL_directory,ECMWF_directory)#,HK_directory,JTCW_directory)


################## for ensamble members 

ECMWF_ECEP<-read.csv("C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/forecast/Input/73W/2020110221/Input/ECMWF_2020110221_73W_ECEP.csv")

impact_dfs <- list()
ftrack_geodb=paste0(main_directory,'typhoon_infographic/shapes/',Typhoon_stormname, '/',Typhoon_stormname,'ensamble_',forecast_time,'_track.gpkg')


if (file.exists(ftrack_geodb)){ file.remove(ftrack_geodb)}


for(ensambles in (unique(ECMWF_ECEP$ENSAMBLE)))
{
  TRACK_DATA<-ECMWF_ECEP  %>% filter(ENSAMBLE==ensambles)
  
  
  TYF<- paste0('ECMWF_',ensambles)
  
  my_track <- track_interpolation(TRACK_DATA) %>% dplyr::mutate(Data_Provider=TYF)
  st_write(obj = my_track,
           dsn = paste0(main_directory,'typhoon_infographic/shapes/',Typhoon_stormname, '/',Typhoon_stormname,'ensamble_',forecast_time,'_track.gpkg'),
           layer ='tc_tracks', append = TRUE)
  
}



if (file.exists(ftrack_geodb)){
  tc_tracks<-st_read(ftrack_geodb)
}



if (!is.null(typhoon_events)) {
  for(ensambles in (unique(ECMWF_ECEP$ENSAMBLE)))
  {
    TRACK_DATA<-ECMWF_ECEP  %>% filter(ENSAMBLE==ensambles)
    
    
    TYF<- paste0('ECMWF_',ensambles)
    
    my_track <- track_interpolation(TRACK_DATA) %>% dplyr::mutate(Data_Provider=TYF)
    
    
    
    ####################################################################################################
    # check if there is a landfall
    
    print("chincking landfall")
    Landfall_check <- st_intersection(php_admin_buffer, my_track)
    cn_rows<-nrow(Landfall_check)
    
    
    print("claculating data")
    new_data<-Model_input_processing(TRACK_DATA,my_track,TYF,Typhoon_stormname)
    
    ####################################################################################################
    print("running modesl")
    
    FORECASTED_IMPACT<-run_prediction_model(data=new_data)
    
    php_admin30<-php_admin3 %>% mutate(GEN_mun_code=adm3_pcode) %>%  left_join(FORECASTED_IMPACT,by="GEN_mun_code")
    
    php_admin3_<-php_admin30 %>% dplyr::arrange(WEA_dist_track) %>%
      dplyr::mutate(impact=ifelse(WEA_dist_track> 100,NA,impact),
                    impact_threshold_passed =ifelse(WEA_dist_track > 100,NA,impact_threshold_passed))
    
    Impact<-php_admin3_  %>% dplyr::mutate('ENSAMBLE'=ensambles) %>%  dplyr::select(GEN_mun_code,impact,ENSAMBLE) %>% st_set_geometry(NULL)
    
    if (cn_rows > 0){
      
      Landfall_check<-Check_landfall_time(php_admin_buffer, my_track)
      Landfall_point<-Landfall_check[1]
      time_for_landfall<- Landfall_check[2]
      etimated_landfall_time<- Landfall_check[3]
      
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
      
      
      ####################################################################################################
      
    }
    else{
      Landfall_point<-NA
      time_for_landfall<- NA
      etimated_landfall_time<- NA
    }
    impact_dfs[[ensambles]] <- Impact
  } 
} ############################ if forecaster loop end here 

# Combining files
all_impact <- bind_rows(impact_dfs)

df1<-all_impact %>% dplyr::select(GEN_mun_code)

df<-all_impact%>%spread(ENSAMBLE, impact)%>% dplyr::select(- GEN_mun_code)

Stats <- function(x){
  Mean <- mean(x, na.rm=TRUE)
  SD <- sd(x, na.rm=TRUE)
  Min <- min(x, na.rm=TRUE)
  Max <- max(x, na.rm=TRUE)
  m10 <- sum(x>10)
  return(c(Mean=Mean, SD=SD, Min=Min, Max=Max,m10))
}



cbind(df1, t(apply(df,1, Stats))) # Where newDF is define as above



write.csv(all_impact, "C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/forecast/Input/ATSANI/all_impact.csv", row.names=F)




















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





for (date_e in unique(tc_tracks$date) ){
  track<-tc_tracks %>% filter(tc_tracks$date==date_e)
  seal_coords <-do.call(rbind, st_geometry(track)) %>% as_tibble() %>% setNames(c("lon","lat")) %>%
    summarize(lat = quantile(lat, 0.5),
              lon = quantile(lon, 0.5),
              lat = quantile(lat, 0.25),
              q20 = quantile(lon, 0.75))
 
}
seal_coords <- do.call(rbind, st_geometry(tc_tracks)) %>% as_tibble() %>% setNames(c("lon","lat"))
df<-tc_tracks %>% 
  group_by(date) %>% 
  summarize(q1 = quantile(dt, 0.25),
            q3 = quantile(dt, 0.75))

