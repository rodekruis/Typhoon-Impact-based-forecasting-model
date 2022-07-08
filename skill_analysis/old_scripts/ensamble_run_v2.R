#!/usr/bin/env Rscript
rm(list=ls())
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
suppressMessages(library(parallel))
suppressMessages(library(parallelMap))

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


geo_variable <- read.csv(paste0(main_directory,"./data/geo_variable.csv"))
wshade <- php_admin3 
# load the rr model
mode_classification <- readRDS(paste0(main_directory,"./models/final_model.rds"))
mode_continious <- readRDS(paste0(main_directory,"./models/final_model_regression.rds"))
mode_classification1 <- readRDS(paste0(main_directory,"./models/xgboost_classify.rds"))
mode_continious1 <- readRDS(paste0(main_directory,"./models/xgboost_regression.rds"))





#------------------------- define functions ---------------------------------

ntile_na <- function(x,n){
  notna <- !is.na(x)
  out <- rep(NA_real_,length(x))
  out[notna] <- ntile(x[notna],n)
  return(out)
}


####################################################################################################
wind_input_processing1<-function(TRACK_DATA,grid_points){
  
  ###################### Model_input_processing ##############
  
  #generate tack with smaller time steps 

  grid_points_adm3<-grid_points  
  TRACK_DATA1<- TRACK_DATA %>% dplyr::mutate(typhoon = paste0(TRACK_DATA$STORMNAME,substr(TRACK_DATA$YYYYMMDDHH, 1, 4)))
  
  wind_grids <- get_grid_winds(hurr_track=TRACK_DATA1, 
                               grid_df=grid_points_adm3,
                               tint = 0.5,
                               gust_duration_cut = 15, 
                               sust_duration_cut = 15)
  
  wind_grids<- wind_grids %>%group_by(pcode) %>%
    dplyr::summarise(vmax_gust= max(vmax_gust),
                     vmax_sust= max(vmax_sust),
                     gust_dur= max(gust_dur),
                     sust_dur= max(sust_dur),
                     dist_track = min(dist_track))%>%
    ungroup()
  
  wind_grids<- wind_grids %>%
    dplyr::mutate(typhoon_name=as.vector(TRACK_DATA$STORMNAME)[1],vmax_gust_mph=vmax_gust*2.23694,vmax_sust_mph=vmax_sust*2.23694) %>%
    dplyr::select(pcode,typhoon_name,vmax_gust,vmax_gust_mph,vmax_sust,vmax_sust_mph,gust_dur,sust_dur,dist_track)
  
  wind_grid<-wind_grids%>%
    dplyr::mutate(Mun_Code=pcode)
  
  
  return(wind_grid)
}



Model_input_processing<-function(TRACK_DATA,grid_points){
  
  ###################### Model_input_processing ##############
  
  #generate tack with smaller time steps 
 
  grid_df<-grid_points  
  
  TRACK_DATA1<- TRACK_DATA %>% dplyr::mutate(typhoon = paste0(TRACK_DATA$STORMNAME,substr(TRACK_DATA$YYYYMMDDHH, 1, 4)))
  
  wind_grids <- get_grid_winds(hurr_track=TRACK_DATA1, grid_df=grid_df, 
                               tint = 0.5,gust_duration_cut = 15, 
                               sust_duration_cut = 15)
  
  wind_grids<- wind_grids %>%group_by(pcode) %>%
    dplyr::summarise(vmax_gust= max(vmax_gust),
                     vmax_sust= max(vmax_sust),
                     gust_dur= max(gust_dur),
                     sust_dur= max(sust_dur),
                     dist_track = min(dist_track))%>%ungroup()
  
  wind_grids<- wind_grids %>%
    dplyr::mutate(Mun_Code=pcode,
                  typhoon_name=as.vector(TRACK_DATA$STORMNAME)[1],
                  vmax_gust_mph=vmax_gust*2.23694,
                  vmax_sust_mph=vmax_sust*2.23694) %>%
    dplyr::select(Mun_Code,
                  typhoon_name,
                  vmax_gust,
                  vmax_gust_mph,
                  vmax_sust,
                  vmax_sust_mph,
                  gust_dur,
                  sust_dur,
                  dist_track)
  
   # track wind should have a unit of knots
  rainfall_<-Read_rainfall_v2(wshade)
  
  
  # BUILD DATA MATRIC FOR NEW TYPHOON 
  data_new_typhoon<-geo_variable %>%
    left_join(material_variable2 %>% dplyr::select(-Region,-Province,-Municipality_City), by = "Mun_Code") %>%
    left_join(data_matrix_new_variables , by = "Mun_Code") %>%
    left_join(wind_grids , by = "Mun_Code")  %>%
    left_join(rainfall_ , by = "Mun_Code")
  
  # source('/home/fbf/data_cleaning_forecast.R')
  data<-clean_typhoon_forecast_data(data_new_typhoon)%>%
    dplyr::select(-GEN_typhoon_name,
                  -GEO_n_households,
                  #-GEN_mun_code,
                  -contains('DAM_'),
                  -GEN_mun_name) %>%
    na.omit() # Randomforests don't handle NAs, you can impute in the future 
  return(data)
}





# load forecast data

# FOR TESTING COMMEMT THE FOLLOWING LINES

#typhoon_info_for_model <- read.csv(paste0(main_directory,"/forecast/Input/typhoon_info_for_model.csv"))
#rain_directory<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='Rainfall',]$filename)
#ECMWF_<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='ECMWF',]$filename)
#Typhoon_stormname<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='ECMWF',]$event)
#forecast_time<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='ECMWF',]$time)
#forecast_time<-str_remove_all(forecast_time, "'")
#rain_directory<-ifelse(identical(character(0), rain_directory),NULL,rain_directory)
#ECMWF_directory<-ifelse(identical(character(0), ECMWF_),NA,ECMWF_)

# FOR TESTING UNCOMMEMT THE FOLLOWING LINES


rain_directory<-paste0(main_directory,"/test_data/rainfall/")# as.character(typhoon_info_for_model[typhoon_info_for_model$source=='Rainfall',]$filename)
Typhoon_stormname<-'SURIGAE'
ECMWF_directory<-paste0(main_directory,"/test_data/") 


################# for testing uncomment the following lines 
#	ECMWF_ECMF <- read.csv(paste0(main_directory,"/test_data/ECMWF_2021041912_SURIGAE_ECMWF.csv"))
#	ECMWF_ECEP <- read.csv(paste0(main_directory,"/test_data/ECMWF_2021041912_SURIGAE_ECEP.csv"))

ECMWF_ECMF<-read.csv(ECMWF_ )                   
ECMWF_ECEP<-read.csv(gsub("_ECMF.csv", "_ECEP.csv", ECMWF_))                   

ECMWF<-ECMWF_ECEP %>%dplyr::select(-VMAX)%>%left_join(ECMWF_ECMF%>%dplyr::select(YYYYMMDDHH,VMAX),by='YYYYMMDDHH')
 

########## filter grid point in the Typhoon Track 

grid_points<-grid_points_adm3%>%dplyr::mutate(pcode=gridid,index=1:nrow(grid_points_adm3),gridid=index)%>%
  dplyr::select(-index)%>%dplyr::filter(between(glat, min(ECMWF_ECMF$LAT, na.rm=T)-1, max(ECMWF_ECMF$LAT, na.rm=T)+1))


#ftrack_geodb=paste0(main_directory,'typhoon_infographic/shapes/',Typhoon_stormname, '/',Typhoon_stormname,'ensamble_',forecast_time,'_track.gpkg')
#if (file.exists(ftrack_geodb)){ file.remove(ftrack_geodb)}

run_ensamble <- function(ensamble){
    TRACK_DATA<-ECMWF  %>% filter(ENSAMBLE==ensamble)%>% drop_na()
    event_name<- paste0('ECMWF_',ensamble)
    TYF<- event_name
    new_data<-wind_input_processing1(TRACK_DATA,grid_points)%>% dplyr::mutate(Ty_year=TYF)
    #wind_dfs[[event_name]] <- new_data
}

######################################################################
################# calculate wind field for all variable ##############
################# PART WHICH NEEDS OPTIMIZATION FOR MODEL RUN TIME #########


parallelStartSocket(cpus = detectCores()-2)

wind_grids_ <- lapply(ensambles, run_ensamble)

#system.time(lapply(ensambles, run_ensamble))

parallelStop()

###########################

wind_grids_ <- bind_rows(wind_dfs_)

rainfall_<-Read_rainfall_v2(wshade)


# BUILD DATA MATRIC FOR NEW TYPHOON 
data_new_typhoon<-geo_variable %>%
  left_join(material_variable2 %>% dplyr::select(-Region,-Province,-Municipality_City), by = "Mun_Code") %>%
  left_join(data_matrix_new_variables , by = "Mun_Code") %>%
  left_join(wind_grids_ , by = "Mun_Code")  %>%
  left_join(rainfall_ , by = "Mun_Code")

# source('/home/fbf/data_cleaning_forecast.R')
data_<-clean_typhoon_forecast_data(data_new_typhoon)%>%
  dplyr::select(-GEN_typhoon_name,
                -GEO_n_households,
                #-GEN_mun_code,
                -contains('DAM_'),
                -GEN_mun_name) %>%na.omit() # 

 

########## run prediction ##########

rm.predict.pr <- predict(mode_classification,
                         data = data_,
                         predict.all = FALSE,
                         num.trees = mode_classification$num.trees, type = "response",
                         se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)

FORECASTED_IMPACT<-as.data.frame(rm.predict.pr$predictions) %>%
  dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact_threshold_passed=rm.predict.pr$predictions) %>%
  left_join(new_data , by = "index") %>%
  dplyr::select(GEN_mun_code,impact_threshold_passed,WEA_dist_track,WEA_vmax_sust_mhp)%>%drop_na()

#colnames(FORECASTED_IMPACT) <- c(GEN_mun_code,paste0(TYF,'_impact_threshold_passed'),WEA_dist_track)

rm.predict.pr <- predict(mode_continious,
                         data = data_, 
                         predict.all = FALSE,
                         num.trees = mode_continious$num.trees, type = "response",
                         se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)

FORECASTED_IMPACT_rr<-as.data.frame(rm.predict.pr$predictions) %>%
  dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact=rm.predict.pr$predictions) %>% 
  dplyr::mutate(priority_index=ntile_na(ifelse(impact<0,NA,impact),5))%>%
  left_join(new_data , by = "index") %>%
  dplyr::select(GEN_mun_code,impact,priority_index)%>%drop_na()

df_imact_forecast<-FORECASTED_IMPACT %>% left_join(FORECASTED_IMPACT_rr,by='GEN_mun_code')



 



all_impact <- df_imact_forecast%>%dplyr::mutate(class_B = case_when(impact >= 5 & impact < 8 ~ "C",impact >= 8 & impact < 10 ~ "B",impact >= 10 ~ "A",TRUE ~ 'Low'))

class_<-c("A","B","C","Low")


#write.csv(all_impact,paste0(main_directory,"all_impact.csv"), row.names=F)


df<-all_impact%>%group_by(GEN_mun_code) %>% group_map(~sapply(1:length(class_),function(i){.x %>% filter(class_B ==class_[i]) %>% nrow*100/52}))
df <- data.frame(matrix(unlist(df), nrow=length(df), byrow=TRUE))

names(df)<-class_
df$pcode<-unique(all_impact$GEN_mun_code)

php_impact<-php_admin3%>%left_join(df,by=c('adm3_pcode'='pcode'))


all_wind <- wind_grids_

all_wind<-all_wind%>%dplyr::mutate(dist50=ifelse(WEA_dist_track >= 50,0,1))
    
track <- track_interpolation(ECMWF_ECMF)  

 

php_admin4<-php_admin3%>%left_join(aggregate(all_wind$dist50, by=list(adm3_pcode=all_wind$GEN_mun_code), FUN=sum)%>%
                                     dplyr::mutate(probability=100*x/52)%>%
                                     dplyr::select(adm3_pcode,probability),by='adm3_pcode')%>%
  left_join(aggregate(all_wind$WEA_vmax_sust_mhp, by=list(adm3_pcode=all_wind$GEN_mun_code), FUN=max)%>%
              dplyr::mutate(VMAX=x)%>%dplyr::select(adm3_pcode,VMAX))%>%
  left_join(geo_variable%>%dplyr::select(Mun_Code,with_coast),by = c('adm3_pcode'='Mun_Code'))#%>%filter(with_coast==1)


#---------------------- vistualize landfall location probability -------------------------------

tmap_mode(mode = "view")
tm_shape(php_admin4) + tm_polygons(col = "probability", name='adm3_en',
                                       palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
                                       breaks=c(0,10,40,60,90,100),colorNA=NULL,
                                       labels=c('   < 10%','10 - 40%','40 - 60%','60 - 90%','   > 90%'),
                                       title="Probability for distance from Track < 50km",
                                   alpha = 0.75,
                                       border.col = "black",lwd = 0.01,lyt='dotted')+
  tm_shape(track) +  
  tm_symbols(size=0.01,border.alpha = 0.75,col='#bdbdbd') +
  tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
  #tm_shape(goni_track) +tm_dots(size=0.1,border.alpha = 0.25,col='#bdbdbd')+
  tm_format("NLD")

#---------------------- vistualize impact probability -------------------------------

tmap_mode(mode = "view")

tm_shape(php_impact) + tm_polygons(col = "B", name='adm3_en',
                                   palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
                                   breaks=c(0,50,70,80,90,100),colorNA=NULL,
                                   labels=c('   < 50%','70 - 70%','70 - 80%','80 - 90%','   > 90%'),
                                   title="Probability for 10%distance from Track < 50km",
                                   alpha = 0.75,
                                   border.col = "black",lwd = 0.01,lyt='dotted')+
  tm_shape(track) +  
  tm_symbols(size=0.01,border.alpha = 0.75,col='#bdbdbd') +
  tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
  #tm_shape(goni_track) +tm_dots(size=0.1,border.alpha = 0.25,col='#bdbdbd')+
  tm_format("NLD")

#---------------------- vistualize maximum wind speed -------------------------------
 
tmap_mode(mode = "view")
tm_shape(php_admin4) + tm_polygons(col = "VMAX", name='adm3_en',
                                   palette=c('#fee5d9','#fc9272','#fb6a4a','#de2d26','#a50f15'),
                                   breaks=c(0,25,40,60,100),colorNA=NULL,
                                   labels=c('   < 25 m/s','25 - 40 m/s','40 - 60 m/s','   > 60 m/s'),
                                   title="Sustained wind speed mps",
                                   alpha = 0.75,
                                   border.col = "black",lwd = 0.01,lyt='dotted')+
  #tm_shape(track) +  
  tm_symbols(size=0.02,border.alpha = 0.75,col='#bdbdbd') +
  tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
  #tm_shape(goni_track) +tm_dots(size=0.1,border.alpha = 0.25,col='#bdbdbd')+
  tm_format("NLD")




 
 


























