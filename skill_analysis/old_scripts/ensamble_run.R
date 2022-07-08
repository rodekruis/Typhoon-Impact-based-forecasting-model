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
 
# load the rr model
mode_classification <- readRDS(paste0(main_directory,"./models/final_model.rds"))
mode_continious <- readRDS(paste0(main_directory,"./models/final_model_regression.rds"))
mode_classification1 <- readRDS(paste0(main_directory,"./models/xgboost_classify.rds"))
mode_continious1 <- readRDS(paste0(main_directory,"./models/xgboost_regression.rds"))



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
wshade <- php_admin3


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
                     dist_track = min(dist_track))%>%
    ungroup()
  
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


############################################################################
# FOR EACH FORECASTER interpolae track data
############################################################################

#dir.create(file.path(paste0(main_directory,'typhoon_infographic/shapes/', Typhoon_stormname)), showWarnings = FALSE)

#typhoon_events<-c(UCL_directory,ECMWF_directory)#,HK_directory,JTCW_directory)


################## for ensamble members 


ECMWF_ECEP<-read.csv('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/forecast/Input/GONI/2020103112/Input/ECMWF_2020103112_GONI_ECEP.csv')
ECMWF_ECMF<-read.csv('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/forecast/Input/GONI/2020103112/Input/ECMWF_2020103112_GONI_ECMF.csv')
                    

ECMWF<-ECMWF_ECEP %>%dplyr::select(-VMAX)%>%left_join(ECMWF_ECMF%>%dplyr::select(YYYYMMDDHH,VMAX),by='YYYYMMDDHH')


########## filter grid point in the Typhoon Track 

grid_points<-grid_points_adm3%>%dplyr::mutate(index=1:nrow(grid_points_adm3),gridid=index)%>%
  dplyr::select(-index)%>%dplyr::filter(between(glat, min(ECMWF_ECMF$LAT, na.rm=T)-1, max(ECMWF_ECMF$LAT, na.rm=T)+1))

#ftrack_geodb=paste0(main_directory,'typhoon_infographic/shapes/',Typhoon_stormname, '/',Typhoon_stormname,'ensamble_',forecast_time,'_track.gpkg')


#if (file.exists(ftrack_geodb)){ file.remove(ftrack_geodb)}

impact_dfs <- list()
wind_dfs <- list()

for(ensambles in (unique(ECMWF$ENSAMBLE)))
{
  TRACK_DATA<-ECMWF  %>% filter(ENSAMBLE==ensambles)%>% drop_na()
  
  event_name<- paste0('ECMWF_',ensambles)
  TYF<- event_name
  
  #my_track <- track_interpolation(TRACK_DATA) %>% dplyr::mutate(Data_Provider=TYF)
  #st_write(obj = my_track, dsn = paste0(main_directory,'data/historicaltrack_track.gpkg'), layer ='tc_tracks', append = TRUE)
  
  new_data<-Model_input_processing(TRACK_DATA,grid_points)%>% dplyr::mutate(Ty_year=TYF)
  
  # if we want to calculate only wind fields 
 # new_data<-wind_input_processing(TRACK_DATA,grid_points)%>% dplyr::mutate(Ty_year=TYF)
  wind_dfs[[event_name]] <- new_data
  
  
  ########## run prediction ##########
  
  rm.predict.pr <- predict(mode_classification,
                           data = new_data,
                           predict.all = FALSE,
                           num.trees = mode_classification$num.trees, type = "response",
                           se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)
  
  FORECASTED_IMPACT<-as.data.frame(rm.predict.pr$predictions) %>%
    dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact_threshold_passed=rm.predict.pr$predictions) %>%
    left_join(data , by = "index") %>%
    dplyr::select(GEN_mun_code,impact_threshold_passed,WEA_dist_track,WEA_vmax_sust_mhp)%>%drop_na()
  
  #colnames(FORECASTED_IMPACT) <- c(GEN_mun_code,paste0(TYF,'_impact_threshold_passed'),WEA_dist_track)
  
  rm.predict.pr <- predict(mode_continious, data = new_data, predict.all = FALSE, num.trees = mode_continious$num.trees, type = "response",
                           se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)
  
  FORECASTED_IMPACT_rr<-as.data.frame(rm.predict.pr$predictions) %>%
    dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact=rm.predict.pr$predictions) %>% 
    dplyr::mutate(priority_index=ntile_na(ifelse(impact<0,NA,impact),5))%>%
    left_join(data , by = "index") %>%
    dplyr::select(GEN_mun_code,impact,priority_index)%>%drop_na()
  
  df_imact_forecast<-FORECASTED_IMPACT %>% left_join(FORECASTED_IMPACT_rr,by='GEN_mun_code')
  
  impact_dfs[[ensambles]] <- df_imact_forecast%>% dplyr::mutate(Ty_year=TYF)
  
  
}


all_wind <- bind_rows(wind_dfs)

all_impact <- bind_rows(impact_dfs)%>%
  dplyr::mutate(class_B = case_when(impact >= 5 & impact < 8 ~ "C",
                                    impact >= 8 & impact < 10 ~ "B", 
                                    impact >= 10 ~ "A",TRUE ~ 'Low'))

class_<-c("A","B","C","Low")

df<-all_impact%>%group_by(GEN_mun_code) %>% group_map(~sapply(1:length(class_),function(i){.x %>% filter(class_B ==class_[i]) %>% nrow*100/52}))
df <- data.frame(matrix(unlist(df), nrow=length(df), byrow=TRUE))

names(df)<-class_
df$pcode<-unique(all_impact$GEN_mun_code)

php_impact<-php_admin3%>%left_join(df,by=c('adm3_pcode'='pcode'))

all_wind<-all_wind%>%dplyr::mutate(dist50=ifelse(WEA_dist_track >= 50,0,1))
    
track <- track_interpolation(ECMWF_ECMF)  

 

php_admin4<-php_admin3%>%left_join(aggregate(all_wind$dist50, by=list(adm3_pcode=all_wind$GEN_mun_code), FUN=sum)%>%
                                     dplyr::mutate(probability=100*x/52)%>%
                                     dplyr::select(adm3_pcode,probability),by='adm3_pcode')%>%
  left_join(aggregate(all_wind$WEA_vmax_sust_mhp, by=list(adm3_pcode=all_wind$GEN_mun_code), FUN=max)%>%
              dplyr::mutate(VMAX=x)%>%dplyr::select(adm3_pcode,VMAX))%>%
  left_join(geo_variable%>%dplyr::select(Mun_Code,with_coast),by = c('adm3_pcode'='Mun_Code'))#%>%filter(with_coast==1)


#---------------------- vistualize stations and risk areas -------------------------------

tmap_mode(mode = "view")
tm_shape(php_admin4) + tm_polygons(col = "probability", name='adm3_en',
                                       palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
                                       breaks=c(0,50,70,80,90,100),colorNA=NULL,
                                       labels=c('   < 50%','70 - 70%','70 - 80%','80 - 90%','   > 90%'),
                                       title="Probability for distance from Track < 50km",
                                   alpha = 0.75,
                                       border.col = "black",lwd = 0.01,lyt='dotted')+
  tm_shape(track) +  
  tm_symbols(size=0.01,border.alpha = 0.75,col='#bdbdbd') +
  tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
  #tm_shape(goni_track) +tm_dots(size=0.1,border.alpha = 0.25,col='#bdbdbd')+
  tm_format("NLD")



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

#---------------------- vistualize stations and risk areas -------------------------------
 
tmap_mode(mode = "view")
tm_shape(php_admin4) + tm_polygons(col = "VMAX", name='adm3_en',
                                   palette=c('#fee5d9','#fc9272','#fb6a4a','#de2d26','#a50f15'),
                                   breaks=c(0,25,40,60,100),colorNA=NULL,
                                   labels=c('   < 25 m/s','25 - 40 m/s','40 - 60 m/s','   > 60 m/s'),
                                   title="Sustained wind speed mps",
                                   alpha = 0.75,
                                   border.col = "black",lwd = 0.01,lyt='dotted')+
  tm_shape(track) +  
  tm_symbols(size=0.02,border.alpha = 0.75,col='#bdbdbd') +
  tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
  #tm_shape(goni_track) +tm_dots(size=0.1,border.alpha = 0.25,col='#bdbdbd')+
  tm_format("NLD")


#e7e1ef
#c994c7
#dd1c77

 
 
 

php_admin40<-php_admin3%>%left_join(aggregate(all_impact$impact10, by=list(adm3_pcode=all_impact$GEN_mun_code), FUN=sum)%>%
                                     dplyr::mutate(probability=100*x/52)%>%
                                     dplyr::select(adm3_pcode,probability),by='adm3_pcode')%>%
  left_join(aggregate(all_impact$impact, by=list(adm3_pcode=all_impact$GEN_mun_code), FUN=mean)%>%dplyr::mutate(mean_impact=x)%>%
              dplyr::select(adm3_pcode,mean_impact),by='adm3_pcode')%>%filter(mean_impact>2)






#---------------------- vistualize stations and risk areas -------------------------------
tmap_mode(mode = "view")
tm_shape(php_admin40) + tm_polygons(col = "probability", name='adm3_en',
                                   palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
                                   breaks=c(0,25,50,75,90,100),colorNA=NULL,
                                   labels=c('   < 25%','25 - 50%','50 - 75%','75 - 90%','   > 90%'),
                                   title="Probability for Passing 10% treshold",
                                   alpha = 0.75,
                                   border.col = "black",lwd = 0.01,lyt='dotted')+
  tm_shape(track) +  
  tm_symbols(size=0.01,border.alpha = 0.75,col='#bdbdbd') +
  tm_shape(php_admin3) + tm_borders(lwd = .5,col='#bdbdbd') + 
  #tm_shape(goni_track) +tm_dots(size=0.1,border.alpha = 0.25,col='#bdbdbd')+
  tm_layout(
    "Western Grebe",
    legend.title.size=1.5,
    legend.text.size = 0.8,
    legend.position = c("left","bottom"),
    legend.bg.color = "white",
    legend.bg.alpha = 1)+
  tm_format("NLD")
 


#---------------------- vistualize stations and risk areas -------------------------------
tmap_mode(mode = "view")
tm_shape(php_admin40) + tm_polygons(col = "mean_impact", name='adm3_en',
                                    palette=c('#fee5d9','#fcbba1','#fc9272','#fb6a4a','#de2d26','#a50f15'),
                                    #palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
                                    breaks=c(0,2,5,7.5,9.9,10),colorNA=NULL,
                                    labels=c('   < 2%','2 - 5%','5 - 7.5%','7.5 - 10%','   > 10%'),
                                    title="Predicted Impact",
                                    alpha = 0.75,
                                    border.col = "black",lwd = 0.01,lyt='dotted')+
  tm_shape(track) +  
  tm_symbols(size=0.01,border.alpha = 0.75,col='#bdbdbd') +
  tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
  #tm_shape(goni_track) +tm_dots(size=0.1,border.alpha = 0.25,col='#bdbdbd')+
  tm_layout(
    "Western Grebe",
    legend.title.size=1.5,
    legend.text.size = 0.8,
    legend.position = c("left","bottom"),
    legend.bg.color = "white",
    legend.bg.alpha = 1)+
  tm_format("NLD")


write.csv(all_wind_goni,paste0(main_directory,"data/historical_typhoon_wind/all_wind_goni.csv"), row.names=F)



dis_data['act_rt5']<-100*with(dis_data, rowSums(select(dis_data, starts_with("ensm")) > rowMeans(select(dis_data, starts_with("rt5")), na.rm = TRUE)))/11

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

