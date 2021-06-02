#!/usr/bin/env Rscript
#options(warn=-1)
suppressMessages(library(stringr))
suppressMessages(library(ggplot2))
suppressMessages(library(dplyr))
suppressMessages(library(tidyr))
suppressMessages(library(tmap))
suppressMessages(library(httr))
suppressMessages(library(sf))
suppressMessages(library(raster))
suppressMessages(library(ranger))
suppressMessages(library(rlang))
suppressMessages(library(plyr))
suppressMessages(library(lubridate))
suppressMessages(library(rNOMADS))
suppressMessages(library(ncdf4))

args = commandArgs(trailingOnly=TRUE)
rainfall_error = args[1]

path='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/'
#path='home/fbf/'
main_directory<-path


###########################################################################
# ------------------------ import DATA  -----------------------------------
setwd(path)
source('lib_r/settings.R')
source('lib_r/data_cleaning_forecast.R')
source('lib_r/prepare_typhoon_input.R')

source('lib_r/track_interpolation.R')


source('lib_r/Read_rainfall_v2.R')
#source('lib_r/Model_input_processing.R')
source('lib_r/run_prediction_model.R')
source('lib_r/Make_maps_ens.R')

source('lib_r/Check_landfall_time.R')


# ------------------------ import DATA  -----------------------------------

php_admin3 <- st_read(dsn=paste0(main_directory,'data-raw'),layer='phl_admin3_simpl2')
php_admin1 <- st_read(dsn=paste0(main_directory,'data-raw'),layer='phl_admin1_gadm_pcode')
wshade <- php_admin3
material_variable2 <- read.csv(paste0(main_directory,"data/material_variable2.csv"))
data_matrix_new_variables <- read.csv(paste0(main_directory,"data/data_matrix_new_variables.csv"))
geo_variable <- read.csv(paste0(main_directory,"data/geo_variable.csv"))

wshade <- php_admin3 
# load the rr model
mode_classification <- readRDS(paste0(main_directory,"./models/final_model.rds"))
mode_continious <- readRDS(paste0(main_directory,"./models/final_model_regression.rds"))
mode_classification1 <- readRDS(paste0(main_directory,"./models/xgboost_classify.rds"))
mode_continious1 <- readRDS(paste0(main_directory,"./models/xgboost_regression.rds"))




# load forecast data
typhoon_info_for_model <- read.csv(paste0(main_directory,"/forecast/Input/typhoon_info_for_model.csv"))
#typhoon_events <- read.csv(paste0(main_directory,'/forecast/Input/typhoon_info_for_model.csv')) 


rain_directory<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='Rainfall',]$filename)
windfield_data<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='windfield',]$filename)
ECMWF_<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='ecmwf',]$filename)

Output_folder<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='Output_folder',]$filename)
forecast_time<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='ecmwf',]$time)
 
#------------------------- define functions ---------------------------------

ntile_na <- function(x,n){
  notna <- !is.na(x)
  out <- rep(NA_real_,length(x))
  out[notna] <- ntile(x[notna],n)
  return(out)
}

####################################################################################################
########### rainfall download function 

wind_grid <- read.csv(windfield_data)%>%dplyr::mutate(dis_track_min=ifelse(dis_track_min<1,1,dis_track_min),Mun_Code=adm3_pcode,pcode=as.factor(substr(adm3_pcode, 1, 10)))

# track wind should have a unit of knots
rainfall_<-Read_rainfall_v2(wshade)

###################################################################### 

typhoon_hazard<-wind_grid%>%
  left_join(rainfall_,by = "Mun_Code")%>%
  dplyr::mutate(typhoon_name=name,
                rainfall_24h=rainfall_24h,
                rain_mean=rainfall_24h,
                dist_track=dis_track_min,
                gust_dur=0,
                sust_dur=0,
                vmax_gust=v_max*1.49,  #sus to gust convrsion 1.49 -- 10 min average 
                vmax_gust_mph=v_max*1.49*2,23694, #mph 1.9 is factor to drive gust and sustained wind 
                vmax_sust_mph=v_max*2,23694,
                vmax_sust=v_max)%>%     #1.21 is conversion factor for 10 min average to 1min average 
  dplyr::select(Mun_Code,vmax_gust,vmax_gust_mph,vmax_sust_mph,vmax_sust,dist_track,rainfall_24h,gust_dur,sust_dur,rain_mean,storm_id,typhoon_name)



# BUILD DATA MATRIC FOR NEW TYPHOON 
data_new_typhoon<-geo_variable %>%
  left_join(material_variable2 %>% dplyr::select(-Region,-Province,-Municipality_City), by = "Mun_Code") %>%
  left_join(data_matrix_new_variables , by = "Mun_Code") %>%
  left_join(typhoon_hazard , by = "Mun_Code") 


data<-clean_typhoon_forecast_data_ensamble(data_new_typhoon)%>%na.omit() # Randomforests don't handle NAs, you can impute in the future 

model_input<-data%>%dplyr::select(-GEN_typhoon_name,
                                  -GEN_typhoon_id,
                                  -GEO_n_households,
                                  -GEN_mun_code,
                                  -index,
                                  #-GEN_mun_code,
                                  -contains('DAM_'),
                                  -GEN_mun_name)

TRACK_DATA<-read.csv(ECMWF_)%>%dplyr::mutate(STORMNAME='NA',
                                             YYYYMMDDHH=format(strptime(YYYYMMDDHH, format = "%Y-%m-%d %H:%M:%S"), '%Y%m%d%H%00'))
 
track <- track_interpolation(TRACK_DATA) 

 
########## run prediction ##########


rm.predict.pr <- predict(mode_continious,
                         data = model_input, 
                         predict.all = FALSE,
                         num.trees = mode_continious$num.trees, type = "response",
                         se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)

df_imact_forecast<-as.data.frame(rm.predict.pr$predictions) %>%
  dplyr::mutate(index= 1:length(rm.predict.pr$predictions),
                impact=rm.predict.pr$predictions) %>%
  left_join(data , by = "index") %>%
  dplyr::mutate(dist50=ifelse(WEA_dist_track >= 50,0,1),
                impact1=ifelse(impact >= 80,80,impact),
                Damaged_houses=as.integer(GEO_n_households*impact1*0.01),
                dm_low=ifelse(Damaged_houses > 0,1,0),
                dm_110k=ifelse(Damaged_houses >= 110000,4,0),
                dm_60k=ifelse(Damaged_houses >= 62000,1,0),
                dm_90k=ifelse(Damaged_houses >= 90000,2,0))%>%filter(WEA_dist_track<100)%>%dplyr::select(index,
                GEN_mun_code,
                GEN_mun_name,
                GEO_n_households,
                GEN_typhoon_name,
                GEN_typhoon_id,
                WEA_dist_track,
                WEA_vmax_sust_mhp,
                #GEN_mun_code,
                impact,
                dm_90k,
                dm_60k,
                dm_110k,
                dm_low,
                dist50,
                Damaged_houses
                #GEN_typhoon_name,
                #GEN_typhoon_id,
  )%>%drop_na()

df_damage<-aggregate(df_imact_forecast$dm_90k, by=list(adm3_pcode=df_imact_forecast$GEN_mun_code), FUN=sum)%>%
  dplyr::mutate(probability_90k=50*x/number_ensambles)%>%dplyr::select(adm3_pcode,probability_90k)%>%
  left_join(aggregate(df_imact_forecast$dm_60k, by=list(adm3_pcode=df_imact_forecast$GEN_mun_code), FUN=sum)%>%
  dplyr::mutate(probability_60k=100*x/number_ensambles)%>%dplyr::select(adm3_pcode,probability_60k),by='adm3_pcode')%>%
  left_join(aggregate(df_imact_forecast$dm_110k, by=list(adm3_pcode=df_imact_forecast$GEN_mun_code), FUN=sum)%>%
              dplyr::mutate(probability_110k=25*x/number_ensambles)%>%dplyr::select(adm3_pcode,probability_110k),by='adm3_pcode')%>%
  left_join(aggregate(df_imact_forecast$dist50, by=list(adm3_pcode=df_imact_forecast$GEN_mun_code), FUN=sum)%>%
              dplyr::mutate(probability_dist50=100*x/number_ensambles)%>%dplyr::select(adm3_pcode,probability_dist50),by='adm3_pcode')%>%
  dplyr::mutate(GEN_mun_code=adm3_pcode)


event_impact<-df_imact_forecast%>%left_join(df_damage,by='GEN_mun_code')
event_impact<-php_admin3%>%left_join(event_impact,by='adm3_pcode')

######################
Typhoon_stormname<-as.character(unique(wind_grid$name)[1])

maps<-Make_maps_ens(php_admin1,event_impact,track,TYF='ECMWF',Typhoon_stormname)
  



####################################################################################################
# ------------------------ save impact data to file   -
 
tmap_save(maps, 
          filename = paste0(Output_folder,'Impact_','_',forecast_time,'_',  Typhoon_stormname,'.png'),
          width=20, height=24,dpi=600,
          units="cm")

####################################################################################################
# ------------------------ save impact data to file   -----------------------------------
 
write.csv(event_impact, file = paste0(Output_folder,'Impact_','_',forecast_time,'_',  Typhoon_stormname,'.csv'))
 
file_names<- c(paste0(Output_folder,'Impact_','_',forecast_time,'_',  Typhoon_stormname,'.png'),
               paste0(Output_folder,'Impact_','_',forecast_time,'_',  Typhoon_stormname,'.csv'))
 
write.table(file_names, file =paste0(main_directory, 'forecast/',Typhoon_stormname,'_',forecast_time,'_file_names.csv'),append=TRUE, col.names = FALSE)





# 
# 
# #---------------------- vistualize landfall location probability -------------------------------
# 
# tmap_mode(mode = "view")
# tm_shape(event_impact) + tm_polygons(col = "probability_dist50", name='adm3_en',
#                                      palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
#                                      breaks=c(0,5,10,20,40,50),colorNA=NULL,
#                                      labels=c('   < 5%','5 - 10%','10 - 20%','20 - 40%','   > 50%'),
#                                      title=paste0("Tphoon- ", typhoonname," Probability for \n distance from Track < 50km"),
#                                      alpha = 0.75,
#                                      border.col = "black",lwd = 0.01,lyt='dotted')+
#   tm_shape(track) + tm_symbols(size=0.08,border.alpha = 0.75,col='#0c2c84') +
#   tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
#   
#   tm_format("NLD")
# 
# #---------------------- vistualize impact probability -------------------------------
# 
# tmap_mode(mode = "view")#view
# 
# tm_shape(event_impact) + tm_polygons(col = "probability_90k", name='adm3_en',
#                                      palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
#                                      breaks=c(0,40,50,60,90,100),colorNA=NULL,
#                                      labels=c('   < 40%','40 - 50%','50 - 60%','60 - 90%','   > 90%'),
#                                      title= " Probability for #Dam. Buld >90k",
#                                      alpha = 0.75,
#                                      border.col = "black",lwd = 0.01,lyt='dotted')+
#   tm_shape(track) +  
#   tm_symbols(size=0.1,border.alpha = 0.75,col='#0c2c84') +
#   tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
#   #tm_shape(goni_track) +tm_dots(size=0.1,border.alpha = 0.25,col='#bdbdbd')+
#   tm_format("NLD")
# 
# 
# 
# 


