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
best_track_27ty<-read.csv(paste0(main_directory,"./data-raw/best_track_27ty.csv"))
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


######################################################
 
files<-list.files('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model//data/past_typhoon_wind/',
                  pattern='_intensity.csv', all.files=FALSE,full.names=TRUE)

listdf<-list()
for(file_name in files){
  
  typhoon_wind <- read.csv(file_name)%>%dplyr::select(-typhoon)%>%
    dplyr::mutate(dis_track_min=ifelse(dis_track_min<1,1,dis_track_min),
                  typhoon=as.factor(strsplit(strsplit(file_name,'/')[[1]][11],'_')[[1]][1]),
                  Mun_Code=adm3_pcode,
                  pcode=as.factor(substr(adm3_pcode, 1, 10)))

  listdf[[file_name]] <-typhoon_wind
  
  
}
past_typhoon_wind<-bind_rows(listdf)

#################################################################

past_typhoon_rain <- read.csv(paste0(main_directory,"./data/past_typhoon_wind/PHL_admin3_zonal_statistics_2021_05_13.csv"))
listdf<-list()
for(typhoonnames in unique(past_typhoon_rain$typhoon_name)){
  
  typhoon_rain <- past_typhoon_rain%>%filter(typhoon_name==typhoonnames)%>%
    group_by(pcode)%>%
    dplyr::summarise(rain_max=max(value)/10,rain_mean = mean(value)/10)%>%ungroup()

    listdf[[typhoonnames]] <-typhoon_rain%>%dplyr::mutate(Mun_Code=paste0(pcode,'0'),typhoon=as.factor(typhoonnames))
  
}

past_typhoon_rain<-bind_rows(listdf)

###################################################################### 

past_typhoon_hazard<-past_typhoon_wind%>%
  left_join(past_typhoon_rain,by = c("Mun_Code","typhoon"))%>%
  dplyr::mutate(typhoon_name=typhoon,
                rainfall_24h=rain_max,
                dist_track=dis_track_min,
                gust_dur=0,
                sust_dur=0,
                vmax_gust=v_max*1.21*1.9*1.94384,  #knot(1.94384) and 1.21 is conversion factor for 10 min average to 1min average
                vmax_gust_mph=v_max*1.21*1.9*2,23694, #mph 1.9 is factor to drive gust and sustained wind 
                vmax_sust_mph=v_max*1.21*2,23694,
                vmax_sust=v_max*1.21*1.94384)%>%     #knot(1.94384) and 1.21 is conversion factor for 10 min average to 1min average 
  dplyr::select(Mun_Code,vmax_gust,vmax_gust_mph,vmax_sust_mph,vmax_sust,dist_track,rainfall_24h,gust_dur,sust_dur,rain_mean,storm_id,typhoon_name)





# BUILD DATA MATRIC FOR pre disaster indicators
data_pre_disaster<-geo_variable%>%
  left_join(material_variable2 %>% dplyr::select(-Region,-Province,-Municipality_City), by = "Mun_Code") %>%
  left_join(data_matrix_new_variables , by = "Mun_Code")


# BUILD DATA MATRIC FOR NEW TYPHOON 
data_new_typhoon<-past_typhoon_hazard%>%left_join(data_pre_disaster,by="Mun_Code")


data<-clean_typhoon_forecast_data_ensamble(data_new_typhoon)%>%na.omit() # Randomforests don't handle NAs, you can impute in the future 

model_input<-data%>%dplyr::select(-GEN_typhoon_name,
                -GEN_typhoon_id,
                -GEO_n_households,
                #-GEN_mun_code,
                -contains('DAM_'),
                -GEN_mun_name) 


########## run prediction ##########

rm.predict.pr <- predict(mode_classification,
                         data = model_input,
                         predict.all = FALSE,
                         num.trees = mode_classification$num.trees, type = "response",
                         se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)

FORECASTED_IMPACT<-as.data.frame(rm.predict.pr$predictions) %>%
  dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact_threshold_passed=rm.predict.pr$predictions) %>%
  left_join(data , by = "index") %>%
  dplyr::select(index,
                GEN_mun_code,
                GEO_n_households,
                impact_threshold_passed,
                GEN_typhoon_name,
                GEN_typhoon_id,
                WEA_dist_track,
                WEA_vmax_sust_mhp)%>%drop_na()

#colnames(FORECASTED_IMPACT) <- c(GEN_mun_code,paste0(TYF,'_impact_threshold_passed'),WEA_dist_track)

rm.predict.pr <- predict(mode_continious,
                         data = model_input, 
                         predict.all = FALSE,
                         num.trees = mode_continious$num.trees, type = "response",
                         se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)

FORECASTED_IMPACT_rr<-as.data.frame(rm.predict.pr$predictions) %>%
  dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact=rm.predict.pr$predictions) %>% 
  dplyr::mutate(priority_index=ntile_na(ifelse(impact<0,NA,impact),5))%>%
  left_join(data , by = "index") %>%
  dplyr::select(index,
                GEN_mun_code,
                GEO_n_households,
                GEN_typhoon_name,
                GEN_typhoon_id,
                WEA_dist_track,
                WEA_vmax_sust_mhp,
                #GEN_mun_code,
                impact,
                #GEN_typhoon_name,
                #GEN_typhoon_id,
                priority_index)%>%drop_na()


#df_imact_forecast<-FORECASTED_IMPACT %>% left_join(FORECASTED_IMPACT_rr,by='index')
df_imact_forecast<-FORECASTED_IMPACT_rr# %>% left_join(FORECASTED_IMPACT_rr,by='index')

class_<-c("A","B","C","Low")

# all_impact <- df_imact_forecast%>%
#   dplyr::mutate(dist50=ifelse(WEA_dist_track >= 50,0,1),Damaged_houses=as.integer(GEO_n_households*impact*0.01),
#                 class_B = case_when(impact >= 5 & impact < 8 ~ "C",impact >= 8 & impact < 10 ~ "B",impact >= 10 ~ "A",TRUE ~ 'Low'))

all_impact <- df_imact_forecast%>%
  dplyr::mutate(dist50=ifelse(WEA_dist_track >= 50,0,1),Damaged_houses=as.integer(GEO_n_households*impact*0.01),
                class_B = case_when(impact >= 5 ~ "C",impact >= 8 ~ "B",impact >= 10 ~ "A",TRUE ~ 'Low'))

listdf<-list()
for(typhoonnames in unique(all_impact$GEN_typhoon_name)){
 
  
  typhoon_impact <- all_impact%>%filter(GEN_typhoon_name==typhoonnames)
  
  number_ensambles<-length(unique(typhoon_impact$GEN_typhoon_id))
  
  df <- typhoon_impact %>%group_by(GEN_mun_code)%>%
    group_map(~sapply(1:length(class_),function(i){.x %>% filter(class_B ==class_[i])%>% nrow*100/number_ensambles}))
  
 
  
  df <- data.frame(matrix(unlist(df), nrow=length(df), byrow=TRUE))
  names(df)<-class_
 df$adm3_pcode<-unique(typhoon_impact$GEN_mun_code)
  
  
  
  ######################
  
  df2<-aggregate(typhoon_impact$dist50, by=list(adm3_pcode=typhoon_impact$GEN_mun_code), FUN=sum)%>%
    dplyr::mutate(probability=100*x/number_ensambles)%>%dplyr::select(adm3_pcode,probability)
  
 listdf[[typhoonnames]] <-df%>%dplyr::mutate(typhoon=as.factor(typhoonnames))%>%left_join(df2,by="adm3_pcode")
  
}

past_impact<-bind_rows(listdf)%>%
  left_join(data%>%dplyr::select(GEN_mun_code,GEO_n_households),by=c('adm3_pcode'='GEN_mun_code'))


past_impact<-unique(past_impact)


#' dict_typhoones={'bopha':'2012-12-04 04:45:00',
#'   #'durian':'2006-11-30 00:00:00',
#'   #'fengshen':'2008-06-21 12:00:00',
#'   #'ketsana':'2009-09-26 00:00:00',
#'   #'washi':'2011-12-16 00:00:00',
#'   'haiyan':'2013-11-08 00:00:00',
#'   'hagupit':'2014-12-06 23:00:00',
#'   'haima':'2016-10-19 23:00:00',
#'   'nock-ten':'2016-12-25 18:00:00',
#'   'mangkhut':'2018-09-15 01:40:00',
#'   'kammuri':'2019-12-02 20:00:00',
#'   #'phanfone':'2019-12-24 00:00:00',
#'   #'vongfong':'2020-05-14 00:00:00',
#'   #'molave':'2020-10-25 18:10:00',
#'   'goni':'2020-11-01 05:00:00'}
    

#'bopha','durian','fengshen','ketsana','washi','haiyan','hagupit','haima','nock-ten','mangkhut','kammuri','phanfone','vongfong','molave','goni'

typhoonname='haima'
TRACK_DATA1<- best_track_27ty%>%
  filter(STORMNAME==toupper(typhoonname))%>%
  dplyr::select(YYYYMMDDHH,LAT,LON,VMAX,STORMNAME)#typhoon = paste0(TRACK_DATA$STORMNAME,substr(TRACK_DATA$YYYYMMDDHH, 1, 4)))

track <- track_interpolation(TRACK_DATA1)
#past_impact<-unique(past_impact)

event_impact<-past_impact%>% filter(typhoon==typhoonname,probability>0) 
 

event_impact<-php_admin3%>%left_join(event_impact,by='adm3_pcode')

#YYYYMMDDHH,LAT,LON,VMAX,STORMNAME
#---------------------- vistualize landfall location probability -------------------------------

tmap_mode(mode = "view")
tm_shape(event_impact) + tm_polygons(col = "probability", name='adm3_en',
                                       palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
                                       breaks=c(0,5,10,20,40,50),colorNA=NULL,
                                       labels=c('   < 5%','5 - 10%','10 - 20%','20 - 40%','   > 50%'),
                                       title=paste0("Tphoon- ", typhoonname," Probability for \n distance from Track < 50km"),
                                   alpha = 0.75,
                                       border.col = "black",lwd = 0.01,lyt='dotted')+
  tm_shape(track) +  
  tm_symbols(size=0.08,border.alpha = 0.75,col='#0c2c84') +
  tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
  
  tm_format("NLD")

#---------------------- vistualize impact probability -------------------------------

tmap_mode(mode = "view")#view

tm_shape(event_impact) + tm_polygons(col = "A", name='adm3_en',
                                   palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
                                   breaks=c(0,50,70,80,90,100),colorNA=NULL,
                                   labels=c('   < 50%','70 - 70%','70 - 80%','80 - 90%','   > 90%'),
                                   title=paste0("Tphoon- ", typhoonname," Probability for >10 % DAMAGE"),
                                   alpha = 0.75,
                                   border.col = "black",lwd = 0.01,lyt='dotted')+
  tm_shape(track) +  
  tm_symbols(size=0.01,border.alpha = 0.75,col='#0c2c84') +
  tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
  #tm_shape(goni_track) +tm_dots(size=0.1,border.alpha = 0.25,col='#bdbdbd')+
  tm_format("NLD")

tmap_mode(mode = "view")

tm_shape(event_impact) + tm_polygons(col = "B", name='adm3_en',
                                     palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
                                     breaks=c(0,50,70,80,90,100),colorNA=NULL,
                                     labels=c('   < 50%','70 - 70%','70 - 80%','80 - 90%','   > 90%'),
                                     title=paste0("Tphoon- ", typhoonname," Probability for >8 % DAMAGE"),
                                     alpha = 0.75,
                                     border.col = "black",lwd = 0.01,lyt='dotted')+
  tm_shape(track) +  
  tm_symbols(size=0.01,border.alpha = 0.75,col='#0c2c84') +
  tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
  #tm_shape(goni_track) +tm_dots(size=0.1,border.alpha = 0.25,col='#bdbdbd')+
  tm_format("NLD")

tmap_mode(mode = "view")

tm_shape(event_impact) + tm_polygons(col = "C", name='adm3_en',
                                     palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
                                     breaks=c(0,50,70,80,90,100),colorNA=NULL,
                                     labels=c('   < 50%','70 - 70%','70 - 80%','80 - 90%','   > 90%'),
                                     title=paste0("Tphoon- ", typhoonname," Probability for >5 % DAMAGE"),
                                     alpha = 0.75,
                                     border.col = "black",lwd = 0.01,lyt='dotted')+
  tm_shape(track) +  
  tm_symbols(size=0.01,border.alpha = 0.75,col='#0c2c84') +
  tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
  #tm_shape(goni_track) +tm_dots(size=0.1,border.alpha = 0.25,col='#bdbdbd')+
  tm_format("NLD")

#---------------------- vistualize impact probability -------------------------------

tmap_mode(mode = "view")

tm_shape(php_impact) + tm_polygons(col = "A", name='adm3_en',
                                   palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
                                   breaks=c(0,50,70,80,90,100),colorNA=NULL,
                                   labels=c('   < 50%','70 - 70%','70 - 80%','80 - 90%','   > 90%'),
                                   title=paste0("Tphoon- ", typhoonname,"Probability for >10 % DAMAGE"),
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




 
 




php_admin4<-php_admin3%>%left_join(aggregate(all_wind$dist50, by=list(adm3_pcode=all_wind$GEN_mun_code), FUN=sum)%>%
                                     dplyr::mutate(probability=100*x/52)%>%dplyr::select(adm3_pcode,probability),by='adm3_pcode')%>%
  left_join(aggregate(all_wind$WEA_vmax_sust_mhp, by=list(adm3_pcode=all_wind$GEN_mun_code), FUN=max)%>%
              dplyr::mutate(VMAX=x)%>%dplyr::select(adm3_pcode,VMAX))%>%
  left_join(geo_variable%>%dplyr::select(Mun_Code,with_coast),by = c('adm3_pcode'='Mun_Code'))























