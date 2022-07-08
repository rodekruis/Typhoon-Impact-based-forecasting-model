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
suppressMessages(library(rlang))
suppressMessages(library(MLmetrics))
suppressMessages(library(plyr))
suppressMessages(library(lubridate))
suppressMessages(library(rNOMADS))
suppressMessages(library(ncdf4))
suppressMessages(library(parallel))
suppressMessages(library(xgboost))

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

source('lib_r/prepare_typhoon_input.R')
source('lib_r/track_interpolation.R')
source('lib_r/Read_rainfall_v2.R')
source('lib_r/Model_input_processing.R')
source('lib_r/run_prediction_model.R')
source('lib_r/Make_maps.R')
library(huxtable)
source('lib_r/Check_landfall_time.R')

#source('home/fbf/prepare_typhoon_input.R')
#source('home/fbf/settings.R')
#source('home/fbf/data_cleaning_forecast.R')

geo_variable <- read.csv(paste0(main_directory,"./data/geo_variable.csv"))
best_track_27ty<-read.csv(paste0(main_directory,"./data-raw/best_track_27ty.csv"))
wshade <- php_admin3 
# load the rr model

xgmodel<-readRDS(paste0(main_directory,"/models/operational/xgboost_regression_v2.RDS"), refhook = NULL)

#------------------------- define functions ---------------------------------

ntile_na <- function(x,n){
  notna <- !is.na(x)
  out <- rep(NA_real_,length(x))
  out[notna] <- ntile(x[notna],n)
  return(out)
}


######################################################
 
typhoon_wind<-read.csv('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/trigger_level_analysis/KAMMURI_all_intensity.csv')


past_typhoon_wind<-typhoon_wind%>%
  dplyr::mutate(Mun_Code=adm3_pcode) 


#################################################################

past_typhoon_rain <- read.csv(paste0(main_directory,"./data/past_typhoon_wind/PHL_admin3_zonal_statistics_2021_05_13.csv"))%>%
  filter(typhoon_name=='kammuri')%>%group_by(pcode)%>%
  dplyr::summarise(rain_max=max(value)/10,rain_mean = mean(value)/10)%>%ungroup()%>%
  dplyr::mutate(Mun_Code=paste0(pcode,'0'))%>%dplyr::select(-pcode)


###################################################################### 

past_typhoon_hazard <- past_typhoon_wind%>%
  left_join(past_typhoon_rain,by = "Mun_Code")%>%
  dplyr::mutate(typhoon_name=typhoon,
                rainfall_24h=rain_max,
                ranfall_sum=rain_max,
                dist_track=dis_track_min,
                gust_dur=0,
                sust_dur=0,
                vmax_gust=v_max*1.49*1.21,  #sus to gust convrsion 1.49 -- 10 min average 
                vmax_gust_mph=v_max*1.49*2,23694*1.21, #mph 1.9 is factor to drive gust and sustained wind 
                vmax_sust_mph=v_max*2,23694*1.21,#mph
                vmax_sust=v_max*1.21)%>%     #1.21 is conversion factor for 10 min average to 1min average 
  dplyr::select(Mun_Code,vmax_gust,vmax_gust_mph,vmax_sust_mph,vmax_sust,dist_track,rainfall_24h,gust_dur,sust_dur,ranfall_sum,storm_id,typhoon_name)





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
                                  -GEN_mun_code,
                                  -index,
                                  #-GEN_mun_code,
                                  #-contains("INT_"),
                                  -contains('DAM_'),
                                  -GEN_mun_name)


####################################################################################################
# ------------------------run prediction   -----------------------------------


test_x       <- data.matrix(model_input)
xgb_test     <- xgb.DMatrix(data=test_x)

y_predicted  <- predict(xgmodel, xgb_test)


df_imact_forecast_all <- as.data.frame(y_predicted)%>%
  dplyr::mutate(index= 1:length(y_predicted),impact=y_predicted)%>%left_join(data , by = "index")%>%
  dplyr::mutate(region=substr(GEN_mun_code,1,4))#%>%filter(region%in%c('PH05','PH08'))



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

typhoonname='kammuri112600 ' #df_imact_forecast_all%>% filter(GEN_typhoon_name==typhoonname)


  
df_imact_forecast<-df_imact_forecast_all%>%dplyr::mutate(dist50=ifelse(WEA_dist_track >= 50,0,1),
                                                  impact1=ifelse(impact > 100,100,impact),
                                                  Damaged_houses=as.integer(GEO_n_households*impact1*0.01),
                                                  )%>%
  filter(WEA_dist_track<100)%>%dplyr::select(index,
                GEN_mun_code,
                GEN_mun_name,
                GEO_n_households,
                GEN_typhoon_name,
                GEN_typhoon_id,
                WEA_dist_track,
                WEA_vmax_sust_mhp,
                #GEN_mun_code,
                impact,
                dist50,
                Damaged_houses
                #GEN_typhoon_name,
                #GEN_typhoon_id,
  )%>%drop_na()

####################################################################################################
# ------------------------ calculate  probability only for region 5 and 8  -----------------------------------

df_imact_forecast <- df_imact_forecast_all%>%#filter(region%in%c('PH05','PH08'))%>%
  filter(GEN_typhoon_name=='kammuri112600')

df_imact_forecast<-df_imact_forecast%>%dplyr::mutate(dist50=ifelse(WEA_dist_track >= 50,0,1),
                                                         impact1=ifelse(impact > 100,100,impact),
                                                         Damaged_houses=as.integer(GEO_n_households*impact1*0.01),
)%>%
  filter(WEA_dist_track<100)%>%dplyr::select(index,
                                             GEN_mun_code,
                                             GEN_mun_name,
                                             GEO_n_households,
                                             GEN_typhoon_name,
                                             GEN_typhoon_id,
                                             WEA_dist_track,
                                             WEA_vmax_sust_mhp,
                                             #GEN_mun_code,
                                             impact,
                                             dist50,
                                             Damaged_houses
                                             #GEN_typhoon_name,
                                             #GEN_typhoon_id,
  )%>%drop_na()


df_imact_forecast_85<-df_imact_forecast%>%filter(region%in%c('PH05','PH08'))%>%
  dplyr::summarise(CDamaged_houses = sum(Damaged_houses))%>%
  dplyr::mutate(DM_CLASS = ifelse(CDamaged_houses >= 80000,5,
                                  ifelse(CDamaged_houses >= 50000,4,
                                         ifelse(CDamaged_houses >= 30000,3,
                                                ifelse(CDamaged_houses >= 10000,2,
                                                       ifelse(CDamaged_houses >= 5000,1, 0))))))%>%
  ungroup()%>%group_by(GEN_typhoon_name)%>%
  dplyr::summarise(VH_80K = round(100*sum(DM_CLASS>=5)/52),
                   H_50K = round(100*sum(DM_CLASS>=4)/52),
                   H_30K = round(100*sum(DM_CLASS>=3)/52),
                   M_10K = round(100*sum(DM_CLASS >=2)/52),
                   L_5K = round(100*sum(DM_CLASS>=1)/52))%>%dplyr::rename(Typhoon_name=GEN_typhoon_name)%>% 
  mutate(
    Typhoon_name = toupper(Typhoon_name)
  )%>%
  as_hux()%>%
  set_text_color(1, everywhere, "blue")%>%
  theme_article()%>%set_caption("PROBABILITY FOR THE NUMBER OF COMPLETELY DAMAGED BUILDINGS") #(Region 5 and 8)
  



####################################################################################################
# ------------------------ calculate and plot probability   -----------------------------------

df_imact_forecast%>%group_by(GEN_typhoon_name,GEN_typhoon_id)%>%
  dplyr::summarise(CDamaged_houses = sum(Damaged_houses))%>%
  dplyr::mutate(DM_CLASS = ifelse(CDamaged_houses >= 100000,5,
                             ifelse(CDamaged_houses >= 80000,4,
                                    ifelse(CDamaged_houses >= 70000,3,
                                           ifelse(CDamaged_houses >= 50000,2,
                                                  ifelse(CDamaged_houses >= 30000,1, 0))))))%>%
  ungroup()%>%group_by(GEN_typhoon_name)%>%
  dplyr::summarise(VH_100K = round(100*sum(DM_CLASS>=5)/52),
            H_80K = round(100*sum(DM_CLASS>=4)/52),
            H_70K = round(100*sum(DM_CLASS>=3)/52),
            M_50K = round(100*sum(DM_CLASS >=2)/52),
            L_30K = round(100*sum(DM_CLASS>=1)/52))%>%dplyr::rename(Typhoon_name=GEN_typhoon_name)%>% 
  mutate(
    Typhoon_name = toupper(Typhoon_name)
  )%>%
  as_hux()%>%
  set_text_color(1, everywhere, "blue")%>%
  theme_article()%>%set_caption("PROBABILITY FOR THE NUMBER OF COMPLETELY DAMAGED BUILDINGS")%>% 
  set_text_color(2, 1, "red")%>%
  set_text_color(2, 5, "red")%>%
  set_text_color(3, 1:4, "red")%>%
  set_text_color(4, 1, "red")%>% 
  set_text_color(4, 4:5, "red")%>% 
  set_text_color(7, 1:2, "red")%>% 
  set_text_color(7, 4:5, "red")%>% 
  set_text_color(9, 1, "red")%>%
  set_text_color(9, 5, "red")




####################################################################################################
 
 


aggregate(df_imact_forecast$Damaged_houses, by=list(adm3_pcode=df_imact_forecast$GEN_typhoon_id), FUN=sum)%>%
  mutate(Stu = ifelse(x >= 10000, 4,
                      ifelse(x >= 5000, 3,
                             ifelse(x >= 2000, 2, 1))))%>%group_by(Stu)%>%
  summarise(Vhigh = sum(Stu>=4)/51,
            high = sum(Stu>=3)/51,
            medium = sum(Stu >=2)/51,
            low = sum(Stu<=1)/51)


aggregate(df_imact_forecast$Damaged_houses, by=list(adm3_pcode=df_imact_forecast$GEN_typhoon_id), FUN=sum)%>%
  dplyr::mutate(dm_low=ifelse(x <20000,1,0),
                dm_80k=ifelse(x >= 80000,4,0),
                dm_20k=ifelse(x >= 10000,1,0),
                dm_50k=ifelse(x >= 50000,2,0),
                probability_50k=50*x/number_ensambles)%>%dplyr::select(adm3_pcode,probability_50k)



TRACK_DATA1<- best_track_27ty%>%
  filter(STORMNAME==toupper(typhoonname))%>%
  dplyr::select(YYYYMMDDHH,LAT,LON,VMAX,STORMNAME)#typhoon = paste0(TRACK_DATA$STORMNAME,substr(TRACK_DATA$YYYYMMDDHH, 1, 4)))

track <- track_interpolation(TRACK_DATA1)
#past_impact<-unique(past_impact)

df_imact_forecast<-df_imact_forecast_all%>% filter(GEN_typhoon_name==typhoonname) 

number_ensambles<-length(unique(df_imact_forecast$GEN_typhoon_id))

####################################################################################################
# ------------------------ calculate probability   -----------------------------------

df_damage  <- aggregate(df_imact_forecast$dm_50k, by=list(adm3_pcode=df_imact_forecast$GEN_mun_code), FUN=sum)%>%
  dplyr::mutate(probability_50k=50*x/number_ensambles)%>%dplyr::select(adm3_pcode,probability_50k)%>%
  left_join(aggregate(df_imact_forecast$dm_20k, by=list(adm3_pcode=df_imact_forecast$GEN_mun_code), FUN=sum)%>%
              dplyr::mutate(probability_20k=100*x/number_ensambles)%>%dplyr::select(adm3_pcode,probability_20k),by='adm3_pcode')%>%
  left_join(aggregate(df_imact_forecast$dm_80k, by=list(adm3_pcode=df_imact_forecast$GEN_mun_code), FUN=sum)%>%
              dplyr::mutate(probability_80k=25*x/number_ensambles)%>%dplyr::select(adm3_pcode,probability_80k),by='adm3_pcode')%>%
  left_join(aggregate(df_imact_forecast$dist50, by=list(adm3_pcode=df_imact_forecast$GEN_mun_code), FUN=sum)%>%
              dplyr::mutate(probability_dist50=100*x/number_ensambles)%>%dplyr::select(adm3_pcode,probability_dist50),by='adm3_pcode')%>%
  dplyr::mutate(GEN_mun_code=adm3_pcode)


past_impact <- df_imact_forecast%>%left_join(df_damage,by='GEN_mun_code')
past_impact<-php_admin3%>%left_join(past_impact,by='adm3_pcode')

#YYYYMMDDHH,LAT,LON,VMAX,STORMNAME
#---------------------- vistualize landfall location probability -------------------------------

tmap_mode(mode = "view")
tm_shape(past_impact) + tm_polygons(col = "probability_dist50", name='adm3_en',
                                       palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
                                       breaks=c(0,5,10,20,40,50),colorNA=NULL,
                                       labels=c('   < 5%','5 - 10%','10 - 20%','20 - 40%','   > 50%'),
                                       title=paste0("Tphoon- ", typhoonname," Probability for \n distance from Track < 50km"),
                                   alpha = 0.75,
                                       border.col = "black",lwd = 0.01,lyt='dotted')+
  tm_shape(track) +  
  tm_symbols(size=0.02,border.alpha = 0.75,col='#0c2c84') +
  tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
  
  tm_format("NLD")

#---------------------- vistualize impact probability -------------------------------

tmap_mode(mode = "view")#view

tm_shape(event_impact) + tm_polygons(col = "probability_90k", name='adm3_en',
                                   palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
                                   breaks=c(0,5,10,20,40,50),colorNA=NULL,
                                   labels=c('   < 5%','5 - 10%','10 - 20%','20 - 40%','   > 50%'),
                                   title=paste0("Tphoon- ", typhoonname," Probability for >10 % DAMAGE"),
                                   alpha = 0.75,
                                   border.col = "black",lwd = 0.01,lyt='dotted')+
  tm_shape(track) +  
  tm_symbols(size=0.02,border.alpha = 0.75,col='#0c2c84') +
  tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
  #tm_shape(goni_track) +tm_dots(size=0.1,border.alpha = 0.25,col='#bdbdbd')+
  tm_format("NLD")



