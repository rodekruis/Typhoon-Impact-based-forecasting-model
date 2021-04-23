rm(list=ls())
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
suppressMessages(library(readr))
suppressMessages(library(janitor))
 

path='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/'
 
main_directory<-path

####################################################################################################

setwd(path)
source('lib_r/settings.R')
source('lib_r/data_cleaning_forecast.R')
 
source('lib_r/prepare_typhoon_input.R')
source('lib_r/track_interpolation.R')
source('lib_r/Read_rainfall_v2.R')
source('lib_r/Model_input_processing.R')
source('lib_r/run_prediction_model.R')
source('lib_r/Make_maps.R')

source('lib_r/Check_landfall_time.R')

#------------------------- define functions ---------------------------------

ntile_na <- function(x,n){
  notna <- !is.na(x)
  out <- rep(NA_real_,length(x))
  out[notna] <- ntile(x[notna],n)
  return(out)
} 




wind_input_processing<-function(TRACK_DATA,my_track,Typhoon_stormname){
  
  ###################### Model_input_processing ##############
  
  #generate tack with smaller time steps 
  my_track <- my_track
  Typhoon_stormname<-Typhoon_stormname
  grid_points_adm3<-grid_points_adm3 # %>% filter(gridid==pcod_SB)
  TRACK_DATA1<- TRACK_DATA %>% dplyr::mutate(typhoon = paste0(TRACK_DATA$STORMNAME,substr(TRACK_DATA$YYYYMMDDHH, 1, 4)))
  wind_grids <- get_grid_winds(hurr_track=TRACK_DATA1, grid_df=grid_points_adm3, tint = 0.5,gust_duration_cut = 15, sust_duration_cut = 15)
  
  wind_grids<- wind_grids %>%
    dplyr::mutate(typhoon_name=as.vector(TRACK_DATA$STORMNAME)[1],fips = as.numeric(substr(gridid, 3, 11)),vmax_gust_mph=vmax_gust*2.23694,vmax_sust_mph=vmax_sust*2.23694) %>%
    dplyr::select(typhoon_name,fips,gridid,vmax_gust,vmax_gust_mph,vmax_sust,vmax_sust_mph,gust_dur,sust_dur,dist_track,glon,glat)
  
  wind_grid<-wind_grids%>%
    dplyr::mutate(Mun_Code=gridid)%>%
    dplyr::select(-fips,-gridid)
  
  
  return(wind_grid)
}


grid_points_adm3<-read.csv(paste0(main_directory,"data-raw/grid_points_admin3.csv"), sep=",")

#setwd(dirname(rstudioapi::getSourceEditorContext()$path))
setwd('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/data')

#YYYYMMDDHH,LAT,LON,VMAX,STORMNAME

Typhoon_events<- readr::read_csv("./Impact data/Typhoon_events_with_impact_data.csv")%>%
  dplyr::mutate(International_Name=toupper(International_Name),STORMNAME1=paste0(International_Name,year))

historical_typhoon_wind<- read.csv("./historical_typhoon_wind/ibtracs.WP.list.v04r00.csv")%>%
  clean_names()%>%dplyr::mutate(name=toupper(name),
                                YYYYMMDDHH=format(strptime(iso_time, format = "%Y-%m-%d %H:%M:%S"), '%Y%m%d%H%S'),
                                LAT=lat,
                                LON=lon,
                                prcen=usa_pres,
                                VMAX=usa_wind,#ifelse(dist2land<1,0.84*usa_wind,ifelse(dist2land>50,0.93*usa_wind,0.87*usa_wind)),
                                season=as.numeric(season),
                                VMAX2=wmo_wind,#ifelse(!is.na(wmo_wind),wmo_wind,V_usa),
                                STORMNAME1=paste0(name,season),
                                STORMNAME=name)%>%
  dplyr::select(YYYYMMDDHH,LAT,LON,VMAX,prcen,STORMNAME1,STORMNAME,season,nature,landfall,dist2land)%>%
  dplyr::filter(season>2005 & STORMNAME1 %in% unique(Typhoon_events$STORMNAME1))


 
goni<-historical_typhoon_wind%>%filter(STORMNAME=='GONI'&season=='2020')

TRACK_DATA<-goni   

TYF<- 'goni'

my_track <- track_interpolation(TRACK_DATA) %>% dplyr::mutate(Data_Provider=TYF)

 

new_data<-wind_input_processing(TRACK_DATA,my_track,event_name)%>% dplyr::mutate(Ty_year=TYF)

####################################################################################################

 
############################################################################
# FOR EACH FORECASTER interpolae track data
############################################################################

 



 

ftrack_geodb=paste0(main_directory,'data/historicaltrack_track.gpkg')


if (file.exists(ftrack_geodb)){ file.remove(ftrack_geodb)}

wind_dfs <- list()
for(event_name in (unique(historical_typhoon_wind$STORMNAME1)))
{
  TRACK_DATA<-historical_typhoon_wind  %>% filter(STORMNAME1==event_name)
  
  
  TYF<- event_name
  
  my_track <- track_interpolation(TRACK_DATA) %>% dplyr::mutate(Data_Provider=TYF)
  
  st_write(obj = my_track,
           dsn = paste0(main_directory,'data/historicaltrack_track.gpkg'), 
           layer ='tc_tracks', 
           append = TRUE)
  
  new_data<-wind_input_processing(TRACK_DATA,my_track,event_name)%>% dplyr::mutate(Ty_year=TYF)
  
  wind_dfs[[event_name]] <- new_data
  
}

# Combining files
 
all_wind <- bind_rows(wind_dfs)

write.csv(all_wind,paste0(main_directory,"data/historical_typhoon_wind/historical_typhoon_windfield.csv"), row.names=F)













