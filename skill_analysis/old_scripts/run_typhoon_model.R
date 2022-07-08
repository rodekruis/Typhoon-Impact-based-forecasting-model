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
source('lib_r/Check_landfall_time.R')

source('lib_r/Make_maps.R')

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
# load the rr model

mode_classification <- readRDS(paste0(main_directory,"models/final_model.rds"))
mode_continious <- readRDS(paste0(main_directory,"models/final_model_regression.rds"))

# load forecast data
typhoon_info_for_model <- read.csv(paste0(main_directory,"/forecast/Input/typhoon_info_for_model.csv"))
#typhoon_events <- read.csv(paste0(main_directory,'/forecast/Input/typhoon_info_for_model.csv')) 
wshade <- php_admin3

rain_directory<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='Rainfall',]$filename)

UCL_<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='UCL',]$filename)
ECMWF_<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='ECMWF',]$filename)
HK_<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='HK',]$filename)
JTCW_<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='JTCW',]$filename)
RSMC_<-as.character(typhoon_info_for_model[typhoon_info_for_model$source=='RSMC',]$filename)

Typhoon_stormname<-as.character(unique(typhoon_info_for_model$event))#[typhoon_info_for_model$source=='UCL',]$event)
#Typhoon_stormname='VAMCO'
forecast_time<-as.character(unique(typhoon_info_for_model$time))#as.character(typhoon_info_for_model[typhoon_info_for_model$source=='UCL',]$time)
forecast_time<-str_remove_all(forecast_time, "'")

rain_directory<-ifelse(identical(character(0), rain_directory),NULL,rain_directory)
UCL_directory<-ifelse(identical(character(0), UCL_),NA,UCL_)
ECMWF_directory<-ifelse(identical(character(0), ECMWF_),NA,ECMWF_)
HK_directory<-ifelse(identical(character(0), HK_),NA,HK_)
JTCW_directory<-ifelse(identical(character(0), JTCW_),NA,JTCW_)
RSMC_directory<-ifelse(identical(character(0), RSMC_),NA,RSMC_)

############################################################################
# FOR EACH FORECASTER interpolae track data
############################################################################

dir.create(file.path(paste0(main_directory,'typhoon_infographic/shapes/', Typhoon_stormname)), showWarnings = FALSE)

typhoon_events<-c(UCL_directory,ECMWF_directory,JTCW_directory,RSMC_directory,HK_directory)


ftrack_geodb=paste0(main_directory,'typhoon_infographic/shapes/',Typhoon_stormname, '/',Typhoon_stormname,'1_',forecast_time,'_track.gpkg')

####################################################################################################

if (file.exists(ftrack_geodb)){ 
  file.remove(ftrack_geodb)
  }

####################################################################################################
for(forecaster in (typhoon_events)){
  if (file.exists(forecaster)){
  
  #TRACK_DATA<-ECMWF_ECEP %>% filter(ENSAMBLE==1) #ECMWF_ECEP %>% filter(ENSAMBLE==ensambles)
    TRACK_DATA<-read.csv(forecaster)
  TYF<- str_split(str_split(forecaster,"/")[[1]][length(str_split(forecaster,"/")[[1]])],"_")[[1]][1]
  if (TYF=="UCL")
  {

    UCL_DATA<-NA #read.csv(forecaster)
  }

  my_track <- track_interpolation(TRACK_DATA) %>% dplyr::mutate(Data_Provider=TYF)
  st_write(obj = my_track, dsn = paste0(main_directory,'typhoon_infographic/shapes/',Typhoon_stormname, '/',Typhoon_stormname,'1_',forecast_time,'_track.gpkg'),
           layer ='tc_tracks', append = TRUE)
  }
}

####################################################################################################

if (file.exists(ftrack_geodb)){
  tc_tracks<-st_read(ftrack_geodb)
}

####################################################################################################
# FOR EACH FORECASTER RUN IMPACT DMODEL , FOR NOW HK AND JTCW ARE EXCLUDED 
####################################################################################################

typhoon_events<-c(JTCW_directory)#,HK_directory)

if (!is.null(typhoon_events)) {
  for(forecaster in (typhoon_events))
    {
    if (file.exists(forecaster)){
      
       

        TRACK_DATA<-read.csv(forecaster)
        #TYF<- paste0('ECMWF_',ensambles)
        TYF<- str_split(str_split(forecaster,"/")[[1]][length(str_split(forecaster,"/")[[1]])],"_")[[1]][1]
        
	  if (TYF=="UCL")
	  {

		UCL_DATA<-NA #read.csv(forecaster)
	  }
        
        my_track <- track_interpolation(TRACK_DATA) %>% dplyr::mutate(Data_Provider=TYF)
        
        
      
      ####################################################################################################
      # check if there is a landfall
      
      print("chincking landfall")
      Landfall_check <- st_intersection(php_admin1, my_track)
      cn_rows<-nrow(Landfall_check)
      
      if (cn_rows > 0){
        print("claculating data")
        new_data<-Model_input_processing(TRACK_DATA,my_track,TYF,Typhoon_stormname)
        
        ####################################################################################################
        print("running modesl")
        
        FORECASTED_IMPACT<-run_prediction_model(data=new_data)
        
        php_admin30<-php_admin3 %>% mutate(GEN_mun_code=adm3_pcode) %>%  left_join(FORECASTED_IMPACT,by="GEN_mun_code")
        
        php_admin3_<-php_admin30 %>% dplyr::arrange(WEA_dist_track) %>%
          dplyr::mutate(impact=ifelse(WEA_dist_track> 100,NA,impact),
                        impact_threshold_passed =ifelse(WEA_dist_track > 100,NA,impact_threshold_passed))
        
        Impact<-php_admin3_  %>%   dplyr::select(GEN_mun_code,impact,impact_threshold_passed,WEA_dist_track) %>% st_set_geometry(NULL)
       
        php_admin4 <- php_admin3_ %>%  dplyr::mutate(dam_perc_comp_prediction_lm_quantile = ntile_na(impact,5)) %>% filter(WEA_dist_track < 300)
        
        region2<-extent(php_admin4)

        typhoon_region = st_bbox(c(xmin =as.vector(region2@xmin), xmax = as.vector(region2@xmax),
                                   ymin = as.vector(region2@ymin), ymax =as.vector(region2@ymax)),
                                 crs = st_crs(php_admin1)) %>% st_as_sfc()
        
       
        st_write(obj = php_admin4,
                 dsn = paste0(main_directory,'typhoon_infographic/shapes/',Typhoon_stormname, '/',Typhoon_stormname,'_',forecast_time,'_impact.gpkg'),
                 layer =TYF,
                 update = TRUE)

        ####################################################################################################
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
        #paste0('home/fbf/forecast/Impact_',as.vector(TRACK_DATA[1,]$YYYYMMDDHH),'_',as.vector(TRACK_DATA[1,]$STORMNAME),'.csv'))
        file_names<- c(paste0(main_directory,'forecast/Output/',Typhoon_stormname,'/Impact_',TYF,'_',forecast_time,'_',  Typhoon_stormname,'.png'),
                       paste0(main_directory,'forecast/Output/',Typhoon_stormname,'/Impact_',TYF,'_',forecast_time,'_', Typhoon_stormname,'.csv'))
        # paste0('home/fbf/forecast/Impact_',as.vector(TRACK_DATA[1,]$YYYYMMDDHH),'_',as.vector(TRACK_DATA[1,]$STORMNAME),'.png'),
        # paste0('home/fbf/forecast/Impact_',as.vector(TRACK_DATA[1,]$YYYYMMDDHH),'_',as.vector(TRACK_DATA[1,]$STORMNAME),'.csv'))
        write.table(file_names, file =paste0(main_directory, 'forecast/',Typhoon_stormname,'_',forecast_time,'_file_names.csv'),append=TRUE, col.names = FALSE)
        
      }
      else{
        file_names<- c(paste0('Nolandfall','_',Typhoon_stormname,'.png'), paste0('Nolandfall','_',Typhoon_stormname,'.csv'))
        write.table(file_names, file =paste0(main_directory, 'forecast/Output/',Typhoon_stormname,'_',forecast_time,'_file_names.csv'),append=TRUE, col.names = FALSE)
      }
    } 
  } ############################ if forecaster loop end here 
}  ################## close the typhone loop 

