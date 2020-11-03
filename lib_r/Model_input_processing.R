#!/usr/bin/env Rscript
#options(warn=-1)
Model_input_processing<-function(TRACK_DATA,my_track,TYF,Typhoon_stormname){
  
  ###################### Model_input_processing ##############
  
  #generate tack with smaller time steps 
  my_track <- my_track
  
  Typhoon_stormname<-Typhoon_stormname
  
  
  
  
  TRACK_DATA1<- TRACK_DATA %>% dplyr::mutate(typhoon = paste0(TRACK_DATA$STORMNAME,substr(TRACK_DATA$YYYYMMDDHH, 1, 4)))
  
  wind_grids <- get_grid_winds(hurr_track=TRACK_DATA1, grid_df=grid_points_adm3, tint = 3,gust_duration_cut = 15, sust_duration_cut = 15)
  
  wind_grids<- wind_grids %>%
    dplyr::mutate(typhoon_name=as.vector(TRACK_DATA$STORMNAME)[1],fips = as.numeric(substr(gridid, 3, 11)),vmax_gust_mph=vmax_gust*2.23694,vmax_sust_mph=vmax_sust*2.23694) %>%
    dplyr::select(typhoon_name,fips,gridid,vmax_gust,vmax_gust_mph,vmax_sust,vmax_sust_mph,gust_dur,sust_dur,dist_track,glon,glat)
  
  wind_grid<-wind_grids%>%
    dplyr::mutate(Mun_Code=gridid)%>%
    dplyr::select(-fips,-gridid)
  
  # track wind should have a unit of knots
  rainfall_<-Read_rainfall_v2(wshade)
  
  
  # BUILD DATA MATRIC FOR NEW TYPHOON 
  data_new_typhoon<-geo_variable %>%
    left_join(material_variable2 %>% dplyr::select(-Region,-Province,-Municipality_City), by = "Mun_Code") %>%
    left_join(data_matrix_new_variables , by = "Mun_Code") %>%
    left_join(wind_grid , by = "Mun_Code")  %>%
    left_join(rainfall_ , by = "Mun_Code")
  
  # source('/home/fbf/data_cleaning_forecast.R')
  data_new_typhoon<-clean_typhoon_forecast_data(data_new_typhoon)
  
  data <- data_new_typhoon %>%
    dplyr::select(-GEN_typhoon_name,
                  -GEO_n_households,
                  #-GEN_mun_code,
                  -contains('DAM_'),
                  -GEN_mun_name) %>%
    na.omit() # Randomforests don't handle NAs, you can impute in the future 
  return(data)
}