#!/usr/bin/env Rscript
#options(warn=-1)

track_interpolation<- function(TRACK_DATA){
  
  ########### track_interpolation ###################
                    TRACK_DATA<- TRACK_DATA %>% dplyr::mutate(typhoon = paste0(TRACK_DATA$STORMNAME,substr(TRACK_DATA$YYYYMMDDHH, 1, 4)))
                    
                    fulltrack<-create_full_track(hurr_track=TRACK_DATA, tint = 1)
                    
                    Mydata<-fulltrack %>% dplyr::mutate(index= 1:nrow(fulltrack))%>% dplyr::select(tclat,tclon,index,typhoon_name,vmax,date)
                    
                    my_track <- st_as_sf(Mydata, coords = c("tclon", "tclat"), crs = "+init=epsg:4326")
  
  return(my_track)
  
}