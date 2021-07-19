#!/usr/bin/env Rscript
#options(warn=-1)

create_full_track2 <- function(hurr_track, tint = 3){ 
  
  typhoon_name <- as.vector(hurr_track$STORMNAME[1])
  
  hurr_track <- hurr_track %>%
    dplyr::mutate(
      index                          = 1:nrow(hurr_track),
      date                           = lubridate::ymd_hm(YYYYMMDDHH),
      tclat                           = abs(as.numeric(LAT)),
      tclon                           = as.numeric(LON),
      tclon                           = ifelse(tclon < 180, tclon, tclon - 180),
      latitude                       = as.numeric(LAT),#as.numeric(LAT),#sprintf("%03d", LAT)#,#as.numeric(as.character(TRACK_DATA$LAT)),
      longitude                      = as.numeric(LON),#as.numeric(LON),# as.numeric(as.character(TRACK_DATA$LON)),
      vmax                           = as.numeric(VMAX)* 0.514444,#1.15078,#mph as.numeric(VMAX),# as.numeric(as.character(TRACK_DATA$VMAX)),
      typhoon_name                   = tolower(STORMNAME),
      prcen                           = as.numeric(prcen), #mph to knotes
      wind                           = as.numeric(VMAX)*0.514444 #mph to knotes 
    ) %>%
    dplyr::select(date                                    ,
                  latitude                                ,
                  tclat                                   ,
                  tclon                                   ,
                  longitude                               ,
                  typhoon_name                            ,
                  wind                                    ,
                  typhoon                                 ,
                  prcen                                   ,
                  vmax      ) 
  
  interp_df <- (floor(nrow(hurr_track)/2)-1)
  #interp_df<- 34 # 108 --30    
  interp_date <- seq(from = min(hurr_track$date),
                     to = max(hurr_track$date),
                     by = tint * 3600) # Date time sequence must use `by` in
  # seconds
  interp_date <- data.frame(date = interp_date)
  
  tclat_spline <- stats::glm(tclat ~ splines::ns(date, df = interp_df),  data = hurr_track)
  interp_tclat <- stats::predict.glm(tclat_spline, newdata = interp_date)  
  tclon_spline <- stats::glm(tclon ~ splines::ns(date, df = interp_df),  data = hurr_track)
  interp_tclon <- stats::predict.glm(tclon_spline, newdata = interp_date)
  
  vmax_spline <- stats::glm(vmax ~ splines::ns(date, df = interp_df),    data = hurr_track)
  interp_vmax <- stats::predict.glm(vmax_spline, newdata = interp_date)
  
  p_spline <- stats::glm(prcen ~ splines::ns(date, df = interp_df),    data = hurr_track)
  interp_p <- stats::predict.glm(p_spline, newdata = interp_date)
  
  full_track <- data.frame(typhoon_name=typhoon_name,date = interp_date, tclat = interp_tclat, tclon = interp_tclon, vmax = interp_vmax, prcen = interp_p)
  return(full_track)
}

track_interpolation<- function(TRACK_DATA){
  
  ########### track_interpolation ###################
                    TRACK_DATA<- TRACK_DATA %>% dplyr::mutate(typhoon = paste0(TRACK_DATA$STORMNAME,substr(TRACK_DATA$YYYYMMDDHH, 1, 4)))
                    
                    fulltrack<-create_full_track(hurr_track=TRACK_DATA, tint = 0.5)
                    
                    Mydata<-fulltrack %>% dplyr::mutate(index= 1:nrow(fulltrack))%>% dplyr::select(tclat,tclon,index,typhoon_name,vmax,date)
                    
                    my_track <- st_as_sf(Mydata, coords = c("tclon", "tclat"), crs = "+init=epsg:4326")
  
  return(my_track)
  
}