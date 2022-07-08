#!/usr/bin/env Rscript
library(stringr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(tmap)
library(viridis)
library(maps)
library(ggmap)
library(httr)
library(sf)

library(raster)
library(rgdal)  ## 
################################
 
library(ranger)
library(caret)
library(randomForest)
library(rlang)
library(RFmarkerDetector)
library(AUCRF)
library(kernlab)
library(ROCR)
library(MASS)
library(glmnet)
library(MLmetrics)
library(plyr)

library(lubridate)

landmask <- readr::read_csv("C:/documents/philipiness/Typhoons/model/new_model/stormwindmodel/data-raw/landseamask_ph1.csv",col_names = c("longitude", "latitude", "land")) %>%
  dplyr::mutate(land = factor(land, levels = c(1, 0), labels = c("land", "water")))


radians_to_degrees<- function(radians){
  degrees  <-  radians *180/pi
  return(degrees)
}

degrees_to_radians<- function(degrees){
  radians <- degrees * pi / 180
  return(radians)
}

create_full_track <- function(hurr_track, tint = 3){ 
  

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
      wind                           = as.numeric(VMAX)*0.514444
    ) %>%
    dplyr::select(date                                    ,
            latitude                                ,
            tclat                                   ,
            tclon                                   ,
            longitude                               ,
            typhoon_name                            ,
            wind                                    ,
            typhoon                                 ,
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
  typhoon_name <- as.vector(TRACK_DATA1$STORMNAME[1])
  full_track <- data.frame(typhoon_name=typhoon_name,date = interp_date, tclat = interp_tclat, tclon = interp_tclon, vmax = interp_vmax)
  return(full_track)
}



latlon_to_km<- function(tclat_1, tclon_1, tclat_2, tclon_2, Rearth = 6378.14){
  tclat_1 <- degrees_to_radians(tclat_1)
  tclon_1 <- degrees_to_radians(tclon_1)
  tclat_2 <- degrees_to_radians(tclat_2)
  tclon_2 <- degrees_to_radians(tclon_2)
  
  delta_L <- tclon_1 - tclon_2
  delta_tclat <- tclat_1 - tclat_2
  
  hav_L <- sin(delta_L / 2) ^ 2
  hav_tclat <- sin(delta_tclat / 2) ^ 2
  
  hav_gamma <- hav_tclat + cos(tclat_1) * cos(tclat_2) * hav_L
  gamma <- 2 * asin(sqrt(hav_gamma))
  
  dist <- Rearth * gamma
  return(dist)
}

calc_forward_speed<- function(tclat_1, tclon_1, time_1, tclat_2, tclon_2, time_2){
  dist <- latlon_to_km(tclat_1, tclon_1, tclat_2, tclon_2) * 1000
  time <- as.numeric(difftime(time_2, time_1, units = "secs"))
  tcspd <- dist / time
  return(tcspd)
}

##Calculate the direction of gradient winds at each location

calc_bearing<- function(tclat_1, tclon_1, tclat_2, tclon_2){
  tclat_1 <- degrees_to_radians(tclat_1)
  tclon_1 <- degrees_to_radians(tclon_1)
  tclat_2 <- degrees_to_radians(tclat_2)
  tclon_2 <- degrees_to_radians(tclon_2)
  
 
  
  S <- cos(tclat_2) * sin(tclon_1 - tclon_2)
  C <- cos(tclat_1) * sin(tclat_2) - sin(tclat_1) * cos(tclat_2) * cos(tclon_1 - tclon_2)
  
  theta_rad <- atan2(S, C)
  theta <- radians_to_degrees(theta_rad) + 90
  theta <- theta %% 360 # restrict to be between 0 and 360 degrees
  return(theta)
}


remove_forward_speed<- function(vmax, tcspd){
  vmax_sfc_sym <- vmax - 0.5 * tcspd
  vmax_sfc_sym[vmax_sfc_sym < 0] <- 0
  return(vmax_sfc_sym)
}

calc_gradient_speed<- function(vmax_sfc_sym, over_land){
  reduction_factor <- 0.9
  if(over_land){
    reduction_factor <- reduction_factor * 0.8
  }
  vmax_gl <- vmax_sfc_sym / reduction_factor
  return(vmax_gl)
}


check_over_land<- function(tclat, tclon){
  lat_diffs <- abs(tclat - landmask$latitude)
  closest_grid_lat <- landmask$latitude[which(lat_diffs ==
                                                min(lat_diffs))][1]
  
  lon_diffs <- abs(tclon - (360 - landmask$longitude))
  closest_grid_lon <- landmask$longitude[which(lon_diffs ==
                                                 min(lon_diffs))][1]
  
  over_land <- landmask %>%
    dplyr::filter_(~ latitude == closest_grid_lat &
                     longitude == closest_grid_lon) %>%
    dplyr::mutate_(land = ~ land == "land") %>%
    dplyr::select_(~ land)
  over_land <- as.vector(over_land$land[1])
  
  return(over_land)
}


#Rmax: Radius from the storm center to the point at which the maximum wind occurs (km)

will7a<- function(vmax_gl, tclat){
  Rmax <- 46.4 * exp(-0.0155 * vmax_gl + 0.0169 * tclat)
  return(Rmax)
}

# X1, which is a parameter needed for the Willoughby model. This is done using equation 10a from Willoughby et al. (2006):
# X1=317.1a^2.026Vmax,G+1.915?

will10a<-function(vmax_gl, tclat){
  X1 <- 317.1 - 2.026 * vmax_gl + 1.915 * tclat
  return(X1)
}
#
#Next, the code calculates another Willoughby parameter, n. This is done with equation 10b from Willoughby et al. (2006):
#n=0.406+0.0144Vmax,G? 0.0038? 

will10b<- function(vmax_gl, tclat){
  n <- 0.4067 + 0.0144 * vmax_gl - 0.0038 * tclat
  return(n)
}
#Next, the code calculates another Willoughby parameter, A, with equation 10c from Willoughby et al. (2006)
will10c<- function(vmax_gl, tclat){
  A <- 0.0696 + 0.0049 * vmax_gl - 0.0064 * tclat
  A[A < 0 & !is.na(A)] <- 0
  return(A)
}

will3_right<- function(n, A, X1, Rmax){
  eq3_right <- (n * ((1 - A) * X1 + 25 * A)) /
    (n * ((1 - A) * X1 + 25 * A) + Rmax)
  return(eq3_right)
}


will3_deriv_func<- function(xi, eq3_right){
  deriv <- 70 * 9 * xi ^ 8 - 315 * 8 * xi ^ 7 + 540 * 7 * xi ^ 6 -
    420 * 6 * xi ^ 5 + 126 * 5 * xi ^ 4
  func <- 70 * xi ^ 9 - 315 * xi ^ 8 + 540 * xi ^ 7 - 420 * xi ^ 6 +
    126 * xi ^ 5 - eq3_right
  deriv_func <-c(deriv, func)
  return(deriv_func)
}

solve_for_xi<- function(xi0 = 0.5, eq3_right, eps = 10e-4, itmax = 100){
  if(is.na(eq3_right)){
    return(NA)
  } else{
    i <- 1
    xi <- xi0
    while(i <= itmax){
      deriv_func <- will3_deriv_func(xi, eq3_right)
      if(abs(deriv_func[2]) <= eps){ break }
      xi <- xi - deriv_func[2] / deriv_func[1]
    }
    if(i < itmax){
      return(xi)
    } else{
      warning("Newton-Raphson did not converge.")
      return(NA)
    }
  }
}


##While the Newton-Raphson method can sometimes perform poorly in finding global maxima, in this case the function for which we are trying ##to find the root is well-behaved. Across tropical storms from 1988 to 2015, the method never failed to converge, and identified roots ##were consistent across storms (typically roots are for ξ of 0.6--0.65). We also tested using the optim function in the stats R package ##and found similar estimated roots but slower convergence times than when using the Newton-Raphson method.

##Now an equation from the Willoughby et al. 2006 paper can be used to calculate R1 (Willoughby, Darling, and Rahn 2006):

#R1?=? Rmax? ? ??(R2? ? ? R1)

#For this function, the package code assumes that R2 − R1 (the width of the transition region) is 25 kilometers when Rmax is larger than 20 #kilometers and 15 kilometers otherwise.

calc_R1<- function(Rmax, xi){
  R2_minus_R1 <- ifelse(Rmax > 20, 25, 15)
  R1 <- Rmax - xi * R2_minus_R1
  return(R1)
}



will1<- function(cdist, Rmax, R1, R2, vmax_gl, n, A, X1, X2 = 25){
  
  if(is.na(Rmax) || is.na(vmax_gl) ||
     is.na(n) || is.na(A) || is.na(X1)){
    return(NA)
  } else {
    
    Vi <- vmax_gl * (cdist / Rmax) ^ n
    Vo <- vmax_gl * ((1 - A) * exp((Rmax - cdist)/X1) + A * exp((Rmax - cdist) / X2))
    
    if(cdist < R1){
      wind_gl_aa <- Vi
    } else if (cdist > R2){
      wind_gl_aa <- Vo
    } else {
      eps <- (cdist - R1) / (R2 - R1)
      w <- 126 * eps ^ 5 - 420 * eps ^ 6 + 540 * eps ^ 7 - 315 *
        eps ^ 8 + 70 * eps ^ 9
      wind_gl_aa <- Vi * (1 - w) + Vo * w
    }
    
    wind_gl_aa[wind_gl_aa < 0 & !is.na(wind_gl_aa)] <- 0
    
    return(wind_gl_aa)
  }
}

##Estimate surface level winds from gradient winds

gradient_to_surface<- function(wind_gl_aa, cdist){
  if(cdist <= 100){
    reduction_factor <- 0.9
  } else if(cdist >= 700){
    reduction_factor <- 0.75
  } else {
    reduction_factor <- 0.90 - (cdist - 100) * (0.15/ 600)
  }
  # Since all counties are over land, reduction factor should
  # be 20% lower than if it were over water
  reduction_factor <- reduction_factor * 0.8
  wind_sfc_sym <- wind_gl_aa * reduction_factor
  return(wind_sfc_sym)
}


add_inflow<- function(gwd, cdist, Rmax){
  if(is.na(gwd) | is.na(cdist) | is.na(Rmax)){
    return(NA)
  }
  
  # Calculate inflow angle over water based on radius of location from storm
  # center in comparison to radius of maximum winds (Phadke et al. 2003)
  if(cdist < Rmax){
    inflow_angle <- 10 + (1 + (cdist / Rmax))
  } else if(Rmax <= cdist & cdist < 1.2 * Rmax){
    inflow_angle <- 20 + 25 * ((cdist / Rmax) - 1)
  } else {
    inflow_angle <- 25
  }
  
  # Add 20 degrees to inflow angle since location is over land, not water
  overland_inflow_angle <- inflow_angle + 20
  
  # Add inflow angle to gradient wind direction
  gwd_with_inflow <- (gwd + overland_inflow_angle) %% 360
  
  return(gwd_with_inflow)
}


add_forward_speed <- function(wind_sfc_sym, tcspd_u, tcspd_v, swd, cdist, Rmax){
  # Calculate u- and v-components of surface wind speed
  swd<- swd * pi / 180
  wind_sfc_sym_u <- wind_sfc_sym * cos(swd)
  wind_sfc_sym_v <-  wind_sfc_sym * sin(swd)
  
  # Add back in component from forward motion of the storm
  correction_factor <- (Rmax * cdist) / (Rmax ^ 2 + cdist ^ 2)
  
  # Add tangential and forward speed components and calculate
  # magnitude of this total wind
  wind_sfc_u <- wind_sfc_sym_u + correction_factor * tcspd_u
  wind_sfc_v <- wind_sfc_sym_v + correction_factor * tcspd_v
  wind_sfc <- sqrt(wind_sfc_u ^ 2 + wind_sfc_v ^ 2)
  
  # Reset any negative values to 0
  wind_sfc <- ifelse(wind_sfc > 0, wind_sfc, 0)
  
  return(wind_sfc)
}


add_wind_radii <-function(full_track = create_full_track()){
  
  with_wind_radii <- full_track %>%
    dplyr::mutate_(tcspd = ~ calc_forward_speed(tclat, tclon, date,
                                                dplyr::lead(tclat),
                                                dplyr::lead(tclon),
                                                dplyr::lead(date)),
                   tcdir = ~ calc_bearing(tclat, tclon,
                                          dplyr::lead(tclat),
                                          dplyr::lead(tclon)),
                   tcspd_u = ~ tcspd * cos(tcdir* pi / 180),
                   tcspd_v = ~ tcspd * sin(tcdir* pi / 180),
                   vmax_sfc_sym = ~ remove_forward_speed(vmax, tcspd),
                   over_land = ~ mapply(check_over_land, tclat, tclon),
                   vmax_gl = ~ mapply(calc_gradient_speed,
                                      vmax_sfc_sym = vmax_sfc_sym,
                                      over_land = over_land),
                   Rmax = ~ will7a(vmax_gl, tclat),
                   X1 = ~ will10a(vmax_gl, tclat),
                   n = ~ will10b(vmax_gl, tclat),
                   A = ~ will10c(vmax_gl, tclat),
                   eq3_right = ~ will3_right(n, A, X1, Rmax),
                   xi = ~ mapply(solve_for_xi, eq3_right = eq3_right),
                   R1 = ~ calc_R1(Rmax, xi),
                   R2 = ~ ifelse(Rmax > 20, R1 + 25, R1 + 15)
    ) %>%
    dplyr::select_(quote(-vmax), quote(-tcspd), quote(-vmax_sfc_sym),
                   quote(-over_land), quote(-eq3_right), quote(-xi))
  return(with_wind_radii)
}

calc_grid_wind <- function(grid_point, with_wind_radii = add_wind_radii()){
  
  grid_wind <- dplyr::mutate_(with_wind_radii,
                              # Calculated distance from storm center to location
                              cdist = ~ latlon_to_km(tclat, tclon,
                                                     grid_point$glat, grid_point$glon),
                              # Calculate gradient winds at the point
                              wind_gl_aa = ~ mapply(will1, cdist = cdist, Rmax = Rmax,
                                                    R1 = R1, R2 = R2, vmax_gl = vmax_gl,
                                                    n = n, A = A, X1 = X1),
                              # calculate the gradient wind direction (gwd) at this
                              # grid point
                              chead = ~ calc_bearing(tclat, tclon,
                                                     grid_point$glat, - grid_point$glon),
                              gwd = ~ (90 + chead) %% 360,
                              # Bring back to surface level (surface wind reduction factor)
                              wind_sfc_sym = ~ mapply(gradient_to_surface,
                                                      wind_gl_aa = wind_gl_aa,
                                                      cdist = cdist),
                              # Get surface wind direction
                              swd = ~ mapply(add_inflow, gwd = gwd, cdist = cdist,
                                             Rmax = Rmax),
                              # Add back in storm forward motion component
                              windspeed = ~ add_forward_speed(wind_sfc_sym,
                                                              tcspd_u, tcspd_v,
                                                              swd, cdist, Rmax)) %>%
    dplyr::select_(~ date, ~ windspeed,~ cdist)
  return(grid_wind)
}

get_grid_winds<- function(hurr_track , grid_df ,tint = 3,gust_duration_cut = 15,sust_duration_cut = 15){
  full_track <- create_full_track(hurr_track = hurr_track, tint = tint)
  with_wind_radii <- add_wind_radii(full_track = full_track)
  
  grid_winds <- plyr::adply(grid_df, 1, calc_and_summarize_grid_wind,
                            with_wind_radii = with_wind_radii,
                            tint = tint, gust_duration_cut = gust_duration_cut,
                            sust_duration_cut = sust_duration_cut)
  return(grid_winds)
}


summarize_grid_wind <- function(grid_wind, tint = 3, gust_duration_cut = 15,
                                sust_duration_cut = 15){
  grid_wind_summary <- grid_wind %>%
    dplyr::mutate_(gustspeed = ~ windspeed * 1.49) %>%
    # Determine max of windspeed and duration of wind over 20
    dplyr::summarize_(vmax_gust = ~ max(gustspeed, na.rm = TRUE),
                      vmax_sust = ~ max(windspeed, na.rm = TRUE),
                      dist_track = ~ min(cdist, na.rm = TRUE),
                      gust_dur = ~ 60 * sum(gustspeed > gust_duration_cut,
                                            na.rm = TRUE),
                      sust_dur = ~ 60 * sum(windspeed > sust_duration_cut,
                                            na.rm = TRUE)) %>%
    dplyr::mutate_(gust_dur = ~ gust_dur * tint,
                   sust_dur = ~ sust_dur * tint)
  grid_wind_summary <- as.matrix(grid_wind_summary)
  return(grid_wind_summary)
}

calc_and_summarize_grid_wind <- function(grid_point,
                                         with_wind_radii = add_wind_radii(),
                                         tint = 3, gust_duration_cut = 15,
                                         sust_duration_cut = 15){
  grid_wind <- calc_grid_wind(grid_point = grid_point,
                              with_wind_radii = with_wind_radii)
  grid_wind_summary <- summarize_grid_wind(grid_wind = grid_wind, tint = tint,
                                           gust_duration_cut = gust_duration_cut,
                                           sust_duration_cut = sust_duration_cut)
  return(grid_wind_summary)
  
}



#------------------------- new typhoon ---------------------------------

# ------------------------ calculate  variables  -----------------------------------
typhoon_events <- read.csv('C://documents//philipiness//Typhoons//model//new_model//input//forecast//typhoon_info_for_model.csv')
grid_points_adm3<-read.csv("C:/documents/philipiness/Typhoons/model/new_model/grid_points_admin3.csv", sep=",")
# ------------------------ import boundary layer   -----------------------------------
php_admin3 <- st_read(dsn='C:/documents/philipiness/Typhoons/model/new_model/data-raw',layer='phl_admin3_simpl2')
php_admin1 <- st_read(dsn='C:/documents/philipiness/Typhoons/model/new_model/data-raw',layer='phl_admin1_gadm_pcode')

for(i in 1:nrow(typhoon_events)){
  
  TRACK_DATA<- read.csv(as.character(typhoon_events$filename[i]))
  
  TRACK_DATA1<- TRACK_DATA %>% 
    dplyr::mutate(typhoon = paste0(TRACK_DATA$STORMNAME,substr(TRACK_DATA$YYYYMMDDHH, 1, 4)))
  
  fulltrack<-create_full_track(hurr_track=TRACK_DATA1, tint = 3)
  
  Mydata<-fulltrack %>% 
    dplyr::mutate(index= 1:nrow(fulltrack))%>%
    dplyr::select(tclat,tclon,index,typhoon_name,vmax,date)
  
  my_track <- st_as_sf(Mydata, coords = c("tclon", "tclat"), crs = "+init=epsg:4326")
  
  # check if there is a landfall
  Landfall_check <- st_intersection(php_admin1, my_track)
  
  if (nrow(Landfall_check) > 0){
    
    Typhoon_stormname<-as.character(typhoon_events$event[i])

    Landfall_point<-Landfall_check[1,]
######################################## caculate time for landfll
    dt<-lubridate::with_tz(lubridate::ymd_hms(format(Landfall_check$date[1], format="%Y-%m-%d %H:%M:%S")), "PST8PDT")
    dt1<-lubridate::with_tz(lubridate::ymd_hm(format(TRACK_DATA$YYYYMMDDHH[1], format="%Y%m%d%H%M")), "PST8PDT")
    time_for_landfall<- as.numeric(difftime(dt,dt1,units="hours"))



wind_grids <- get_grid_winds(hurr_track=TRACK_DATA1, grid_df=grid_points_adm3, tint = 3,gust_duration_cut = 15, sust_duration_cut = 15)

wind_grids<- wind_grids %>%
  dplyr::mutate(typhoon_name=as.vector(TRACK_DATA$STORMNAME)[1],fips = as.numeric(substr(gridid, 3, 11)),vmax_gust_mph=vmax_gust*2.23694,vmax_sust_mph=vmax_sust*2.23694) %>%
  dplyr::select(typhoon_name,fips,gridid,vmax_gust,vmax_gust_mph,vmax_sust,vmax_sust_mph,gust_dur,sust_dur,dist_track,glon,glat)


wind_grid<-wind_grids%>%
  dplyr::mutate(Mun_Code=gridid)%>%
  dplyr::select(-fips,-gridid)



# track wind should have a unit of knots


##

NOAA_rain <- brick("C:/documents/philipiness/Typhoons/model/new_model/input/forecast/rainfall_forecast.grib2")

names(NOAA_rain) = paste("rain",outer(1:45,'0',paste,sep="-"),sep="-")


crs1 <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0" 
wshade <- readOGR('C:/documents/philipiness/Typhoons/model/new_model/data-raw', layer = 'phl_admin3_simpl2')
wshade <- spTransform(wshade, crs1)

e <- extent(110,150,5,40) # clip data to polygon around PAR

NOAA_rain <- crop(NOAA_rain, e)

rainfall <-extract(NOAA_rain, y=wshade, method='bilinear',fun=max,df=TRUE,sp=TRUE) 

rainfall_<-rainfall@data %>%
  dplyr::mutate(Mun_Code=adm3_pcode,rainfall_24h=rain.1.0 + rain.2.0 + rain.3.0 + rain.4.0 + rain.5.0 + rain.6.0+rain.7.0 + rain.8.0 + rain.9.0 + rain.10.0 + rain.11.0 + rain.12.0) %>%
  dplyr::select(Mun_Code,rainfall_24h)




# BUILD DATA MATRIC FOR NEW TYPHOON 

setwd('C://documents//philipiness//Typhoons//model//new_model')

material_variable2 <- read.csv("data/material_variable2.csv")
data_matrix_new_variables <- read.csv("data/data_matrix_new_variables.csv")
geo_variable <- read.csv("data/geo_variable.csv")
data_new_typhoon<-geo_variable %>%
  left_join(material_variable2 %>% dplyr::select(-Region,-Province,-Municipality_City), by = "Mun_Code") %>%
  left_join(data_matrix_new_variables , by = "Mun_Code") %>%
  left_join(wind_grid , by = "Mun_Code")  %>%
  left_join(rainfall_ , by = "Mun_Code")

source('C:/documents/philipiness/Typhoons/model/new_model/data_cleaning_forecast.R')
data_new_typhoon<-clean_typhoon_forecast_data(data_new_typhoon)
 


data <- data_new_typhoon %>%
  dplyr::select(-GEN_typhoon_name,
                -GEO_n_households,
                #-GEN_mun_code,
                -contains('DAM_'),
                -GEN_mun_name) %>%
  na.omit() # Randomforests don't handle NAs, you can impute in the future 



# load the model
mode_classification <- readRDS("C://documents//philipiness//Typhoons//model//new_model//final_model.rds")

rm.predict.pr <- predict(mode_classification, data = data, predict.all = FALSE, num.trees = mode_classification$num.trees, type = "response",
                         se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)

FORECASTED_IMPACT<-as.data.frame(rm.predict.pr$predictions) %>%
  dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact_threshold_passed=rm.predict.pr$predictions) %>%
  left_join(data , by = "index") %>%
  dplyr::select(GEN_mun_code,impact_threshold_passed,WEA_dist_track)

#write.csv(data_new_typhoon, file = 'C:/documents/philipiness/Typhoons/model/new_model/output/new_typhoon.csv')

# load the rr model
mode_continious <- readRDS("C://documents//philipiness//Typhoons//model//new_model//final_model_regression.rds")

rm.predict.pr <- predict(mode_continious, data = data, predict.all = FALSE, num.trees = mode_continious$num.trees, type = "response",
                         se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)

FORECASTED_IMPACT_rr<-as.data.frame(rm.predict.pr$predictions) %>%
  dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact=rm.predict.pr$predictions) %>%
  left_join(data , by = "index") %>%
  dplyr::select(GEN_mun_code,impact,WEA_dist_track)




php_admin3<-php_admin3 %>%  dplyr::mutate(GEN_mun_code=adm3_pcode)%>%  dplyr::select(adm3_en,GEN_mun_code,geometry)

php_admin3<-php_admin3 %>%  left_join(FORECASTED_IMPACT, by = "GEN_mun_code")
php_admin3<-php_admin3 %>%  left_join(FORECASTED_IMPACT_rr, by = "GEN_mun_code")



php_admin3<-php_admin3 %>% dplyr::arrange(WEA_dist_track.x) %>%
  dplyr::mutate(impact=ifelse(WEA_dist_track.x > 100,NA,impact),impact_threshold_passed =ifelse(WEA_dist_track.x > 100,NA,impact_threshold_passed))

Impact<-php_admin3  %>%   dplyr::select(GEN_mun_code,impact,impact_threshold_passed,WEA_dist_track.x)

Impact <- Impact %>% st_set_geometry(NULL)

####################################################################################################
#  Then calculate damage quantile. ----


ntile_na <- function(x,n)
{
  notna <- !is.na(x)
  out <- rep(NA_real_,length(x))
  out[notna] <- ntile(x[notna],n)
  return(out)
}


php_admin4 <- php_admin3 %>% 
  mutate(dam_perc_comp_prediction_lm_quantile = ntile_na(impact,5))  %>% filter(WEA_dist_track.x < 300)


# ------------------------ make a map   -----------------------------------
php_admin3<-php_admin3 %>% arrange(WEA_dist_track.x)

region<- st_bbox(php_admin3[1,])
typhoon_region = st_bbox(c(xmin = region[["xmin"]]-2, xmax = 2+region[["xmax"]],
                           ymin = region[["ymin"]]-2, ymax =2+ region[["ymax"]]),
                         crs = st_crs(php_admin1)) %>% st_as_sfc()


subtitle =paste0("Predicted damage per Manucipality
Source of wind speed forecast Tropical Storm Risk (UCL)
Only municipalities within 100km of forecasted typhoon track are included
Prediction is about completely damaged houses only\n",'Estimated time to Landfall: ',dt,' PST (',time_for_landfall,' hrs)')

tmap_mode(mode = "plot")
impact_map= 
  tm_shape(php_admin4) + 
  tm_fill(col = "impact",showNA=FALSE, border.col = "black",lwd = 3,lyt='dotted',
          breaks = c(0,0.1,1,2,5,9.5,10),
          title='Predicted % of Damaged ',
          labels=c(' No Damage',' < 1%',' 1 to 2%',' 2 to 5%',' 5 to 10%',' > 10%'),
          palette = c('#ffffff','#fdd0a2','#fdae6b','#fd8d3c','#e6550d','#a63603')) + #,style = "cat")+
          
  tm_borders(col = NA, lwd = .25, lty = "solid", alpha = .25, group = NA) +
  #tm_polygons(col = "dam_perc_comp_prediction_lm_quantile", border.col = "black",lwd = 0.1,lyt='dotted',style = "cat")+
  tm_shape(my_track) + tm_symbols(size=0.1,border.alpha = .25,col="blue") +
  tm_shape(Landfall_point) + tm_symbols(size=0.25,border.alpha = .25,col="red") +   
  tm_scale_bar(breaks = c(0, 100, 200), text.size = .5,
               color.light = "#f0f0f0",
               position = c(0,0))+
  tm_layout(legend.outside= TRUE, legend.outside.position=c("left"),
            main.title=subtitle, main.title.size=.8,asp=.8)

ph_map = tm_shape(php_admin1) + tm_polygons(border.col = "black",lwd = 0.1,lyt='dotted',alpha =0.2) +
  tm_shape(typhoon_region) + tm_borders(lwd = 2,col='red') +
  tm_compass(type = "8star", position = c("left", "top")) +
  tm_credits("The maps used do not imply \nthe expression of any opinion on \nthe part of the International Federation\n of the Red Cross and Red Crescent Societies\n concerning the legal status of a\n territory or of its authorities.",
             position = c("left", "BOTTOM"),size = 0.9) +
  tm_layout(inner.margins=c(.04,.03, .02, .01)) 

map1<- tmap_arrange(impact_map,ph_map,ncol = 2,widths = c(.8, .2))




# ------------------------ save to file   -----------------------------------

## save an image ("plot" mode)
tmap_save(map1, filename = paste0('C:/documents/philipiness/Typhoons/model/new_model/input/forecast/Impact_',as.vector(TRACK_DATA[1,]$YYYYMMDDHH),'_',as.vector(TRACK_DATA[1,]$STORMNAME),'.png'))
write.csv(Impact, file = paste0('C:/documents/philipiness/Typhoons/model/new_model/input/forecast/Impact_',as.vector(TRACK_DATA[1,]$YYYYMMDDHH),'_',as.vector(TRACK_DATA[1,]$STORMNAME),'.csv'))


file_names<- c(paste0('C:/documents/philipiness/Typhoons/model/new_model/input/forecast/Impact_',as.vector(TRACK_DATA[1,]$YYYYMMDDHH),'_',as.vector(TRACK_DATA[1,]$STORMNAME),'.png'),
               paste0('C:/documents/philipiness/Typhoons/model/new_model/input/forecast/Impact_',as.vector(TRACK_DATA[1,]$YYYYMMDDHH),'_',as.vector(TRACK_DATA[1,]$STORMNAME),'.csv'))

write.table(file_names, file = 'C:/documents/philipiness/Typhoons/model/new_model/input/forecast/file_names.csv',append=TRUE, col.names = FALSE)

}
  
  
  else{
    
    Typhoon_stormname<-as.character(typhoon_events$event[i])
    
    file_names<- c(paste0('Nolandfall','_',Typhoon_stormname,'.png'),
                   paste0('Nolandfall','_',Typhoon_stormname,'.csv'))
    
    
    
    write.table(file_names, file = 'C:/documents/philipiness/Typhoons/model/new_model/input/forecast/file_names.csv',append=TRUE, col.names = FALSE)
    
    
}
  
  
}

