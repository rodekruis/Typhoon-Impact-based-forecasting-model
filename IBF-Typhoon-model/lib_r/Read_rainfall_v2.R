#!/usr/bin/env Rscript
#options(warn=-1)
Read_rainfall_v2<-function(wshade){
  
########### Read_rainfall ###################
            if(file.exists(paste0(rain_directory, "/rainfall_24.nc") )) {
              
              e <- raster::extent(114,145,4,28) # clip data to polygon around 
              NOAA_rain <- brick(paste0(rain_directory, "/rainfall_24.nc"))
              NOAA_rain <- crop(NOAA_rain, e)
              rainfall <-raster::extract(NOAA_rain, y=wshade, method='bilinear',fun=max,df=TRUE,sp=TRUE)
              rainfall_<-rainfall@data %>%
                dplyr::mutate(Mun_Code=adm3_pcode,rainfall_24h=apply(rainfall@data[,7:ncol(rainfall@data)], 1, FUN=max)) %>%
                dplyr::select(Mun_Code,rainfall_24h) #%>% left_join(rainfall_1,by='Mun_Code')
          
            }
  
            else{
              
              urls.out <- CrawlModels(abbrev = "gfs_0p25", depth = 2) # to avoid error if GFS model out put is not yet uploaded we use GFS model results for previous time step
              model.parameters <- ParseModelPage(urls.out[2])  #Get a list of forecasts, variables and levels
              levels <- c("surface") #What region of the atmosphere to get data for
              variables <- c("PRATE")  #What data to return
              dir.create(file.path("forecast", "rainfall"), showWarnings = FALSE)
    
    # Get the data
          for (i in 2:length(head(model.parameters$pred,n=72))){
            grib.info <- GribGrab(urls.out[2], model.parameters$pred[i], levels, variables,local.dir = "forecast/rainfall")
          }
                    
          file_list <- list.files("forecast/rainfall")
          xx <- raster::stack()   # Read each grib file to a raster and stack it to xx  
          
          
          for (files in file_list)  {
            fn <- file.path("forecast/rainfall", files)
            r2 <- raster(fn)
            x1 <- crop(r2, e)
            xx <- raster::stack(xx, x1)
          }     
          xx[xx < 0] <- NA  # Remove noise from the data
          
          NOAA_rain<-xx*3600*3 # unit conversion kg/m2/s to mm/3hr
          names(NOAA_rain) = paste("rain",outer(1:length(file_list),'0',paste,sep="-"),sep="-")
          rainfall <-extract(NOAA_rain, y=wshade, method='bilinear',fun=max,df=TRUE,sp=TRUE) 
          rainfall_<-rainfall@data %>%
            dplyr::mutate(Mun_Code=adm3_pcode,rainfall_24h=rain.1.0 + rain.2.0 + rain.3.0 + rain.4.0 + rain.5.0 + rain.6.0+rain.7.0 + rain.8.0 + rain.9.0 + rain.10.0 + rain.11.0 + rain.12.0) %>%
            dplyr::select(Mun_Code,rainfall_24h)
        } 
        
  
  return(rainfall_)
  
}

