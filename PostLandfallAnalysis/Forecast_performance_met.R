library(readr)
library(dplyr)

library("ggplot2")

setwd("C:\\Users\\ATeklesadik\\OneDrive - Rode Kruis\\Documents\\documents\\Typhoon-Impact-based-forecasting-model\\PostLandfallAnalysis\\")

 
typhones_data <- read.csv("C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/PostLandfallAnalysis/vamco.csv") %>%
  dplyr::mutate(model_name='BestTrack',
                forecast_time='observed',
                cyc_speed=VMAX,
                lat=LAT,lon=LON,
                #time_var=format(strptime(Synoptic_Time, format = "%d/%m/%Y %H:%M"), '%Y-%m-%d  %H:%M'),
                time_var=format(strptime(YYYYMMDDHH, format = "%Y%m%d%H%M"), '%Y-%m-%d  %H:%M')) %>%
  dplyr::select(model_name,cyc_speed,lat,lon,forecast_time,time_var)
 
 

vamco_all <- read_csv("vamco_all.csv")  %>% 
  dplyr::mutate(time_var=format(strptime(time, format = "%Y/%m/%d, %H:%M"), '%Y-%m-%d  %H:%M')) %>%
  dplyr::mutate(forecast_time=format(strptime(forecast_time, format = "%Y/%m/%d, %H:%M"), '%Y-%m-%d  %H:%M')) %>%
  dplyr::select(Mtype,product,model_name,cyc_speed,lat,lon,vhr,forecast_time,time_var) %>%
  dplyr::mutate(cyc_speed=ifelse(model_name=='ecmwf',cyc_speed*1.94384449,cyc_speed))


vamco_all_ <- vamco_all%>% filter(Mtype=='forecast')%>%
  dplyr::select(model_name,cyc_speed,lat,lon,forecast_time,time_var)


total <- rbind(vamco_all_, typhones_data)

landfall<-format(strptime("2020/11/11,  12:00", format = "%Y/%m/%d, %H:%M"), '%Y-%m-%d  %H:%M')
 
landfall_time<-strptime("2020/11/11,  12:00", format = "%Y/%m/%d, %H:%M")

colfunc <- colorRampPalette(c("#de2d26","#fc9272","#fee0d2"))

 
#"ecmwf" "cens"      "cmc"       "gefs"      "gfs"       "jma-geps"  "jma"       "jma-gsm" 
for (model_names in unique(vamco_all_$model_name)){
  data1=vamco_all_ %>% filter(model_name==model_names) %>% filter(forecast_time<landfall)
  df<-as.data.frame(dplyr::bind_rows(data1,typhones_data)) %>% mutate(cyc_speed=cyc_speed*1.852)
  
  
  scatwinddam<- ggplot() + geom_line(data=df, aes(x=as.POSIXct(df$time_var),
                                                  y=cyc_speed,group=forecast_time,
                                                  color=forecast_time))+
    #scale_x_discrete(name="time",breaks = levels(as.factor(df$time_var))[c(T, rep(F, 3))])+
     
      
    scale_y_continuous(name="Forecasted wind speed (Km/h)") + coord_cartesian(ylim=c(0,300)) +
    geom_hline(yintercept=175, color = "red") +
    geom_vline(xintercept = as.POSIXct(landfall_time),linetype="dotted", color = "red", size=2)+
      xlab("time")+
    ggtitle(paste0('Genesis of Typhoon Vamco forecasted Maximum \n wind speed (10 minutes average) -', toupper(model_names)))+
    scale_colour_manual(name = "Forecast Time", values = c(colfunc(length(unique(df$forecast_time))-1),'#3182bd'))+
    theme(text = element_text(size=15,margin =  margin(0, 2, 0, 2, "cm")), 
          axis.text.x = element_text( size = 12, angle = 0,hjust = 1),
          axis.text.y = element_text( size = 12, angle = 0),plot.title=element_text(size=18),
          plot.background = element_rect(fill = "transparent",colour = NA),
          plot.margin = margin(1, 1, 1, 1, "cm")) 
  
  ggsave(filename=paste0(model_names,'_vamco_forecast.png'),plot=scatwinddam, width = 30, height = 20, units = "cm")
  
  
  
}




typhones_data <- read.csv("C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/PostLandfallAnalysis/GONI2.csv") %>%
  dplyr::mutate(model_name='BestTrack',
                forecast_time='observed',
                cyc_speed=VMAX,
                lat=LAT,lon=LON,
                #time_var=format(strptime(Synoptic_Time, format = "%d/%m/%Y %H:%M"), '%Y-%m-%d  %H:%M'),
                time_var=format(strptime(YYYYMMDDHH, format = "%Y%m%d%H%M"), '%Y-%m-%d  %H:%M')) %>%
  dplyr::select(model_name,cyc_speed,lat,lon,forecast_time,time_var)



goni_all <- read_csv("goni_all.csv")  %>% 
  dplyr::mutate(time_var=format(strptime(time, format = "%Y/%m/%d, %H:%M"), '%Y-%m-%d  %H:%M')) %>%
  dplyr::mutate(forecast_time=format(strptime(forecast_time, format = "%Y/%m/%d, %H:%M"), '%Y-%m-%d  %H:%M')) %>%
  dplyr::select(Mtype,product,model_name,cyc_speed,lat,lon,vhr,forecast_time,time_var) %>%
  dplyr::mutate(cyc_speed=ifelse(model_name=='ecmwf',cyc_speed*1.94384449,cyc_speed))


goni_all_ <- goni_all%>% filter(Mtype=='forecast')%>%
  dplyr::select(model_name,cyc_speed,lat,lon,forecast_time,time_var)


total <- rbind(goni_all_, typhones_data)

landfall<-format(strptime("2020/11/01,  00:30", format = "%Y/%m/%d, %H:%M"), '%Y-%m-%d  %H:%M')

landfall_time<-strptime("2020/11/01,  00:30", format = "%Y/%m/%d, %H:%M")

colfunc <- colorRampPalette(c("#de2d26","#fc9272","#fee0d2"))


#"ecmwf" "cens"      "cmc"       "gefs"      "gfs"       "jma-geps"  "jma"       "jma-gsm" 
for (model_names in unique(goni_all_$model_name)){
  data1=goni_all_ %>% filter(model_name==model_names) %>% filter(forecast_time<landfall)
  df<-as.data.frame(dplyr::bind_rows(data1,typhones_data)) %>% mutate(cyc_speed=cyc_speed*1.852)
  
  
  scatwinddam<-ggplot() + geom_line(data=df, aes(x=as.POSIXct(df$time_var), y=cyc_speed,group=forecast_time,color=forecast_time))+
    scale_y_continuous(name="Forecasted wind speed (Km/h)") + coord_cartesian(ylim=c(0,300)) +
    geom_hline(yintercept=175, color = "red") +
    geom_vline(xintercept = as.POSIXct(landfall_time),linetype="dotted", color = "red", size=2)+
    xlab("time")+
    ggtitle(paste0('Genesis of Typhoon Goni forecasted Maximum \n wind speed (10 minutes average) -', toupper(model_names)))+
    scale_colour_manual(name = "Forecast Time", values = c(colfunc(length(unique(df$forecast_time))-1),'#3182bd'))+
    theme(text = element_text(size=15,margin =  margin(0, 2, 0, 2, "cm")), axis.text.x = element_text( size = 12, angle = 60,hjust = 1),
          axis.text.y = element_text( size = 12, angle = 0),plot.title=element_text(size=18),
          plot.background = element_rect(fill = "transparent",colour = NA),
          plot.margin = margin(1, 1, 1, 1, "cm"),) 
  
  ggsave(filename=paste0(model_names,'_Goni_forecast.png'),plot=scatwinddam, width = 30, height = 20, units = "cm")
  
  
  
}


