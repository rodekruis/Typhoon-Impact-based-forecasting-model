library(readr)
library(dplyr)

library("ggplot2")

setwd("C:\\Users\\ATeklesadik\\OneDrive - Rode Kruis\\Documents\\documents\\Typhoon-Impact-based-forecasting-model\\")

NDRRMC.nov11.matched <- read_csv("C:\\Users\\ATeklesadik\\OneDrive - Rode Kruis\\Documents\\documents\\Typhoon-Impact-based-forecasting-model\\PostLandfallAnalysis\\NDRRMC-nov11-matched.csv") %>% 
  dplyr::mutate(Mun_code=pcode,C_damaged=as.numeric(totally_damaged)) %>%
  dplyr::select(Mun_code,C_damaged)

impact <- read_csv("C:\\Users\\ATeklesadik\\OneDrive - Rode Kruis\\Documents\\documents\\Typhoon-Impact-based-forecasting-model\\PostLandfallAnalysis\\Impact_GONI OBSERVED_2020111012_GONI.csv") %>% 
  dplyr::mutate(Mun_code=GEN_mun_code) 

 

material_variable2 <- read_csv("C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/data/material_variable3.csv") %>% 
  dplyr::mutate(NH=Housing_Units,Mun_code=Mun_Code)%>% 
  left_join(NDRRMC.nov11.matched,by='Mun_code') %>%  
  dplyr::mutate(adm3_pcode=Mun_code,impact_ob=100*C_damaged/NH) %>% 
  left_join(impact,by='Mun_code') %>% dplyr::mutate(diff=impact_ob-impact) #%>%  dplyr::select(adm3_pcode,impact_ob,impact,diff)

dam_com<- php_admin3 %>% left_join(material_variable2,by='adm3_pcode') 




TRACK_DATA<-GONI2
Typhoon_stormname<-'GONI'
TYF<-"GONI OBSERVED"
my_track <- track_interpolation(TRACK_DATA) %>% dplyr::mutate(Data_Provider=TYF)
  


php_admin4 <- dam_com %>% 
  dplyr::mutate(dam_perc_comp_prediction_lm_quantile = ntile_na(impact_ob,5)) %>% filter(WEA_dist_track < 300)

region2<-extent(php_admin4)

typhoon_region = st_bbox(c(xmin =as.vector(region2@xmin), xmax = as.vector(region2@xmax),
                           ymin = as.vector(region2@ymin), ymax =as.vector(region2@ymax)),
                         crs = st_crs(php_admin1)) %>% st_as_sfc()


subtitle =paste0("Predicted damage per Municipality for ", Typhoon_stormname,'\n',
                 "Source of wind speed forecast ",TYF,'\n',
                 "Only Areas within 100km of forecasted track are included",'\n',
                 "Prediction is about completely damaged houses only")


impact_map=tm_shape(php_admin4) + 
  tm_fill(col = "impact_ob",showNA=FALSE, border.col = "black",lwd = 3,lyt='dotted',
          breaks = c(0,0.1,1,2,5,9.5,10),
          title='observed % of Damaged ',
          labels=c(' No Damage',' < 1%',' 1 to 2%',' 2 to 5%',' 5 to 10%',' > 10%'),
          palette = c('#ffffff','#fdd0a2','#fdae6b','#fd8d3c','#e6550d','#a63603')) + #,style = "cat")+
  tm_borders(col = NA, lwd = .25, lty = "solid", alpha = .25, group = NA) +
  tm_shape(my_track) + tm_symbols(col='Data_Provider',size=0.1,border.alpha = .25) +
  tm_compass(type = "8star", position = c("right", "top")) +
  tm_scale_bar(breaks = c(0, 100, 200), text.size = .5, color.light = "#f0f0f0",
               position = c(0,.1))+
  tm_credits("The maps used do not imply the expression of any opinion on the part of the International Federation of the \nRed Cross and Red Crescent Societies concerning the legal status of a territory or of its authorities.",
             position = c("left", "BOTTOM"),size = 0.6) + 
  tm_layout(legend.show = FALSE)#legend.outside= TRUE,            legend.outside.position=c("left"),            inner.margins=c(.01,.04, .02, .01),            main.title=subtitle, main.title.size=.8,asp=.8)


impact_map2=tm_shape(php_admin4) + 
  tm_fill(col = "impact_ob",showNA=FALSE, border.col = "black",lwd = 3,lyt='dotted',
          breaks = c(0,0.1,1,2,5,9.5,10),
          title='observed % of Damaged ',
          labels=c(' No Damage',' < 1%',' 1 to 2%',' 2 to 5%',' 5 to 10%',' > 10%'),
          palette = c('#ffffff','#fdd0a2','#fdae6b','#fd8d3c','#e6550d','#a63603')) +
  tm_shape(tc_tracks) + tm_symbols(col='Data_Provider',size=0.1,border.alpha = .25) +
  tm_layout(legend.only = TRUE, legend.position=c("left", "top"))#, main.title=subtitle, main.title.size=.8,asp=.8)



ph_map = tm_shape(php_admin1) + tm_polygons(border.col = "black",lwd = 0.1,lyt='dotted',alpha =0.2) +
  tm_shape(typhoon_region) + tm_borders(lwd = 2,col='red') +
  #tm_credits("The maps used do not imply \nthe expression of any opinion on \nthe part of the International Federation\n of the Red Cross and Red Crescent Societies\n concerning the legal status of a\n territory or of its authorities.",
  tm_layout(legend.show = FALSE)#inner.margins=c(.04,.03, .04, .04)) 

ph_map2 = tm_shape(php_admin1)+ tm_polygons(border.col = "white",lwd = 0.01,lyt='dotted',alpha =0.2) +
  #tm_shape(typhoon_region) +# tm_borders(lwd = 2,col='red') +
  tm_credits( subtitle,position = c("left", "top"),size = 0.7) +
  tm_logo(paste0(path,'logos/combined_logo.png'),#https://www.510.global/wp-content/uploads/2017/07/510-LOGO-WEBSITE-01.png', 
          height=3, position = c("right", "top"))+
  tm_layout(legend.show = FALSE)

map1<- tmap_arrange(ph_map,ph_map2,impact_map2,impact_map,nrow=2,ncol = 2,widths = c(.3, .7),heights = c(.3, .7))
work_directory<-"C:\\Users\\ATeklesadik\\OneDrive - Rode Kruis\\Documents\\documents\\Typhoon-Impact-based-forecasting-model\\PostLandfallAnalysis\\"
tmap_save(map1, 
          filename = paste0(work_directory,'observed damage.png'),
          width=20, height=24,dpi=600,
          units="cm")  
 
typhones_data <- read.csv("C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/PostLandfallAnalysis/typhones_data.csv") %>%
  dplyr::mutate(model_name='BestTrack',
                cyc_speed=VMAX,
                lat=LAT,lon=LON,
                time_var=format(strptime(Synoptic_Time, format = "%d/%m/%Y %H:%M"), '%Y-%m-%d  %H:%M')) %>%
  dplyr::mutate(forecast_time='observed') %>% filter(stormname=='VAMCO')%>% dplyr::select(model_name,cyc_speed,lat,lon,forecast_time,time_var)
 
 


goni_all <- read_csv("skill_analysis/data/multicyclone/goni_all.csv")  %>% 
  dplyr::mutate(time_var=format(strptime(time, format = "%Y/%m/%d, %H:%M"), '%Y-%m-%d  %H:%M')) %>%
  dplyr::mutate(forecast_time=format(strptime(forecast_time, format = "%Y/%m/%d, %H:%M"), '%Y-%m-%d  %H:%M')) %>%
  dplyr::select(Mtype,product,model_name,cyc_speed,lat,lon,vhr,forecast_time,time_var) %>%
  dplyr::mutate(cyc_speed_1=ifelse(model_name=='ecmwf',cyc_speed*3.6*1.149,cyc_speed))%>%
  dplyr::mutate(cyc_speed=ifelse(model_name=='ecmwf',cyc_speed*1.94384449,cyc_speed))



vamco_all <- read_csv("skill_analysis/data/multicyclone/vamco_all.csv")  %>% 
  dplyr::mutate(time_var=format(strptime(time, format = "%Y/%m/%d, %H:%M"), '%Y-%m-%d  %H:%M')) %>%
  dplyr::mutate(forecast_time=format(strptime(forecast_time, format = "%Y/%m/%d, %H:%M"), '%Y-%m-%d  %H:%M')) %>%
  dplyr::select(Mtype,product,model_name,cyc_speed,lat,lon,vhr,forecast_time,time_var) %>%
  dplyr::mutate(cyc_speed=ifelse(model_name=='ecmwf',cyc_speed*1.94384449,cyc_speed))


vamco_all_ <- vamco_all%>% filter(Mtype=='forecast')%>% 
  dplyr::select(model_name,cyc_speed,lat,lon,forecast_time,time_var) %>% 
  append(typhones_data)

#%>% 
  arrange(time_var,model_name)#%>% fill(cyc_speed, .direction = "down")#  mutate(dis = na.locf(cyc_speed))



landfall<-format(strptime("2020/11/11,  00:00", format = "%Y/%m/%d, %H:%M"), '%Y-%m-%d  %H:%M')
 
landfall_time<-strptime("2020/11/11,  00:00", format = "%Y/%m/%d, %H:%M")

colfunc <- colorRampPalette(c("#de2d26","#fc9272","#fee0d2"))

 
#"ecmwf" "cens"      "cmc"       "gefs"      "gfs"       "jma-geps"  "jma"       "jma-gsm" 
for (model_names in unique(vamco_all_$model_name)){
  data1=vamco_all_ %>% filter(Mtype=='forecast' & model_name==model_names) %>% filter(forecast_time<landfall)
  df<-as.data.frame(dplyr::bind_rows(data1,typhones_data)) %>% mutate(cyc_speed=cyc_speed*1.852)
  
  
  scatwinddam<-ggplot() + geom_line(data=df, aes(x=time_var, y=cyc_speed,group=forecast_time,color=forecast_time))+
    scale_x_discrete(name="time",breaks = levels(as.factor(df$time_var))[c(T, rep(F, 3))])+
    scale_y_continuous(name="Forecasted wind speed (Km/h)") + coord_cartesian(ylim=c(0,300)) +
    geom_hline(yintercept=175, linetype="dashed", color = "red") +
    #geom_vline(xintercept = landfall_time, linetype="dotted",  color = "blue", size=1.5)+
    ggtitle(paste('Genesis of Typhoon GONI forecasted Maximum wind speed (10 minutes average)  '),model_names) +
    scale_colour_manual(name = "Forecast Time", values = c(colfunc(length(unique(df$forecast_time))-1),'#3182bd'))+
    theme(text = element_text(size=15), axis.text.x = element_text( size = 12, angle = 60,hjust = 1),
          axis.text.y = element_text( size = 12, angle = 0),plot.title=element_text(size=18),
          plot.background = element_rect(fill = "transparent",colour = NA)) 
  
  ggsave(filename=paste0(model_names,' forecast.png'),plot=scatwinddam, width = 30, height = 20, units = "cm")
  
  
  
}

