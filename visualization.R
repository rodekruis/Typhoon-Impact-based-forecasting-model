
suppressMessages(library(tmap))
suppressMessages(library(viridis))
suppressMessages(library(maps))
suppressMessages(library(ggmap))
suppressMessages(library(httr))
suppressMessages(library(sf))
suppressMessages(library(raster))
suppressMessages(library(rgdal))
suppressMessages(library(ranger))

path='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/'
 

main_directory<-path
source('lib_r/settings.R')

df2 <- data %>%
  mutate(more_than_10_perc_damage = ifelse(DAM_comp_houses_perc > 0.1, 1, 0),perc_damage=DAM_comp_houses_perc,
         more_than_10_perc_damage_ = as.factor(more_than_10_perc_damage)) %>%
  select(-index,
         GEN_typhoon_name,
         GEN_typhoon_name_year,
         GEN_mun_code,
         GEN_prov_name,
         GEN_mun_name,
         -contains('DAM_')) %>%
  na.omit()

xgmodel_r<-readRDS(file.path(getwd(),"/models/xgboost_regression.RDS"))
xgmodel_c<-readRDS(file.path(getwd(),"/models/xgboost_classify.RDS"))


df_<-df%>%dplyr::select(-perc_damage,-contains("INT_"))

df_1<-df%>%dplyr::select(-more_than_10_perc_damage,-contains("INT_"))
#create tasks
traintask <- makeClassifTask (data = df_, target = "more_than_10_perc_damage")
traintask2 <- makeRegrTask (data = df_1, target = "perc_damage")


xgpred <- predict(xgmodel_c, testtask)
xgpred2 <- predict(xgmodel_r, traintask2)
df3<-as.data.frame(cbind(xgpred$data,xgpred2$data))

names(df3)<-c('id','T_c','P_c','id2','T_r','P_r')

df4<-cbind(df2%>%select(GEN_mun_code,GEN_prov_name,GEN_mun_name,GEN_typhoon_name_year,GEN_typhoon_name,perc_damage),df3)


#Create boxplot of percentage of completely damaged houses per typhoon
ggplot(df4, aes(x=GEN_typhoon_name,y=perc_damage))+
  geom_boxplot(width=0.6, fill="orangered") +
  scale_y_continuous(limits=c(0,100),name="Percentage of houses completely damaged") +
  scale_x_discrete(name="Typhoon") +
  coord_flip() +
  theme_bw() +
  theme(text=element_text(size=16))

ggplot(df4, aes(x=GEN_typhoon_name,y=P_r))+
  geom_boxplot(width=0.6, fill="orangered") +
  scale_y_continuous(limits=c(0,100),name="Percentage of houses completely damaged") +
  scale_x_discrete(name="Typhoon") +
  coord_flip() +
  theme_bw() +
  theme(text=element_text(size=16))

ggplot(df4, aes(y = perc_damage, x = P_r)) +
  geom_point(aes(color = factor(GEN_typhoon_name)))+
  scale_y_continuous(name="Simulated % of houses completely damaged(MLM)") +
  scale_x_continuous(name="Observed % of houses completely damaged") +
  stat_smooth(method = "lm",
              col = "#C42126",
              se = FALSE,
              size = 1)


ggplot(df4, aes(x = T_c, y = P_c)) +
  geom_point(aes(color = factor(GEN_typhoon_name)))




Haiyan<-df2%>%filter(GEN_typhoon_name=='Haiyan')


php_admin4<-php_admin3%>%left_join(Haiyan%>%dplyr::mutate(adm3_pcode=GEN_mun_code),by='adm3_pcode')

php_admin4<-php_admin3%>%left_join(aggregate(all_wind_goni1$dist50, by=list(adm3_pcode=all_wind_goni1$Mun_Code), FUN=sum)%>%
                                     dplyr::mutate(probability=100*x/52)%>%dplyr::select(adm3_pcode,probability),by='adm3_pcode')%>%
  left_join(aggregate(all_wind_goni1$vmax_gust, by=list(adm3_pcode=all_wind_goni1$Mun_Code), FUN=max)%>%
              dplyr::mutate(VMAX=x)%>%dplyr::select(adm3_pcode,VMAX))%>%filter(probability>=10)


#---------------------- vistualize stations and risk areas -------------------------------

tmap_mode(mode = "view")
tm_shape(php_admin4) + tm_polygons(col = "probability", name='adm3_en',
                                   palette=c('#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043'),
                                   breaks=c(0,25,50,75,90,100),colorNA=NULL,
                                   labels=c('   < 25%','25 - 50%','50 - 75%','75 - 90%','   > 90%'),
                                   title="Probability for distance from Track < 50km",
                                   alpha = 0.75,
                                   border.col = "black",lwd = 0.01,lyt='dotted')+
  #tm_shape(my_track) +  
  tm_symbols(size=0.2,border.alpha = 0.75,col='#bdbdbd') +
  tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
  #tm_shape(goni_track) +tm_dots(size=0.1,border.alpha = 0.25,col='#bdbdbd')+
  tm_format("NLD")




#---------------------- vistualize stations and risk areas -------------------------------



Haiyan<-df2%>%filter(GEN_typhoon_name=='Haiyan')





#a50f15

php_admin4<-php_admin3%>%left_join(Haiyan%>%dplyr::mutate(adm3_pcode=GEN_mun_code),by='adm3_pcode')

tmap_mode(mode = "view")
tm_shape(php_admin4) + tm_polygons(col = "WEA_vmax_sust_mhp", name='adm3_en',
                                   palette=c('#fee5d9','#fcbba1','#fc9272','#fb6a4a','#de2d26','#a50f15'),
                                   breaks=c(0,10,25,50,75,90,110),colorNA=NULL,
                                   #labels=c('   < 25%','25 - 50%','50 - 75%','75 - 90%','   > 90%'),
                                   title="Sustained wind speed mph",
                                   alpha = 0.75,
                                   border.col = "black",lwd = 0.01,lyt='dotted')+
  #tm_shape(my_track) +  
  tm_shape(php_admin3) + tm_borders(lwd = .4,col='#bdbdbd') + 
  tm_symbols(size=0.002,border.alpha = 0.75,col='#bdbdbd') +
  #tm_shape(goni_track) +tm_dots(size=0.1,border.alpha = 0.25,col='#bdbdbd')+
  tm_format("NLD")




 