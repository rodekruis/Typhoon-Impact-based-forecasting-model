
library(dplyr)
library(ggplot2)
library(tidyr)
library(purrr)
library(stringr)
library("openxlsx")
library(readxl)
library(zoo)
library(xts)
library(lubridate)
library(glmnet)
library(tidyverse)
library(caret)
#---------------------- Load inmpact data -------------------------------

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

baseline <- function(newdata,threshold) {
  newdata <- newdata %>% mutate(X2=WAE_vmax_kph)
  path <- './data/BASILE_MODEL.xlsx'
  sheets<- excel_sheets(path = path)
  for(i in 1:7)
  {  if(as.character(sheets[i])=="C1_M"){
    mydf <- openxlsx::read.xlsx(path, sheet = i, startRow = 1, colNames = FALSE) %>% mutate(WAE_vmax_kph=X2) %>% select(X1,WAE_vmax_kph)
    #mydf <- mydf %>% mutate(WAE_vmax_kph=X2)
    #names(mydf)[2] <- "WAE_vmax_kph"
    cvfit_2 <- lm(X1 ~ poly(WAE_vmax_kph, degree = 5, raw = TRUE), data=mydf)
    
    y_predicted <- predict(cvfit_2, newdata = newdata) 
    newdata <- newdata  %>% dplyr::mutate(pre_damage=ifelse (((newdata$WAE_vmax_kph >80) & (y_predicted > threshold)), 1, 0))
    newdata <- newdata  %>% dplyr::mutate(DAM_Strong.Roof.Strong.Wall=pre_damage*MAT_Strong.Roof.Strong.Wall*100/TP)
  }
    if(as.character(sheets[i])=="CHB_L_W"){     ### CHB_L_W    x<-seq(100,350,10)
      mydf <- openxlsx::read.xlsx(path, sheet = i, startRow = 1, colNames = FALSE) %>% mutate(WAE_vmax_kph=X2) %>% select(X1,WAE_vmax_kph)
      cvfit_2 <- lm(X1 ~ poly(WAE_vmax_kph, degree = 5, raw = TRUE), data=mydf)
      y_predicted <- predict(cvfit_2, newdata = newdata) 
      
      newdata <- newdata  %>% mutate(pre_damage=ifelse (((newdata$WAE_vmax_kph >80) & (y_predicted > threshold)), 1, 0))
     # newdata <- newdata %>% mutate(pre_damage=ifelse ((newdata$WAE_vmax_kph <80 | y_predicted < 0), 0, y_predicted))
      newdata <- newdata  %>% dplyr::mutate(DAM_Strong.Roof.Light.Wall=pre_damage*MAT_Strong.Roof.Light.Wall*100/TP)
    }
    if(as.character(sheets[i])=="CWS_L_W"){ ### CWS_L_W    x<-seq(100,350,10)
      mydf <- openxlsx::read.xlsx(path, sheet = i, startRow = 1, colNames = FALSE) %>% mutate(WAE_vmax_kph=X2) %>% select(X1,WAE_vmax_kph)
      cvfit_2 <- lm(X1 ~ poly(WAE_vmax_kph, degree = 5, raw = TRUE), data=mydf)
      y_predicted <- predict(cvfit_2, newdata = newdata) 
      
      newdata <- newdata  %>% mutate(pre_damage=ifelse (((newdata$WAE_vmax_kph >80) & (y_predicted > threshold)), 1, 0))
     # newdata <- newdata %>% mutate(pre_damage=ifelse (newdata$WAE_vmax_kph >200, 1, y_predicted)) %>%        mutate(pre_damage=ifelse ((newdata$WAE_vmax_kph <80 | y_predicted < 0), 0, y_predicted))
      newdata <- newdata  %>% dplyr::mutate(DAM_Strong.Roof.Salvage.Wall=pre_damage*MAT_Strong.Roof.Salvage.Wall*100/TP)
      
    }
    if(as.character(sheets[i])=="C1_L_S"){
      mydf <- openxlsx::read.xlsx(path, sheet = i, startRow = 1, colNames = FALSE) %>% mutate(WAE_vmax_kph=X2) %>% select(X1,WAE_vmax_kph)
      cvfit_2 <- lm(X1 ~ poly(WAE_vmax_kph, degree = 5, raw = TRUE), data=mydf)
      y_predicted <- predict(cvfit_2, newdata = newdata) 
      newdata <- newdata  %>% mutate(pre_damage=ifelse (((newdata$WAE_vmax_kph >80) & (y_predicted > threshold)), 1, 0))
      #newdata <- newdata %>% mutate(pre_damage=ifelse ((newdata$WAE_vmax_kph <80 | y_predicted < 0), 0, y_predicted))
      newdata <- newdata  %>% dplyr::mutate(DAM_Light.Roof.Strong.Wall=pre_damage*MAT_Light.Roof.Strong.Wall*100/TP)
      
    }
    if(as.character(sheets[i])=="W1_L"){
      mydf <- openxlsx::read.xlsx(path, sheet = i, startRow = 1, colNames = FALSE) %>% mutate(WAE_vmax_kph=X2) %>% select(X1,WAE_vmax_kph)
      cvfit_2 <- lm(X1 ~ poly(WAE_vmax_kph, degree = 5, raw = TRUE), data=mydf)
      y_predicted <- predict(cvfit_2, newdata = newdata) 
      newdata <- newdata  %>% mutate(pre_damage=ifelse (((newdata$WAE_vmax_kph >80) & (y_predicted > threshold)), 1, 0))
     # newdata <- newdata %>%  mutate(pre_damage=ifelse ((newdata$WAE_vmax_kph <80 | y_predicted < 0), 0, y_predicted))
      
      newdata <- newdata  %>% dplyr::mutate(DAM_Light.Roof.Light.Wall=pre_damage*MAT_Light.Roof.Light.Wall*100/TP)
    }
    if(as.character(sheets[i])=="W3_L"){     #####W3-L    x<-seq(50,180,10)
      mydf <- openxlsx::read.xlsx(path, sheet = i, startRow = 1, colNames = FALSE) %>% mutate(WAE_vmax_kph=X2) %>% select(X1,WAE_vmax_kph)
      
      cvfit_2 <- lm(X1 ~ poly(WAE_vmax_kph, degree = 5, raw = TRUE), data=mydf)
      y_predicted <- predict(cvfit_2, newdata = newdata) 
      
      newdata <- newdata  %>% mutate(pre_damage=ifelse (((newdata$WAE_vmax_kph >50) & (y_predicted > threshold)), 1, 0))
     # newdata <- newdata %>% mutate(pre_damage=ifelse (newdata$WAE_vmax_kph >200, 1, y_predicted)) %>%         mutate(pre_damage=ifelse ((newdata$WAE_vmax_kph <80 | y_predicted < 0), 0, y_predicted))
      newdata <- newdata  %>% dplyr::mutate(DAM_Salvaged.Roof.Light.Wall=pre_damage*MAT_Salvaged.Roof.Light.Wall*100/TP)
      
    }
    if(as.character(sheets[i])=="N_L"){    ########### N-L   #    x<-seq(80,200,10)
      
      cvfit_2 <- lm(X1 ~ poly(WAE_vmax_kph, degree = 5, raw = TRUE), data=mydf)
      mydf <- openxlsx::read.xlsx(path, sheet = i, startRow = 1, colNames = FALSE) %>% mutate(WAE_vmax_kph=X2) %>% select(X1,WAE_vmax_kph)
      y_predicted <- predict(cvfit_2, newdata = newdata)
      
      newdata <- newdata  %>% mutate(pre_damage=ifelse (((newdata$WAE_vmax_kph >80) & (y_predicted > threshold)), 1, 0))
      #newdata <- newdata %>% mutate(pre_damage=ifelse (newdata$WAE_vmax_kph >200, 1, y_predicted)) %>%         mutate(pre_damage=ifelse (newdata$WAE_vmax_kph <80, 0, y_predicted))
      newdata <- newdata  %>% dplyr::mutate(DAM_Salvaged.Roof.Salvaged.Wall=pre_damage*MAT_Salvaged.Roof.Salvage.Wall*100/TP)
    }
    
  }
  
  return(newdata)
  
}

# %>%  dplyr::mutate(Pcode=PCODE) %>%    dplyr::select(Pcode,Hotspot)
#cvfit_2 <- cv.glmnet(x = poly(mydf$X1, degree = 5, raw = TRUE),  y = mydf$X2)
#assign(paste("model",as.character(sheets[i]), sep = "_"), cvfit_2)
#saveRDS(cvfit_2, paste0("C:/documents/philipiness/basline_model_data/model",as.character(sheets[i]),'.rds'))


x<-seq(50,400,1)

newdata <- data.frame(WAE_vmax_kph = x)

 

source('data_source.R')

setwd('..//')

data<-clean_typhoon_data()

typhoon_names<- c("Bopha","conson","Durian","Fengshen"	,"Fung-wong","Goni","Hagupit","Haima","Haiyan","Kalmaegi",
                  "ketsana","Koppu","Krosa","Lingling","Mangkhut","Mekkhala","Melor","Nari","Nesat","Nock-Ten",
                  "Rammasun","Sarika","Utor")

# Define grid for grid search here we are using the hyper parameters family and alpha , number of foldes we use the defult 10


# Data data preparation please ignore the name " more_than_10_perc_damage" used it for nameing convection 

df <- data %>%
  dplyr::mutate(WAE_vmax_kph = 1.60934*WEA_vmax_sust_mhp,
                TP=(MAT_Strong.Roof.Light.Wall+
                      MAT_Light.Roof.Light.Wall+
                      MAT_Strong.Roof.Salvage.Wall+
                      MAT_Salvaged.Roof.Light.Wall+
                      MAT_Strong.Roof.Strong.Wall+
                      MAT_Light.Roof.Strong.Wall+
                      MAT_Salvaged.Roof.Salvage.Wall)) %>%
  dplyr::select(#-index,
                GEN_typhoon_name,
                GEN_typhoon_name_year,
                GEN_mun_code,
                #-GEN_prov_name,
                GEN_mun_name,
                WAE_vmax_kph,
                TP,
                DAM_comp_houses_perc,
                #contains('WEA_'),
                GEO_n_households,
                contains('MAT_')
                ) %>% na.omit() 




baseline_prediction<-baseline(newdata=df,threshold=0.55) %>%
  dplyr::mutate(Damage_Baseline=(DAM_Strong.Roof.Light.Wall+
                                DAM_Light.Roof.Light.Wall+
                                DAM_Strong.Roof.Salvage.Wall+
                                DAM_Salvaged.Roof.Light.Wall+
                                DAM_Strong.Roof.Strong.Wall+
                                DAM_Light.Roof.Strong.Wall+
                                DAM_Salvaged.Roof.Salvaged.Wall))%>%
  dplyr::select(Damage_Baseline)

df2<-cbind(df,baseline_prediction)%>%dplyr::select(-contains('MAT_'),
                                                   contains('GEO'))

ggplot(df2, aes(x = DAM_comp_houses_perc, y = Damage_Baseline)) +
  geom_point(aes(color = factor(GEN_typhoon_name)))+
  scale_y_continuous(limits=c(0,20),name="Simulated % of houses completely damaged(BLM)") +
  scale_x_continuous(limits=c(0,20),name="Observed % of houses completely damaged") +
  stat_smooth(method = "lm",
              col = "#C42126",
              se = FALSE,
              size = 1)






