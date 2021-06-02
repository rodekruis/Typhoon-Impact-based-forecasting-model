
# IMPORT LIBRARY
library(stringr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(httr)
library(sf)
library(raster)
library(randomForest)
library(rlang)
library(plyr)
library(lubridate)
library(ranger)
library(tmap)
library(parallel)
library(parallelMap)
library(caret)
  
library(xgboost)
library(mlr) 
library(tune)



main_directory<-'C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/'

source('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/lib_r/settings.R')
source('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/lib_r/data_cleaning_forecast.R')
source('C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/lib_r/track_interpolation.R')


geo_variable <- read.csv("C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/data/geo_variable.csv")
#mode_classification1 <- readRDS(paste0(main_directory,"./models/xgboost_classify.rds"))
#mode_continious1 <- readRDS(paste0(main_directory,"./models/xgboost_regression.rds"))

#------------------------- define functions ---------------------------------

ntile_na <- function(x,n){
  notna <- !is.na(x)
  out <- rep(NA_real_,length(x))
  out[notna] <- ntile(x[notna],n)
  return(out)
}



# BUILD WIND DATA MATRIC FOR NEW TYPHOON 
results_folder="C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/data/past_typhoon_windfields"

name_sid <- read.csv("C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/temp/typhoon_name_sid_.csv") %>% dplyr::select(-id)

past_typhoon_wind <- read.csv("C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/data/historical_typhoon_wind/historical_typhoons_wind.csv")%>%
  dplyr::mutate(dis_track_min=ifelse(dis_track_min<1,1,dis_track_min),
                Mun_Code=adm3_pcode,pcode=as.factor(substr(adm3_pcode, 1, 10))) 

past_typhoon_wind <-name_sid %>% left_join(past_typhoon_wind,by="storm_id")%>%drop_na()%>%
  dplyr::mutate(typhoon_name=toupper(paste0(typhoon_name,substr(storm_id,1,4))))%>% dplyr::select(-X)


#################################################################
# BUILD RAIN DATA MATRIC FOR NEW TYPHOON 
past_typhoon_rain1 <- read.csv("C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/data/historical_typhoon_wind/PHL_admin3_zonal_statistics_2021_05_13.csv") %>%
  filter(typhoon_name %in% c('KAMMURI2019','PHANFONE2019','VONGFONG2020','MOLAVE2020','GONI2020')) %>% mutate(Mun_Code=pcode,typhoon=typhoon_name,ranfall=value)%>% dplyr::select(Mun_Code,typhoon,ranfall)
past_typhoon_rain2 <- read.csv("C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/data/all_rainfall.csv") %>% mutate(ranfall=rainfll_max)%>% dplyr::select(Mun_Code,typhoon,ranfall)


 

past_typhoon_rain<-bind_rows(past_typhoon_rain1,past_typhoon_rain2 )%>%dplyr::mutate(rainfll_max=ranfall,
                                                                                     typhoon_name=toupper(typhoon))%>%dplyr::select(-typhoon)


###################################################################### 



# BUILD DATA MATRIC FOR pre disaster indicators
data_pre_disaster <- geo_variable%>%
  left_join(material_variable2 %>% dplyr::select(-Region,-Province,-Municipality_City), by = "Mun_Code") %>%
  left_join(data_matrix_new_variables , by = "Mun_Code") %>%
  dplyr::mutate(coast_length= ifelse(is.na(coast_length),0, coast_length))

# BUILD impact DATA MATRIC 

impact <- read.csv("C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/data/IMpact_data_philipines_final4.csv")%>%
  na.omit()%>%
  dplyr::mutate(Mun_Code=pcode,typhoon_name=as.factor(toupper(paste0(typhoon,Year))))%>% dplyr::select(-pcode)#%>%na.omit()

names(impact)<-c("id","typhoon", "Year","Totally", "Partially","total", "Mun_Code", "typhoon_name") 

# BUILD DATA MATRIC FOR NEW TYPHOON 
data_typhoon <- as.data.frame(merge(tibble(Mun_Code = data_pre_disaster$Mun_Code),tibble( typhoon_name=unique(impact$typhoon_name))))%>%
  left_join(impact,by=c("Mun_Code",'typhoon_name'))%>%
  left_join(data_pre_disaster,by="Mun_Code") %>%
  left_join(past_typhoon_wind ,by = c("Mun_Code","typhoon_name"))%>%
  left_join(past_typhoon_rain,by=c("Mun_Code",'typhoon_name'))%>%
  dplyr::mutate(dist_track=dis_track_min,
                Municipality_City =Mun_name,
                gust_dur=0,
                sust_dur=0,
                rainfll_max=ifelse(is.na(rainfll_max), 50,rainfll_max),
                coast_length=ifelse(is.na(coast_length), 0,coast_length),
                ranfall=ifelse(is.na(ranfall), 50,ranfall),
                ranfall_sum=ranfall,
                vmax_gust=v_max*1.21*1.49*1.94384,  #knot(1.94384) and 1.21 is conversion factor for 10 min average to 1min average
                vmax_gust_mph=v_max*1.21*1.49*2,23694, #mph 1.9 is factor to drive gust and sustained wind
                vmax_sust_mph=v_max*1.21*2,23694,
                vmax_sust=v_max*1.21*1.94384)%>%
  dplyr::select(-Year,-id,-Mun_name,-v_max)%>%na.omit()


 


data_new_typhoon <- clean_typhoon_data(data_typhoon)%>%na.omit()%>%filter(DAM_comp_houses_perc<80)

# Randomforests don't handle NAs, you can impute in the future 

# CHECK DATA CONSISTANCE 

data_new_typhoon%>%filter(DAM_comp_houses_perc>30 & WEA_dist_track >100 )%>%dplyr::select(DAM_comp_houses_perc)%>%
  arrange(DAM_comp_houses_perc)


model_input <- data_new_typhoon %>% dplyr::mutate(more_than_10_perc_damage = ifelse(DAM_comp_houses_perc > 0.1, 1, 0),
                                                  more_than_10_perc_damage =as.factor(more_than_10_perc_damage),
                                                  perc_damage=DAM_comp_houses_perc)%>%dplyr::select(-index,
                                                                                                    -GEN_typhoon_name_year,
                                                                                                    -GEN_typhoon_name, 
                                                                                                    -GEN_mun_code, 
                                                                                                    -GEN_mun_name,
                                                                                                    -contains('DAM_'))%>%na.omit() # Randomforests don't handle NAs, you can impute in the future



####################################################################
####################################################################


ctrl <- makeTuneControlRandom(maxit = 1000L)

params <- makeParamSet(makeIntegerParam("max_depth",lower = 3L,upper = 15L),
                       makeNumericParam("min_child_weight",lower = 1L,upper = 20L),
                       makeNumericParam("subsample",lower = 0.5,upper = 1),
                       makeNumericParam("eta",lower = 0.03,upper = .6),
                       #makeNumericParam("gamma",lower = 0,upper = 100),
                       makeNumericParam("alpha",lower = 0, upper = 2, trafo = function(x) 10^x-1),
                       #makeNumericParam("lambda", lower = 0, upper = 2, trafo = function(x) 10^x-1),
                       makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

df<-model_input
set.seed(20)

# Simple xgboost with self sampled test and training set
train_ind <- sample(seq_len(nrow(df)), size = 0.8*nrow(df))

df_train1 <- df[train_ind, ]
df_test1 <- df[-train_ind, ]


#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "response",par.vals = list( objective="binary:logistic", 
                                                                                eval_metric="error",nrounds=300L))

lrn2 <- makeLearner("regr.xgboost",predict.type = "response",par.vals = list( objective="reg:squarederror",
                                                                              eval_metric="mae", nrounds=300L))

####################################################################
####################################################################
####################################################################


parallelStartSocket(cpus = detectCores()-2)

df_train<-df_train1%>%dplyr::select(-more_than_10_perc_damage,-contains("INT_"))
df_test<-df_test1%>%dplyr::select(-more_than_10_perc_damage,-contains("INT_"))


#create tasks
traintask2 <- makeRegrTask (data = df_train, target = "perc_damage")
testtask2 <- makeRegrTask (data = df_test, target = "perc_damage")

reg_tune <- tuneParams(learner = lrn2,
                       task = traintask2, 
                       resampling = makeResampleDesc("CV", iters  = 10L),
                       measures = mse, 
                       par.set = params, 
                       control = ctrl,
                       show.info = T)

###################################################
###################################################

lrn_tune <- setHyperPars(lrn2, par.vals =reg_tune$x )

xgmodel <- train(learner = lrn_tune, task = traintask2)


xgpred <- predict(xgmodel, testtask2)
xgpred2 <- predict(xgmodel, traintask2)

y <- xgpred$data$truth

y_predicted <- xgpred$data$response

# CALCULATE THE STATISTICS 
                            #R2
R2_ <- R2(y_predicted, y)

                       #MEAN ABSOLUTE ERROR
MAE_ <- MAE(y_predicted, y)

###Symmetric mean absolute percentage error
Smbe1 <- (abs(y - y_predicted))/0.5*(abs(y)+abs(y_predicted))
Smbe1[is.infinite(Smbe1)]<-NA

SMAPE<-100*mean(Smbe1, na.rm = TRUE)

RMSE_ <- sqrt(mean((y - y_predicted)^2))#ROOT MEAN SQUARE ERROR 

result1 <- 100*abs((y_predicted-y)/y)
result1[is.infinite(result1)]<-NA

MAPE<-mean(result1,na.rm=TRUE) #MEAN ABSOLUTE PERCENTAGE ERROR
MBE_ <- mean(y_predicted-y) #MEAN BIAS ERROR

#normalized root meat squared error

NRMSE  <- sqrt(sum((y_predicted-y)^2)/sum((y_predicted-mean(y))^2))
stat_list <- c(R2_,RMSE_,NRMSE,MAE_,MAPE,MBE_,SMAPE)
y <- xgpred2$data$truth
y_predicted <- xgpred2$data$response

# CALCULATE THE STATISTICS 

R2_ <- R2(y_predicted, y) #R2
MAE_ <- MAE(y_predicted, y) #MEAN ABSOLUTE ERROR

Smbe1 <- (abs(y - y_predicted))/0.5*(abs(y)+abs(y_predicted)) ###Symmetric mean absolute percentage error
Smbe1[is.infinite(Smbe1)]<-NA

SMAPE<-100*mean(Smbe1, na.rm = TRUE)

RMSE_ <- sqrt(mean((y - y_predicted)^2)) #ROOT MEAN SQUARE ERROR

result1 <- 100*abs((y_predicted-y)/y)
result1[is.infinite(result1)]<-NA
MAPE<-mean(result1,na.rm=TRUE)#MEAN ABSOLUTE PERCENTAGE ERROR


MBE_ <- mean(y_predicted-y)#MEAN BIAS ERROR

NRMSE  <- sqrt(sum((y_predicted-y)^2)/sum((y_predicted-mean(y))^2))#normalized root meat squared error

stat_list2 <- c(R2_,RMSE_,NRMSE,MAE_,MAPE,MBE_,SMAPE)

df_score_regr<-cbind(as.data.frame(stat_list)%>%
                       dplyr::rename('score_test'='stat_list'),as.data.frame(stat_list2)%>%
                       dplyr::rename('score_train'='stat_list2'))
rownames(df_score_regr)<-c('R2','RMSE','NRMSE','MAE','MAPE','MBE','SMAPE')



