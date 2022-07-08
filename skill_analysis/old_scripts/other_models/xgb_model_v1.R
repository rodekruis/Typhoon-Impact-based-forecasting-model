
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


model_mtric <- function(y_predicted,y){
  
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
  
  return(df_score_regr)
}




# BUILD WIND DATA MATRIC FOR NEW TYPHOON 
results_folder="C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/data/past_typhoon_windfields"

name_sid <- read.csv("C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/temp/typhoon_name_sid.csv") %>% dplyr::select(-id)

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
#################################
#################################
#################################

#########calculate return period of Events

library(xts) 
library(zoo)
library(extRemes)

df_impact<-impact%>%dplyr::select("Totally", "Year","Partially","total","typhoon_name")%>%group_by(typhoon_name)%>%
  dplyr::summarise(Totally=sum(Totally),Partial=sum(Partially),Total=sum(total),Year=as.Date(paste0(max(Year),'-01-01')))%>%  filter((Totally < 150000  )& (Totally >500) )#typhoon_name!='HAIYAN2013'


impact_df2<-apply.yearly(xts(df_impact$Totally, order.by=df_impact$Year), max)

RT<-return.level(fevd(impact_df2, data.frame(impact_df2), units = "num"), span=14,return.period = c(2, 3,5,7,10,15,20), do.ci = FALSE)
print(RT)

#####################
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
                vmax_gust=v_max*1.49*1.94384,  #knot(1.94384) and 1.21 is conversion factor for 10 min average to 1min average
                vmax_gust_mph=v_max*1.49*2,23694, #mph 1.9 is factor to drive gust and sustained wind
                vmax_sust_mph=v_max*2,23694,
                vmax_sust=v_max*1.94384)%>%
  dplyr::select(-Year,-id,-Mun_name,-v_max)%>%na.omit()


 


data_new_typhoon <- clean_typhoon_data(data_typhoon)%>%na.omit()%>%filter(DAM_comp_houses_perc<80)

# Randomforests don't handle NAs, you can impute in the future 

# CHECK DATA CONSISTANCE 

data_new_typhoon%>%filter(DAM_comp_houses_perc>30 & WEA_dist_track >100 )%>%dplyr::select(DAM_comp_houses_perc)%>%
  arrange(DAM_comp_houses_perc)


model_input1 <- data_new_typhoon %>% dplyr::mutate(more_than_10_perc_damage = ifelse(DAM_comp_houses_perc > 0.1, 1, 0),
                                                  more_than_10_perc_damage =as.factor(more_than_10_perc_damage),
                                                  perc_damage=DAM_comp_houses_perc)%>%dplyr::select(-index,
                                                                                                    -GEN_typhoon_name_year,
                                                                                                    -GEN_typhoon_name,
                                                                                                    -GEO_n_households,
                                                                                                    -GEN_mun_code, 
                                                                                                    -GEN_mun_name,
                                                                                                    -contains('DAM_'))%>%na.omit() # Randomforests don't handle NAs, you can impute in the future



####################################################################
####################################################################

combined_input_data<-read.csv("C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/IBF-Typhoon-model/data/model_input/combined_input_data.csv")

model_input1 <- combined_input_data %>% dplyr::mutate(more_than_10_perc_damage = ifelse(DAM_perc_dmg > 10, 1, 0),
                                                   more_than_10_perc_damage =as.factor(more_than_10_perc_damage),
                                                   perc_damage=DAM_perc_dmg)%>%dplyr::select(-typhoon,
                                                                                                     -Mun_Code, 
                                                                                                     -contains('DAM_'))%>%na.omit() # Randomforests don't handle NAs, you can impute in the future





ctrl <- makeTuneControlRandom(maxit = 1000L)

params <- makeParamSet(makeIntegerParam("max_depth",lower = 6L,upper = 10L),
                       makeNumericParam("min_child_weight",lower = 3L,upper = 5L),
                       makeNumericParam("subsample",lower = 0.5,upper = 1),
                       makeNumericParam("eta",lower = 0.03,upper = .3),
                       #makeNumericParam("gamma",lower = 0,upper = 100),
                       makeNumericParam("alpha",lower = 0, upper = 2, trafo = function(x) 10^x-1),
                       #makeNumericParam("lambda", lower = 0, upper = 2, trafo = function(x) 10^x-1),
                       makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

df<-model_input1%>%dplyr::select(-more_than_10_perc_damage)#,-contains("INT_"))
set.seed(20)

# Simple xgboost with self sampled test and training set
train_ind <- sample(seq_len(nrow(df)), size = 1*nrow(df))

df_train1 <- df[train_ind, ]
df_test1 <- df[-train_ind, ]

train_x <- data.matrix(df_train1 %>% dplyr::select(-perc_damage))
train_y <- as.numeric(as.character(df_train1$perc_damage))

xgb_train <- xgb.DMatrix(data=train_x, label=train_y)

test_x <- data.matrix(df_test1 %>%dplyr::select(-perc_damage))
test_y <- as.numeric(as.character(df_test1$perc_damage))

xgb_test <- xgb.DMatrix(data=test_x, label=test_y)


param_ <- list(booster = "gbtree",
               objective = "reg:squarederror",
               alpha= 1.565309,
               eta=0.03185399,
               gamma=0,
               max_depth=7,
               min_child_weight= 4.8785,
               subsample= 0.7326575,
               colsample_bytree= 0.8185566)



# Optimizing nrounds with cross validation function
xgb_cv_model <- xgb.cv( params = param_,
                        data = xgb_train,
                        nrounds = 50,
                        nfold = 5,
                        showsd = T,
                        stratified = T,
                        print_every_n = 10,
                        early_stopping_roudns = 30,
                        metrics='mae',
                        maximize = F)

### iterations to stop over fitting 

d<-as.data.frame(xgb_cv_model$evaluation_log)%>%dplyr::select(iter,train_mae_mean,test_mae_mean)

ggplot(d) + 
  geom_line(aes(x = iter, y = train_mae_mean), col = "red") +
  geom_line(aes(x = iter, y = test_mae_mean), col = "blue")+
  scale_y_continuous(sec.axis = sec_axis(~.*1, name="test_mae_mean") )

# Min error on 50, train model
#min(xgb_cv_model$evaluation_log$test_rmse_mean)


#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "response",par.vals = list( objective="binary:logistic", 
                                                                                eval_metric="error",nrounds=50L))

lrn2 <- makeLearner("regr.xgboost",predict.type = "response",par.vals = list( objective="reg:squarederror",
                                                                              eval_metric="mae", nrounds=50L))

####################################################################
####################################################################
####################################################################



df_train<-df_train1
df_test<-df_test1 


#create tasks
traintask2 <- makeRegrTask (data = df_train, target = "perc_damage")
testtask2 <- makeRegrTask (data = df_test, target = "perc_damage")

parallelStartSocket(cpus = detectCores()-2)

reg_tune <- tuneParams(learner = lrn2,
                       task = traintask2, 
                       resampling = makeResampleDesc("CV", iters  = 10L),
                       measures = mse, 
                       par.set = params, 
                       control = ctrl,
                       show.info = T)


parallelStop()

###################################################
###################################################
#optimal parameters 

param_ <- list(booster = "gbtree",
               objective = "reg:squarederror",
               alpha= 1.565309,
               eta=0.03185399,
               gamma=0,
               max_depth=7,
               min_child_weight= 4.8785,
               subsample= 0.7326575,
               colsample_bytree= 0.8185566)



#lrn_tune <- setHyperPars(lrn2, par.vals =reg_tune$x )
#xgmodel <- train(learner = lrn_tune, task = traintask2)
#xgpred2 <- predict(xgmodel, traintask2)
#y <- xgpred$data$truth

#y_predicted <- xgpred$data$response

xgb_model <- xgb.train(params = param_,
                       data=xgb_train,
                       nrounds=50,
                       verbose=TRUE)


saveRDS(xgb_model,file.path(getwd(),"/models/xgboost_regression_v2.RDS"))

xgmodel=readRDS(file.path(getwd(),"/models/xgboost_regression_v2.RDS"), refhook = NULL)



test_x <- data.matrix(df_test1 %>%dplyr::select(-perc_damage))

xgb_test <- xgb.DMatrix(data=test_x)


y_predicted = predict(xgmodel, xgb_test)

y<-test_y

# CALCULATE THE STATISTICS 

df_score<- model_mtric(y_predicted,y)
