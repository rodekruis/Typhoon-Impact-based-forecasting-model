# ------------------------  Tuning xgboost with MLR --------------------------------------
# first optimize model parameter by resampling 20-80 for classfier and regression model

rm(list=ls())
library(dplyr)
library(ranger)
library(caret)
library(randomForest)
library(broom)
library(xgboost)
library(mlr)
library(parallel)
library(parallelMap)
library(tune)
library(workflows)
library(tidymodels)
library(plot3D)
library(mlrMBO)
#parallelStartSocket(cpus = detectCores()-1)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

source('data_source.R')
setwd('..//')


#raw_data <- read.csv('datamatrix_Hazard_impact_static.csv', stringsAsFactors = FALSE)
data <- clean_typhoon_data()

input_prep<-function(data){
  df <- data %>%
    mutate(more_than_10_perc_damage = ifelse(DAM_comp_houses_perc > 0.1, 1, 0),
           more_than_10_perc_damage =as.factor(more_than_10_perc_damage),
           perc_damage=DAM_comp_houses_perc) %>%
    dplyr::select(-index,
           -GEN_typhoon_name,
           -GEN_typhoon_name_year,
           -GEN_mun_code,
           -GEN_prov_name,
           -GEN_mun_name,
           -contains('DAM_')) %>%
    na.omit() # Randomforests don't handle NAs, you can impute in the future
  return(df)
  
}

ctrl <- makeTuneControlRandom(maxit = 1000L)

params <- makeParamSet(makeIntegerParam("max_depth",lower = 3L,upper = 15L),
                       makeNumericParam("min_child_weight",lower = 1L,upper = 20L),
                       makeNumericParam("subsample",lower = 0.5,upper = 1),
                       makeNumericParam("eta",lower = 0.03,upper = .6),
                       #makeNumericParam("gamma",lower = 0,upper = 100),
                       makeNumericParam("alpha",lower = 0, upper = 2, trafo = function(x) 10^x-1),
                       #makeNumericParam("lambda", lower = 0, upper = 2, trafo = function(x) 10^x-1),
                       makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

df<-input_prep(data)%>%dplyr::mutate(more_than_10_perc_damage =as.factor(more_than_10_perc_damage))
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

df_train<-df_train1%>%dplyr::select(-perc_damage,-contains("INT_"))

df_test<-df_test1%>%dplyr::select(-perc_damage,-contains("INT_"))

df_<-df%>%dplyr::select(-perc_damage,-contains("INT_"))
#create tasks
traintask <- makeClassifTask (data = df_train, target = "more_than_10_perc_damage")

testtask <- makeClassifTask (data = df_, target = "more_than_10_perc_damage")





parallelStartSocket(cpus = detectCores()-2)



classifier_tune <- tuneParams(learner = lrn,
                     task = traintask, 
                     resampling = makeResampleDesc("CV",stratify = T, iters  = 10L),
                     measures = acc, 
                     par.set = params, 
                     control = ctrl,
                     show.info = T)

###################################################
#tuned parameters
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  max_depth=14,
  eta=0.08320083,
  alpha=7.732376,
  min_child_weight=1.002875,
  subsample=0.8241503,
  colsample_bytree= 0.5125735
  
)


###################################################


lrn_tune <- setHyperPars(lrn, par.vals =classifier_tune$x )
xgmodel <- train(learner = lrn_tune, task = traintask)

xgpred <- predict(xgmodel, testtask)
xgpred2 <- predict(xgmodel, traintask)

err_matrix<-confusionMatrix(xgpred$data$response,xgpred$data$truth)
err_matrix2<-confusionMatrix(xgpred2$data$response,xgpred2$data$truth)

df_score<-rbind(as.data.frame(err_matrix$byClass)%>%dplyr::rename('score_test'='err_matrix$byClass'),as.data.frame(err_matrix$overall)%>%
                  dplyr::rename('score_test'='err_matrix$overall'))
df_score2<-rbind(as.data.frame(err_matrix2$byClass)%>%
                   dplyr::rename('score_train'='err_matrix2$byClass'),as.data.frame(err_matrix2$overall)%>%
                   dplyr::rename('score_train'='err_matrix2$overall'))

df_score_classfier<-cbind(df_score2,df_score)

parallelStop()

# save model to binary local file


saveRDS(xgmodel,file.path(getwd(),"/models/xgboost_classify.RDS"))

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
#tuned parameters
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  max_depth=9,
  eta=0.0442109,
  alpha=62.45896,
  min_child_weight=14.78415,
  subsample=0.6996492,
  colsample_bytree= 0.506948

)


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

saveRDS(xgmodel,file.path(getwd(),"/models/xgboost_regression.RDS"))

xgmodelc=readRDS(file.path(getwd(),"/models/xgboost_classify.RDS"), refhook = NULL)
xgmodelr=readRDS(file.path(getwd(),"/models/xgboost_regression.RDS"), refhook = NULL)
parallelStop()
####################################################################
####################################################################
####################################################################



typhoon_names<-as.character(unique(data$GEN_typhoon_name))





df_score_clas_list<-list()

for(typhoons in typhoon_names){
  df_train<-input_prep(data%>%dplyr::filter(GEN_typhoon_name!=typhoons))%>%
    dplyr::select(-perc_damage,-contains("INT_"))%>%
    dplyr::mutate(more_than_10_perc_damage=as.factor(more_than_10_perc_damage))
  df_test<-input_prep(data%>%filter(GEN_typhoon_name==typhoons))%>%
    dplyr::select(-perc_damage,-contains("INT_"))%>%
    dplyr::mutate(more_than_10_perc_damage=as.factor(more_than_10_perc_damage))
  #create tasks
  traintask <- makeClassifTask (data = df_train, target = "more_than_10_perc_damage")
  testtask <- makeClassifTask (data = df_test, target = "more_than_10_perc_damage")

  # Recreate optimal model
  lrn_tune <- setHyperPars(lrn, par.vals =classifier_tune$x )
  xgmodel <- train(learner = lrn, task = traintask)

  xgpred <- predict(xgmodel, testtask)
  xgpred2 <- predict(xgmodel, traintask)
  
  err_matrix<-confusionMatrix(xgpred$data$response,xgpred$data$truth)
  err_matrix2<-confusionMatrix(xgpred2$data$response,xgpred2$data$truth)
  
  df_score<-rbind(as.data.frame(err_matrix$byClass)%>%
                    dplyr::rename('score_test'='err_matrix$byClass'),as.data.frame(err_matrix$overall)%>%
                    dplyr::rename('score_test'='err_matrix$overall'))%>%dplyr::mutate(typhoon=typhoons)
  df_score2<-rbind(as.data.frame(err_matrix2$byClass)%>%
                     dplyr::rename('score_train'='err_matrix2$byClass'),as.data.frame(err_matrix2$overall)%>%
                     dplyr::rename('score_train'='err_matrix2$overall'))
  
  df_score_clas_list[[typhoons]]<-cbind(df_score2,df_score)%>%dplyr::mutate(index=rownames(df_score))
  
  
}

df_score_clas_list<-bind_rows(df_score_clas_list)

########################################################



df_score_reg_list<-list()
for(typhoons in typhoon_names){
  
  df_train<-input_prep(data%>%dplyr::filter(GEN_typhoon_name!=typhoons))%>%dplyr::select(-more_than_10_perc_damage)
  
  df_test<-input_prep(data%>%filter(GEN_typhoon_name==typhoons))%>%dplyr::select(-more_than_10_perc_damage)
  
  #create tasks
  traintask2 <- makeRegrTask (data = df_train, target = "perc_damage")
  testtask2 <- makeRegrTask (data = df_test, target = "perc_damage")
  
  

  
  # Recreate optimal model
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
  
  df_score<-cbind(as.data.frame(stat_list)%>%
                    dplyr::rename('score_test'='stat_list'),as.data.frame(stat_list2)%>%
                    dplyr::rename('score_train'='stat_list2'))%>%
    dplyr::mutate(typhoon=typhoons,index=c('R2','RMSE','NRMSE','MAE','MAPE','MBE','SMAPE'))
  
  rownames(df_score)<-c('R2','RMSE','NRMSE','MAE','MAPE','MBE','SMAPE')
  
  
  df_score_reg_list[[typhoons]]<-df_score
  
}

df_score_reg_list<-bind_rows(df_score_reg_list)



df<-input_prep(data)


data.dam_per_typ <- data %>%drop_na()%>%
  group_by(GEN_typhoon_name) %>%
  summarise(n_observations    = n(),
            n_municipalties   = length(unique(GEN_mun_code)),
            min_damage        = min(DAM_comp_houses_perc),
            total_damage        = sum(DAM_comp_houses),
            Q1_damage         = quantile(DAM_comp_houses_perc,0.25),
            Q2_damage         = quantile(DAM_comp_houses_perc,0.5),
            Q3_damage         = quantile(DAM_comp_houses_perc,0.75),
            max_damage        = max(DAM_comp_houses_perc),
            Max_WIND        = max(WEA_vmax_sust_mhp),
            mean_damage       = mean(DAM_comp_houses_perc))

#Graph with number of municipalities per typhoon
#graph.nr_mun_per_typ <- 
ggplot(data.dam_per_typ %>% arrange(desc(n_municipalties)), aes(x=GEN_typhoon_name, y=total_damage))+
   geom_col(fill='orangered', width=0.6)+
   theme_bw()+
   geom_text(aes(label=n_municipalties), vjust=-0.4) +
   xlab("Typhoon") +
   #ylab("Number of municipalities with damage") +
  ylab("Total number of Completely Damaged Houses") +
  coord_flip() +
  theme_bw() +
  theme(text=element_text(size=16))
#   
# 
# ggsave(filename='PIC_nr_mun_per_typ.png',plot=graph.nr_mun_per_typ, width = 25, height = 20, units = "cm")


#Create boxplot of percentage of completely damaged houses per typhoon
boxplot.dats <-
  ggplot(data, aes(x=GEN_typhoon_name,y=DAM_comp_houses_perc))+
  geom_boxplot(width=0.6, fill="orangered") +
  scale_y_continuous(limits=c(0,100),name="Percentage of houses completely damaged") +
  scale_x_discrete(name="Typhoon") +
  coord_flip() +
  theme_bw() +
  theme(text=element_text(size=16))

boxplot.dats <-
  ggplot(data, aes(x=GEN_typhoon_name,y=WEA_vmax_sust_mhp))+
  geom_boxplot(width=0.6, fill="lightblue") +
  scale_y_continuous(limits=c(0,150),name="Maximum sustained wind speed mph") +
  scale_x_discrete(name="Typhoon") +
  coord_flip() +
  theme_bw() +
  theme(text=element_text(size=16))

boxplot.dats <-
  ggplot(data, aes(x=GEN_typhoon_name,y=WEA_rainfall_max))+
  geom_boxplot(width=0.6, fill="blue") +
  scale_y_continuous(limits=c(0,450),name="Maximum Rain fall mm/day") +
  scale_x_discrete(name="Typhoon") +
  coord_flip() +
  theme_bw() +
  theme(text=element_text(size=16))






DF<-data%>%select(GEN_typhoon_name,contains('GEO_'),-GEO_Area,-GEO_n_households,-GEO_landslide_y,-GEO_landslide_o,
                  -GEO_landslide_r,-GEO_stormsurge_y,-GEO_ruggedness_stdev,GEO_slope_stdev,-GEO_stormsurge_r,
                  -GEO_stormsurge_o,-GEO_landslide_r,-GEO_perimeter,-GEO_pop_density_u5_and_o60)

par(mfrow=c(3,4))
for(i in 2:12) {
  counts <- table(DF[,i])
  name <- substr(names(DF)[i],5,90)
  boxplot(DF[,i], main=name,cex.main=1.25)

  
}

x <- iris[,1:4]
y <- iris[,5]
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

ggsave(filename='Boxplot damage per typhoon.png',plot=boxplot.dam_per_typ1, width = 35, height = 20, units = "cm")
