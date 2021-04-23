
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
library(smoof)


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

ctrl <- makeTuneControlRandom(maxit = 100L)

params <- makeParamSet(makeIntegerParam("max_depth",lower = 3L,upper = 15L),
                       makeNumericParam("min_child_weight",lower = 1L,upper = 20L),
                       makeNumericParam("subsample",lower = 0.5,upper = 1),
                       makeNumericParam("eta",lower = 0.03,upper = .6),
                       #makeNumericParam("gamma",lower = 0,upper = 100),
                       makeNumericParam("alpha",lower = 0, upper = 2, trafo = function(x) 10^x-1),
                       #makeNumericParam("lambda", lower = 0, upper = 2, trafo = function(x) 10^x-1),
                       makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

df<-input_prep(data)%>%dplyr::mutate(more_than_10_perc_damage =as.factor(more_than_10_perc_damage))


# objective function: we want to maximise the log likelihood by tuning most parameters
obj.fun  <- smoof::makeSingleObjectiveFunction(
  name = "xgb_cv_bayes",
  fn =   function(x){
    set.seed(12345)
    cv <- xgb.cv(params = list(
      booster          = "gbtree",
      eta              = x["eta"],
      max_depth        = x["max_depth"],
      min_child_weight = x["min_child_weight"],
      gamma            = x["gamma"],
      subsample        = x["subsample"],
      colsample_bytree = x["colsample_bytree"],
      objective        = 'count:poisson', 
      eval_metric     = "poisson-nloglik"),
      data = mydb_xgbmatrix,
      nround = 30,
      folds=  cv_folds,
      monotone_constraints = myConstraint$sens,
      prediction = FALSE,
      showsd = TRUE,
      early_stopping_rounds = 10,
      verbose = 0)
    
    cv$evaluation_log[, max(test_poisson_nloglik_mean)]
  },
  par.set = makeParamSet(
    makeNumericParam("eta",              lower = 0.001, upper = 0.05),
    makeNumericParam("gamma",            lower = 0,     upper = 5),
    makeIntegerParam("max_depth",        lower= 1,      upper = 10),
    makeIntegerParam("min_child_weight", lower= 1,      upper = 10),
    makeNumericParam("subsample",        lower = 0.2,   upper = 1),
    makeNumericParam("colsample_bytree", lower = 0.2,   upper = 1)
  ),
  minimize = FALSE
)



