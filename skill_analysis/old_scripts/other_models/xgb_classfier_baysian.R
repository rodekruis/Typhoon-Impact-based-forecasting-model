
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



df<-input_prep(data)%>%dplyr::mutate(more_than_10_perc_damage =as.factor(more_than_10_perc_damage))

# Simple xgboost with self sampled test and training set
train_ind <- sample(seq_len(nrow(df)), size = 0.8*nrow(df))

df_train1 <- df[train_ind, ]
df_test1 <- df[-train_ind, ]


#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "response",par.vals = list( objective="binary:logistic", 
                                                                                eval_metric="error",nrounds=100L))

lrn2 <- makeLearner("regr.xgboost",predict.type = "response",par.vals = list( objective="reg:squarederror",
                                                                              eval_metric="mae", nrounds=100L))



####################################################################
####################################################################
####################################################################

df_train<-df_train1%>%dplyr::select(-perc_damage,-contains("INT_"))

df_test<-df_test1%>%dplyr::select(-perc_damage,-contains("INT_"))

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
      colsample_bytree = x["colsample_bytree"], #     objective        = 'count:poisson', #
      objective="binary:logistic",
      eval_metric="logloss", #     eval_metric     = "poisson-nloglik"
      ),
      data = df_train,#mydb_xgbmatrix,
      nround =   x["nrounds"],
      folds=  cv_folds,
      prediction = FALSE,
      showsd = TRUE,
      early_stopping_rounds = 10,
      verbose = 0)
    
    cv$evaluation_log[, min(test_logloss_mean)]
  },
  par.set = makeParamSet(
    makeNumericParam("eta",              lower = 0.001, upper = 0.05),
    makeNumericParam("gamma",            lower = 0,     upper = 5),
    makeIntegerParam("max_depth",        lower= 1,      upper = 10),
    makeIntegerParam("min_child_weight", lower= 1,      upper = 10),
    makeNumericParam("subsample",        lower = 0.2,   upper = 1),
    makeNumericParam("colsample_bytree", lower = 0.2,   upper = 1),
    makeIntegerParam("nrounds", lower = 100, upper = 1000)
  ),
  minimize = TRUE
)

# generate an optimal design with only 10  points
des = generateDesign(n=10,
                     par.set = getParamSet(obj.fun), 
                     fun = lhs::randomLHS)  ## . If no design is given by the user, mlrMBO will generate a maximin Latin Hypercube Design of size 4 times the number of the black-box function's parameters.
# i still want my favorite hyperparameters to be tested

simon_params <- data.frame(max_depth = 6,
                           colsample_bytree= 0.8,
                           subsample = 0.8,
                           min_child_weight = 3,
                           eta  = 0.01,
                           gamma = 0) %>% as_tibble()
#final design  is a combination of latin hypercube optimization and my own preferred set of parameters
final_design =  simon_params  %>% bind_rows(des)
# bayes will have 10 additional iterations
control = makeMBOControl()
control = setMBOControlTermination(control, iters = 10)
switch_generate_interim_data <- TRUE

if ( switch_generate_interim_data){
  run = mbo(fun = obj.fun, 
            design = final_design,  
            control = control, 
            show.info = TRUE)
  write_rds( run, ("run.rds"))
} else {
  run <- read_rds(("run.rds")) 
}


