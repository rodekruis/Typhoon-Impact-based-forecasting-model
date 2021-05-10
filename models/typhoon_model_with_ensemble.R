rm(list=ls())
library(dplyr)
library(ranger)
library(caret)
library(randomForest)
library(broom)
library(xgboost)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

source('data_source.R')
setwd('..//')
raw_data<-clean_typhoon_data()

#raw_data <- read.csv('datamatrix_Hazard_impact_static.csv', stringsAsFactors = FALSE)
data <- clean_typhoon_data()

input_prep<-function(data){
  df <- data %>%
    mutate(more_than_10_perc_damage = ifelse(DAM_comp_houses_perc > 0.1, 1, 0),perc_damage=DAM_comp_houses_perc,
           more_than_10_perc_damage_ = as.factor(more_than_10_perc_damage)) %>%
    select(-index,
           -GEN_typhoon_name,
           -GEN_typhoon_name_year,
           -GEN_mun_code,
           -GEN_prov_name,
           -GEN_mun_name,
           -contains('DAM_')) %>%
    na.omit() # Randomforests don't handle NAs, you can impute in the future
  return(df)
  
}

 
df <- input_prep(data)


# ------------------------ Random forest -----------------------------------

# Basic random forest to check performance and number of trees
rf <- randomForest(more_than_10_perc_damage ~ ., data = df)
plot(rf)

# Define grid for grid search
hyper_grid <- expand.grid(
  mtry       = seq(16, 26, by = 2),
  node_size  = seq(4, 16, by = 2),
  sampe_size = c(.55, .632, .70),
  OOB_RMSE   = 0
)

# Loop over grid, train model and save error
for(i in 1:nrow(hyper_grid)) {
  print(i)

  # train model
  model <- ranger(
    formula         = more_than_10_perc_damage ~ .,
    data            = df,
    num.trees       = 500,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sampe_size[i],
    seed            = 123
  )

  # add OOB error to grid
  hyper_grid$OOB_RMSE[i] <- model$prediction.error
}

# Create the optimal model based on the best score from the grid search
optimal_model <- ranger(more_than_10_perc_damage ~ ., data = df,
                        num.trees = 500, mtry = 24, min.node.size = 14,
                        importance = 'impurity')

# Check the variable importance
optimal_model$variable.importance %>%
  tidy() %>%
  dplyr::arrange(desc(x)) %>%
  dplyr::top_n(25) %>%
  ggplot(aes(reorder(names, x), x)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top 25 important variables")

# And confusion matrix
optimal_model$confusion.matrix

# --------------- XGBoost ------------------------------------------

set.seed(7)

# Simple xgboost with self sampled test and training set
train_ind <- sample(seq_len(nrow(df)), size = 0.8*nrow(df))

df_train <- df[train_ind, ]
df_test <- df[-train_ind, ]

train_x <- data.matrix(df_train %>% select(-more_than_10_perc_damage))
train_y <- as.numeric(as.character(df_train$more_than_10_perc_damage))

xgb_train <- xgb.DMatrix(data=train_x, label=train_y)

test_x <- data.matrix(df_test %>% select(-more_than_10_perc_damage))
test_y <- as.numeric(as.character(df_test$more_than_10_perc_damage))

xgb_test <- xgb.DMatrix(data=test_x, label=test_y)

params <- list(booster = "gbtree",
               objective = "binary:logistic",
               eta=0.3, gamma=0,
               max_depth=6,
               min_child_weight=1,
               subsample=1,
               colsample_bytree=1)

xgb_model <- xgb.train(params = params,
                       data=xgb_train,
                       nrounds=100,
                       verbose=TRUE)

pred = predict(xgb_model, xgb_test)

pred_bin = as.factor(round(pred))
confusionMatrix(as.factor(test_y), pred_bin)

# Optimizing nrounds with cross validation function
xgb_cv_model <- xgb.cv( params = params,
                        data = xgb_train,
                        nrounds = 100,
                        nfold = 5,
        showsd = T,
        stratified = T,
        print_every_n = 10,
        early_stopping_roudns = 30,
        maximize = F)

# Min error on 72, train model
min(xgb_cv_model$evaluation_log$test_error_mean)


xgb_opt_nround <- xgb.train (params = params, data = xgb_train, nrounds = 72,
           watchlist = list(val=xgb_test, train=xgb_train), print_every_n = 10, early_stopping_rounds = 10, maximize = F , eval_metric = "error")

xgbpred <- predict (xgb_opt_nround, xgb_test)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

# Check conf matrix and var importance
confusionMatrix (as.factor(xgbpred), as.factor(test_y))

mat <- xgb.importance (feature_names = colnames(df),model = xgb_opt_nround)
xgb.plot.importance (importance_matrix = mat[1:30])



# ------------------------  Tuning xgboost with MLR --------------------------------------
library(mlr)
library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())


#create tasks
traintask <- makeClassifTask (data = df_train, target = "more_than_10_perc_damage")
testtask <- makeClassifTask (data = df_test, target = "more_than_10_perc_damage")

#create tasks
traintask2 <- makeClassifTask (data = df_train, target = "perc_damage")
testtask2 <- makeClassifTask (data = df_test, target = "perc_damage")

#create learner
lrn <- makeLearner("classif.xgboost",
                   predict.type = "response",
                   par.vals = list( objective="binary:logistic",
                                    eval_metric="error",
                                    nrounds=300L))




lrn2 <- makeLearner("regr.xgboost",
                    predict.type = "response",
                    par.vals = list( objective="reg:logistic",
                                     eval_metric="mae",
                                     nrounds=300L))


 

#set parameter space getParamSet("regr.xgboost")


params <- makeParamSet(makeDiscreteParam("booster",values = c("gbtree","gbdart")),
                       makeIntegerParam("max_depth",lower = 3L,upper = 10L),
                       makeNumericParam("min_child_weight",lower = 1L,upper = 10L),
                       makeNumericParam("subsample",lower = 0.5,upper = 1),
                       makeNumericParam("eta",lower = 0.3,upper = 1),
                       #makeNumericParam("gamma",lower = 0,upper = 100),
                       makeNumericParam("alpha",lower = 0, upper = 2, trafo = function(x) 10^x-1),
                       makeNumericParam("lambda", lower = 0, upper = 2, trafo = function(x) 10^x-1),
                       makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=10L)

#search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, 
                     par.set = params, control = ctrl, show.info = T)

# Check new accuracy
 


typhoon_names<-as.character(unique(data$GEN_typhoon_name))


hyper_grid_C <- expand.grid(
  typhoon=typhoon_names,
  booster=NA,
  max_depth=0,
  min_child_weight=0,
  subsample=0,
  eta=0,
  lambda=0,
  colsample_bytree=0,
  accuracy=0
)


i<-1
df_score_list<-list()
for(typhoons in typhoon_names){
  df_train<-input_prep(data%>%filter(GEN_typhoon_name!=typhoons))
  df_test<-input_prep(data%>%filter(GEN_typhoon_name==typhoons))
  
  mytune <- tuneParams(learner = lrn,
                       task = traintask, 
                       resampling = rdesc,
                       measures = acc, 
                       par.set = params, 
                       control = ctrl,
                       show.info = T)

  # Recreate optimal model
  lrn_tune <- setHyperPars(lrn, par.vals = mytune$x)
  xgmodel <- train(learner = lrn, task = traintask)
  xgpred <- predict(xgmodel, testtask)
  
  confusionMatrix(xgpred$data$response,xgpred$data$truth)
  
  hyper_grid_C$accuracy[i]<- mytune$y
  hyper_grid_C$booster[i]<- mytune$x[1]
  hyper_grid_C$max_depth[i]<- mytune$x[2]
  hyper_grid_C$min_child_weight[i]<- mytune$x[3]
  hyper_grid_C$subsample[i]<- mytune$x[4]
  hyper_grid_C$eta[i]<- mytune$x[5]
  hyper_grid_C$lambda[i]<- mytune$x[6]
  hyper_grid_C$colsample_bytree[i]<- mytune$x[7]
  
  lrn_tune <- setHyperPars(lrn, par.vals =mytune$x )
  xgmodel <- train(learner = lrn_tune, task = traintask)
  
  xgpred <- predict(xgmodel, testtask)
  xgpred2 <- predict(xgmodel, traintask)
  
  err_matrix<-confusionMatrix(xgpred$data$response,xgpred$data$truth)
  err_matrix2<-confusionMatrix(xgpred2$data$response,xgpred2$data$truth)
  
  df_score<-rbind(as.data.frame(err_matrix$byClass)%>%
                    rename('score_test'='err_matrix$byClass'),as.data.frame(err_matrix$overall)%>%
                    rename('score_test'='err_matrix$overall'))%>%dplyr::mutate(typhoon=typhoons)
  df_score2<-rbind(as.data.frame(err_matrix2$byClass)%>%
                    rename('score_train'='err_matrix2$byClass'),as.data.frame(err_matrix2$overall)%>%
                    rename('score_train'='err_matrix2$overall'))
  
  df_score_list[[i]]<-cbind(df_score2,df_score)
  
  
  
  
  
  
  i<-i+1
  
  
}

hyper_grid_R <- expand.grid(
  typhoon=typhoon_names,
  booster=NA,
  max_depth=0,
  min_child_weight=0,
  subsample=0,
  eta=0,
  lambda=0,
  colsample_bytree=0,
  accuracy=0
)


mytune2 <- tuneParams(learner = lrn2,
                      task = traintask, 
                      resampling = rdesc,
                      measures = acc, 
                      par.set = params, 
                      control = ctrl,
                      show.info = T)
 


# save model to binary local file
#xgb.save(bst, "xgboost.model")

# ------------------- Ensemble --------------------------------------
library(caretEnsemble)

trControl <- trainControl(
  method="cv",
  number=7,
  savePredictions="final",
  # index=createResample(train.full.dt$OverallQual, 7),
  allowParallel =TRUE,
  classProbs=TRUE
)

xgbTreeGrid <- expand.grid(nrounds = 72, max_depth = seq(5, 8,by = 1), eta = 0.1, gamma = 0, colsample_bytree = 1.0,  subsample = 1.0, min_child_weight = seq(1, 4, by=1))
rangerGrid <- expand.grid(.splitrule='gini', .mtry = 18, .min.node.size = 12)

modelList <<- caretList(
  x = train_x,
  y = as.factor(ifelse(train_y, 'yes', 'no')),
  # y = train_y,
  trControl=trControl,
  metric="error",
  tuneList=list(
    ## Do not use custom names in list. Will give prediction error with greedy ensemble. Bug in caret.
    xgbTree = caretModelSpec(method="xgbTree",  tuneGrid = xgbTreeGrid, nthread = 8),
    ranger=caretModelSpec(method="ranger", tuneGrid = rangerGrid)
  )
)

set.seed(333)
greedyEnsemble <- caretEnsemble(
  modelList,
  metric="error",
  trControl=trainControl(
    number=7, method = "cv"
  ))



summary(greedyEnsemble)
