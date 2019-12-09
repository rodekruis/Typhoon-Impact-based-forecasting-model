
source('C://documents//philipiness//Typhoons//model//new_model//data_source.R')

library(boot)
library(AUCRF)
library(RFmarkerDetector)
library(kernlab)
library(ROCR)
library(MASS)
library(glmnet)
library(MLmetrics)
# function to obtain bootstrapped stat coefficients and determine significance 

############# ------------------------ linear regression -----------------------------------
OLS_coef_bootstrap <- function(data, bootstrap_iterations)
{ 
  bootstrapped_coefs <- boot(data, statistic=stat_coef, R=bootstrap_iterations)
  output <- NULL
  
  for (i in 1:ncol(bootstrapped_coefs$t))
  {
    var_output <- c(bootstrapped_coefs$t0[i],mean(bootstrapped_coefs$t[,i]),quantile(bootstrapped_coefs$t[,i],probs=c( 0.1, 0.5, 0.95, 0.99)))
    output <- cbind(output, var_output)
  }
 
  rownames(output) <- c("Original","Bootstrapped","1%","50%","95%","99%")
  colnames(output) <-  c('R2_','RMSE_','NRMSE','MAE_','MAPE','MBE_','SMAPE')
 
  return(output)}




# function to return stat coefficients


stat_coef <- function(data, indices) {
  
  d <- data[indices,] 
  
  x <- d %>% dplyr::select(-'more_than_10_perc_damage') %>% data.matrix()
  y <- d$more_than_10_perc_damag
  y_predicted <- predict(cvfit, s =  cvfit$lambda.min, newx = x)
  
  # CALCULATE THE STATISTICS 
#R2
  R2_ <- R2(y_predicted, y)
#MEAN ABSOLUTE ERROR
  MAE_ <- MAE(y_predicted, y)
  
###Symmetric mean absolute percentage error
  Smbe1 <- (abs(y - y_predicted))/0.5*(abs(y)+abs(y_predicted))
  Smbe1[is.infinite(Smbe1)]<-NA
  
  SMAPE<-100*mean(Smbe1, na.rm = TRUE)
#ROOT MEAN SQUARE ERROR
  RMSE_ <- sqrt(mean((y - y_predicted)^2))
  
  result1 <- 100*abs((y_predicted-y)/y)
  
  result1[is.infinite(result1)]<-NA
#MEAN ABSOLUTE PERCENTAGE ERROR
  MAPE<-mean(result1,na.rm=TRUE)
  
#MEAN BIAS ERROR
  MBE_ <- mean(y_predicted-y)
  
#normalized root meat squared error
  NRMSE  <- sqrt(sum((y_predicted-y)^2)/sum((y_predicted-mean(y))^2))
  stat_list <- c(R2_,RMSE_,NRMSE,MAE_,MAPE,MBE_,SMAPE)
  return(stat_list)
  } 


data<-clean_typhoon_data()

typhoon_names<- c("Bopha","conson","Durian","Fengshen"	,"Fung-wong","Goni","Hagupit","Haima","Haiyan","Kalmaegi",
  "ketsana","Koppu","Krosa","Lingling","Mangkhut","Mekkhala","Melor","Nari","Nesat","Nock-Ten",
  "Rammasun","Sarika","Utor")

# Define grid for grid search here we are using the hyper parameters family and alpha , number of foldes we use the defult 10
hyper_grid3 <- expand.grid(
  typhoon=typhoon_names,
  family_t=c("gaussian","poisson"),
  alpha_t=c(0.1,0.5,1),
  lambda_min=0,
  mae_sd=0,
  mae=0
)

# Data data preparation please ignore the name " more_than_10_perc_damage" used it for nameing convection 

df <- data %>%
  dplyr::mutate(more_than_10_perc_damage = DAM_comp_houses_perc) %>%
  dplyr::select(-index,
                #-GEN_typhoon_name,
                -GEN_typhoon_name_year,
                -GEN_mun_code,
                -GEN_prov_name,
                -GEN_mun_name,
                #-GEO_n_households,
                -contains('DAM_')) %>%
  na.omit() 

seed= 123
i<- 1
# grid search 
for(i in 1:nrow(hyper_grid3)) {
  output <- NULL
  # train model  check ranger documentation 
  typ = as.vector(hyper_grid3$typhoon[i])
  
  df_train <- df[df$GEN_typhoon_name!=typ,] %>% dplyr::select(-GEN_typhoon_name,-GEO_n_households)
  
  df_val <- df[df$GEN_typhoon_name ==typ,] %>% dplyr::select(-GEN_typhoon_name,-GEO_n_households)
  
 
  

  
  
  x <- df_train %>% dplyr::select(-more_than_10_perc_damage) %>% data.matrix()
  y <- df_train$more_than_10_perc_damage

  cvfit <- cv.glmnet(x,y,alpha=hyper_grid3$alpha_t[i],family=as.vector(hyper_grid3$family_t[i]),type.measure="mae")
  

  hyper_grid3$lamda_min[i]<- cvfit$lambda.min
  
  hyper_grid3$mae[i]<- min(cvfit$cvm)
  hyper_grid3$mae_sd[i]<- cvfit$cvsd[cvfit$lambda == cvfit$lambda.min]
  
  coefs_lr <- OLS_coef_bootstrap(df_val,1000)
  coefs_lr <- as.data.frame(coefs_lr)
  

  #output <- cbind(output, coefs)
  coefs_lr$grid_ind <-i
  coefs_lr$typhoon <-typ

  write.table(coefs_lr, "C:/documents/philipiness/Typhoons/Journal_paper_ibf/stat_per_typhoon_linear_regression_3.csv",
              sep = ",", col.names = !file.exists("C:/documents/philipiness/Typhoons/Journal_paper_ibf/stat_per_typhoon_linear_regression.csv"), append = T)
  
  
}

write.table(hyper_grid_llr,"C:/documents/philipiness/Typhoons/Journal_paper_ibf/hypergrid_linear_regression.csv")


#############--------------random forest reggression-----------------
# function to obtain bootstrapped stat coefficients and determine significance 
OLS_coef_bootstrap4 <- function(data, bootstrap_iterations)
{ 
  bootstrapped_coefs <- boot(data, statistic=stat_coef4, R=bootstrap_iterations)
  output <- NULL
  
  for (i in 1:ncol(bootstrapped_coefs$t))
  {
    var_output <- c(bootstrapped_coefs$t0[i],mean(bootstrapped_coefs$t[,i]),quantile(bootstrapped_coefs$t[,i],probs=c( 0.1, 0.5, 0.95, 0.99)))
    output <- cbind(output, var_output)
  }
  rownames(output) <- c("Original","Bootstrapped","1%","50%","95%","99%")
  colnames(output) <-  c('R2_','RMSE_','NRMSE','MAE_','MAPE','MBE_','SMAPE')
  return(output)}
stat_coef4 <- function(data, indices) {
  
  d <- data[indices,] 
  
  x <- d %>% dplyr::select(-'more_than_10_perc_damage') %>% data.matrix()
  y <- d$more_than_10_perc_damag
  
  predictions1_v <- predict(model_forest_reg, d)
    y_predicted<-predictions1_v$predictions 
  
  # CALCULATE THE STATISTICS 
  #R2
  R2_ <- R2(y_predicted, y)
  #MEAN ABSOLUTE ERROR
  MAE_ <- MAE(y_predicted, y)
  
  ###Symmetric mean absolute percentage error
  Smbe1 <- (abs(y - y_predicted))/0.5*(abs(y)+abs(y_predicted))
  Smbe1[is.infinite(Smbe1)]<-NA
  
  SMAPE<-100*mean(Smbe1, na.rm = TRUE)
  #ROOT MEAN SQUARE ERROR
  RMSE_ <- sqrt(mean((y - y_predicted)^2))
  
  result1 <- 100*abs((y_predicted-y)/y)
  
  result1[is.infinite(result1)]<-NA
  #MEAN ABSOLUTE PERCENTAGE ERROR
  MAPE<-mean(result1,na.rm=TRUE)
  
  #MEAN BIAS ERROR
  MBE_ <- mean(y_predicted-y)
  
  #normalized root meat squared error
  NRMSE  <- sqrt(sum((y_predicted-y)^2)/sum((y_predicted-mean(y))^2))
  stat_list <- c(R2_,RMSE_,NRMSE,MAE_,MAPE,MBE_,SMAPE)
  return(stat_list) } 

##################### import data

data<-clean_typhoon_data()

typhoon_names<- c("Bopha","conson","Durian","Fengshen"	,"Fung-wong","Goni","Hagupit","Haima","Haiyan","Kalmaegi",
                  "ketsana","Koppu","Krosa","Lingling","Mangkhut","Mekkhala","Melor","Nari","Nesat","Nock-Ten",
                  "Rammasun","Sarika","Utor")

# Define grid for grid search here we are using the hyper parameters family and alpha , number of foldes we use the defult 10
hyper_grid_r <- expand.grid(
  mtry       = 0,
  node_size  = 0,
  sampe_fraction = 0,
  typhoon=typhoon_names,
  mse=0)

# Data data preparation please ignore the name " more_than_10_perc_damage" used it for nameing convection 

df <- data %>%
  dplyr::mutate(more_than_10_perc_damage = DAM_comp_houses_perc) %>%
  dplyr::select(-index,
                #-GEN_typhoon_name,
                -GEN_typhoon_name_year,
                -GEN_mun_code,
                -GEN_prov_name,
                -GEN_mun_name,
                #-GEO_n_households,
                -contains('DAM_')) %>%
  na.omit() 

seed= 123
 
for(i in 1:nrow(hyper_grid_r)) 
  {

  # train model  check ranger documentation 
  typ = as.vector(hyper_grid_r$typhoon[i])
  
  df_1 <- df[df$GEN_typhoon_name!=typ,] %>% dplyr::select(-GEN_typhoon_name,-GEO_n_households)
  df_val <- df[df$GEN_typhoon_name ==typ,] %>% dplyr::select(-GEN_typhoon_name,-GEO_n_households)
  
  typhoon_regression = makeRegrTask(data = df_1, target = "more_than_10_perc_damage")
  
  # Tuning
  tunned_f = tuneRanger(typhoon_regression, measure = NULL, num.trees = 500, num.threads = NULL,
                        tune.parameters = c("mtry", "min.node.size","sample.fraction"), save.file.path = NULL)
  
  
  hyper_grid_r$mtry[i]            = tunned_f$recommended.pars$mtry
  hyper_grid_r$min.node.size[i]   = tunned_f$recommended.pars$min.node.size
  hyper_grid_r$sample.fraction[i] = tunned_f$recommended.pars$sample.fraction
  hyper_grid_r$mse[i] =tunned_f$recommended.pars$mse
  
  # Model with the new tuned hyperparameters
  model_forest_reg <- ranger(
    formula         = more_than_10_perc_damage ~ .,
    data            = df_1,
    num.trees       = 500,
    mtry            = tunned_f$recommended.pars$mtry,
    min.node.size   = tunned_f$recommended.pars$min.node.size,
    sample.fraction = tunned_f$recommended.pars$sample.fraction,
    seed            = 123)

  # bootstrapping with 1000 replications 
  coefs_rf_rg <- OLS_coef_bootstrap4(df_val,1000)
  coefs_rf_rg <- as.data.frame(coefs_rf_rg)
  #output <- cbind(output, coefs)
  coefs_rf_rg$grid_ind <-i
  coefs_rf_rg$typhoon <-typ
  
  write.table(coefs_rf_rg,"C:/documents/philipiness/Typhoons/Journal_paper_ibf/stat_per_typhoon_forest_reg_3.csv",sep = ",",
              col.names = !file.exists("C:/documents/philipiness/Typhoons/Journal_paper_ibf/stat_per_typhoon_forest_reg_3.csv"), append = T)
  
}

# write result of hypergrid parameter search
write.table(hyper_grid_r,"C:/documents/philipiness/Typhoons/Journal_paper_ibf/hypergrid_forest_reg.csv")


############ --------------------- Lasso logistic regression ------------------------------
# function to obtain bootstrapped stat coefficients and determine significance 
OLS_coef_bootstrap2 <- function(data, bootstrap_iterations)
  { 

    bootstrapped_coefs <- boot(data, statistic=stat_coef2, R=bootstrap_iterations)
      output <- NULL
      
      for (i in 1:ncol(bootstrapped_coefs$t))
      {
        var_output <- c(bootstrapped_coefs$t0[i],mean(bootstrapped_coefs$t[,i]),quantile(bootstrapped_coefs$t[,i],probs=c( 0.1, 0.5, 0.95, 0.99),na.rm = TRUE))
        output <- cbind(output, var_output)
      }
      
      rownames(output) <- c("Original","Bootstrapped","1%","50%","95%","99%")
      colnames(output) <-  c('accuracy_','accuracy_lb','accuracy_ub','kappa_','F1.score_','auc_roc')
      
      return(output)
  
  }




# function to return stat coefficients

stat_coef2 <- function(data, indices) {
  d <- data[indices,] 
    # x <- d %>% dplyr::select(-'more_than_10_perc_damage') %>% data.matrix()  # y <- d$more_than_10_perc_damag
    #d<-df_val
    xval <- d %>% dplyr::select(-more_than_10_perc_damage) %>% data.matrix()
    yval <- d$more_than_10_perc_damage
    
    # CALCULATE THE STATISTICS 
    
    lasso.predict.pr <- predict(cv.lasso, newx=xval, s=cv.lasso$lambda.min, type = "response")
    # Plot confusion matrix:
    lasso.predict.pr[lasso.predict.pr >= "0.50"] <- 1
    lasso.predict.pr[lasso.predict.pr < "0.50"] <- 0
    
    pred <- prediction(lasso.predict.pr, yval) 
    
    auc <- performance(pred, measure= 'auc')
    auc_roc <- auc@y.values[[1]]
    
    confusionmatrix.lasso <- confusionMatrix(as.factor(lasso.predict.pr), as.factor(yval))
    
    # Calculate accuracy: 
    accuracy_ <- confusionmatrix.lasso[["overall"]][["Accuracy"]]
    kappa_ <- confusionmatrix.lasso[["overall"]][["Kappa"]]
    accuracy_lb <- confusionmatrix.lasso[["overall"]][["AccuracyLower"]]
    accuracy_ub <- confusionmatrix.lasso[["overall"]][["AccuracyUpper"]]
    
    #F1.score_ <- F1_Score(yval, lasso.predict.pr, positive = "1") 
    
    upscr_= (2*confusionmatrix.lasso[["byClass"]][["Precision"]]*confusionmatrix.lasso[["byClass"]][["Recall"]])
    subs_= sum(confusionmatrix.lasso[["byClass"]][["Precision"]], confusionmatrix.lasso[["byClass"]][["Recall"]])
    F1.score_ = ifelse(subs_ > 0, (upscr_/subs_), 0) 
    
    # 'MLmetrics
    stat_list<-c(accuracy_,accuracy_lb,accuracy_ub,kappa_,F1.score_,auc_roc)
  
    return(stat_list) 
   

}

  
data<-clean_typhoon_data()
#LL_result<-LL_regression(data)
data <- data %>%
  dplyr::mutate(more_than_10_perc_damage = ifelse(DAM_comp_houses_perc > 10, 1, 0), more_than_10_perc_damage = as.factor(more_than_10_perc_damage)) %>%
  dplyr::select(-index,
                #-GEN_typhoon_name,
                -GEN_typhoon_name_year,
                -GEN_mun_code,
                -GEN_prov_name,
                -contains('DAM_'),
                -GEN_mun_name) %>%
  na.omit() # Randomforests don't handle NAs, you can impute in the future 


# for the classification problem make sure the training dataset contains typhones with at least 1 entry with damage above the threshold value 
typhoon_names<- c("Bopha","conson","Durian","Fengshen"	,"Fung-wong","Goni","Hagupit","Haima","Haiyan","Kalmaegi",
                  "ketsana","Koppu","Krosa","Lingling","Mangkhut","Mekkhala","Melor","Nari","Nesat","Nock-Ten",
                  "Rammasun","Sarika","Utor")
typhoon_names_ <-c()

for(typ1 in typhoon_names){
  # train model  check ranger documentation 
  df_val <- data[data$GEN_typhoon_name ==typ1,]
  df_val1<-data %>% filter(more_than_10_perc_damage == '1') #, Sepal.Width <= 3)
  
  if (length(df_val1$more_than_10_perc_damage) > 0){
     
    typhoon_names_ <- c(typhoon_names_,typ1)  }

}


#hyper_grid_lm1 <-hyper_grid_lm

hyper_grid_llr <- expand.grid(
  typhoon  = typhoon_names_,
  type_measure = c("auc", "class","mae"),
  alpha_t = c(0.1,0.5,1),
  obe=0,
  obe_sd=0,
  lamda_min=0
  )


library(glmnet) 

df<-data

for(i in 1:nrow(hyper_grid_llr)) {
  # train model  check ranger documentation 
  print("logistic regression lasio -classification")
  set.seed(123)
  typ = as.vector(hyper_grid_llr$typhoon[i])
  
  type_measure=as.vector(hyper_grid_llr$type_measure[i])
  
  df_1 <- df[df$GEN_typhoon_name!=typ,] %>% dplyr::select(-GEN_typhoon_name,-GEO_n_households)
  df_val <- df[df$GEN_typhoon_name ==typ,] %>% dplyr::select(-GEN_typhoon_name,-GEO_n_households)
  

  
  xtrain <- df_1 %>% dplyr::select(-more_than_10_perc_damage) %>% data.matrix()
  ytrain <- df_1$more_than_10_perc_damage
  
  # Fit model on outertrain (with 10-fold cross-validation to determine optimal lambda): 
  
  cv.lasso <- cv.glmnet(xtrain, ytrain,  alpha=hyper_grid_llr$alpha_t[i], family="binomial",type.measure=type_measure) 
  
  hyper_grid_llr$lamda_min[i]<- cvfit$lambda.min
  
  hyper_grid_llr$obe[i]<- min(cvfit$cvm)
  hyper_grid_llr$obe_sd[i]<- cvfit$cvsd[cvfit$lambda == cvfit$lambda.min]
  
  # Make predictions on outer testset: 
  
  ##################
  # 
  tryCatch(
    expr = {
      coefs_1 <- OLS_coef_bootstrap2(df_val,1000)
      coefs_1 <- as.data.frame(coefs_1)
      #output <- cbind(output, coefs_1)
      coefs_1$grid_ind <-i
      coefs_1$typhoon <-typ
      write.table(coefs_1,"C:/documents/philipiness/Typhoons/Journal_paper_ibf/stat_per_typhoon_lasio_logistic.csv",
                  sep = ",", col.names = !file.exists("C:/documents/philipiness/Typhoons/Journal_paper_ibf/stat_per_typhoon_lasio_logistic.csv"), append = T)
      
       },
    error = function(e) {print('Caught an error!') }
  )
}

write.table(hyper_grid_llr,"C:/documents/philipiness/Typhoons/Journal_paper_ibf/hypergrid_lasio_logistic.csv")



############--------------------- RANDOM FOREST classfication ------------------------------
# function to obtain bootstrapped stat coefficients and determine significance 
OLS_coef_bootstrap3 <- function(data, bootstrap_iterations)
{ 
  
  bootstrapped_coefs <- boot(data, statistic=stat_coef3, R=bootstrap_iterations)
  output <- NULL
  for (i in 1:ncol(bootstrapped_coefs$t))
  {
    var_output <- c(bootstrapped_coefs$t0[i],mean(bootstrapped_coefs$t[,i]),quantile(bootstrapped_coefs$t[,i],probs=c( 0.1, 0.5, 0.95, 0.99),na.rm = TRUE))
    output <- cbind(output, var_output)
  }
  
  rownames(output) <- c("Original","Bootstrapped","1%","50%","95%","99%")
  colnames(output) <-  c('accuracy_','accuracy_lb','accuracy_ub','kappa_','F1.score_','auc_roc')
  return(output)
  
}

# function to return stat coefficients
stat_coef3 <- function(data, indices) {
  d <- data[indices,] 
  yval <- as.vector(d$more_than_10_perc_damag)
  model_pr <- predict(model_forest, data = d , predict.all = FALSE, num.trees = model_forest$num.trees, type = "response")
  y_predicted <-as.numeric(as.character(model_pr$predictions))
  yval<- as.numeric(as.character(yval))
  confusionmatrix.lasso<-confusionMatrix(as.factor(y_predicted),as.factor(yval))
  

 
  # CALCULATE THE STATISTICS 
  pred <- prediction(y_predicted, yval)
  
  auc <- performance(pred, measure= 'auc')
  auc_roc <- auc@y.values[[1]]
  
 
  
  # Calculate accuracy: 
  accuracy_ <- confusionmatrix.lasso[["overall"]][["Accuracy"]]
  kappa_ <- confusionmatrix.lasso[["overall"]][["Kappa"]]
  accuracy_lb <- confusionmatrix.lasso[["overall"]][["AccuracyLower"]]
  accuracy_ub <- confusionmatrix.lasso[["overall"]][["AccuracyUpper"]]
  
  upscr_= (2*confusionmatrix.lasso[["byClass"]][["Precision"]]*confusionmatrix.lasso[["byClass"]][["Recall"]])
  subs_= sum(confusionmatrix.lasso[["byClass"]][["Precision"]], confusionmatrix.lasso[["byClass"]][["Recall"]])
  F1.score_ = ifelse(subs_ > 0, (upscr_/subs_), 0) 

  stat_list<-c(accuracy_,accuracy_lb,accuracy_ub,kappa_,F1.score_,auc_roc)
  

  return(stat_list) 
  
  
}

# re_import data

source('C://documents//philipiness//Typhoons//model//new_model//data_source.R')

data<-clean_typhoon_data()

data <- data %>%
  dplyr::mutate(more_than_10_perc_damage = ifelse(DAM_comp_houses_perc > 10, 1, 0), more_than_10_perc_damage = as.factor(more_than_10_perc_damage)) %>%
  dplyr::select(-index,
                #-GEN_typhoon_name,
                -GEN_typhoon_name_year,
                #-GEO_n_households,
                #-GEN_mun_code,
                -GEN_prov_name,
                -contains('DAM_'),
                -GEN_mun_name) %>%
  na.omit() # Randomforests don't handle NAs, you can impute in the future 

# for the classification problem make sure the training dataset contains typhones with at least 1 entry with damage above the threshold value 
typhoon_names<- c("Bopha","conson","Durian","Fengshen"	,"Fung-wong","Goni","Hagupit","Haima","Haiyan","Kalmaegi",
                  "ketsana","Koppu","Krosa","Lingling","Mangkhut","Mekkhala","Melor","Nari","Nesat","Nock-Ten",
                  "Rammasun","Sarika","Utor")
typhoon_names_ <-c()

for(typ1 in typhoon_names){
  # train model  check ranger documentation 
  df_val <- data[data$GEN_typhoon_name ==typ1,]
  df_val1<-df_val %>% filter(more_than_10_perc_damage == '1') #, Sepal.Width <= 3)
  
  if (length(df_val1$more_than_10_perc_damage) > 0){
    print(length(df_val1$more_than_10_perc_damage))
    typhoon_names_ <- c(typhoon_names_,typ1)  }
  
}



# Basic random forest to check performance and number of trees
# Define grid for grid search
hyper_grid <- expand.grid(
  mtry       = 0,
  node_size  = 0,
  sampe_size = 0,
  #sampe_split = c("gini" , "extratrees"),
  typhoon=typhoon_names_,
  auc=0)

 
for(i in 1:nrow(hyper_grid)) 
{
  print("random forest -classification")
  # train model  check ranger documentation 
  typ = as.vector(hyper_grid$typhoon[i])
  
  df_1 <- data[data$GEN_typhoon_name!=typ,] %>% dplyr::select(-GEN_typhoon_name,-GEO_n_households)
  df_val <- data[data$GEN_typhoon_name ==typ,] %>% dplyr::select(-GEN_typhoon_name,-GEO_n_households)
  
  
  typhoon_class = makeClassifTask(data = df_1, target = "more_than_10_perc_damage")
  
  
  # Tuning
  tunned_f = tuneRanger(typhoon_class, measure = NULL, num.trees = 500, num.threads = NULL,
                        tune.parameters = c("mtry", "min.node.size","sample.fraction"), save.file.path = NULL)
  
  
  hyper_grid$mtry[i]            = tunned_f$recommended.pars$mtry
  hyper_grid$node_size[i]   = tunned_f$recommended.pars$min.node.size
  hyper_grid$sampe_size[i] = tunned_f$recommended.pars$sample.fraction
  hyper_grid$auc[i] =tunned_f$recommended.pars$auc
  
  # Model with the new tuned hyperparameters
  model_forest <- ranger(
    formula         = more_than_10_perc_damage ~ .,
    data            = df_1,
    num.trees       = 500,
    mtry            = tunned_f$recommended.pars$mtry,
    min.node.size   = tunned_f$recommended.pars$min.node.size,
    sample.fraction = tunned_f$recommended.pars$sample.fraction,
    seed            = 123  )
  
  
  
 
  
  # Make predictions on outer testset: 

  tryCatch(
    expr = {
      coefs_rf_c <- OLS_coef_bootstrap3(df_val,1000)
      
      coefs_rf_c <- as.data.frame(coefs_rf_c)
      
      #output <- cbind(output, coefs_1)
      coefs_rf_c$grid_ind <-i
      coefs_rf_t$typhoon <-typ
      
      write.table(coefs_rf_c,"C:/documents/philipiness/Typhoons/Journal_paper_ibf/stat_per_typhoon_random forest_classfier_3.csv",
                  sep = ",", col.names = !file.exists("C:/documents/philipiness/Typhoons/Journal_paper_ibf/stat_per_typhoon_random forest_classfier_3.csv"), append = T)
      
    },
    error = function(e) {print('Caught an error!') }
  )
}


# write result of hypergrid parameter search
write.table(hyper_grid,"C:/documents/philipiness/Typhoons/Journal_paper_ibf/hypergrid_forest_classfier.csv")







