#------------------------  Load required packages ------------------------------

# Load required packages: 

install.packages(c("AUCRF", "caret", "glmnet", "kernlab", "MLmetrics", "naniar", "pROC", "psych", "readxl",
                   "RFmarkerDetector", "ROCR", "tidyimpute", "visdat"))
library(dplyr)
library(ranger)
library(caret)
library(randomForest)
library(broom)
library(xgboost)

library(rlang)
library(tibble)
library(reshape2)
library(rasterVis)
library(RColorBrewer)
library(utils)
library(zoo)
library(readr)
library(utils)
library(readxl)

library(RFmarkerDetector)
library(AUCRF)

library(kernlab)
library(ROCR)
library(MASS)
library(glmnet)
 
library(MLmetrics)
library(plyr)
library(psych)
library(corrplot)
library(visdat)
library(naniar)
library(lubridate)
library(tidyimpute)
library(ggplot2)
library(dplyr)
library(tidyr)
library("xlsx")


library(stringr)

# import code to clean data 
source('C://documents//philipiness//Typhoons//model//new_model//data_source.R')
data1<-clean_typhoon_data()

setwd('C://documents//philipiness//Typhoons//model//new_model')

df <- data1 %>%
  dplyr::mutate(more_than_10_perc_damage = ifelse(DAM_comp_houses_perc > 10, 1, 0), more_than_10_perc_damage = as.factor(more_than_10_perc_damage)) %>%
  dplyr::select(-index,
         #-GEN_typhoon_name,
         -GEN_typhoon_name_year,
         #-GEN_mun_code,
         -GEN_prov_name,
         #-contains('DAM_'),
         -GEN_mun_name) %>%
  na.omit() # Randomforests don't handle NAs, you can impute in the future

summary(df)
str(df)
#threshold for classification 
threshold=5

# Make total affect variable binary: GEN_dist_track
df<-df %>% filter(GEN_dist_track < 200) #, Sepal.Width <= 3)
df <- df %>% na.omit(GEO_n_households)

data<-df
# Standardize the variables: 

for (i in 3:ncol(data))
{
  data[,i] <- (as.numeric(data[,i])-mean(as.numeric(data[,i])))/sd(as.numeric(data[,i]))
} 

correlations <- round(cor(data[,3:length(data)]),2)

corrplot(correlations)


data<-data %>% dplyr::select(-GEO_n_households,-GEN_typhoon_name) %>%
  na.omit()

data$more_than_10_perc_damage <- as.factor(data$more_than_10_perc_damage)





#--------------------- Conditional density plots of total impact vs each independent variable: 
for (i in 1:length(data)) {
  cdplot(more_than_10_perc_damage ~ data[,i], data = data,  bw = "nrd0", n = 512, main = names(data[i]))
}

data.dam_per_typ <- df %>%
  group_by(GEN_typhoon_name) %>%
  summarise(n_observations    = n(),
            n_municipalties   = length(unique(GEN_mun_code)),
            min_damage        = min(DAM_comp_houses_perc),
            Q1_damage         = quantile(DAM_comp_houses_perc,0.25),
            Q2_damage         = quantile(DAM_comp_houses_perc,0.5),
            Q3_damage         = quantile(DAM_comp_houses_perc,0.75),
            max_damage        = max(DAM_comp_houses_perc),
            mean_damage       = mean(DAM_comp_houses_perc))

#Graph with number of municipalities per typhoon
 graph.nr_mun_per_typ <- 
   ggplot(df, aes(x=GEN_typhoon_name, y=n_municipalties))+
   geom_col(fill='grey', width=0.8)+
   theme_bw()+
   geom_text(aes(label=n_municipalties), vjust=-0.4) +
   xlab("Typhoon") +
   ylab("Number of municipalities with damage")+
   theme(axis.text.x=element_text(angle =- 90, vjust = 0.5))
  

ggsave(filename='C:/Typhoons/graphs/PIC_nr_mun_per_typ.png',plot=graph.nr_mun_per_typ, width = 25, height = 20, units = "cm")

main_dir<-'C:/Typhoons/graphs/'
#Create boxplot of percentage of completely damaged houses per typhoon
boxplot.dam_per_typ1 <-
  ggplot(data, aes(x=GEN_typhoon_name,y=DAM_comp_houses_perc))+
  geom_boxplot(width=0.8, fill="orangered") +
  scale_y_continuous(limits=c(0,100),name="Percentage of houses completely damaged") +
  scale_x_discrete(name="Typhoon") +
  coord_flip() +
  theme_bw() +
  theme(text=element_text(size=20))

ggsave(filename='C:/Typhoons/graphs/Boxplot damage per typhoon.png',plot=boxplot.dam_per_typ1, width = 35, height = 20, units = "cm")

typhoon_names<-levels(droplevels(data$GEN_typhoon_name[!duplicated(data$GEN_typhoon_name)]))

#Make correlation matrices ----
cor.matrix.tot <- round(cor(data[,8:length(data)]),2)


cor.dam.windspeed <- as.matrix(cor.matrix.tot[9:ncol(cor.matrix.tot),5])
write.xlsx(cor.matrix.tot, file="Correlation_matrices.xlsx",sheetName="Total",row.names=TRUE)

for (typ in typhoon_names)
{   
  #Make correlation matrix
  name <- paste("cor.matrix.", typ, sep="")
  cor.temp.loop <- round(
    cor( data %>%
           filter(GEN_typhoon_name == typ) %>%
           select(-c(index,GEN_typhoon_name,GEN_mun_code,GEN_mun_name,GEN_prov_code)) 
    )
    ,2)
  #assign(name,cor.temp.loop)
  
  
  cor.dam.windspeed <- cbind(cor.dam.windspeed,cor.temp.loop[9:ncol(cor.matrix.tot),5])
  write.xlsx(cor.temp.loop, file="Correlation_matrices.xlsx",sheetName=typ,append=TRUE,row.names=TRUE)
}

colnames(cor.dam.windspeed) <- c('Total',typhoon_names)
write.xlsx(cor.dam.windspeed,"Corr_damage_others_per_typhoon.xlsx")
#write.csv(cor.dam.windspeed, 'Corr_damage_others_per_typhoon.xlsx_1.csv')




setwd('C://documents//philipiness//Typhoons//model//new_model')
 
df <- data1 %>%
  dplyr::mutate(more_than_10_perc_damage = ifelse(DAM_comp_houses_perc > 10, 1, 0), more_than_10_perc_damage = as.factor(more_than_10_perc_damage)) %>%
  dplyr::select(-index,
                #-GEN_typhoon_name,
                -GEN_typhoon_name_year,
                #-GEN_mun_code,
                -GEN_prov_name,
                #-contains('DAM_'),
                -GEN_mun_name) %>%
  na.omit() # Randomforests don't handle NAs, you can impute in the future


# Make total affect variable binary: GEN_dist_track
df<-df %>% filter(GEN_dist_track < 200) #, Sepal.Width <= 3)
df <- df %>% na.omit(GEO_n_households)

data<-df



 # --------------------- Lasso logistic regression ------------------------------

data<-clean_typhoon_data()
source('lasso_logistic_regression.R')

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



typhoon_names<-c("Bopha","Durian","Fengshen","Hagupit","Haima","Haiyan","Mangkhut",
                 "Nesat","Rammasun")

# Start for-loop for (nested) 5-fold crossvalidation: 
nfolds <- 5 

#hyper_grid_lm1 <-hyper_grid_lm
hyper_grid_llr <- expand.grid(
  fold       = seq(1, nfolds, by = 1),
  type_measure=c("auc", "mse","mae"),
  typhoon  = typhoon_names,
  AUC.lasso= 0,
  AUC.lassov= 0,
  F1.lasso= 0,
  F1.lassov= 0,
  accuracy.lasso= 0,
  accuracy.lassov=0)


'%ni%'<-Negate('%in%')
hyper_grid_llr_backup<-hyper_grid_llr

df<-data


for(i in 1:nrow(hyper_grid_llr)) {
  # train model  check ranger documentation 
  set.seed(123)
  
  typ = as.vector(hyper_grid_llr$typhoon[i])
  type.measure=as.vector(hyper_grid_llr$type_measure[i])
  j=hyper_grid_llr$fold[i]
  
  df_1 <- df[df$GEN_typhoon_name!=typ,] %>% dplyr::select(-GEN_typhoon_name,-GEO_n_households)
  df_val <- df[df$GEN_typhoon_name ==typ,] %>% dplyr::select(-GEN_typhoon_name,-GEO_n_households)
  
  folds <- sample(rep(1:nfolds, length=nrow(df_1), nrow(df_1)))
  train <- df_1[folds != j,]
  test <- df_1[folds == j,]
  
  xtrain <- data.matrix(train %>% dplyr::select(-more_than_10_perc_damage))
  ytrain <- as.numeric(as.character(train$more_than_10_perc_damage))
  
  xtest <- data.matrix(test %>% dplyr::select(-more_than_10_perc_damage))
  ytest <- as.numeric(as.character(test$more_than_10_perc_damage))
  
  xval <- data.matrix(df_val %>% dplyr::select(-more_than_10_perc_damage))
  yval <- as.numeric(as.character(df_val$more_than_10_perc_damage))
  
  
  # Fit model on outertrain (with 10-fold cross-validation to determine optimal lambda): 
  
  cv.lasso <- cv.glmnet(xtrain, ytrain, alpha = 1, family="binomial",type.measure=type.measure) 
  
  
  betas <- coef(cv.lasso, cv.lasso$lambda.min,exact=TRUE)
  #write.table(as.matrix(betas), file="C://documents//philipiness//Typhoons//model//new_model//coef.csv", sep = ",", append = T)
  #write.xlsx(as.matrix(betas), file="C://documents//philipiness//Typhoons//model//new_model//llr_model_coef.xlsx",sheetName=i,append=TRUE,row.names=TRUE)
  write.xlsx(as.matrix(betas), file="C://documents//philipiness//Typhoons//model//new_model//llr_model_coef.xlsx",sheetName=paste0("Sheet",i),append=TRUE,row.names=TRUE)
  
  # Make predictions on outer testset: 
  lasso.predict.pr <- predict(cv.lasso, newx=xtest, s=cv.lasso$lambda.min, type = "response")
  lasso.predict.val <- predict(cv.lasso, newx=xval, s=cv.lasso$lambda.min, type = "response")
  # Calculate Area under the curve: 
  pred <- prediction(lasso.predict.pr, test$more_than_10_perc_damage) 
  
  auc <- performance(pred, measure= 'auc')
  
  hyper_grid_llr$AUC.lasso[i] <- auc@y.values[[1]]
  
  # Calculate Area under the curve: 
  vali_date<-prediction(lasso.predict.val, df_val$more_than_10_perc_damage) 
  
  auc_v <- performance(vali_date, measure= 'auc')
  hyper_grid_llr$AUC.lassov[i] <- auc_v@y.values[[1]]
  
  # Plot area under the curve: 
  perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
  #par(mfrow=c(2,2))
  #plot(perf, col=rainbow(7), main="ROC curve_t", xlab="1-Specificity",      ylab="Sensitivity")    
  #abline(0, 1) 
  
  # Plot area under the curve: 
  perfv <- performance(vali_date, measure = "tpr", x.measure = "fpr") 
  #plot(perfv, col=rainbow(7), main=paste0("ROC curve",typ), xlab="1-Specificity", ylab="Sensitivity")
  #abline(0, 1)
  
  
  # Plot confusion matrix:
  lasso.predict.pr[lasso.predict.pr >= "0.50"] <- 1
  lasso.predict.pr[lasso.predict.pr < "0.50"] <- 0
  
  lasso.predict.val[lasso.predict.val >= "0.50"] <- 1
  lasso.predict.val[lasso.predict.val < "0.50"] <- 0
  
  confusionmatrix.lasso <- confusionMatrix(as.factor(lasso.predict.pr), as.factor(ytest))
  confusionmatrix.lasso_v <- confusionMatrix(as.factor(lasso.predict.val), as.factor(yval))
  
  #fourfoldplot(confusionmatrix.lasso$table, main = "Confusion matrix_tr")
  #fourfoldplot(confusionmatrix.lasso_v$table, main = "Confusion matrix_val")
  #cm.lasso1 <- as.data.frame(confusionmatrix.lasso[["table"]])[,3]
  #cm.lasso2 <- as.data.frame(confusionmatrix.lasso_v[["table"]])[,3]
  #cm.lasso[,i] <- as.matrix(cm.lasso1)
  # cm.lassov[,i] <- as.matrix(cm.lasso2)
  
  # Calculate accuracy: 
  hyper_grid_llr$accuracy.lasso[i] <- confusionmatrix.lasso[["overall"]][["Accuracy"]]
  hyper_grid_llr$accuracy.lassov[i] <- confusionmatrix.lasso_v[["overall"]][["Accuracy"]]
  hyper_grid_llr$accuracy.lasso[i] <- confusionmatrix.lasso[["overall"]][["Accuracy"]]
  hyper_grid_llr$accuracy.lassov[i] <- confusionmatrix.lasso_v[["overall"]][["Accuracy"]]
  
  # Calculate F1 score: 
  hyper_grid_llr$F1.lasso[i] <- F1_Score(test$more_than_10_perc_damage, lasso.predict.pr, positive = "1")
  hyper_grid_llr$F1.lassov[i] <- F1_Score(df_val$more_than_10_perc_damage, lasso.predict.val, positive = "1")
  
}

write.table(hyper_grid_llr, file="C://documents//philipiness//Typhoons//model//new_model//hyper_grid_llr.csv", sep = ",", append = T)


library(reticulate)
use_python('C:\\anaconda\\python.exe')

sns <- import('seaborn')
plt <- import('matplotlib.pyplot')
pd <- import('pandas')
#specifying which version of python to use
#building a seaborn pairplot using pairplot()
hyper_grid_llr[[]]

# "fold"            "type_measure"    "typhoon"         "AUC.lasso"       "AUC.lassov"      "F1.lasso"       
# "F1.lassov"       "accuracy.lasso"  "accuracy.lassov"
sns$pairplot(r_to_py(hyper_grid_llr), hue = "typhoon")
#display the plot
plt$show()

 # Results of performance metrics: 
 apply(AUC.lasso, 2, mean)
 apply(F1.lasso, 2, mean)
 apply(accuracy.lasso, 2, mean)
 rowMeans(cm.lasso) # mean confusion matrix 
 
 
 
 # ------------------------ Random forest -----------------------------------
 source('C://documents//philipiness//Typhoons//model//new_model//data_source.R')
 typhoon_names<-c("Bopha","Durian","Fengshen","Hagupit","Haima","Haiyan","Mangkhut","Nesat","Rammasun")
 data<-clean_typhoon_data()
 # Basic random forest to check performance and number of trees
 # Define grid for grid search
 hyper_grid <- expand.grid(
   mtry       = seq(8, 26, by = 2), #seq(16, 26, by = 2),
   node_size  = seq(4, 16, by = 2),
   sampe_size = c(.55, .632, .70),
   sampe_split = c("gini" , "extratrees"),
   typhoon=typhoon_names,
   OOB_RMSE   = 0,
   Accurcy_t =0,
   Accurcy_v =0
 )
 
 data <- data %>%
   dplyr::mutate(more_than_10_perc_damage = ifelse(DAM_comp_houses_perc > 10, 1, 0), more_than_10_perc_damage = as.factor(more_than_10_perc_damage)) %>%
   dplyr::select(-index,
                 -GEN_typhoon_name,
                 -GEN_typhoon_name_year,
                 #-GEN_mun_code,
                 -GEN_prov_name,
                 -contains('DAM_'),
                 -GEN_mun_name) %>%
   na.omit() # Randomforests don't handle NAs, you can impute in the future 
 
 df<-data
 
nfolds <- 5 
'%ni%'<-Negate('%in%')
   
for(i in 1:nrow(hyper_grid)) {
  print(i)
     
     # train model  check ranger documentation 
     typ = as.vector(hyper_grid$typhoon[i])
 
     df_1 <- df[df$GEN_typhoon_name!=typ,] %>% dplyr::select(-GEN_typhoon_name,-GEO_n_households)
     df_val <- df[df$GEN_typhoon_name ==typ,] %>% dplyr::select(-GEN_typhoon_name,-GEO_n_households)
     
     model <- ranger(
       formula         = more_than_10_perc_damage ~ .,
       data            = df_1,
       num.trees       = 500,
       splitrule       = hyper_grid$sampe_split[i],
       mtry            = hyper_grid$mtry[i],
       min.node.size   = hyper_grid$node_size[i],
       sample.fraction = hyper_grid$sampe_size[i],
       seed            = 123
     )
     
     # add OOB error to grid
     #fourfoldplot(model$confusion.matrix, main = "Confusion matrix_tr")
     
     hyper_grid$OOB_RMSE[i] <- model$prediction.error
     pred1<-confusionMatrix(predict(model, df_1)$predictions, as.factor(df_1$more_than_10_perc_damage))
     pred2<-confusionMatrix(predict(model, df_val)$predictions, as.factor(df_val$more_than_10_perc_damage))
     hyper_grid$Accuracy_t[i] <- pred1[["overall"]][["Accuracy"]]
     hyper_grid$Accuracy_v[i] <- pred2[["overall"]][["Accuracy"]]
     #saveRDS(model, paste0("C:/Typhoons/models/randomforest_models/model_",i,".rds"))
   }
 
 
 # load best model super_model <- readRDS("./final_model.rds") maybe this is not needed 
 
 
 # Create the optimal model based on the best score from the grid search
 optimal_model <- ranger(more_than_10_perc_damage ~ ., data = df,
                         num.trees = 500, mtry = 18, min.node.size = 4,sample.fraction=.632,splitrule="gini",
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
 # 
 
 # save the model to disk
 saveRDS(optimal_model, "C://documents//philipiness//Typhoons//model//new_model//final_model.rds")
 

 # load the model
 RANDOMFOREST_model <- readRDS("C://documents//philipiness//Typhoons//model//new_model//final_model.rds")
 
 
 
 rm.predict.pr <- predict(RANDOMFOREST_model, data = df, predict.all = FALSE, num.trees = optimal_model$num.trees, type = "response",
         se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)
 
 
 
 
 FORECASTED_IMPACT<-as.data.frame(rm.predict.pr$predictions) %>%
   dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact=rm.predict.pr$predictions) %>%
  left_join(df , by = "index") %>%
   dplyr::select(GEN_mun_code,impact)
 
#plot graphs
 
 ggplot(data = hyper_grid, mapping = aes(x = Accuracy_v, y = Accuracy_t)) +
   geom_point(alpha = 0.5, aes(color = typhoon))
 
 
 ggplot(data = hyper_grid, mapping = aes(x = Accuracy_v, y = Accuracy_t, color = mtry)) +
   geom_point() +
   facet_wrap(facets =  vars(typhoon))
 
 ggplot(data = hyper_grid, mapping = aes(x = Accuracy_v, y = Accuracy_t, color = node_size)) +
   geom_point() +
   facet_wrap(facets =  vars(typhoon))
 ggplot(data = hyper_grid, mapping = aes(x = Accuracy_v, y = Accuracy_t, color = sampe_size)) +
   geom_point() +
   facet_wrap(facets =  vars(typhoon))
 ggplot(data = hyper_grid, mapping = aes(x = Accuracy_v, y = Accuracy_t, color = sampe_split)) +
   geom_point() +
   facet_wrap(facets =  vars(typhoon))
 
 
 
 

 
 
 
 tuneRanger(task, measure = NULL, iters = 70, iters.warmup = 30,time.budget = NULL, num.threads = NULL, num.trees = 1000,
            parameters = list(replace = FALSE, respect.unordered.factors ="order"), tune.parameters = c("mtry", "min.node.size","sample.fraction"),
            save.file.path = NULL, build.final.model = TRUE,
            show.info = getOption("mlrMBO.show.info", TRUE))
 

  # other way of tunning it 
 typhoon_class = makeClassifTask(data = df_1, target = "more_than_10_perc_damage")
  # Estimate runtime
 estimateTimeTuneRanger(typhoon_class)
 # Tuning
 res = tuneRanger(typhoon_class, measure = list(multiclass.brier), num.trees = 1000, num.threads = 2, iters = 70,
                  tune.parameters = c("mtry", "min.node.size","sample.fraction"), save.file.path = NULL)
 
 
 # ------------------------ linear model -----------------------------------
 data<-clean_typhoon_data()

 typhoon_names<-c("Bopha","Durian","Fengshen","Hagupit","Haima","Haiyan","Mangkhut",
                  "Nesat","Rammasun")

df <- data %>%
   dplyr::mutate(more_than_10_perc_damage = DAM_comp_houses_perc) %>%
  dplyr::select(-index,
                 #-GEN_typhoon_name,
                 -GEN_typhoon_name_year,
                 -GEN_mun_code,
                 -GEN_prov_name,
                 -GEN_mun_name,
                 -contains('DAM_')) %>%
   na.omit() # Randomforests don't handle NAs, you can impute in the future 



 
 levels(droplevels(df$GEN_typhoon_name[!duplicated(df$GEN_typhoon_name)]))
 # Start for-loop for (nested) 5-fold crossvalidation: 
 set.seed(123)
 nfolds<-5
 
 #hyper_grid_lm1 <-hyper_grid_lm
 hyper_grid_lm <- expand.grid(
   typhoon  = typhoon_names,
   alpha_       = seq(.1, 1, by = .1),
   R2= 0,
   RMSE= 0,
   MAE= 0,
   R2v= 0,
   RMSEv= 0,
   MAEv= 0,
   R21= 0,
   RMSE1= 0,
   MAE1= 0)
 
 for(i in 1:nrow(hyper_grid_lm)) {
   print(i)
   idx=i
   typ = as.vector(hyper_grid_lm$typhoon[i])
   alpha_ =hyper_grid_lm$alpha_[i]
   j=hyper_grid_lm$fold[i]
   
   #df_1 <- data.standardized[data.standardized$GEN_typhoon_name!=typ,] %>% dplyr::select(-GEN_typhoon_name)
   
   df_1_ <- df[df$GEN_typhoon_name!=typ,] %>% dplyr::select(-GEN_typhoon_name)
   df_val_ <-df[df$GEN_typhoon_name ==typ,] %>% dplyr::select(-GEN_typhoon_name)
   
   #df_val <- data.standardized[data.standardized$GEN_typhoon_name ==typ,] %>% dplyr::select(-GEN_typhoon_name)

 
     train_ <- df_1_#[folds != j,] # non normalized data
     test_ <- df_1_#[folds == j,]

     mod_coefs <- NULL
     
     xtrain <-df_1_ %>% dplyr::select(-more_than_10_perc_damage)
     ytrain <- as.numeric(as.character(df_1_$more_than_10_perc_damage))
     
     xtval <- df_val_ %>% dplyr::select(-more_than_10_perc_damage)
     
     xtval <- data.matrix(xtval)
     xtrain <- data.matrix(xtrain)
     
     ytvalt <- as.numeric(as.character(df_val_$more_than_10_perc_damage))
     
   # Build the model
     
   model_lm <- lm(more_than_10_perc_damage ~., data = df_1_)
   cvfit <- cv.glmnet(xtrain, ytrain,alpha = alpha_, type.measure = "mse", nlambda = 100)

   #train.control <- trainControl(method = "repeatedcv",  number = 5, repeats = 3)   # Train the model
   #model_lm_cv <- train(more_than_10_perc_damage ~., data = train, method = "lm",trControl = train.control)
   
   # Summarize the results
   # Make predictions and compute the R2, RMSE and MAE
   
   #predictions <- model_lm %>% predict(test)
  # predictions_v<- predict(model_lm,xtval)
   predictions1_v <- predict(cvfit, newx=xtval, s= "lambda.min")
   


   
   #predictions <- (predictions)*sd(as.numeric(df$more_than_10_perc_damage)) + mean(as.numeric(df$more_than_10_perc_damage))
   #predictions_v <- (predictions_v)*sd(as.numeric(df$more_than_10_perc_damage)) + mean(as.numeric(df$more_than_10_perc_damage))
 
 
   #hyper_grid_lm$R2v[idx]<-R2(predictions_v, ytvalt)
   #hyper_grid_lm$MAEv[idx]<-MAE(predictions_v, ytvalt)
   #hyper_grid_lm$RMSEv[idx]<-RMSE(predictions_v, ytvalt)

   hyper_grid_lm$R21[idx]<-R2(predictions1_v, ytvalt)
   hyper_grid_lm$MAE1[idx]<-MAE(predictions1_v, ytvalt)
   hyper_grid_lm$RMSE1[idx]<-RMSE(predictions1_v, ytvalt)
   

      
   mod_coefs <- coef(cvfit, cvfit$lambda.min,exact=TRUE)
   colnames(mod_coefs) <- paste0(typ,j)
   write.table(as.matrix(mod_coefs), file="C://documents//philipiness//Typhoons//model//new_model//lm_model_coef.csv", sep = ",", append = T)
   #write.xlsx(as.matrix(mod_coefs), file="C://documents//philipiness//Typhoons//model//new_model//lm_model_coef.xlsx",sheetName=i,append=TRUE,row.names=TRUE)

    }
 r21
 RMSE1
 (mae1)
   
   
 ggplot(data = hyper_grid_lm, mapping = aes(x = alpha_, y = R21)) +
   geom_point() +
   facet_wrap(facets =  vars(typhoon))
 
 
 
 ggplot(data = hyper_grid_lm, mapping = aes(x = alpha_, y = R21)) +
   geom_point(alpha = 0.5, aes(color = typhoon))
 
 ggplot(data = hyper_grid_lm, mapping = aes(x = alpha_, y = RMSE1)) +
   geom_point(alpha = 0.5, aes(color = typhoon))
 
 ggplot(data = hyper_grid_lm, mapping = aes(x = alpha_, y = MAE1)) +
   geom_point(alpha = 0.5, aes(color = typhoon))
 