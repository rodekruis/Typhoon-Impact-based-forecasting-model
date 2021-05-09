library(pso)
#Take start time to measure time of random search algorithm
start.time <- Sys.time()

# Create empty lists
lowest_error_list = list()
parameters_list = list()

# Create 10,000 rows with random hyperparameters
set.seed(20)

params <- makeParamSet(makeDiscreteParam("booster",values = c("gbtree","gbdart")),
                       makeIntegerParam("max_depth",lower = 3L,upper = 10L),
                       makeNumericParam("min_child_weight",lower = 1L,upper = 10L),
                       makeNumericParam("subsample",lower = 0.5,upper = 1),
                       #makeNumericParam("gamma",lower = 0,upper = 100),
                       makeNumericParam("alpha",lower = 0, upper = 2, trafo = function(x) 10^x-1),
                       makeNumericParam("lambda", lower = 0, upper = 2, trafo = function(x) 10^x-1),
                       makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))



for (iter in 1:10000){
  param <- list(booster ="gbtree",
                objective = "binary:logistic",
                max_depth = sample(3:10, 1),
                eta = runif(1, .01, .3),
                subsample = runif(1, .7, 1),
                gamma = runif(1, .1, 50),
                lambda = runif(1, 0.1, 50),
                colsample_bytree = runif(1, .5, 1),
                min_child_weight = sample(0:10, 1)
  )
  parameters <- as.data.frame(param)
  parameters_list[[iter]] <- parameters
}

# Create object that contains all randomly created hyperparameters
parameters_df = do.call(rbind, parameters_list)

traintask <- makeClassifTask (data = df_train, target = "more_than_10_perc_damage")
testtask <- makeClassifTask (data = df_test, target = "more_than_10_perc_damage")


train_x <- data.matrix(df_train %>% select(-more_than_10_perc_damage))
train_y <- as.numeric(as.character(df_train$more_than_10_perc_damage))

xgb_train <- xgb.DMatrix(data=train_x, label=train_y)



test_x <- data.matrix(df_test %>% select(-more_than_10_perc_damage))
test_y <- as.numeric(as.character(df_test$more_than_10_perc_damage))

xgb_test <- xgb.DMatrix(data=test_x, label=test_y)



# Use randomly created parameters to create 10,000 XGBoost-models
for (row in 1:nrow(parameters_df)){
  set.seed(20)
  mdcv <- xgb.train(data=xgb_train,
                    booster = as.character(parameters_df$booster[row]),
                    objective = "binary:logistic",
                    max_depth = parameters_df$max_depth[row],
                    eta = parameters_df$eta[row],
                    subsample = parameters_df$subsample[row],
                    colsample_bytree = parameters_df$colsample_bytree[row],
                    min_child_weight = parameters_df$min_child_weight[row],
                    nrounds= 300,
                    eval_metric = "error",
                    early_stopping_rounds= 30,
                    print_every_n = 100 )
  lowest_error <- as.data.frame(1 - min(mdcv$evaluation_log$val_error))
  lowest_error_list[[row]] <- lowest_error
}

# Create object that contains all accuracy's
lowest_error_df = do.call(rbind, lowest_error_list)

# Bind columns of accuracy values and random hyperparameter values
randomsearch = cbind(lowest_error_df, parameters_df)

# Quickly display highest accuracy
max(randomsearch$`1 - min(mdcv$evaluation_log$val_error)`)

# Stop time and calculate difference
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

write_csv(randomsearch, "data/randomsearch.csv")

# install.packages("pacman", verbose = F, quiet = T)
pacman::p_load(caret, tidyverse, readr, readxl, parallel, doParallel, gridExtra, plyr, pso, GA, DEoptim, GGally, xgboost, broom, knitr, kableExtra, tictoc, install = T)
ggcorr(data%>%select(-contains('INT'),-contains('GEN'),-contains('MAT')), label = TRUE, palette = "RdBu", name = "Correlation", hjust = 0.95, label_size = 2, label_round = 1)







# Simple xgboost with self sampled test and training set
train_ind <- sample(seq_len(nrow(df)), size = 0.8*nrow(df))

df_train <- df[train_ind, ]
df_test <- df[-train_ind, ]

training_set <- df_train
test_set <- df_test

# Training Parameters
CV_folds <- 5 # number of folds
CV_repeats <- 3 # number of repeats
minimum_resampling <- 5 # minimum number of resamples

# trainControl object for standard repeated cross-validation
train_control <- caret::trainControl(method = "repeatedcv", 
                                     number = CV_folds, 
                                     repeats = CV_repeats, 
                                     verboseIter = FALSE, 
                                     returnData = FALSE) 

# trainControl object for repeated cross-validation with grid search
adapt_control_grid <- caret::trainControl(method = "adaptive_cv", number = CV_folds, repeats = CV_repeats, 
                                          adaptive = list(min = minimum_resampling, # minimum number of resamples tested before model is excluded
                                                          alpha = 0.05, # confidence level used to exclude parameter settings
                                                          method = "gls", # generalized least squares
                                                          complete = TRUE), 
                                          search = "grid", # execute grid search
                                          verboseIter = FALSE, returnData = FALSE) 

# trainControl object for repeated cross-validation with random search
adapt_control_random <- caret::trainControl(method = "adaptive_cv", number = CV_folds, repeats = CV_repeats, 
                                            adaptive = list(min = minimum_resampling, # minimum number of resamples tested before model is excluded
                                                            alpha = 0.05, # confidence level used to exclude parameter settings
                                                            method = "gls", # generalized least squares
                                                            complete = TRUE), 
                                            search = "random", # execute random search
                                            verboseIter = FALSE, returnData = FALSE) 


# Set parameter settings for search algorithm
max_iter <- 10 # maximum number of iterations
pop_size <- 10 # population size
# Create custom function for assessing solutions

#create tasks
traintask <- makeClassifTask (data = df_train, target = "more_than_10_perc_damage")
testtask <- makeClassifTask (data = df_test, target = "more_than_10_perc_damage")

#create learner
lrn <- makeLearner("classif.xgboost",
                   predict.type = "response",
                   par.vals = list( objective="binary:logistic",
                                    eval_metric="error",
                                    nrounds=100L, 
                                    eta=0.1))


 

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=10L)

#search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)




eval_function_XGBoost_Linear <- function(x, data,train_settings){
  x1 <- x[1];
  x2 <- x[2];
  x3 <- x[3];
  x4 <- x[4];
  x5 <- x[5];
  x6 <- x[6];
  x7 <- x[7];
  x8 <- x[8];
  suppressWarnings(#Create dataframe with proportion of each solid component
    XGBoost_Linear_model <- caret::train(more_than_10_perc_damage ~.,
                                       data = df_train,
                                       method = "xgbDART",
                                       trControl = train_settings,
                                       verbose = FALSE,
                                       silent = 1,
                                       rate_drop=0,
                                       skip_drop=0,
                                       tuneGrid = expand.grid(
                                         nrounds = round(x1),
                                         eta = 10^x2,
                                         alpha = 10^x3,
                                         gamma= 10^x4,
                                         max_depth= round(x5),
                                         min_child_weight = 10^x6,
                                         colsample_bytree=10^x7,
                                         subsample=10^x8)
  ))

  return(XGBoost_Linear_model$results$error) # minimize RMSEmytune$x 
  #return(XGBoost_Linear_model$results$RMSE) # minimize RMSEmytune$x Accuracy

}

# Define minimum and maximum values for each input
nrounds_min_max <- c(100,10^3)
eta_min_max <- c(-2,-1)
alpha_min_max <- c(-3,1)
gamma_min_max <- c(-3,1)
max_depth_min_max<- c(4,12)
min_child_weight_min_max<- c(-1,1)
colsample_bytree_min_max<- c(-0.3,-0.1)
subsample_weight_min_max<- c(-0.3,-0.1)
set.seed(1)
n_cores <- detectCores()-1

PSO_T0 <- Sys.time()
# Run search algorithm

PSO_model_XGBoost_Linear <- pso::psoptim(
  par = rep(NA, 4),
  fn = eval_function_XGBoost_Linear, 
  lower = c(nrounds_min_max[1], eta_min_max[1], alpha_min_max[1], gamma_min_max[1], max_depth_min_max[1], min_child_weight_min_max[1],colsample_bytree_min_max[1],subsample_weight_min_max[1]),
  upper = c(nrounds_min_max[2], eta_min_max[2], alpha_min_max[2], gamma_min_max[2], max_depth_min_max[2], min_child_weight_min_max[2],colsample_bytree_min_max[2],subsample_weight_min_max[2]), 
  control = list(
    trace = 1, #  produce tracing information on the progress of the optimization
    maxit = max_iter, # maximum number of iterations
    REPORT = 1, #  frequency for reports
    trace.stats = T,
    s = pop_size, # Swarm Size,
    maxit.stagnate = round(0.75*max_iter), # maximum number of iterations without improvement
    vectorize = T,
    type = "SPSO2011" # method used
  ),
  data = training_set,
  train_settings = train_control
)

PSO_T1 <- Sys.time()
PSO_T1-PSO_T0
PSO_summary <- data.frame(
  Iteration = PSO_model_XGBoost_Linear$stats$it,
  Mean = PSO_model_XGBoost_Linear$stats$f %>% sapply(FUN = mean),
  Median = PSO_model_XGBoost_Linear$stats$f %>% sapply(FUN = median),
  Best = PSO_model_XGBoost_Linear$stats$error %>% sapply(FUN = min)
)
PSO_summary %>% 
  gather(key = "Parameter", value = "Value", - Iteration) %>% 
  ggplot(mapping = aes(x = Iteration, y = Value, col = Parameter)) +
  geom_line() +
  geom_point() +
  theme_bw() +
  theme(aspect.ratio = 0.9) +
  scale_x_continuous(breaks = PSO_model_XGBoost_Linear$stats$it, minor_breaks = NULL) +
  labs(x = "Iteration", y = "RMSE", title = "RMSE values at each iteration", subtitle = "Results using Particle Swarm Optimization") +
  scale_color_brewer(type = "qual", palette = "Set1")
