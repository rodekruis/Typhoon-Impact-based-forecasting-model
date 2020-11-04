#!/usr/bin/env Rscript
#options(warn=-1)

run_prediction_model<-function(data){
  
  ########## run prediction ##########
  
  rm.predict.pr <- predict(mode_classification, data = data, predict.all = FALSE, num.trees = mode_classification$num.trees, type = "response",
                           se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)
  
  FORECASTED_IMPACT<-as.data.frame(rm.predict.pr$predictions) %>%
    dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact_threshold_passed=rm.predict.pr$predictions) %>%
    left_join(data , by = "index") %>%
    dplyr::select(GEN_mun_code,impact_threshold_passed,WEA_dist_track)
  
  #colnames(FORECASTED_IMPACT) <- c(GEN_mun_code,paste0(TYF,'_impact_threshold_passed'),WEA_dist_track)
  
  rm.predict.pr <- predict(mode_continious, data = data, predict.all = FALSE, num.trees = mode_continious$num.trees, type = "response",
                           se.method = "infjack", quantiles = c(0.1, 0.5, 0.9), seed = NULL, num.threads = NULL, verbose = TRUE)
  
  FORECASTED_IMPACT_rr<-as.data.frame(rm.predict.pr$predictions) %>%
    dplyr::mutate(index= 1:length(rm.predict.pr$predictions),impact=rm.predict.pr$predictions) %>% 
    dplyr::mutate(priority_index=ntile_na(ifelse(impact<0,NA,impact),5))%>%
    left_join(data , by = "index") %>%
    dplyr::select(GEN_mun_code,impact,priority_index)
  
  df_imact_forecast<-FORECASTED_IMPACT %>% left_join(FORECASTED_IMPACT_rr,by='GEN_mun_code')
  
  #colnames(FORECASTED_IMPACT_rr) <- c(GEN_mun_code,paste0(TYF,'_impact'),paste0(TYF,'_priority_index'))
  
  return(df_imact_forecast)
}
