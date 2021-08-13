# Functions for calculating the damage probabilities


get_damage_probability <- function(damaged_houses, threshold, n_ensemble) {
  return(round(100 * sum(damaged_houses >= threshold) / n_ensemble))
}


get_total_impact_forecast <- function(df_impact_forecast, damage_thresholds, organization){
  
  n_ensemble <- length(unique(df_impact_forecast[["GEN_typhoon_id"]])) # usually 52
  
  # Get the total summarized impact
  # For some reason if <- is used here the method can't find this var
  df_total_impact_forecast = df_impact_forecast %>%
    group_by(GEN_typhoon_name, GEN_typhoon_id) %>%
    dplyr::summarise(CDamaged_houses = sum(Damaged_houses)) %>%
    group_by(GEN_typhoon_name) %>%
    dplyr::summarise(
      # TODO: Is there a more elegant way to do this?
      c1=get_damage_probability(CDamaged_houses, cerf_damage_thresholds[1], n_ensemble), 
      c2=get_damage_probability(CDamaged_houses, cerf_damage_thresholds[2], n_ensemble), 
      c3=get_damage_probability(CDamaged_houses, cerf_damage_thresholds[3], n_ensemble), 
      c4=get_damage_probability(CDamaged_houses, cerf_damage_thresholds[4], n_ensemble), 
      c5=get_damage_probability(CDamaged_houses, cerf_damage_thresholds[5], n_ensemble), 
    ) 
  
  # Rename the columns
  # TODO: Are the prefixes really needed?
  cname_prefix = c("VH", "H", "H", "M", "L")
  cnames = paste0(cname_prefix, "_", damage_thresholds / 1000, "k")
  colnames(df_total_impact_forecast) = c("Tyhoon_name", cnames)
  
  # Save to CSV
  write.csv(df_total_impact_forecast,
            file = paste0(Output_folder, organization, "_TRIGGER_LEVEL_",
                          forecast_time, "_", Typhoon_stormname, ".csv"),
            row.names=FALSE)
  
  # Print results to terminal
  # TODO: Doesn't seem to work at the moment
  df_total_impact_forecast %>%
    as_hux() %>%
    set_text_color(1, everywhere, "blue") %>%
    theme_article() %>%
    set_caption(
      paste0(
        organization,
        " PROBABILITY FOR THE NUMBER OF COMPLETELY DAMAGED BUILDINGS"
      )
    )
  
  # Get the impact by municiaplity
  # TODO: do these change between CERF / DREF?
  probabilities = c(0.95, 0.80, 0.70, 0.60, 0.50)
  cnames =  paste0("p", int(probabilities*100))
  df_mun_impact_forecast = df_impact_forecast %>%
    group_by(GEN_typhoon_name, GEN_mun_code) %>%
    dplyr::summarise(
      c1=quantile(Damaged_houses, c(probabilities[1])),
      c2=quantile(Damaged_houses, c(probabilities[2])),
      c3=quantile(Damaged_houses, c(probabilities[3])),
      c4=quantile(Damaged_houses, c(probabilities[4])),
      c5=quantile(Damaged_houses, c(probabilities[5])),
    )
  # Rename the columns
  colnames(df_mun_impact_forecast) = c("Tyhoon_name", "Municipality", cnames)
  # Save to CSV
  write.csv(df_mun_impact_forecast,
            file = paste0(Output_folder, organization, "_municipality_breakdown_",
                          forecast_time, "_", Typhoon_stormname, ".csv"),
            row.names=FALSE)
  
  return(df_total_impact_forecast)
}
