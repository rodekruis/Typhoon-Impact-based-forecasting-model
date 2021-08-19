# Functions for calculating the damage probabilities


get_total_impact_forecast <- function(df_impact_forecast, damage_thresholds,
                                      probabilities, organization) {
  # Set column names -- assume that threshold vector length is 5
  cnames <- paste0(">=", damage_thresholds / 1000, "k")
  damage_thresholds_named <- setNames(damage_thresholds, cnames)
  # Get the total summarized impact
  # For some reason if <- is used here the method can't find this var
  df_total_impact_forecast <- df_impact_forecast %>%
    group_by(GEN_typhoon_name, GEN_typhoon_id) %>%
    dplyr::summarise(CDamaged_houses = sum(Damaged_houses)) %>%
    group_by(GEN_typhoon_name) %>%
    dplyr::summarise(
      purrr::map_dfc(
        damage_thresholds_named, ~ get_damage_probability(CDamaged_houses, .x)
      )
    ) %>%
    ungroup() %>%
    dplyr::rename(Typhoon_name = GEN_typhoon_name)

  # Save to CSV
  write.csv(df_total_impact_forecast,
    file = paste0(
      Output_folder, organization, "_TRIGGER_LEVEL_",
      forecast_time, "_", Typhoon_stormname, ".csv"
    ),
    row.names = FALSE
  )

  # Print results to terminal
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
  cnames <- paste0("p", int(probabilities * 100))
  probabilities_named <- setNames(probabilities, cnames)

  df_mun_impact_forecast <- df_impact_forecast %>%
    group_by(GEN_typhoon_name, GEN_mun_code, GEN_mun_name) %>%
    dplyr::summarise(
      purrr::map_dfc(
        probabilities_named, ~ get_percentile(Damaged_houses, .x)
      )
    ) %>%
    ungroup() %>%
    add_row(
      GEN_mun_name = "TOTAL",
      dplyr::summarise(., across(where(is.numeric), sum))
    ) %>%
    dplyr::rename(
      Typhoon_name = GEN_typhoon_name,
      Municipality_code = GEN_mun_code,
      Municipality_name = GEN_mun_name
    )

  # Save to CSV
  write.csv(df_mun_impact_forecast,
    file = paste0(
      Output_folder, organization, "_municipality_breakdown_",
      forecast_time, "_", Typhoon_stormname, ".csv"
    ),
    row.names = FALSE
  )

  return(df_total_impact_forecast)
}


get_damage_probability <- function(damaged_houses, threshold) {
  return(round(100 * sum(damaged_houses >= threshold) / length(damaged_houses)))
}


get_percentile <- function(damaged_houses, percentile) {
  return(quantile(damaged_houses, c(1 - percentile)))
}
