#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
suppressMessages(library(dplyr))
suppressMessages(library(tidyr))
suppressMessages(library(tmap))
suppressMessages(library(sf))
suppressMessages(library(geojsonsf))
suppressMessages(library(raster))
suppressMessages(library(rlang))
suppressMessages(library(lubridate))
suppressMessages(library(ncdf4))
suppressMessages(library(huxtable))
suppressMessages(library(xgboost))
suppressMessages(library(readr))
rainfall_error <- args[1]
sf_use_s2(FALSE)
# path='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/IBF-Typhoon-model/'
path <- "/home/fbf/"
# path='./'
main_directory <- path

###########################################################################
# ------------------------ import DATA  -----------------------------------
setwd(path)
source("lib_r/settings.R")
source("lib_r/data_cleaning_forecast.R")
source("lib_r/prepare_typhoon_input.R")

source("lib_r/track_interpolation.R")


source("lib_r/Read_rainfall_v2.R")
source("lib_r/Model_input_processing.R")
source("lib_r/run_prediction_model.R")
source("lib_r/Make_maps.R")

source("lib_r/Check_landfall_time.R")
source("lib_r/damage_probability.R")

# ------------------------ import DATA  -----------------------------------

# php_admin3 <- st_read(dsn=paste0(main_directory,'data-raw'),layer='phl_admin3_simpl2')
php_admin3 <- geojsonsf::geojson_sf(
  paste0(main_directory, "data-raw/phl_admin3_simpl2.geojson")
)

# php_admin1 <- st_read(dsn=paste0(main_directory,'data-raw'),layer='phl_admin1_gadm_pcode')
php_admin1 <- geojsonsf::geojson_sf(
  paste0(main_directory, "data-raw/phl_admin1_gadm_pcode.geojson")
)

wshade <- php_admin3
material_variable2 <- read.csv(
  paste0(main_directory, "data/material_variable2.csv")
)
data_matrix_new_variables <- read.csv(
  paste0(main_directory, "data/data_matrix_new_variables.csv")
)
geo_variable <- read.csv(
  paste0(main_directory, "data/geo_variable.csv")
)

wshade <- php_admin3
# load the rr model
# mode_classification <- readRDS(paste0(main_directory,"./models/final_model.rds"))
# mode_continious <- readRDS(paste0(main_directory,"./models/final_model_regression.rds"))
# mode_classification1 <- readRDS(paste0(main_directory,"./models/xgboost_classify.rds"))

xgmodel <- readRDS(
  paste0(main_directory, "/models/operational/xgboost_regression_v2.RDS"),
  refhook = NULL
)

# load forecast data
typhoon_info_for_model <- read.csv(
  paste0(main_directory, "/forecast/Input/typhoon_info_for_model.csv")
)
# typhoon_events <- read.csv(paste0(main_directory,'/forecast/Input/typhoon_info_for_model.csv'))


rain_directory <- as.character(
  typhoon_info_for_model[typhoon_info_for_model[["source"]] == "Rainfall", ][["filename"]]
)
windfield_data <- as.character(
  typhoon_info_for_model[typhoon_info_for_model[["source"]] == "windfield", ][["filename"]]
)
ECMWF_ <- as.character(
  typhoon_info_for_model[typhoon_info_for_model[["source"]] == "ecmwf", ][["filename"]]
)
TRACK_DATA <- read.csv(ECMWF_) # %>%dplyr::mutate(STORMNAME=Typhoon_stormname, YYYYMMDDHH=format(strptime(YYYYMMDDHH, format = "%Y-%m-%d %H:%M:%S"), '%Y%m%d%H%00'))

Output_folder <- as.character(
  typhoon_info_for_model[typhoon_info_for_model[["source"]] == "Output_folder", ][["filename"]]
)
forecast_time <- as.character(
  typhoon_info_for_model[typhoon_info_for_model[["source"]] == "ecmwf", ][["time"]]
)

#------------------------- define functions ---------------------------------

ntile_na <- function(x, n) {
  notna <- !is.na(x)
  out <- rep(NA_real_, length(x))
  out[notna] <- ntile(x[notna], n)
  return(out)
}



####################################################################################################
# ------------------------ prepare model input   -----------------------------------



wind_grid <- read.csv(windfield_data) %>%
  dplyr::mutate(
    dis_track_min = ifelse(dis_track_min < 1, 1, dis_track_min),
    Mun_Code = adm3_pcode,
    pcode = as.factor(substr(adm3_pcode, 1, 10))
  )

rainfall_ <- Read_rainfall_v2(wshade)


typhoon_hazard <- wind_grid %>%
  left_join(rainfall_, by = "Mun_Code") %>%
  dplyr::mutate(
    typhoon_name = name,
    rainfall_24h = rainfall_24h,
    ranfall_sum = rainfall_24h,
    dist_track = dis_track_min,
    gust_dur = 0,
    sust_dur = 0,
    vmax_gust = v_max * 1.49, # sus to gust convrsion 1.49 -- 10 min average
    vmax_gust_mph = v_max * 1.49 * 2, 23694, # mph 1.9 is factor to drive gust and sustained wind
    vmax_sust_mph = v_max * 2, 23694,
    vmax_sust = v_max
  ) %>%
  # 1.21 is conversion factor for 10 min average to 1min average
  dplyr::select(
    Mun_Code, vmax_gust,
    vmax_gust_mph, vmax_sust_mph,
    vmax_sust, dist_track,
    rainfall_24h, gust_dur,
    sust_dur, ranfall_sum,
    storm_id, typhoon_name
  )


# BUILD DATA MATRIC FOR NEW TYPHOON
data_new_typhoon1 <- geo_variable %>%
  left_join(material_variable2 %>%
    dplyr::select(
      -Region,
      -Province,
      -Municipality_City
    ), by = "Mun_Code") %>%
  left_join(data_matrix_new_variables, by = "Mun_Code") %>%
  left_join(typhoon_hazard, by = "Mun_Code") %>%
  na.omit()


typhoon_data_cleaned <- clean_typhoon_forecast_data_ensamble(data_new_typhoon1) # %>%na.omit() # Randomforests don't handle NAs, you can impute in the future

model_input <- typhoon_data_cleaned %>% dplyr::select(
  -GEN_typhoon_name,
  -GEN_typhoon_id,
  -GEO_n_households,
  -GEN_mun_code,
  -index,
  #-GEN_mun_code,
  #-contains("INT_"),
  -contains("DAM_"),
  -GEN_mun_name
)




####################################################################################################
# ------------------------run prediction   -----------------------------------


test_x <- data.matrix(model_input)
xgb_test <- xgb.DMatrix(data = test_x)
y_predicted <- predict(xgmodel, xgb_test)



df_impact_forecast <- as.data.frame(y_predicted) %>%
  dplyr::mutate(
    index = 1:length(y_predicted),
    impact = y_predicted
  ) %>%
  left_join(typhoon_data_cleaned, by = "index") %>%
  dplyr::mutate(
    dist50 = ifelse(WEA_dist_track >= 50, 0, 1),
    e_impact = ifelse(impact > 100, 100, impact),
    region = substr(GEN_mun_code, 1, 4),
    Damaged_houses = as.integer(GEO_n_households * e_impact * 0.01),
  ) %>%
  filter(WEA_dist_track < 500) %>%
  dplyr::select(
    index,
    region,
    GEN_mun_code,
    GEN_mun_name,
    GEO_n_households,
    GEN_typhoon_name,
    GEN_typhoon_id,
    WEA_dist_track,
    WEA_vmax_sust_mhp,
    e_impact,
    dist50,
    Damaged_houses,
  ) %>%
  drop_na() %>%
  # Add 0 damage tracks to any municipalities with missing members
  # Note that after this step the index is NA for the 0 damage members
  tidyr::complete(
    GEN_typhoon_id, 
    nesting(
      region,
      GEN_mun_code,
      GEN_mun_name,
      GEO_n_households,
      GEN_typhoon_name
    ),
    fill = list(
      WEA_vmax_sust_mhp = 0,
      e_impact = 0,
      dist50 = 0,
      Damaged_houses = 0
    )
  )

# For debugging if needed: a boxplot showing ensemble distribution of
# damanged houses per municipality
# boxplot(Damaged_houses~GEN_mun_code, data=df_impact_forecast, range=0, las=2)

Typhoon_stormname <- as.character(unique(wind_grid[["name"]])[1])

####################################################################################################
# ------------------------ calculate  probability only for region 5 and 8  -----------------------------------

# Only select regions 5 and 8
cerf_regions <- c("PH05", "PH08")
cerf_damage_thresholds <- c(80000, 50000, 30000, 10000, 5000)
cerf_probabilities <- c(0.95, 0.80, 0.70, 0.60, 0.50)

df_impact_forecast_CERF <- get_total_impact_forecast(
  df_impact_forecast %>% filter(region %in% cerf_regions),
  cerf_damage_thresholds, cerf_probabilities, "CERF"
)


####################################################################################################
# ------------------------ calculate and plot probability National -----------------------------------

dref_damage_thresholds <- c(100000, 80000, 70000, 50000, 30000)
dref_probabilities <- c(0.95, 0.80, 0.70, 0.60, 0.50)

df_impact_forecast_DREF <- get_total_impact_forecast(
  df_impact_forecast, dref_damage_thresholds, dref_probabilities, "DREF"
)


# ------------------------ calculate average impact vs probability   -----------------------------------

n_ensemble <- length(unique(df_impact_forecast[["GEN_typhoon_id"]])) # usually 52

df_impact_dist50 <- aggregate(
  df_impact_forecast[["dist50"]],
  by = list(GEN_mun_code = df_impact_forecast[["GEN_mun_code"]]),
  FUN = sum
) %>%
  dplyr::mutate(probability_dist50 = 100 * x / n_ensemble) %>%
  dplyr::select(GEN_mun_code, probability_dist50) %>%
  left_join(
    aggregate(
      df_impact_forecast[["e_impact"]],
      by = list(GEN_mun_code = df_impact_forecast[["GEN_mun_code"]]),
      FUN = sum
    ) %>%
      dplyr::mutate(impact = x / n_ensemble) %>%
      dplyr::select(GEN_mun_code, impact),
    by = "GEN_mun_code"
  ) %>%
  left_join(
    aggregate(
      df_impact_forecast[["WEA_dist_track"]],
      by = list(GEN_mun_code = df_impact_forecast[["GEN_mun_code"]]),
      FUN = sum
    ) %>%
      dplyr::mutate(WEA_dist_track = x / n_ensemble) %>%
      dplyr::select(GEN_mun_code, WEA_dist_track),
    by = "GEN_mun_code"
  )

df_impact <- df_impact_forecast %>%
  left_join(df_impact_dist50, by = "GEN_mun_code")


####################################################################################################


# ------------------------ calculate and plot probability   -----------------------------------

# df_impact_forecast%>%group_by(GEN_typhoon_id)%>%
# dplyr::summarise(CDamaged_houses = sum(Damaged_houses))%>%
# dplyr::mutate(DM_CLASS = ifelse(CDamaged_houses >= 100000,4,
# ifelse(CDamaged_houses >= 80000,3,
# ifelse(CDamaged_houses >= 50000,2,
# ifelse(CDamaged_houses >= 30000,1, 0)))))%>%
# ungroup()%>%dplyr::summarise(VH_100K = round(100*sum(DM_CLASS>=4)/52),
# H_80K = round(100*sum(DM_CLASS>=3)/52),
# M_50K = round(100*sum(DM_CLASS >=2)/52),
# L_30K = round(100*sum(DM_CLASS>=1)/52))#%>%as_hux()%>%set_text_color(1, everywhere, "blue")%>%theme_article()%>%set_caption("PROBABILITY FOR THE NUMBER OF COMPLETELY DAMAGED BUILDINGS")






# ------------------------ calculate probability   -----------------------------------

event_impact <- php_admin3 %>%
  left_join(
    df_impact_dist50 %>%
      dplyr::mutate(adm3_pcode = GEN_mun_code),
    by = "adm3_pcode"
  )

track <- track_interpolation(TRACK_DATA) %>%
  dplyr::mutate(Data_Provider = "ECMWF_HRS")

maps <- Make_maps_avg(php_admin1,
  event_impact,
  track,
  TYF = "ECMWF",
  Typhoon_stormname
)

####################################################################################################
# ------------------------ save impact data to file   -

tmap_save(maps,
  filename = paste0(Output_folder, "Average_Impact", "_", forecast_time, "_", Typhoon_stormname, ".png"),
  width = 20, height = 24, dpi = 600, units = "cm"
)

####################################################################################################
# ------------------------ save impact data to file   -----------------------------------

write.csv(data.frame(event_impact) %>% dplyr::select(-c(geometry)),
  file = paste0(Output_folder, "Average_Impact", "_", forecast_time, "_", Typhoon_stormname, ".csv")
)
