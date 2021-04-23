#------------------------  Load required packages ------------------------------
# Load required packages: 
library(dplyr)
library(ranger)
library(caret)
library(randomForest)
library(broom)
library(tidyr)
library(stringr)

clean_typhoon_data <- function() {
  #-------------------------Examine final dataset---------------------------------

# rangers for randowm forest 

#-------------------------import dataset---------------------------------

########################################################################################################## hazard variables 
  
wind_full_grid_27 <- read.csv("data/wind_full_grid_27.csv")
wind<- wind_full_grid_27 %>%
  dplyr::mutate(ty_pcode= paste0(wind_full_grid_27$Mun_Code,tolower(str_sub(wind_full_grid_27$typhoon_name,-16,-5))))

all_rainfall <- read.csv("data/all_rainfall.csv")
rainfall<- all_rainfall %>%
  dplyr::mutate(ty_pcode= paste0(all_rainfall$Mun_Code,tolower(all_rainfall$typhoon)))

HAZARD_DATA<-wind %>%
  left_join(rainfall, by = "ty_pcode")

########################################################################################################## impact data

IMpact_data_philipines_final <- read.csv("data/IMpact_data_philipines_final2.csv",sep=",",encoding="UTF-8")

IMpactl<- IMpact_data_philipines_final %>%
  dplyr::mutate(Mun_Code=pcode,ty_pcode= paste0(IMpact_data_philipines_final$pcode,tolower(IMpact_data_philipines_final$typhoon))) %>%
  dplyr::select(pcode,typhoon,Year,Totally,Partially,total,ty_pcode)

#IMpact2<- IMpactl %>%left_join(material_variable2 %>% dplyr::select(Mun_Code,Housing.Units) , by = "Mun_Code") %>%  dplyr::mutate(Dam_per=(Totally/Housing.Units))
  
########################################################################################## hazard impact catalog for historical events 

hazard_impact<-HAZARD_DATA %>%
  left_join(IMpactl, by = "ty_pcode")

hazard_impact_catalogue<-hazard_impact[!is.na(hazard_impact$Year),] %>%
  dplyr::mutate(Mun_Code= gridid,typhoon=typhoon.x) %>%
  dplyr::select(Mun_Code,
                gridid,
                typhoon_name,
                typhoon,
                vmax_gust,
                vmax_sust,
                vmax_gust_mph,
                vmax_sust_mph,
                sust_dur,
                gust_dur,
                dist_track ,
                rainfll_max,
                ranfall_sum,
                Year,
                Totally,
                Partially,
                total)

########################################################################################################## Geo varialbles 

geo_variable <- read.csv("data/geo_variable.csv")
data_matrix_new_variables <- read.csv("data/data_matrix_new_variables.csv")

########################################################################################################## vulnerability data
material_variable2 <- read.csv("data/material_variable2.csv")
Poverty_2012 <- read.csv("data/Poverty_2012.csv")
population <- read.csv("data/population_2015.csv")


static_data3<-population %>% 
  dplyr::mutate(perc_u5_and_o60=100*(pop_u5_and_o60/total_pop),Mun_Code=pcode) %>%
  left_join(Poverty_2012, by = "pcode")%>% 
  left_join(geo_variable %>% dplyr::select(-Mun_name), by = "Mun_Code") %>%
  left_join(material_variable2 %>% dplyr::select(-Region) , by = "Mun_Code") %>%
  left_join(data_matrix_new_variables %>% dplyr::select(-Mun_Name) , by = "Mun_Code") # %>%   left_join(IMpactl , by = "pcode")



#-------------------------combne and clean dataset---------------------------------
#
datamatrix_Hazard_impact_static<-hazard_impact_catalogue %>%
  left_join(static_data3, by = "Mun_Code")


#-------------------------group dataset---------------------------------
data.training <- datamatrix_Hazard_impact_static %>%
  dplyr::arrange(typhoon_name,Mun_Code) %>%
  dplyr::mutate(
    index                          = 1:nrow(datamatrix_Hazard_impact_static),
    GEN_typhoon_name_year         = typhoon_name,
    GEN_typhoon_name               = typhoon,
    GEO_landslide                  =landslide_per,
    GEO_landslide_y               =Yel_per_LSSAb,
    GEO_landslide_o                 =Or_per_LSblg,
    GEO_landslide_r                 =Red_per_LSbldg,
    GEO_stormsurge                 =stormsurge_per,
    GEO_stormsurge_y                 =Yellow_per_LSbl,
    GEO_stormsurge_o                 =OR_per_SSAbldg,
    GEO_stormsurge_r                 =RED_per_SSAbldg,
    WEA_gust_dur                   =gust_dur,
    WEA_sust_dur                   =sust_dur,
    WEA_vmax_gust                  =vmax_gust,
    WEA_vmax_sust                  =vmax_sust,
    WEA_vmax_gust_mhp              =vmax_gust_mph,
    WEA_vmax_sust_mhp              =vmax_sust_mph,
    WEA_vmax_gust_sqr              =vmax_gust_mph^2,
    WEA_vmax_sust_sqr              =vmax_sust_mph^2,
    WEA_vmax_sust_cube             =vmax_sust_mph^3,
    WEA_vmax_gust_cube             =vmax_gust_mph^3,
    WEA_dist_track                 =dist_track,
    WEA_vmax_sust_mhp              =vmax_sust_mph,
    GEN_mun_code                   = Mun_Code,
    GEN_prov_name                  = Province,
    GEN_mun_name                   = Municipality_City,
    DAM_comp_houses                = Totally,
    DAM_part_houses                = Partially,
    DAM_tot_houses                 = total,
    #DAM_tot_houses_0p25weight      = total_damage_houses_0p25weight,
    DAM_comp_houses_perc           = ifelse(100*(Totally/Housing.Units) > 100, 100,100*(Totally/Housing.Units)) ,
    DAM_part_houses_perc           = ifelse(100*(Partially/Housing.Units) > 100, 100,100*(Partially/Housing.Units)) ,
    DAM_tot_houses_perc            = ifelse(100*(total/Housing.Units) > 100, 100,100*(total/Housing.Units)) ,
    #DAM_tot_houses_0p25weight_perc = total_damage_houses_0p25weight_perc,
    WEA_windspeed                  = vmax_sust_mph,
    WEA_windspeed_squared          = vmax_sust_mph^2,
    WEA_windspeed_cube             = vmax_sust_mph^3,
    WEA_rainfall_max               = rainfll_max,
    WEA_rainfall_sum               = ranfall_sum,
    WEA_rainfall_max1             = ifelse(WEA_rainfall_max > 350, 1,0), #ifelse(WEA_rainfall_max <= 350, 0, 0)),
    #POS_distance_first_impact      = distance_first_impact,
    #POS_distance_typhoon           = distance_typhoon_km,
    GEO_slope_mean                 = mean_slope,
    GEO_slope_stdev                = slope_stdev,
    GEO_ruggedness_mean            = mean_ruggedness,
    GEO_ruggedness_stdev           = ruggedness_stdev,
    GEO_elevation_mean             = mean_elevation_m,
    GEO_coast_length               = ifelse(is.na(coast_length),0,coast_length),
    GEO_coast_yn                   = with_coast,
    GEO_Area                       = area_km2,
    GEO_perimeter                  = perimeter,
    GEO_poverty_incidence           = Poverty_incidence,
    #GEO_poverty_perc               = poverty_perc,
    #GEO_pop_density_15             = pop_density_15,pop_u5_and_o60,
    GEO_pop_density                 = total_pop/area_km2,
    GEO_pop_density_u5_and_o60      = pop_u5_and_o60/area_km2,
    GEO_pop_u5_and_o60      = pop_u5_and_o60,
    #GEO_pop_15                     = pop_15,
    GEO_n_households               = Housing.Units,
    #GEO_coast_altitude             = ifelse(GEO_coast_yn ==1 & GEO_elevation_mean > 100, 'High coast',
    #                                        ifelse(GEO_coast_yn==1 & GEO_elevation_mean <= 100, 'Low coast', 'No coast')),
    MAT_Strong.Roof.Strong.Wall                =100*(Strong.Roof.Strong.Wall/Housing.Units) ,
    MAT_Strong.Roof.Light.Wall                 =100*(Strong.Roof.Light.Wall/Housing.Units),
    MAT_Strong.Roof.Salvage.Wall               =100*(Strong.Roof.Salvage.Wall/Housing.Units),
    MAT_Light.Roof.Strong.Wall                 =100*(Light.Roof.Strong.Wall/Housing.Units),
    MAT_Light.Roof.Light.Wall                  =100*(Light.Roof.Light.Wall/Housing.Units),
    MAT_Light.Roof.Salvage.Wall                =100*(Light.Roof.Salvage.Wall/Housing.Units),
    MAT_Salvaged.Roof.Strong.Wall              =100*(Salvaged.Roof.Strong.Wall/Housing.Units),
    MAT_Salvaged.Roof.Light.Wall               =100*(Salvaged.Roof.Light.Wall/Housing.Units),
    MAT_Salvaged.Roof.Salvage.Wall             =100*(Salvaged.Roof.Salvage.Wall/Housing.Units),
    #MAT_wall_bamboo                = wall_bamboo,
    #MAT_wall_makeshift             = wall_makeshift,
    #MAT_wall_wood                  = wall_wood,
    #MAT_wall_iron                  = wall_iron,
    #MAT_wall_conc                  = wall_conc,
    #MAT_wall_sum                   = sum_wall,
    #MAT_roof_straw                 = roof_straw,
    #MAT_roof_makeshift             = roof_makeshift,
    #MAT_roof_wood                  = roof_wood,
    #MAT_roof_iron                  = roof_iron,
    #MAT_roof_half_conc             = roof_half_conc,
    #MAT_roof_conc                  = roof_conc,
    #MAT_roof_sum                   = sum_roof,
    INT_rainfall_landslide          = landslide_per * rainfll_max,
    INT_windspeed_Strong.Roof.Strong.Wall            = vmax_sust_mph * Strong.Roof.Strong.Wall,  
    INT_windspeed_Strong.Roof.Light.Wall             = vmax_sust_mph * Strong.Roof.Light.Wall,
    INT_windspeed_Strong.Roof.Salvage.Wall           = vmax_sust_mph * Strong.Roof.Salvage.Wall,
    INT_windspeed_Light.Roof.Strong.Wall             = vmax_sust_mph * Light.Roof.Strong.Wall,  
    INT_windspeed_Light.Roof.Light.Wall              = vmax_sust_mph * Light.Roof.Light.Wall,
    INT_windspeed_Light.Roof.Salvage.Wall            = vmax_sust_mph * Light.Roof.Salvage.Wall,
    INT_windspeed_Salvaged.Roof.Strong.Wall          = vmax_sust_mph * Salvaged.Roof.Strong.Wall,  
    INT_windspeed_Salvaged.Roof.Light.Wall           = vmax_sust_mph * Salvaged.Roof.Light.Wall,
    INT_windspeed_Salvaged.Roof.Salvage.Wall         = vmax_sust_mph * Salvaged.Roof.Salvage.Wall,
    INT_windspeed_cube_Strong.Roof.Strong.Wall       = vmax_sust_mph^3  * Strong.Roof.Strong.Wall,  
    INT_windspeed_cube_Strong.Roof.Light.Wall        = vmax_sust_mph^3  * Strong.Roof.Light.Wall,
    INT_windspeed_cube_Strong.Roof.Salvage.Wall      = vmax_sust_mph^3  * Strong.Roof.Salvage.Wall,
    INT_windspeed_cube_Light.Roof.Strong.Wall        = vmax_sust_mph^3  * Light.Roof.Strong.Wall,  
    INT_windspeed_cube_Light.Roof.Light.Wall         = vmax_sust_mph^3  * Light.Roof.Light.Wall,
    INT_windspeed_cube_Light.Roof.Salvage.Wall       = vmax_sust_mph^3  * Light.Roof.Salvage.Wall,
    INT_windspeed_cube_Salvaged.Roof.Strong.Wall     = vmax_sust_mph^3  * Salvaged.Roof.Strong.Wall,  
    INT_windspeed_cube_Salvaged.Roof.Light.Wall      = vmax_sust_mph^3  * Salvaged.Roof.Light.Wall,
    INT_windspeed_cube_Salvaged.Roof.Salvage.Wall    = vmax_sust_mph^3  * Salvaged.Roof.Salvage.Wall,
    INT_windspeed_coast_yn                           = vmax_sust_mph * with_coast,
    INT_windspeed_cube_coast_yn                      = vmax_sust_mph^3 * with_coast,               
    INT_coast_yn_elevation                           = with_coast * mean_elevation_m
    
  ) %>%
  dplyr::select (index,
                 GEN_mun_code,
                 GEN_prov_name,
                 GEN_mun_name,
                 GEN_typhoon_name,
                 GEN_typhoon_name_year,
                 GEO_n_households,
                 DAM_comp_houses_perc,
                 DAM_comp_houses,
                 DAM_part_houses,
                 DAM_tot_houses,
                 DAM_part_houses_perc,
                 DAM_tot_houses_perc,
                 GEO_landslide,
                 GEO_landslide_y,
                 GEO_landslide_o,
                 GEO_landslide_r,
                 GEO_stormsurge,
                 GEO_stormsurge_y,
                 GEO_stormsurge_o,
                 GEO_stormsurge_r,
                 WEA_gust_dur,
                 WEA_sust_dur,
                 #WEA_vmax_gust,
                 #WEA_vmax_sust,
                 #WEA_vmax_gust_mhp,
                 WEA_vmax_sust_mhp,
                 #WEA_vmax_gust_sqr,
                 WEA_vmax_sust_sqr,
                 WEA_vmax_sust_cube,
                 #WEA_vmax_gust_cube,
                 WEA_dist_track,
                 #WEA_vmax_sust_mhp,
                 #WEA_windspeed,
                 #WEA_windspeed_squared,
                 #WEA_windspeed_cube,
                 WEA_rainfall_max,
                 #WEA_rainfall_max1,
                 #WEA_rainfall_sum,
                 GEO_slope_mean,
                 GEO_slope_stdev,
                 GEO_ruggedness_mean,
                 GEO_ruggedness_stdev,
                 GEO_elevation_mean,
                 GEO_coast_length,
                 GEO_coast_yn,
                 GEO_Area,
                 GEO_perimeter,
                 #GEO_poverty_perc,
                 GEO_poverty_incidence,
                 GEO_pop_density,
                 GEO_pop_density_u5_and_o60,
                 GEO_pop_u5_and_o60,
                 MAT_Strong.Roof.Strong.Wall,
                 MAT_Strong.Roof.Light.Wall,
                 MAT_Strong.Roof.Salvage.Wall,
                 MAT_Light.Roof.Strong.Wall,
                 MAT_Light.Roof.Light.Wall,
                 MAT_Light.Roof.Salvage.Wall,
                 MAT_Salvaged.Roof.Strong.Wall,
                 MAT_Salvaged.Roof.Light.Wall,
                 MAT_Salvaged.Roof.Salvage.Wall,
                 INT_rainfall_landslide,
                 INT_windspeed_Strong.Roof.Strong.Wall,
                 INT_windspeed_Strong.Roof.Light.Wall,
                 INT_windspeed_Strong.Roof.Salvage.Wall,
                 INT_windspeed_Light.Roof.Strong.Wall,
                 INT_windspeed_Light.Roof.Light.Wall,
                 INT_windspeed_Light.Roof.Salvage.Wall,
                 INT_windspeed_Salvaged.Roof.Strong.Wall,
                 INT_windspeed_Salvaged.Roof.Light.Wall,
                 INT_windspeed_Salvaged.Roof.Salvage.Wall,
                 INT_windspeed_cube_Strong.Roof.Strong.Wall,
                 INT_windspeed_cube_Strong.Roof.Light.Wall,
                 INT_windspeed_cube_Strong.Roof.Salvage.Wall,
                 INT_windspeed_cube_Light.Roof.Strong.Wall,
                 INT_windspeed_cube_Light.Roof.Light.Wall,
                 INT_windspeed_cube_Light.Roof.Salvage.Wall,
                 INT_windspeed_cube_Salvaged.Roof.Strong.Wall,
                 INT_windspeed_cube_Salvaged.Roof.Light.Wall,
                 INT_windspeed_cube_Salvaged.Roof.Salvage.Wall,
                 INT_windspeed_coast_yn,
                 INT_windspeed_cube_coast_yn,
                 INT_coast_yn_elevation)
return(data.training)
}
