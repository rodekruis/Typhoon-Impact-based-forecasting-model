library(dplyr)
library(devtools)
library(stringi)

# Read in and clean up `county_centers`
county_points_php2 <- read.csv("data-raw/county_points_php2.csv",
                               sep=";")%>%
  mutate(fips =ADM4_PCODE,
         COUNAME = stri_trans_general(Barangay, "latin-ascii")) %>%
  select(fips, Lat, Lon, Manucipali, pop_Popula) %>%
  rename(gridid = fips,
         state_name = Manucipali,
         gpop = pop_Popula,
         glat = Lat, 
         glon = Lon) %>%
  select(gridid, glat, glon)

use_data(county_points, overwrite = TRUE)
