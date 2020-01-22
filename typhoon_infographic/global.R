# Load all project libraries here
library(dplyr)
library(ggplot2)
library(plotly)
library(readr)
library(shiny)
library(shinydashboard)
library(stringr)

# Load all R functions in the resources folder
for (file in list.files('r_resources')){
  source(file.path('r_resources', file))
}

# Load in the data that should be globally available
df_raw <- read_csv(file.path('data', 'table_mockup_data.csv'))



