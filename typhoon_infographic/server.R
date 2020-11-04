#note: add these libraries
library(leaflet)
library(tmap)
library(readxl)
library(readr)
library(tidyverse)
library(shiny)
library(scales)

server <- function(input, output) {
  #################################################################################
  ### Define reactive elements (values could be read from csv in global.R)
  #################################################################################
  people_in_affected <- reactiveVal(9043000)
  people_in_worst_affected <- reactiveVal(3014000)
  vulnerable_people <- reactiveVal(961000)

  people_in_affected_temp <- 9043000
  people_in_worst_affected_temp <- 3014000
  vulnerable_people_temp <- 961000

  #################################################################################
  ### Functions to generate the colored human icons
  #################################################################################
  calc_n_affected_colors <- function(people_in_affected, people_in_worst_affected, vulnerable_people) {
    total_affected <- people_in_affected + people_in_worst_affected + vulnerable_people
    n1 <- round((people_in_affected / total_affected) * 15, 0)
    n2 <- round((people_in_worst_affected / total_affected) * 15, 0)
    n3 <- round((vulnerable_people / total_affected) * 15, 0)
    total_n <- n1 + n2 + n3
    if (total_n < 15) {
      n1 <- n1 + 1
    } else if (total_n > 15) {
      n1 <- n1 - 1
    }

    all_colors <- c(rep("human_grey", n1), rep("human_blue", n2), rep("human_orange", n3))

    return(all_colors)
  }


  affect_colors_to_divs <- function(affected_colors) {
    full_string <- ""
    i = 0
    for (color in affected_colors) {
      i = i+1
      full_string <- paste0(full_string, "<span class='human_icon_div'><i class='fas fa-male fa-3x icon_", color, "'></i></span>")
      if ((i %% 5) == 0) {
        full_string <- paste0(full_string, "<br />")
      }
    }
    return(full_string)
  }

  n_affected_colors <- reactive({
    calc_n_affected_colors(people_in_affected(), people_in_worst_affected(), vulnerable_people())
  })

  #################################################################################
  ### The top row is generated on the server side because it contains server information
  #################################################################################
  output$infographic_placeholder1 <- renderUI({
    withTags({
      div(
        div(
          id="colored_humans",
          HTML(
            affect_colors_to_divs(n_affected_colors())
        )
        ),
        div(
          id="toprow_text",
          div(
            id="affected_numbers",
            div(
              id="nr_div_people_in_affected",
              span(id="nr_span_people_in_affected", class="big_number", people_in_affected()), br(),
              span(id="nr_text_span_people_in_affected", "people living in affected areas")
            ),
            div(
              id="nr_div_people_in_worst_affected",
              span(id="nr_span_people_in_worst_affected", class="big_number",people_in_worst_affected()), br(),
              span(id="nr_text_span_people_in_worst_affected", "people living in worst affected areas")
            ),
            div(
              id="nr_div_vulnerable_people",
              span(id="nr_span_vulnerable_people", class="big_number",vulnerable_people()), br(),
              span(id="nr_text_span_vulnerable_people", "Vulnerable people in worst affected areas")
            )
          ),
          div(
            id="affected_explanation",
            p(
              id="affected_explanation_text",
              "\nAffected areas experienced moderate wind damage or higher. Worst affected areas experienced widespread damage or higher. Vulnerable population is estimated based on pre-existing socio-economic indicators"
            )
          )
        )
      )
    })
  })
  
  output$infographic_placeholder2 <- renderUI({
    withTags({
      div(
        div(
          id="colored_humans",
          HTML(
            affect_colors_to_divs(n_affected_colors())
          )
        ),    HTML(
          paste0(
            #'<span style="font-size:20px"> False Alarm Ratio: </span> ', fals_alarm_ratio_val(), "<br />",
            '<span style="font-size:30px"> people living in affected areas: </span> ', people_in_affected(), "<br />",
            '<span style="font-size:30px"> people living in worst affected areas: </span> ', people_in_worst_affected(), "<br />",
            '<span style="font-size:30px"> Vulnerable people in worst affected areas: </span> ', people_in_worst_affected(), "<br />","<br />","<br />",
            '<span style="font-size:20px"> Affected areas experienced moderate wind damage or higher: </span> ','', "<br />",
            '<span style="font-size:20px"> Worst affected areas experienced widespread damage or higher.: </span> ','', "<br />",
            '<span style="font-size:20px"> Vulnerable population is estimated based on pre-existing socio-economic indicators: </span> ','', "<br />"
            
            
          )
        )
    )
  })
})
        
  
  output$infographic_placeholder3 <- renderUI({
    HTML(
      paste0(
        #'<span style="font-size:20px"> False Alarm Ratio: </span> ', fals_alarm_ratio_val(), "<br />",
        '<span style="font-size:20px"> people living in affected areas: </span> ', people_in_affected(), "<br />",
        '<span style="font-size:20px"> people living in worst affected areas: </span> ', people_in_worst_affected(), "<br />",
        '<span style="font-size:20px"> Vulnerable people in worst affected areas: </span> ', people_in_worst_affected(), "<br />","<br />",
        '<span style="font-size:20px"> Affected areas experienced moderate wind damage or higher: </span> ','', "<br />",
        '<span style="font-size:20px"> Worst affected areas experienced widespread damage or higher.: </span> ','', "<br />",
        '<span style="font-size:20px"> Vulnerable population is estimated based on pre-existing socio-economic indicators: </span> ','', "<br />"


      )
    )
  })
  


  #################################################################################
  ### Charts of "VUNERABLE POPULATION BREAKDOWN" - second row
  #################################################################################

  #dataset about age population
  pop <- read.csv("data/population_census2015_disaggregated.csv", sep= ";")

  #suming up children population
  pop$children <- pop$F0.4 + pop$F5.9 + pop$F10.14 + pop$M0.4 + pop$M5.9 + pop$M10.14

  #suming up adult population
  pop$adults <- pop$F15.19 + pop$F20.24 + pop$F25.29 + pop$F30.34 + pop$F35.39 +
    pop$F40.44 + pop$F45.49 + pop$F50.54 + pop$F55.59 + pop$F60.64 +
    pop$M15.19 + pop$M20.24 + pop$M25.29 + pop$M30.34 + pop$M35.39 +
    pop$M40.44 + pop$M45.49 + pop$M50.54 + pop$M55.59 + pop$M60.64

  #suming up elderly population
  pop$elderly <- pop$F65.69 + pop$F70.74 + pop$F75.79 + pop$F80. +
    pop$M65.69 + pop$M70.74 + pop$M75.79 + pop$M80.

  #create a summarized table for the pie chart
  table_piechart <- pop %>%
    pivot_longer(
      cols = c("children", "adults", "elderly"),
      names_to = "group",
      values_to = "numbers"
    ) %>%
    group_by(group) %>%
    summarise(numbers = sum(numbers))


  #piechart 1 - Children Population
  output$chartChildren <- renderPlot({
  ggplot(table_piechart, aes(x= "", y = numbers, fill= group))+
    geom_bar(width = 1, stat = "identity") +
    coord_polar("y", start = 0) +
    theme_void() +
    theme(
      legend.position = "none")+
    scale_fill_manual(values = c("grey83", "red2", "grey50"))
    })

  #The number about the children populaton is a piece separated from the chart.
  #It must come, however, from the same dataset.
  output$number_children <- renderText({
    pop_children <- pop %>%
      summarise(children = sum(round(children)))
    comma(pop_children$children[1])
    })

  #piechart 2 - Adult Population
  output$chartAdults <- renderPlot({
    ggplot(table_piechart, aes(x= "", y = numbers, fill= group))+
    geom_bar(width = 1, stat = "identity") +
    coord_polar("y", start = 0) +
    theme_void() +
    theme(
      legend.position = "none")  +
    scale_fill_manual(values = c("red2", "grey83", "grey50"))
    })

  #The number about the adults populaton is a piece separated from the chart.
  #It must come, however, from the same dataset.
  output$number_adults <- renderText({
    pop_adults <- pop %>%
      summarise(adults = sum(round(adults)))
    comma(pop_adults$adults[1])
    })

  #piechart 3 - Elderly Population
  output$chartElderly <- renderPlot({
    ggplot(table_piechart, aes(x= "", y = numbers, fill= group))+
    geom_bar(width = 1, stat = "identity") +
    coord_polar("y", start = 0) +
    theme_void() +
    theme(
      legend.position = "none")  +
    scale_fill_manual(values = c("grey83", "grey50", "red2"))
    })

  #The number about the elderly populaton is a piece separated from the chart.
  #It must come, however, from the same dataset.
  output$number_elderly <- renderText({
    pop_elderly <- pop %>%
      summarise(elderly = sum(round(elderly)))
      comma(pop_elderly$elderly[1])
    })


  #################################################################################
  ### Boxes about food, shelter and water -third row
  #################################################################################

 #dataset
  vulnerable_pop <- read.csv("data/table_mockup_data.csv")

#sum up the vulnerable population in worst affected areas
 vuln_pop <- vulnerable_pop %>%
   summarise(vulnPeople = sum(Vulnerable.People.in.Worst.Affected.Areas))
 as.character(vuln_pop$vulnPeople[1])

 #function to format big numbers as million, billion and trillion
 format_big_num = function(n){
   case_when(
     n >= 1e12 ~ paste(round(n/1e12, digits = 2), 'T'),
     n >= 1e9 ~ paste(round(n/1e9, digits = 2), 'B'),
     n >= 1e6 ~ paste(round(n/1e6, digits = 2), 'M'),
     n >= 1e3 ~ paste(round(n/1e3, digits = 2), 'K'),
     TRUE ~ as.character(n))
 }

  #creating a variable for calories per day. Number of vulnerable people in
  #worst affected areas multiplied by number of calories they should consume
  #per day (1800) and using the function to format the output in a readable way
  calories_perday <- format_big_num(vuln_pop * 1800)

  #the same goes for square meters of shelter
  sqm_shelter <- format_big_num(vuln_pop * 5)

  #the same goes for liters of water per day
  liters_water <- format_big_num(vuln_pop * 2)

 #Food box
 output$foodbox <- renderValueBox({
    valueBox(
      paste0(calories_perday),
      "calories per day",
      icon = icon("utensils"),
      color = "red"
    )
  })

  #Shelter box
   output$shelterbox <- renderValueBox({
    valueBox(
      paste0(sqm_shelter),
      "SQM of shelter",
      icon = icon("home"),
      color = "red"
    )
  })

  #Water Box
  output$waterbox <- renderValueBox({
    valueBox(
      paste0(liters_water),
      "liters of water per day",
      icon = icon("tint"),
      color = "red"
    )
  })

  #################################################################################
  ### Map
  #################################################################################

  output$map1 = renderLeaflet({
    #data for test. change for the real one
    data("World")
    impact
    
    impact_map<-tm_shape(impact) + 
      tm_fill(col = "impact",showNA=FALSE, border.col = "black",lwd = 3,lyt='dotted',
              breaks = c(0,0.1,1,2,5,9.5,10),
              title='Predicted % of Damaged ',
              labels=c(' No Damage',' < 1%',' 1 to 2%',' 2 to 5%',' 5 to 10%',' > 10%'),
              palette = c('#ffffff','#fdd0a2','#fdae6b','#fd8d3c','#e6550d','#a63603')) + #,style = "cat")+
      
      tm_borders(col = NA, lwd = .25, lty = "solid", alpha = .25, group = NA) +
      #tm_polygons(col = "dam_perc_comp_prediction_lm_quantile", border.col = "black",lwd = 0.1,lyt='dotted',style = "cat")+
      #tm_shape(my_track) + tm_symbols(size=0.1,border.alpha = .25,col="blue") +
      #tm_shape(Landfall_point) + tm_symbols(size=0.25,border.alpha = .25,col="red") +   
      #tm_scale_bar(breaks = c(0, 100, 200), text.size = .5, color.light = "#f0f0f0",position = c(0,0))+
      tm_layout(legend.outside= TRUE, legend.outside.position=c("left"))
    
    
    
     #tm <- tm_shape(World) +   tm_polygons("HPI")
     #trick to display interactive maps: library leaflet
     tmap_leaflet(impact_map)
    })

  #################################################################################
  ### Table in the botton
  #################################################################################

  output$information_table <- DT::renderDataTable({
    df_raw
  })
 }



