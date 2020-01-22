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
      full_string <- paste0(full_string, "<span class='human_icon_div'><i class='fas fa-male fa-5x icon_", color, "'></i></span>")
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
              "Affected areas experienced moderate wind damage or higher. Worst affected areas experienced widespread damage or higher. Vulnerable population is estimated based on pre-existing socio-economic indicators"
            )
          )
        )
      )
    })
  })

  #################################################################################
  ### Placeholders for now, can be filled with actual visualisations
  #################################################################################
  output$infographic_placeholder2 <- renderText({
    "infographic_placeholder2"
  })

  output$infographic_placeholder3 <- renderText({
    "infographic_placeholder3"
  })

  output$information_table <- DT::renderDataTable({
    df_raw
  })
}


