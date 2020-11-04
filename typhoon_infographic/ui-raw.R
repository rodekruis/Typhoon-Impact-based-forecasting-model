header <- dashboardHeader(
  title = "Typhoon Kammuri",
  # Trick to put logo in the corner
  tags$li(div(
    class="logo_div",
    img(src = 'logo510-.jpg',
        title = "logo", height = "44px")),
    class = "dropdown")
)

ui_tab_main <- tabItem(
  "tab_main",
  fluidRow(
    id = "main_top_row",
    column(
      id = "infographic_column",
      width = 6,
      fluidRow(
        id = "infographic_toprow",
        column(
          width=12,
          uiOutput("infographic_placeholder1")
        )
      ),
      fluidRow(
        id = "infographic_middlerow",
        column(
          width=12,
          textOutput("")
        )
      ),
      fluidRow(
        id = "infographic_bottomrow",
        column(
          width=12,
          textOutput("")
        )
      )
    ),
    column(
      width = 6,
      img(src="typhoon_map_example.png", height="600px", width="600px")
    )
  ),
  fluidRow(
    column(
      width = 12,
      DT::DTOutput("information_table")
    )
  )
)

body <- dashboardBody(
  # Loads CSS and JS from www/custom.css in
  tags$head(tags$link(rel = "stylesheet",
                      type = "text/css", href = "custom.css")),
  tags$head(tags$script(src="main.js")),
  tabItems(
    ui_tab_main
  )
)

ui <- dashboardPage(
  header,
  dashboardSidebar(
    collapsed=T,
    sidebarMenu(
      menuItem("Main Tab", tabName = "tab_main")
    )
  ),
  body
)
