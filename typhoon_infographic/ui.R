
header <- dashboardHeader(
  title = "IBF TYPHOON",
  # Trick to put logo in the corner
  tags$li(div(
    class="logo_div",
    img(src = 'https://www.510.global/wp-content/uploads/2017/07/510-LOGO-WEBSITE-01.png',#'510logo.png', 
        title = "logo", height = "44px")),
    class = "dropdown"),
  tags$li(div(
    class="logo_div",
    img(src = 'germanRC-.png',
        title = "logo", height = "44px")),
    class = "dropdown"),
  tags$li(div(
    class="logo_div",
    img(src = 'PhilRC.jpg',
        title = "logo", height = "44px")),
    class = "dropdown")
)


ui_tab_main <- tabItem(
  "tab_main",
  
   #first row - colored men, text and numbers
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
      
      #second row - vunerable population breakdown
      fluidRow(
        box(title = "VUNERABLE POPULATION BREAKDOWN",
            width = 12,
        column(
          width=2,
          plotOutput(outputId = "chartChildren", height=150)
        ),
        column(
          width=2,
          h3(textOutput(outputId = "number_children"),
             style = "color: red; font-weight: bold;"),
          tags$h5("children (<15)")
        ),
        column(
          width=2,
          plotOutput(outputId = "chartAdults", height=150)
        ),
        column(
          width=2,
          h3(textOutput(outputId = "number_adults"),
             style = "color: red; font-weight: bold;"),
            tags$h5("adults (15-65)")
        ),
        column(
          width=2,
          plotOutput(outputId= "chartElderly", height=150)
        ),
        column(
          width=2,
          h3(textOutput(outputId = "number_elderly"),
             style = "color: red; font-weight: bold;"),
          tags$h5("elderly (>65)")
               )
        )
      ),
      
      #third row - value boxes (food, shelter, water)
      fluidRow(
        id = "food_shelter_water_boxes",
        column(
          width=12,
          fluidRow(
            valueBoxOutput("foodbox"),
            valueBoxOutput("shelterbox"),
            valueBoxOutput("waterbox")
          )
        )
      )
    ),
    #map
    column(  width = 5,
      leafletOutput("map1", height="615px", width="900px")
    )
  ),
  #table
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